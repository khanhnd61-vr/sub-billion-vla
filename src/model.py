"""
model.py - the VLA-Adapter model wrapper

HYBRID design: the heavy, well-tested backbone (DINOv2+SigLIP vision, Qwen2.5-0.5B LLM, the
multimodal assembly, and the learnable ActionQuery tokens) is REUSED from the parent
VLA-Adapter repo's `prismatic` package. This file owns the *method*: it wires LoRA onto the
frozen VLM, runs the single forward that yields per-layer hidden states, and hands them to the
Bridge policy in policy.py. The forward here is a faithful, single-GPU re-expression of
`run_forward_pass` in `vla-scripts/finetune.py`.

Trainable modules (~207M): LoRA r64 (all-linear) + ActionQuery + Bridge policy + proprio projector.
Vision encoders, projector, and LLM are frozen.
"""
from pathlib import Path

import torch
import torch.nn as nn

import policy                                                                                    # noqa: E402
import constants as C                                                                            # noqa: F401
from peft import LoraConfig, PeftModel, get_peft_model                                           # noqa: E402
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor   # noqa: E402
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig                            # noqa: E402
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction                    # noqa: E402
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor # noqa: E402
from prismatic.models import load                                                                # noqa: E402
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask        # noqa: E402
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics                 # noqa: E402


# Keys differ between the released Prismatic VLM state-dict and the HF OpenVLA module; remap on load.
_REPLACE_MAP = [
    ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
    ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
    ("llm_backbone.llm", "language_model"),
    ("projector.projector.0", "projector.fc1"),
    ("projector.projector.2", "projector.fc2"),
    ("projector.projector.4", "projector.fc3"),
    ("gamma", "scale_factor"),
]


def _register_hf():
    """Register the custom OpenVLA classes with the HF Auto* factories (idempotent)."""
    for fn, args in [
        (AutoConfig.register, ("openvla", OpenVLAConfig)),
        (AutoImageProcessor.register, (OpenVLAConfig, PrismaticImageProcessor)),
        (AutoProcessor.register, (OpenVLAConfig, PrismaticProcessor)),
        (AutoModelForVision2Seq.register, (OpenVLAConfig, OpenVLAForActionPrediction)),
    ]:
        try:
            fn(*args)
        except Exception:
            pass  # already registered


def _rename(state_dict):
    out = {}
    for k, v in state_dict.items():
        for old, new in _REPLACE_MAP:
            if old in k:
                k = k.replace(old, new)
        out[k] = v
    return out


def build_base_vla(device, dtype=torch.bfloat16):
    """Build the OpenVLA module from config and load the pretrained MiniVLM backbone weights into it.
    Returns the *frozen base* model (no LoRA, no action-head)."""
    _register_hf()
    vlm = load(str(C.VLM_PATH), hf_token="", load_for_training=True)        # released Prismatic MiniVLM
    config = AutoConfig.from_pretrained(str(C.CONFIG_JSON))
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=dtype).to(device)
    missing, unexpected = vla.load_state_dict(_rename(vlm.state_dict()), strict=False)
    del vlm
    vla.vision_backbone.set_num_images_in_input(C.NUM_IMAGES)
    return vla


def build_processor():
    _register_hf()
    return AutoProcessor.from_pretrained(str(C.CONFIG_DIR), trust_remote_code=True)


class SubBillionVLA(nn.Module):
    """Frozen VLM + LoRA + ActionQuery + Bridge policy + proprio projector (the trainable VLA-Adapter)."""

    def __init__(self, device, dtype=torch.bfloat16, lora_rank=C.LORA_RANK):
        super().__init__()
        self.device = device
        self.num_patches = C.NUM_PATCHES

        base = build_base_vla(device, dtype)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=C.LORA_DROPOUT,
            target_modules="all-linear",
            init_lora_weights="gaussian",
            modules_to_save=["action_queries"],   # train + checkpoint the ActionQuery embedding cleanly
        )
        self.vla = get_peft_model(base, lora_config)
        self.vla.print_trainable_parameters()

        self.action_head = policy.L1RegressionActionHead(
            input_dim=C.LLM_DIM, hidden_dim=C.LLM_DIM, action_dim=C.ACTION_DIM,
            num_task_tokens=C.NUM_TASK_TOKENS, num_blocks=C.NUM_LLM_LAYERS,
        ).to(device).to(dtype)
        self.proprio_projector = policy.ProprioProjector(
            llm_dim=C.LLM_DIM, proprio_dim=C.PROPRIO_DIM,
        ).to(device).to(dtype)

    def trainable_parameters(self):
        params = [p for p in self.vla.parameters() if p.requires_grad]
        params += list(self.action_head.parameters())
        params += list(self.proprio_projector.parameters())
        return params

    def forward(self, batch, phase="Training"):
        """Faithful single-GPU `run_forward_pass` (L1 path). Returns (loss, metrics_dict)."""
        gt_actions = batch["actions"].to(self.device).to(torch.bfloat16)
        labels = batch["labels"].to(self.device)
        proprio = batch["proprio"].to(self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                labels=labels,
                output_hidden_states=True,
                proprio=proprio,
                proprio_projector=self.proprio_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        # Action-token positions come from the (shifted) labels, exactly as the repo.
        gt_token_ids = labels[:, 1:]
        cur_mask = get_current_action_mask(gt_token_ids)
        nxt_mask = get_next_actions_mask(gt_token_ids)
        B = batch["input_ids"].shape[0]

        # Stack per-layer hidden states: keep the NUM_PATCHES vision tokens + the NUM_TOKENS action queries.
        per_layer = []
        for item in out.hidden_states:                                          # 25 tensors (embed + 24 layers)
            text_hs = item[:, self.num_patches:-1]
            act_hs = text_hs[cur_mask | nxt_mask].reshape(B, 1, C.NUM_TOKENS, -1).to(torch.bfloat16)
            task_hs = item[:, : self.num_patches].reshape(B, 1, self.num_patches, -1)
            per_layer.append(torch.cat((task_hs, act_hs), dim=2))               # (B,1,NUM_PATCHES+64,896)
        multi_layer = torch.cat(per_layer, dim=1)                               # (B,25,NUM_PATCHES+64,896)

        pred_actions = self.action_head.predict_action(
            multi_layer, proprio=proprio, proprio_projector=self.proprio_projector, phase=phase,
        )                                                                       # (B, 8, 7)

        loss = nn.L1Loss()(pred_actions, gt_actions)
        metrics = {
            "loss": loss.item(),
            "curr_action_l1": nn.L1Loss()(pred_actions[:, 0], gt_actions[:, 0]).item(),
            "next_actions_l1": nn.L1Loss()(pred_actions[:, 1:], gt_actions[:, 1:]).item(),
        }
        return loss, metrics

    def save_checkpoint(self, ckpt_dir, step, dataset_statistics, processor):
        """Self-contained checkpoint: LoRA+ActionQuery adapter + Bridge/proprio weights + norm stats + processor.
        Load it back with `load_for_eval`."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vla.save_pretrained(str(ckpt_dir / "adapter"))                     # LoRA + action_queries
        torch.save(self.action_head.state_dict(), ckpt_dir / f"action_head--{step}_checkpoint.pt")
        torch.save(self.proprio_projector.state_dict(), ckpt_dir / f"proprio_projector--{step}_checkpoint.pt")
        save_dataset_statistics(dataset_statistics, ckpt_dir)
        processor.save_pretrained(str(ckpt_dir))


def _find_ckpt_file(ckpt_dir, prefix):
    matches = sorted(Path(ckpt_dir).glob(f"{prefix}--*_checkpoint.pt"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}--*_checkpoint.pt in {ckpt_dir}")
    return matches[-1]


def load_for_eval(ckpt_dir, device, dtype=torch.bfloat16):
    """Reconstruct a merged inference model + action head + proprio projector from a saved checkpoint.
    Returns (vla, action_head, proprio_projector, processor). `vla.predict_action(...)` is ready for the
    repo's `get_vla_action` helper."""
    import json

    ckpt_dir = Path(ckpt_dir)
    base = build_base_vla(device, dtype)
    vla = PeftModel.from_pretrained(base, str(ckpt_dir / "adapter")).merge_and_unload()
    vla.vision_backbone.set_num_images_in_input(C.NUM_IMAGES)

    with open(ckpt_dir / "dataset_statistics.json") as f:
        stats = json.load(f)
    vla.norm_stats = stats                                                      # used by get_vla_action un-normalization

    action_head = policy.L1RegressionActionHead(
        input_dim=C.LLM_DIM, hidden_dim=C.LLM_DIM, action_dim=C.ACTION_DIM,
        num_task_tokens=C.NUM_TASK_TOKENS, num_blocks=C.NUM_LLM_LAYERS,
    ).to(device).to(dtype)
    action_head.load_state_dict(torch.load(_find_ckpt_file(ckpt_dir, "action_head"), map_location=device))
    action_head.eval()

    proprio_projector = policy.ProprioProjector(llm_dim=C.LLM_DIM, proprio_dim=C.PROPRIO_DIM).to(device).to(dtype)
    proprio_projector.load_state_dict(torch.load(_find_ckpt_file(ckpt_dir, "proprio_projector"), map_location=device))
    proprio_projector.eval()

    processor = AutoProcessor.from_pretrained(str(ckpt_dir), trust_remote_code=True)
    return vla, action_head, proprio_projector, processor
