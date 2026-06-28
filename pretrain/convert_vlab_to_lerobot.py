#!/usr/bin/env python
"""Convert a VLAb `smolvla2` checkpoint into a lerobot-0.4.4 `smolvla` pretrained_model dir.

Transfers the 498 shared weights (the full VLM-with-expert, action in/out projections and the
action-time MLP), re-initializes `state_proj` (the two forks route proprioceptive state
differently: VLAb -> action expert [720], lerobot -> VLM prefix [960]), and drops VLAb's in-model
normalization buffers (lerobot computes normalization from the finetuning dataset).

Run with the lerobot-0.4.4 interpreter (sub-billion-vla/.venv-smolvla):
    .venv-smolvla/bin/python convert_vlab_to_lerobot.py <vlab_pretrained_model_dir> <out_dir>
"""
import json
import sys

import torch  # noqa: F401
from safetensors.torch import load_file

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

src_dir, out_dir = sys.argv[1], sys.argv[2]
vc = json.load(open(f"{src_dir}/config.json"))

cfg = SmolVLAConfig(
    chunk_size=vc["chunk_size"],
    n_action_steps=vc["n_action_steps"],
    max_state_dim=vc["max_state_dim"],
    max_action_dim=vc["max_action_dim"],
    num_vlm_layers=vc["num_vlm_layers"],
    expert_width_multiplier=vc["expert_width_multiplier"],
    self_attn_every_n_layers=vc["self_attn_every_n_layers"],
    attention_mode=vc["attention_mode"],
    num_expert_layers=vc["num_expert_layers"],
    vlm_model_name=vc["vlm_model_name"],
    load_vlm_weights=False,
    add_image_special_tokens=vc.get("add_image_special_tokens", False),
    tokenizer_max_length=vc["tokenizer_max_length"],
    num_steps=vc["num_steps"],
    prefix_length=vc["prefix_length"],
    pad_language_to=vc["pad_language_to"],
    min_period=vc["min_period"],
    max_period=vc["max_period"],
    normalization_mapping={
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,
        "ACTION": NormalizationMode.MEAN_STD,
    },
    # LIBERO is a 2-image embodiment; exact features are reconciled from the dataset at finetune time.
    input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))},
)
cfg.device = "cpu"
policy = SmolVLAPolicy(cfg)

vlab_sd = load_file(f"{src_dir}/model.safetensors")
# Keep model.* weights; drop the 6 normalization buffers and the 2 state_proj tensors (shape mismatch).
transfer = {
    k: v for k, v in vlab_sd.items() if k.startswith("model.") and not k.startswith("model.state_proj")
}
missing, unexpected = policy.load_state_dict(transfer, strict=False)
print(f"transferred weights : {len(transfer)} / {len(policy.state_dict())}")
print(f"re-initialized (missing): {list(missing)}")
print(f"unexpected (must be empty): {list(unexpected)}")
assert not unexpected, "Unexpected keys — architecture mismatch!"
assert set(missing) == {"model.state_proj.weight", "model.state_proj.bias"}, missing

policy.save_pretrained(out_dir)

# lerobot 0.4.4 keeps normalization in an external processor pipeline, so a pretrained dir must
# carry the processor files (policy_preprocessor.json / policy_postprocessor.json). Build them
# (structure only — at finetune time lerobot overrides the normalizer/unnormalizer stats from the
# target dataset's `dataset.meta.stats`) and save alongside the model.
preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, dataset_stats=None)
preprocessor.save_pretrained(out_dir)
postprocessor.save_pretrained(out_dir)
print("saved processor pipelines (preprocessor + postprocessor)")
print("saved lerobot smolvla checkpoint ->", out_dir)
