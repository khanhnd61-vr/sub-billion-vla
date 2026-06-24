# sub-billion-vla

## Baseline model

A **minimal, faithful** reproduction of **VLA-Adapter** ([arXiv 2509.09372](https://arxiv.org/abs/2509.09372)).

  - A frozen ~0.5B **Prismatic MiniVLM** (DINOv2 ViT-L + SigLIP SO400M @224px; Qwen2.5-0.5B, hidden **896**, 24 layers) adapted via **LoRA r64**
  -  **64 learnable ActionQuery tokens** (`h_a`) stacked with **512 VLM tokens** (`h_t`). 
  - A **Bridge policy** (`L1RegressionActionHead`), which is a 24-block MLP-ResNet over a learnable action-chunk seed; block *i* cross-attends to layer *i+1*'s `h_t` (gated by `tanh(g)`) and `h_a` (+ proprio), emitting an `(B, 8, 7)` chunk trained with **L1**.
  
Only LoRA + ActionQuery + Bridge + proprio projector train (**≈207M**, ~14%); the rest is frozen.

## Setup

Setup uses [uv](https://docs.astral.sh/uv/). Create the env and install the pinned
deps, then let `src/download.py` vendor the external code (git-clone `VLA-Adapter` +
`LIBERO` into `./vendor`) and fetch the model + dataset from the Hugging Face Hub:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
python src/download.py
```

## Train

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/train.py --run_dir runs/spatial-original
```

Hyperparameter: effective batch **16** (batch 4 × grad-accum 4),
LoRA r64, constant **2e-4**, bf16, ~17.6 GB VRAM (fits a 24 GB RTX 3090). A checkpoint is written
every `--save_freq` (default 1000) steps to `runs/spatial-original/step_<N>_chkpt/`. Live curves:
`tensorboard --logdir runs/spatial-original/tb` (`train/loss` is the L1).

### Checkpoint format (self-contained)
```
step_<N>_chkpt/
├── adapter/                              # LoRA weights + the trained ActionQuery embedding (PEFT)
├── action_head--<N>_checkpoint.pt        # Bridge policy
├── proprio_projector--<N>_checkpoint.pt  # proprio projector
├── dataset_statistics.json              # bounds_q99 normalization stats (for un-normalizing actions)
└── <processor files>
```
`src/eval.py` rebuilds the model by loading the frozen backbone, merging this adapter, and attaching
the two heads (`model.load_for_eval`).

### Resume training
The LoRA path keeps no optimizer state, so just relaunch and point the base at the latest adapter
(or simply restart fresh - it re-converges quickly). On the 3090, host-RAM is tight during nothing
in particular here (we save adapters, not full merged models, so there's no merge-time RAM spike).

## Evaluate

Run 10 tasks × 50 episodes = 500. **Eval requires mujoco==3.3.0.**

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/eval.py --ckpt runs/spatial-original/step_16000_chkpt
```

### Expected result on LIBERO-Spatial

| | SR |
|---|---|
| Paper (Original) | 97.8% |
| Eval authors' released checkpoint | 97.2% |
| Re-trained with this recipe | **97.6%** |

A healthy run shows roughly **1k≈33%, 2k≈45%, 3k≈63%, 4k≈75%**, then a 96–99% plateau from ~11k.

## Notes

- Pro version is not implemented.
- ActionQuery is trained/saved via PEFT `modules_to_save` (cleaner than the parent's manual
  injection); module/param names match the parent so policy weights stay interchangeable.
- `src/model.py`/`src/policy.py` mirror `prismatic/models/action_heads.py` and the L1 branch of
  `run_forward_pass` in `vla-scripts/finetune.py` line-for-line in behavior.
