# sub-billion-vla

## Baseline model

A minimal, faithful reproduction of [VLA-Adapter](https://arxiv.org/abs/2509.09372).

  - A frozen ~0.5B **Prismatic MiniVLM** (DINOv2 ViT-L + SigLIP SO400M @224px; Qwen2.5-0.5B, hidden 896, 24 layers) adapted via **LoRA r64**.
  -  **64 ActionQuery** tokens (`h_a`) stacked with 512 VLM tokens (`h_t`).
  - A **Bridge policy**, a 24-block MLP-ResNet over the learnable action-chunk seed; block *i* cross-attends to layer *i+1*'s `h_t` (gated by `tanh(g)`) and `h_a`.
  
LoRA + ActionQuery + Bridge Policy (~207M, ~14%) are trainable with **L1 Regression**.
Vision encoders, projector, and LLM are frozen.

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
python src/train.py # --batch_size 4 --grad_accumulation_steps 4 --learning_rate 2e-4 --lora_rank 64
```

The default setting consumes ~17 GB VRAM.
A checkpoint is written every `--save_freq` (default 1000) steps to `runs/spatial-original/step_<N>_chkpt/`.
Live curves: `tensorboard --logdir runs/spatial-original/tb` (`train/loss` is the L1).
Prepend `CUDA_VISIBLE_DEVICES=<id>` to pick a GPU (defaults to the first one).

### Checkpoint format
```
step_<N>_chkpt/
├── adapter/                             # LoRA weights + the trained ActionQuery embedding (PEFT)
├── action_head--<N>_checkpoint.pt       # Bridge policy
├── proprio_projector--<N>_checkpoint.pt # proprio projector
├── dataset_statistics.json              # bounds_q99 normalization stats (for un-normalizing actions)
└── <processor files>
```
`src/eval.py` rebuilds the model by loading the frozen backbone, merging this adapter, and attaching
the two heads (`model.load_for_eval`).

## Evaluate

Run 10 tasks × 50 episodes = 500. Eval requires mujoco==**3.3.0**.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/eval.py --ckpt runs/spatial-original/step_20000_chkpt
```

### Expected result on LIBERO-Spatial

| | SR |
|---|---|
| Paper (Original) | 97.8% |
| Athors' released checkpoint | 97.2% |
| Re-trained with this recipe | **97.6%** |

A healthy run shows roughly **1k≈33%, 2k≈45%, 3k≈63%, 4k≈75%**, then a 96–99% plateau from ~11k.

## Notes

- Pro version is not implemented.
- Different version of mujoco may produce different evaluation results.
- This codebase only implements the novel part, which is learnable ActionQuery tokens, the Bridge-Attention policy, and L1-regression train/eval. The heavy infrastructure (DINOv2+SigLIP+Qwen2.5-0.5B backbone, RLDS pipeline, LIBERO simulator) is reused by downloading `VLA-Adapter` and `LIBERO` in `./vendor`.
