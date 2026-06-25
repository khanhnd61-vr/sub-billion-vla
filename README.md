# sub-billion-vla

## 1. VLA-Adapter baseline

A minimal, faithful reproduction of [VLA-Adapter](https://arxiv.org/abs/2509.09372).

  - A frozen ~0.5B **Prismatic MiniVLM** (DINOv2 ViT-L + SigLIP SO400M @224px; Qwen2.5-0.5B, hidden 896, 24 layers) adapted via **LoRA r64**.
  -  **64 ActionQuery** tokens (`h_a`) stacked with 512 VLM tokens (`h_t`).
  - A **Bridge policy**, a 24-block MLP-ResNet over the learnable action-chunk seed; block *i* cross-attends to layer *i+1*'s `h_t` (gated by `tanh(g)`) and `h_a`.
  
LoRA + ActionQuery + Bridge Policy (~207M, ~14%) are trainable with **L1 Regression**.
Vision encoders, projector, and LLM are frozen.

### Setup

Setup uses [uv](https://docs.astral.sh/uv/). Create the env and install the pinned
deps, then let `src/download.py` vendor the external code (git-clone `VLA-Adapter` +
`LIBERO` into `./vendor`) and fetch the model + dataset from the Hugging Face Hub:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
python src/download.py
```

### Train

```bash
python src/train.py # --batch_size 4 --grad_accumulation_steps 4 --learning_rate 2e-4 --lora_rank 64
```

The default setting consumes ~17 GB VRAM.
A checkpoint is written every `--save_freq` (default 1000) steps to `runs/spatial-original/step_<N>_chkpt/`.
Live curves: `tensorboard --logdir runs/spatial-original/tb` (`train/loss` is the L1).
Prepend `CUDA_VISIBLE_DEVICES=<id>` to pick a GPU (defaults to the first one).

#### Checkpoint format
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

### Evaluate

Run 10 tasks × 50 episodes = 500. Eval requires mujoco==**3.3.0**.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/eval.py --ckpt runs/spatial-original/step_20000_chkpt
```

#### Expected result on LIBERO-Spatial

| | SR |
|---|---|
| Paper (Original) | 97.8% |
| Athors' released checkpoint | 97.2% |
| Re-trained with this recipe | **97.6%** |

A healthy run shows roughly **1k≈33%, 2k≈45%, 3k≈63%, 4k≈75%**, then a 96–99% plateau from ~11k.

### Notes

- Pro version is not implemented.
- Different version of mujoco may produce different evaluation results.
- This codebase only implements the novel part, which is learnable ActionQuery tokens, the Bridge-Attention policy, and L1-regression train/eval. The heavy infrastructure (DINOv2+SigLIP+Qwen2.5-0.5B backbone, RLDS pipeline, LIBERO simulator) is reused by downloading `VLA-Adapter` and `LIBERO` in `./vendor`.

## 2. SmolVLA baseline (lerobot)

A second sub-billion baseline: finetune HuggingFace's [SmolVLA](https://arxiv.org/abs/2506.01844)
(~450M); SmolVLM2 backbone + flow-matching action expert).
This path is entirely lerobot-native - its train/eval are single CLI commands that pull
[`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) and the
[`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero) dataset from the Hub,
and shares nothing with the VLA-Adapter stack above. It therefore lives in its own virtualenv.

### Setup
Use a **separate** venv from the VLA-Adapter stack (lerobot needs a much newer transformers/torch
than `requirements.txt` pins):
```bash
uv venv --python 3.10 .venv-smolvla
source .venv-smolvla/bin/activate
uv pip install "lerobot[smolvla,libero]"
```
If the published extras lag the LIBERO integration, install from source instead:
`git clone https://github.com/huggingface/lerobot && cd lerobot && uv pip install -e ".[smolvla,libero]"`.

### Train
Train SmolVLA on the full LIBERO dataset (all four suites combined), mixed precision. This is
lerobot's documented LIBERO recipe: it loads the pretrained SmolVLM2 **VLM** weights and trains the
~100M action expert on LIBERO:
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --steps=40000 \
  --save_freq=5000 \
  --output_dir=outputs/smolvla-libero \
  --job_name=smolvla-libero \
  --batch_size=64
```

This recipe reuses the pretrained VLM and learns the fresh action expert on LIBERO.
Because of the mismatch between smolvla_base's action expert (3-camera embodiment) and HuggingFaceVLA/libero setting (2-image vision input), the recipe cannot reuse smolvla_base's pretrained action expert.

### Evaluate (LIBERO-Spatial)
lerobot-native eval. `--eval.n_episodes` is **per task**, so 50 × 10 tasks = 500 episodes,
matching the VLA-Adapter protocol above:
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 lerobot-eval \
  --policy.path=outputs/smolvla-libero/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --env.max_parallel_tasks=1
```

The SmolVLA policy trained on `HuggingFaceVLA/libero` expects **relative** (delta) end-effector
actions, which is lerobot's default `--env.control_mode`; leave it unless you retrain on an
absolute-action dataset.

#### Result

| Task | SR |
|---|---|
| Object  | T.B.D |
| Goal    | T.B.D |
| Spatial | T.B.D |
| Long    | T.B.D |
