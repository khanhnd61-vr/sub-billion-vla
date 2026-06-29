# Pretrain SmolVLA using VLAb

## Train

Clone [pretraining framework](https://github.com/huggingface/VLAb) at pinned commit, and apply `fix_pad.patch`.
```bash
cd /path/to/sub-billion-vla/pretrain
git clone https://github.com/huggingface/VLAb.git
cd VLAb
git checkout 10558f6f958902c1b5ff5eed76ff5766fab6f64b
git apply ../fix_pad.patch
```

Download [pretrain dataset](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2) and exclude some episodes
```bash
ROOT=/path/to/data/HuggingFaceVLA/community_dataset_v2
hf download HuggingFaceVLA/community_dataset_v2 --local-dir $ROOT --repo-type dataset
EXCLUDE='LegrandFrederic/Orange-brick-lower-resolution|Yotofu/so100_sweeper_shoes|lirislab/guess_who_lighting|lirislab/guess_who_no_cond|roboticshack/team2-guess_who_less_ligth|roboticshack/team2-guess_who_so100|roboticshack/team2-guess_who_so100_edge_case|roboticshack/team2-guess_who_so100_light'
REPO_IDS=$(cd "$ROOT" && find . -name info.json -path '*/meta/*' -printf '%h\n' \
           | sed 's|/meta$||; s|^\./||' | sort -u | grep -Evx "$EXCLUDE" | paste -sd,)
```

Install the venv with [uv](https://github.com/astral-sh/uv)
```bash
cd /path/to/sub-billion-vla/pretrain/VLAb
cp ../pyproject.toml ../uv.lock .
uv sync
```

**Stage 1** - Pretrain
```bash
uv run accelerate launch --config_file accelerate_configs/single_gpu.yaml \
    src/lerobot/scripts/train.py \
    --policy.type=smolvla2 \
    --policy.repo_id=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.use_amp=true \
    --dataset.repo_id="$REPO_IDS" \
    --dataset.root="$ROOT" \
    --dataset.video_backend=pyav \
    --dataset.features_version=2 \
    --output_dir="./outputs/training" \
    --batch_size=32 \
    --num_workers=4 \
    --steps=200000 \
    --wandb.enable=true \
    --wandb.project="smolvla2-training"
```

Convert ckpt
```bash
cd /path/to/sub-billion-vla
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SRC=pretrain/VLAb/outputs/training/checkpoints/020000/pretrained_model
OUT=outputs/converted_smolvla_libero_init/pretrained_model
.venv-smolvla/bin/python pretrain/convert_vlab_to_lerobot.py "$SRC" "$OUT"
```

**Stage 2** - Finetune
```bash
cd /path/to/sub-billion-vla && source .venv-smolvla/bin/activate
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.path=outputs/converted_smolvla_libero_init/pretrained_model \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
  --steps=40000 --save_freq=5000 \
  --output_dir=outputs/smolvla-libero-from-vlab \
  --batch_size=64
```

## Eval
```bash
CKPT=outputs/smolvla-libero-from-vlab/checkpoints/030000/pretrained_model
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "==== $SUITE ===="
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --policy.path=$CKPT \
    --env.type=libero --env.task=$SUITE \
    --eval.batch_size=1 --eval.n_episodes=50 \
    --env.max_parallel_tasks=1 \
    --output_dir=outputs/eval_2stage/030000/$SUITE
done
```


## Results

Compare the results of 1-stage (finetune) and 2-stage (pretrain+finetune) SmolVLA

|  Task   | 1-stage | 2-stage |
| ------: |--------:|-------: |
| Spatial | 64.4    | 68.4    |
| Object  | 69,6    | 0.0     |
| Goal    | 73.2    | 76.2    |
| Long    | 43.2    | 45.6    |
| Average | 62.6    | 47.55   |