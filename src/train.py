"""
train.py - fine-tune VLA-Adapter (Original variant) on LIBERO-Spatial.

Recipe: frozen MiniVLM + LoRA r64 (all-linear) + ActionQuery +
Bridge policy + proprio, trained with L1 regression on 8-step action chunks. AdamW @ constant 2e-4,
bf16, effective batch 16 (batch 4 x grad-accum 4). Converges by ~11-20k steps (peak ~98-99%).

Run from the sub-billion-vla/ directory:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/train.py --run_dir runs/spatial-original
Resume training with --resume_adapter <latest_ckpt>/adapter (LoRA re-inits).
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import constants as C
import data as data_mod
import model as model_mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default="runs/spatial-original")
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--grad_accumulation_steps", type=int, default=C.GRAD_ACCUMULATION_STEPS)
    p.add_argument("--learning_rate", type=float, default=C.LEARNING_RATE)
    p.add_argument("--lora_rank", type=int, default=C.LORA_RANK)
    p.add_argument("--max_steps", type=int, default=C.MAX_STEPS)
    p.add_argument("--num_steps_before_decay", type=int, default=C.NUM_STEPS_BEFORE_DECAY)
    p.add_argument("--shuffle_buffer_size", type=int, default=C.SHUFFLE_BUFFER_SIZE)
    p.add_argument("--save_freq", type=int, default=C.SAVE_FREQ)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--no_image_aug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    print("Building model (frozen MiniVLM + LoRA + Bridge policy) ...")
    model = model_mod.SubBillionVLA(device=device, lora_rank=args.lora_rank)
    processor = model_mod.build_processor()

    print("Building LIBERO-Spatial dataloader ...")
    dataloader, dataset_statistics = data_mod.build_dataloader(
        processor, batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size, image_aug=not args.no_image_aug,
    )

    optimizer = AdamW(model.trainable_parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[args.num_steps_before_decay], gamma=0.1)

    model.train()
    optimizer.zero_grad()
    grad_step = 0
    recent = {"loss": 0.0, "curr_action_l1": 0.0, "next_actions_l1": 0.0}

    print(f"Training up to {args.max_steps} steps (effective batch "
          f"{args.batch_size * args.grad_accumulation_steps}) ...")
    for batch_idx, batch in enumerate(dataloader):
        loss, metrics = model(batch, phase="Training")
        (loss / args.grad_accumulation_steps).backward()
        for k in recent:
            recent[k] += metrics[k] / args.grad_accumulation_steps

        if (batch_idx + 1) % args.grad_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            grad_step += 1

            if grad_step % args.log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"step {grad_step} | L1 {recent['loss']/args.log_freq:.4f} | lr {lr:.2e}")
                writer.add_scalar("train/loss", recent["loss"] / args.log_freq, grad_step)
                writer.add_scalar("train/curr_action_l1", recent["curr_action_l1"] / args.log_freq, grad_step)
                writer.add_scalar("train/next_actions_l1", recent["next_actions_l1"] / args.log_freq, grad_step)
                writer.add_scalar("train/lr", lr, grad_step)
                writer.add_scalar("train/vram_gb", torch.cuda.max_memory_allocated() / 1e9, grad_step)
                recent = {k: 0.0 for k in recent}

            if grad_step % args.save_freq == 0:
                ckpt = run_dir / f"step_{grad_step}_chkpt"
                print(f"saving checkpoint -> {ckpt}")
                model.save_checkpoint(ckpt, grad_step, dataset_statistics, processor)

            if grad_step >= args.max_steps:
                break

    print("done.")
    writer.close()


if __name__ == "__main__":
    main()
