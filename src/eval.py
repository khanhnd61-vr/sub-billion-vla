"""
eval.py - evaluate a trained checkpoint on LIBERO-Spatial (10 tasks x 50 episodes = 500).

HYBRID: the LIBERO simulator + the proven inference/rollout (`run_task` -> `get_action` ->
`get_vla_action`, image prep, proprio normalization, action un-normalization, success counting)
are REUSED from the VLA-Adapter repo. We only build the model from OUR adapter-format checkpoint
(model.load_for_eval) and drive the per-task loop.

IMPORTANT: eval REQUIRES mujoco==3.3.0. Run with EGL rendering:
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python src/eval.py --ckpt runs/spatial-original/step_16000_chkpt
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import tqdm                                                                          # noqa: E402
import constants as C                                                                # noqa: F401
from libero.libero import benchmark                                                  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, run_task        # noqa: E402
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere # noqa: E402
import model as model_mod                                                            # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="a step_*_chkpt dir produced by train.py")
    p.add_argument("--task_suite_name", type=str, default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    cfg = GenerateConfig(
        pretrained_checkpoint=args.ckpt,
        model_family="openvla",
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        num_images_in_input=C.NUM_IMAGES,
        use_proprio=True,
        use_film=False,
        use_l1_regression=True,
        use_minivlm=True,               # MiniVLM prompt, matching training
        use_pro_version=False,
        num_open_loop_steps=C.NUM_ACTIONS_CHUNK,
        center_crop=True,
        unnorm_key=C.DATASET_NAME,      # key inside dataset_statistics.json
        seed=args.seed,
        use_wandb=False,
    )

    set_seed_everywhere(cfg.seed)

    print(f"Loading checkpoint {args.ckpt} ...")
    vla, action_head, proprio_projector, processor = model_mod.load_for_eval(args.ckpt, device)
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name} ({num_tasks} tasks x {cfg.num_trials_per_task} episodes)")

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg, task_suite, task_id, vla, resize_size, processor,
            action_head, proprio_projector, None,
            total_episodes, total_successes, None, None,
        )

    rate = total_successes / total_episodes if total_episodes else 0.0
    print(f"Total episodes: {total_episodes}")
    print(f"Total successes: {total_successes}")
    print(f"Overall success rate: {rate:.4f} ({rate * 100:.1f}%)")


if __name__ == "__main__":
    main()
