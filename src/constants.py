"""
constants.py - config for the VLA-Adapter "Original" Phase-1 recipe (LIBERO-Spatial,
Prismatic MiniVLM = Qwen2.5-0.5B + DINOv2/SigLIP @224px).

STANDALONE: this package does not assume it lives inside the VLA-Adapter repo. Its external code
dependencies (the `prismatic` package + the `experiments` eval glue, and the `libero` simulator)
are *vendored* by src/download.py into ./vendor and put on sys.path here. Model weights / dataset are
downloaded from the Hugging Face Hub into the vendored layout. Every path can be overridden with an
env var so you can point at copies you already have instead of re-downloading.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
# --- Vendored external code (cloned by src/download.py) ---
VENDOR_DIR = Path(os.environ.get("SBVLA_VENDOR_DIR", PROJECT_ROOT / "vendor"))
VLA_ADAPTER_DIR = Path(os.environ.get("SBVLA_VLA_ADAPTER_DIR", VENDOR_DIR / "VLA-Adapter"))
LIBERO_DIR = Path(os.environ.get("SBVLA_LIBERO_DIR", VENDOR_DIR / "LIBERO"))
VLA_ADAPTER_COMMIT = "23fa0c9c159e2aa04341cdd3e924f44061311060"  # pinned for reproducibility
VLA_ADAPTER_URL = "https://github.com/OpenHelix-Team/VLA-Adapter.git"
LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"

for _p in (VLA_ADAPTER_DIR, LIBERO_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

def _prime_libero_platform():
    import importlib

    token = "libero_spatial"
    if any("libero" in a.lower() for a in sys.argv):
        return
    sys.argv.append(token)
    try:
        importlib.import_module("prismatic.vla.constants")
    except Exception:
        pass
    finally:
        if sys.argv and sys.argv[-1] == token:
            sys.argv.remove(token)


_prime_libero_platform()

# --- Artifacts (downloaded by download.py; override with env vars to reuse existing copies) ---
VLM_PATH = Path(os.environ.get(
    "SBVLA_VLM_PATH", VLA_ADAPTER_DIR / "pretrained_models" / "prism-qwen25-extra-dinosiglip-224px-0_5b"))
CONFIG_DIR = Path(os.environ.get("SBVLA_CONFIG_DIR", VLA_ADAPTER_DIR / "pretrained_models" / "configs"))
CONFIG_JSON = CONFIG_DIR / "config.json"
DATA_ROOT_DIR = Path(os.environ.get("SBVLA_DATA_DIR", VLA_ADAPTER_DIR / "data" / "libero"))
DATASET_NAME = "libero_spatial_no_noops"

# --- Action / proprio space (LIBERO) ---
NUM_ACTIONS_CHUNK = 8                           # predict an 8-step action chunk
ACTION_DIM = 7                                  # 3 xyz + 3 axis-angle + 1 gripper
PROPRIO_DIM = 8                                 # 3 eef pos + 3 eef axis-angle + 1 gripper qpos (+1)
NORMALIZATION = "bounds_q99"                    # actions/proprio normalized to [-1, 1] via 1st/99th percentile

# --- Backbone sizes (Qwen2.5-0.5B MiniVLM @224px) ---
LLM_DIM = 896                                   # Qwen2.5-0.5B hidden size
NUM_LLM_LAYERS = 24                             # => 25 hidden-state tensors (embeddings + 24 decoder layers)
TOKENS_PER_IMAGE = 256                          # 224/14 = 16, 16*16 = 256 patches (per DINO and per SigLIP))
NUM_IMAGES = 2                                  # 3rd-person + wrist
NUM_TASK_TOKENS = TOKENS_PER_IMAGE * NUM_IMAGES # 512 vision "task" tokens feeding the Bridge
NUM_PATCHES = NUM_TASK_TOKENS                   # vision tokens occupy the first NUM_PATCHES seq positions

# --- ActionQuery tokens (learnable queries appended to the LLM input) ---
NUM_TOKENS = 64                                 # number of ActionQuery tokens (h_a)
ACTION_TOKEN_BEGIN_IDX = 151386                 # Qwen2.5 vocab boundary marking action-token positions
IGNORE_INDEX = -100
STOP_INDEX = 2

# --- Training recipe ---
LORA_RANK = 64
LORA_ALPHA = 2 * LORA_RANK
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
MAX_STEPS = 200_005
GRAD_ACCUMULATION_STEPS = 4                     # effective batch = 16
NUM_STEPS_BEFORE_DECAY = 200_000                # MultiStepLR x0.1 milestone (=> effectively constant LR)
SHUFFLE_BUFFER_SIZE = 12_000                    # reduced for 31GB host RAM
SAVE_FREQ = 1000
IMAGE_RESOLUTION = (224, 224)
