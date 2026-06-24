"""
download.py - one-shot setup for the standalone package. Fetches everything into ./vendor and the
vendored repo layout; each step is skipped if its destination already exists.

  1. Vendored code  : git clone VLA-Adapter (pinned) + LIBERO  -> ./vendor/{VLA-Adapter,LIBERO}
  2. Backbone       : Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b  (HF)
  3. Dataset        : openvla/modified_libero_rlds  (subdir libero_spatial_no_noops)  (HF)
  4. (optional)     : VLA-Adapter/LIBERO-Spatial    (authors' 97.8% checkpoint)       (HF)

Usage:
    python src/download.py                 # vendored code + backbone + dataset
    python src/download.py --released      # also the authors' finetuned checkpoint
    python src/download.py --code-only     # just clone the vendored repos

If you already have a VLA-Adapter checkout, skip the clone by exporting SBVLA_VLA_ADAPTER_DIR
(and SBVLA_LIBERO_DIR) to point at it before running anything.
"""
import argparse
import subprocess

from huggingface_hub import snapshot_download

import constants as C

RELEASED_CKPT_DIR = C.VLA_ADAPTER_DIR / "outputs" / "LIBERO-Spatial"


def _have(path):
    return path.exists() and any(path.iterdir())


def _git(*args):
    print("  $ git", *args)
    subprocess.run(["git", *args], check=True)


def clone_vendored_code():
    C.VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if _have(C.VLA_ADAPTER_DIR):
        print(f"[skip] VLA-Adapter already at {C.VLA_ADAPTER_DIR}")
    else:
        print(f"[get ] cloning VLA-Adapter @ {C.VLA_ADAPTER_COMMIT[:10]} -> {C.VLA_ADAPTER_DIR}")
        _git("clone", C.VLA_ADAPTER_URL, str(C.VLA_ADAPTER_DIR))
        try:
            _git("-C", str(C.VLA_ADAPTER_DIR), "checkout", C.VLA_ADAPTER_COMMIT)
        except subprocess.CalledProcessError:
            print("[warn] could not check out the pinned commit; staying on the default branch.")

    if _have(C.LIBERO_DIR):
        print(f"[skip] LIBERO already at {C.LIBERO_DIR}")
    else:
        print(f"[get ] cloning LIBERO -> {C.LIBERO_DIR}")
        _git("clone", "--depth", "1", C.LIBERO_URL, str(C.LIBERO_DIR))
    print("      -> next: pip install -r vendor/LIBERO/requirements.txt")


def download_backbone():
    # The VLA-Adapter clone ships a placeholder README in this dir, so a plain "non-empty" (_have)
    # check would wrongly skip the download. Key off config.json, which only the real weights provide.
    if (C.VLM_PATH / "config.json").exists():
        print(f"[skip] backbone already at {C.VLM_PATH}")
        return
    print(f"[get ] backbone -> {C.VLM_PATH}")
    snapshot_download(
        repo_id="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b",
        local_dir=str(C.VLM_PATH),
    )


def download_dataset():
    dest = C.DATA_ROOT_DIR / C.DATASET_NAME
    if _have(dest):
        print(f"[skip] dataset already at {dest}")
        return
    print(f"[get ] LIBERO-Spatial RLDS -> {C.DATA_ROOT_DIR}")
    C.DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="openvla/modified_libero_rlds",
        repo_type="dataset",
        local_dir=str(C.DATA_ROOT_DIR),
        allow_patterns=[f"{C.DATASET_NAME}/*"],
    )


def download_released_ckpt():
    if _have(RELEASED_CKPT_DIR):
        print(f"[skip] released ckpt already at {RELEASED_CKPT_DIR}")
        return
    print(f"[get ] released finetuned ckpt -> {RELEASED_CKPT_DIR}")
    snapshot_download(repo_id="VLA-Adapter/LIBERO-Spatial", local_dir=str(RELEASED_CKPT_DIR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", action="store_true", help="also download the authors' finetuned checkpoint")
    ap.add_argument("--code-only", action="store_true", help="only clone the vendored repos")
    args = ap.parse_args()

    clone_vendored_code()
    if not args.code_only:
        download_backbone()
        download_dataset()
        if args.released:
            download_released_ckpt()
    print("done.")
