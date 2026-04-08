"""
One-time setup script for the RunPod network volume.

Run this on a temporary RunPod pod (cheap CPU pod is fine) with the network
volume mounted at /runpod-volume to download:
  - The ACE-Step XL-turbo base model
  - The VAE
  - The 0.6B language model
  - 5 starter LoRAs

Total disk usage: ~25 GB

Usage on the pod:
    python setup_volume.py
"""

import os
import subprocess
import sys
from pathlib import Path

VOLUME_ROOT = os.environ.get("ACESTEP_VOLUME_ROOT", "/runpod-volume")
CHECKPOINTS_DIR = Path(VOLUME_ROOT) / "checkpoints"
LORAS_DIR = Path(VOLUME_ROOT) / "loras"

MODELS = [
    ("ACE-Step/acestep-v15-xl-turbo", CHECKPOINTS_DIR / "acestep-v15-xl-turbo"),
    ("ACE-Step/acestep-v15-xl-base", CHECKPOINTS_DIR / "acestep-v15-xl-base"),
    ("ACE-Step/Ace-Step1.5", CHECKPOINTS_DIR / "vae"),  # contains VAE
    ("ACE-Step/acestep-5Hz-lm-0.6B", CHECKPOINTS_DIR / "acestep-5Hz-lm-0.6B"),
]

LORAS = [
    ("ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA", "chinese-new-year"),
    ("DisturbingTheField/ACE-Step-v1.5-ambient_dream1-LoRA", "ambient-dream"),
    ("DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs", "raspy-vocal"),
    ("DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA", "acoustic-guitar"),
    ("6san/symphonic_metal_lora_for_ace-step_v15", "symphonic-metal"),
]


def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


def download(repo_id, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists() and any(dest_path.glob("*.safetensors")):
        print(f"[skip] {repo_id} already at {dest_path}")
        return
    dest_path.mkdir(parents=True, exist_ok=True)
    run(f"hf download {repo_id} --local-dir {dest_path}")


def main():
    print(f"Volume root: {VOLUME_ROOT}")
    if not Path(VOLUME_ROOT).exists():
        print(f"ERROR: {VOLUME_ROOT} does not exist. Mount your network volume first.")
        sys.exit(1)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LORAS_DIR.mkdir(parents=True, exist_ok=True)

    # Install hf cli if missing
    run("pip install --upgrade huggingface_hub[cli]")

    print("\n=== Downloading base models ===")
    for repo, dest in MODELS:
        download(repo, dest)

    print("\n=== Downloading LoRAs ===")
    for repo, name in LORAS:
        download(repo, LORAS_DIR / name)

    # Show final layout
    print("\n=== Volume contents ===")
    run(f"du -sh {CHECKPOINTS_DIR}/* {LORAS_DIR}/*")
    print("\nSetup complete. You can now deploy the serverless endpoint.")


if __name__ == "__main__":
    main()
