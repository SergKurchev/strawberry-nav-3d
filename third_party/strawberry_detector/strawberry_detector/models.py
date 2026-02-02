"""
Model downloading and loading utilities.
"""

import subprocess
import sys
from pathlib import Path

import gdown
import requests
from tqdm import tqdm

from . import config


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def patch_depth_anything_repo():
    """
    Patch Depth-Anything-V2 to avoid device mismatch on macOS where input tensors
    can end up on MPS while weights are on CPU.

    This patch is idempotent: safe to run multiple times.
    """
    pe = (
        config.DEPTH_ANYTHING_DIR
        / "metric_depth"
        / "depth_anything_v2"
        / "dinov2_layers"
        / "patch_embed.py"
    )

    if not pe.exists():
        # Repo not present yet, or structure differs — nothing to patch.
        return

    s = pe.read_text(encoding="utf-8")

    needle = "if x.device != self.proj.weight.device:"
    if needle in s:
        return  # already patched

    target = "x = self.proj(x)  # B C H W"
    i = s.find(target)
    if i == -1:
        # Fallback: try without comment
        target = "x = self.proj(x)"
        i = s.find(target)
        if i == -1:
            print("⚠️ patch_embed.py: target line not found, skipping patch")
            return

    line_start = s.rfind("\n", 0, i) + 1
    indent = s[line_start:i]  # indentation of the target line

    insert = (
        f"{indent}if x.device != self.proj.weight.device:\n"
        f"{indent}    x = x.to(self.proj.weight.device)\n\n"
    )

    pe.write_text(s[:line_start] + insert + s[line_start:], encoding="utf-8")
    print(f"✅ Patched Depth-Anything-V2: {pe}")


def ensure_depth_anything_repo():
    """Clone Depth-Anything-V2 repository if not present."""
    if not config.DEPTH_ANYTHING_DIR.exists():
        print("📥 Cloning Depth-Anything-V2 repository...")
        subprocess.run(
            ["git", "clone", config.DEPTH_ANYTHING_REPO_URL, str(config.DEPTH_ANYTHING_DIR)],
            check=True,
        )
        print("✅ Depth-Anything-V2 cloned successfully")

    # Apply patch after clone (or on existing repo)
    patch_depth_anything_repo()

    # Add to path if not already
    depth_path = str(config.DEPTH_ANYTHING_DIR)
    if depth_path not in sys.path:
        sys.path.insert(0, depth_path)


def ensure_depth_weights():
    """Download depth model weights if not present."""
    config.ensure_dirs()
    weights_path = config.get_depth_weights_path()

    if not weights_path.exists():
        print("📥 Downloading Depth-Anything-V2 weights...")
        download_file(config.DEPTH_WEIGHTS_URL, weights_path, "Depth weights")
        print(f"✅ Depth weights saved to {weights_path}")

    return weights_path


def ensure_yolo_weights():
    """Download YOLO weights from Google Drive if not present."""
    config.ensure_dirs()
    weights_path = config.get_yolo_weights_path()

    if not weights_path.exists():
        print("📥 Downloading YOLO strawberry weights...")
        url = f"https://drive.google.com/uc?id={config.YOLO_WEIGHTS_GDRIVE_ID}"
        gdown.download(url, str(weights_path), quiet=False)
        print(f"✅ YOLO weights saved to {weights_path}")

    return weights_path


def ensure_all_models():
    """Ensure all required models are downloaded."""
    ensure_depth_anything_repo()
    ensure_depth_weights()
    ensure_yolo_weights()
    print("✅ All models ready")
