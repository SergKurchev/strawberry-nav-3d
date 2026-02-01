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
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_depth_anything_repo():
    """Clone Depth-Anything-V2 repository if not present."""
    if not config.DEPTH_ANYTHING_DIR.exists():
        print("📥 Cloning Depth-Anything-V2 repository...")
        subprocess.run(
            ["git", "clone", config.DEPTH_ANYTHING_REPO_URL, str(config.DEPTH_ANYTHING_DIR)],
            check=True
        )
        print("✅ Depth-Anything-V2 cloned successfully")
    
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
