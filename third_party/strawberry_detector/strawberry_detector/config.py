"""
Configuration and paths for strawberry detector.
"""

import os
from pathlib import Path

# Base directory (strawberry_detector package root)
PACKAGE_DIR = Path(__file__).parent.parent.absolute()

# Checkpoints directory
CHECKPOINTS_DIR = PACKAGE_DIR / "checkpoints"

# Depth-Anything-V2 settings
DEPTH_ANYTHING_REPO_URL = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_ANYTHING_DIR = PACKAGE_DIR / "Depth-Anything-V2"
DEPTH_WEIGHTS_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth"
DEPTH_WEIGHTS_FILENAME = "depth_anything_v2_metric_hypersim_vitl.pth"
DEPTH_ENCODER = "vitl"  # Vision Transformer Large

# YOLO Strawberry Segmentation settings
YOLO_WEIGHTS_GDRIVE_ID = "10cpgTPpNocwytHg77AqKypWX-yOdhsGY"
YOLO_WEIGHTS_FILENAME = "strawberry_yolo_best.pt"

# Class names for strawberry detection (from YOLO model)
# Note: These are the actual class names from the trained model
CLASS_NAMES = {
    0: "strawberry",      # Unripe/green strawberry
    1: "ripe_strawberry", # Ripe red strawberry
}

# Default device
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def get_depth_weights_path() -> Path:
    """Get path to depth model weights."""
    return CHECKPOINTS_DIR / DEPTH_WEIGHTS_FILENAME


def get_yolo_weights_path() -> Path:
    """Get path to YOLO model weights."""
    return CHECKPOINTS_DIR / YOLO_WEIGHTS_FILENAME


def ensure_dirs():
    """Ensure required directories exist."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
