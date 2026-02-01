"""
Depth estimation wrapper using Depth-Anything-V2.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from . import config
from .models import ensure_depth_anything_repo, ensure_depth_weights


class DepthEstimator:
    """
    Depth estimation using Depth-Anything-V2 metric model.
    
    Estimates depth in meters from a single RGB image.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize depth estimator.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        # Ensure model files are present
        ensure_depth_anything_repo()
        weights_path = ensure_depth_weights()
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Import Depth-Anything model
        depth_path = str(config.DEPTH_ANYTHING_DIR)
        if depth_path not in sys.path:
            sys.path.insert(0, depth_path)
        
        from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        
        # Model configuration for Hypersim ViT-Large
        model_configs = {
            'vitl': {
                'encoder': 'vitl',
                'features': 256,
                'out_channels': [256, 512, 1024, 1024]
            }
        }
        
        # Load model
        self.model = DepthAnythingV2(**model_configs[config.DEPTH_ENCODER])
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        
        print(f"✅ Depth model loaded on {self.device}")
    
    def estimate(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Estimate depth from an image.
        
        Args:
            image: Image path or numpy array (BGR or RGB)
        
        Returns:
            Depth map in meters with shape (H, W)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Run inference
        with torch.no_grad():
            depth = self.model.infer_image(img)
        
        return depth
    
    def estimate_with_stats(self, image: Union[str, Path, np.ndarray]) -> dict:
        """
        Estimate depth and return statistics.
        
        Args:
            image: Image path or numpy array
        
        Returns:
            Dict with depth map and statistics
        """
        depth = self.estimate(image)
        
        return {
            "depth_map": depth,
            "shape": depth.shape,
            "min_meters": float(depth.min()),
            "max_meters": float(depth.max()),
            "mean_meters": float(depth.mean()),
        }
