"""
Utility functions for image loading and visualization.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import requests


def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """
    Load image from various sources.
    
    Args:
        source: File path, URL, or numpy array
    
    Returns:
        Image as BGR numpy array
    """
    if isinstance(source, np.ndarray):
        return source.copy()
    
    source = str(source)
    
    # Check if URL
    if source.startswith(("http://", "https://")):
        return load_image_from_url(source)
    
    # Load from file
    img = cv2.imread(source)
    if img is None:
        raise ValueError(f"Could not load image from: {source}")
    
    return img


def load_image_from_url(url: str) -> np.ndarray:
    """Download and load image from URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    img_array = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Could not decode image from URL: {url}")
    
    return img


def visualize_results(image: np.ndarray, detections: list, 
                      depth_map: np.ndarray = None) -> np.ndarray:
    """
    Draw detection results on image.
    
    Args:
        image: Original BGR image
        detections: List of detection dicts
        depth_map: Optional depth map for overlay
    
    Returns:
        Annotated image
    """
    result = image.copy()
    
    colors = {
        "ripe_strawberry": (0, 255, 0),  # Green for ripe
        "strawberry": (0, 165, 255),      # Orange for unripe/generic
    }
    
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        
        # Get color
        class_name = det.get("class_name", "unknown")
        color = colors.get(class_name, (255, 0, 0))
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        depth_str = ""
        if "depth" in det:
            depth_cm = det["depth"].get("mean_cm", 0)
            depth_str = f" {depth_cm:.0f}cm"
        
        label = f"{det['id']}: {class_name}{depth_str}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw mask outline if available
        if "mask" in det and "polygon" in det["mask"]:
            polygon = np.array(det["mask"]["polygon"], dtype=np.int32)
            if len(polygon) > 0:
                cv2.polylines(result, [polygon], True, color, 2)
    
    return result


def save_visualization(image: np.ndarray, output_path: Union[str, Path]):
    """Save visualization image."""
    cv2.imwrite(str(output_path), image)
