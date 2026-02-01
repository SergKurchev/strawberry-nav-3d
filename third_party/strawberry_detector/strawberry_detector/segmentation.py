"""
YOLO-based strawberry segmentation.
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from . import config
from .models import ensure_yolo_weights


class StrawberrySegmenter:
    """
    Strawberry detection and segmentation using YOLOv8.
    """
    
    def __init__(self, weights_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize segmenter.
        
        Args:
            weights_path: Path to YOLO weights. Downloads if None.
            device: Device to use ('cuda', 'cpu', or auto)
        """
        from ultralytics import YOLO
        
        # Get weights path
        if weights_path is None:
            weights_path = ensure_yolo_weights()
        
        # Load model
        self.model = YOLO(str(weights_path))
        
        # Set device if specified
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✅ YOLO model loaded on {self.device}")
    
    def segment(self, image: Union[str, Path, np.ndarray], 
                conf_threshold: float = 0.25) -> List[dict]:
        """
        Detect and segment strawberries in an image.
        
        Args:
            image: Image path or numpy array
            conf_threshold: Confidence threshold for detections
        
        Returns:
            List of detections with bboxes, masks, and metadata
        """
        # Load image to get original size
        if isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
        else:
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            orig_h, orig_w = img.shape[:2]
        
        # Run inference
        results = self.model(image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i, box in enumerate(boxes):
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detection = {
                    "id": len(detections),
                    "class_id": cls_id,
                    "class_name": config.CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "confidence": round(conf, 4),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "center_x": int((x1 + x2) / 2),
                        "center_y": int((y1 + y2) / 2),
                    }
                }
                
                # Get mask if available
                if masks is not None and i < len(masks):
                    mask = masks[i].data.cpu().numpy()[0]
                    
                    # Resize mask to original image size
                    mask_h, mask_w = mask.shape
                    if mask_h != orig_h or mask_w != orig_w:
                        mask = cv2.resize(
                            mask.astype(np.float32), 
                            (orig_w, orig_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                        mask = (mask > 0.5).astype(np.float32)
                    
                    # Get polygon points from resized mask
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    polygon = []
                    if contours:
                        # Get largest contour
                        largest = max(contours, key=cv2.contourArea)
                        polygon = largest.reshape(-1, 2).tolist()
                    
                    detection["mask"] = {
                        "binary_mask": mask,  # Will be removed before JSON serialization
                        "polygon": polygon,
                        "area": int(mask.sum()),
                    }
                
                detections.append(detection)
        
        return detections
    
    def get_mask_for_detection(self, detection: dict) -> Optional[np.ndarray]:
        """Get binary mask from detection dict."""
        if "mask" in detection and "binary_mask" in detection["mask"]:
            return detection["mask"]["binary_mask"]
        return None
