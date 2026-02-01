"""
Strawberry Detection & Depth Estimation

A tool for detecting strawberries with segmentation and depth estimation.
"""

__version__ = "1.0.0"

from .detector import StrawberryDetector, detect

__all__ = ["StrawberryDetector", "detect", "__version__"]
