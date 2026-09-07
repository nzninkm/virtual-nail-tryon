"""Bridge module for nail segmentation pipeline."""
import numpy as np
from src.pipeline.ai_segmenter import YOLONailSegmenter


class NailSegmenter:
    """Standard nail segmenter interface for try-on pipeline."""

    def __init__(self, *args, **kwargs):
        """Initialize underlying YOLO segmentation engine."""
        self._model = YOLONailSegmenter(*args, **kwargs)

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segment nail regions from input image.

        Args:
            image_bgr: Input BGR image array.

        Returns:
            Binary mask (uint8).
        """
        return self._model.segment_nails(image_bgr)
