"""Bridge module for nail segmentation pipeline."""
import inspect
from typing import Optional
import numpy as np

try:
    from src.pipeline.ai_segmenter import YOLONailSegmenter as AINailSegmenter
except ImportError:
    try:
        from src.pipeline.ai_segmenter import AINailSegmenter
    except ImportError as exc:
        raise ImportError(
            "Could not import nail segmenter class from src.pipeline.ai_segmenter."
        ) from exc


class NailSegmenter:
    """Standard nail segmenter wrapper for pipeline consistency."""

    def __init__(self, *args, **kwargs):
        """Initialize underlying AI segmentation model."""
        self._model = AINailSegmenter(*args, **kwargs)

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Segment nail regions from the input image by dynamically resolving method names.

        Args:
            image_bgr: Input BGR image as a numpy array.

        Returns:
            Binary mask of segmented nails (uint8).
        """
        # Candidate method names commonly used in YOLO nail segmenter implementations
        candidate_methods = [
            "segment_nails",
            "segment",
            "get_mask",
            "predict_mask",
            "predict",
            "infer",
            "process",
            "__call__",
        ]

        for method_name in candidate_methods:
            if hasattr(self._model, method_name):
                method = getattr(self._model, method_name)
                if callable(method):
                    result = method(image_bgr)
                    # Handle return types: mask or tuple of (mask, metadata)
                    if isinstance(result, tuple):
                        return result[0]
                    return result

        available_callables = [
            attr for attr, val in inspect.getmembers(self._model, predicate=callable)
            if not attr.startswith("_")
        ]
        raise AttributeError(
            f"Loaded segmenter model has no known inference method. "
            f"Available public methods: {available_callables}"
        )
