"""AI-driven semantic segmentation module for precision nail boundary detection."""
import os
import cv2
import numpy as np
from typing import Optional


class AINailSegmenter:
    """Inference engine supporting deep-learning nail mask segmentation models."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.net = None

        if model_path and os.path.exists(model_path):
            try:
                # Support loading pre-trained ONNX models via OpenCV DNN
                self.net = cv2.dnn.readNetFromONNX(model_path)
                print(f"[INFO] Loaded AI segmentation model from: {model_path}")
            except Exception as err:
                print(f"[WARNING] Failed to load ONNX model ({err}). Falling back to algorithmic segmenter.")
                self.net = None

    def segment_roi(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Segment the nail plate within a given fingertip ROI using deep neural net inference.
        Returns binary mask (uint8: 0 or 255).
        """
        if self.net is None or roi_bgr.size == 0:
            return None

        h, w = roi_bgr.shape[:2]
        # Preprocessing: standard 256x256 blob with normalization
        blob = cv2.dnn.blobFromImage(
            roi_bgr,
            scalefactor=1.0 / 255.0,
            size=(256, 256),
            mean=(0.485, 0.456, 0.406),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()

        # Extract single channel probability map
        prob_map = output[0, 0, :, :]
        prob_map = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob_map > 0.5).astype(np.uint8) * 255
        return mask
