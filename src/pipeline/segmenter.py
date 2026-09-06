"""Hybrid nail plate segmentation combining adaptive vision heuristics with optional deep AI models."""
import cv2
import numpy as np
from typing import Tuple, Optional

from src.pipeline.ai_segmenter import AINailSegmenter


class NailSegmenter:
    """Adaptive nail segmentation engine with AI priority and fallback algorithmic routines."""

    def __init__(self, ai_model_path: Optional[str] = "assets/models/nail_seg.onnx"):
        self.ai_engine = AINailSegmenter(ai_model_path)

    def segment_nail(
        self,
        roi_bgr: np.ndarray,
        tip_local: Tuple[int, int],
        dip_local: Tuple[int, int],
    ) -> np.ndarray:
        """
        Segment nail area using AI if available, otherwise apply skin-differential corridor heuristics.
        """
        if roi_bgr.size == 0:
            return np.zeros((10, 10), dtype=np.uint8)

        # 1. Primary path: Deep-learning AI segmentation
        ai_mask = self.ai_engine.segment_roi(roi_bgr)
        if ai_mask is not None and np.sum(ai_mask) > 100:
            return ai_mask

        # 2. Fallback path: Adaptive color-space differential segmentation
        return self._segment_heuristic(roi_bgr, tip_local, dip_local)

    def _segment_heuristic(
        self,
        roi_bgr: np.ndarray,
        tip_local: Tuple[int, int],
        dip_local: Tuple[int, int],
    ) -> np.ndarray:
        """Algorithmic color-space and morphological fallback mask generator."""
        h, w = roi_bgr.shape[:2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        # Sample local skin baseline from DIP joint proximity
        skin_y = max(0, min(h - 1, dip_local[1]))
        skin_x = max(0, min(w - 1, dip_local[0]))
        skin_patch = lab[max(0, skin_y - 4):min(h, skin_y + 5), max(0, skin_x - 4):min(w, skin_x + 5)]
        skin_mean_b = np.mean(skin_patch[:, :, 2]) if skin_patch.size > 0 else 145.0

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(lab[:, :, 0])

        # Differential thresholding: Nail plate is typically less yellow than skin
        b_channel = lab[:, :, 2]
        nail_candidate = (b_channel < (skin_mean_b - 5)).astype(np.uint8) * 255

        # Refine with Saturation and Value constraints
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        sat_val_mask = ((s_channel > 20) & (s_channel < 180) & (v_channel > 60)).astype(np.uint8) * 255
        combined = cv2.bitwise_and(nail_candidate, sat_val_mask)

        # Directional corridor mask around fingertip axis
        corridor = np.zeros((h, w), dtype=np.uint8)
        cv2.line(corridor, dip_local, tip_local, 255, thickness=int(max(10, w * 0.45)))
        cv2.circle(corridor, tip_local, int(max(8, w * 0.25)), 255, -1)

        gated = cv2.bitwise_and(combined, corridor)

        # Clean-up morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(gated, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Retain largest connected component
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 40:
                cv2.drawContours(final_mask, [largest], -1, 255, -1)

        return final_mask
