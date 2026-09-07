"""AI-driven nail segmentation using pre-trained YOLOv8 segmentation models."""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from ultralytics import YOLO


class YOLONailSegmenter:
    """Performs precise nail segmentation using a pre-trained YOLO segmentation network."""

    DEFAULT_MODEL = "mnemic/nails_seg_yolov8"

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """Initialize YOLO model for nail segmentation."""
        target_model = model_path if model_path else self.DEFAULT_MODEL
        print(f"[INFO] Loading YOLO nail segmentation model: {target_model}")
        self.model = YOLO(target_model)
        self.conf_threshold = conf_threshold

    def segment_nails(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segment all nails in the image.

        Returns:
            accumulated_mask: Binary uint8 mask (H, W) covering all detected nails (values 0 or 255).
            individual_masks: List of binary uint8 masks for each detected nail.
        """
        h, w = image_bgr.shape[:2]
        accumulated_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks = []

        # Run inference
        results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)

        if not results or results[0].masks is None:
            print("[WARNING] No nails detected by YOLO segmentation model.")
            return accumulated_mask, individual_masks

        # Extract masks
        masks_tensor = results[0].masks.data  # Tensor of shape (N, H_out, W_out)
        
        for mask_t in masks_tensor:
            # Convert to numpy uint8
            mask_np = (mask_t.cpu().numpy() * 255).astype(np.uint8)
            
            # Resize mask to original image dimensions if needed
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

            # Morphological smoothing to remove noise along nail boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

            individual_masks.append(mask_np)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask_np)

        return accumulated_mask, individual_masks
