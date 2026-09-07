"""AI-driven nail segmentation using pre-trained YOLOv8 segmentation models."""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO


class YOLONailSegmenter:
    """Performs precise nail segmentation using a pre-trained YOLOv8 segmentation network."""

    MODEL_FILENAME = "nails_seg_s_yolov8_v1.pt"
    MODEL_DOWNLOAD_URL = (
        "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
    )

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """
        Initialize the YOLO nail segmenter.
        Automatically downloads weights if not present locally.
        """
        resolved_weights_path = self._resolve_model_path(model_path)
        print(f"[INFO] Loading YOLO nail segmentation weights: {resolved_weights_path}")
        self.model = YOLO(resolved_weights_path)
        self.conf_threshold = conf_threshold

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve model weights path and download checkpoint if necessary."""
        if model_path and os.path.exists(model_path):
            return model_path

        # Determine project root and models directory
        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        local_weights = models_dir / self.MODEL_FILENAME

        if not local_weights.exists():
            print(f"[INFO] Model weights not found locally. Downloading from {self.MODEL_DOWNLOAD_URL} ...")
            try:
                urllib.request.urlretrieve(self.MODEL_DOWNLOAD_URL, str(local_weights))
                print(f"[INFO] Download completed and saved to: {local_weights}")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download YOLO nail weights from Hugging Face: {exc}"
                ) from exc

        return str(local_weights)

    def segment_nails(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segment all visible nails in the input image.

        Args:
            image_bgr: Input image in BGR format (H, W, 3).

        Returns:
            accumulated_mask: Combined binary uint8 mask of all nails (H, W) with values 0 or 255.
            individual_masks: List of individual binary uint8 masks for each detected nail.
        """
        h, w = image_bgr.shape[:2]
        accumulated_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks: List[np.ndarray] = []

        # Run model inference
        results = self.model(image_bgr, conf=self.conf_threshold, verbose=False)

        if not results or results[0].masks is None:
            print("[WARNING] No nails detected by YOLO segmentation model.")
            return accumulated_mask, individual_masks

        # Extract predicted segmentation masks
        masks_tensor = results[0].masks.data

        for mask_t in masks_tensor:
            mask_np = (mask_t.cpu().numpy() * 255).astype(np.uint8)

            # Resize to original input image dimensions if necessary
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

            # Smooth mask boundaries using morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

            individual_masks.append(mask_np)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask_np)

        return accumulated_mask, individual_masks
