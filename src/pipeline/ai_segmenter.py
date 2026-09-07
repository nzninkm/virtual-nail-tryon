"""AI-driven nail segmentation using YOLOv8 with pre-processing and morphological refinements."""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import cv2
import numpy as np
from ultralytics import YOLO


class YOLONailSegmenter:
    """Performs nail segmentation using YOLOv8 with robust mirror downloading and adaptive pre-processing."""

    MODEL_FILENAME = "nails_seg_s_yolov8_v1.pt"
    DOWNLOAD_URLS = [
        "https://hf-mirror.com/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt",
        "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt",
    ]

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """
        Initialize segmenter and verify local model presence.

        Args:
            model_path: Optional custom path to model weights.
            conf_threshold: Minimum confidence score for detection.
        """
        resolved_weights = self._resolve_model_path(model_path)
        print(f"[INFO] Loading YOLO nail segmentation weights: {resolved_weights}")
        self.model = YOLO(resolved_weights)
        self.conf_threshold = conf_threshold

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve model weights path and download checkpoint if necessary."""
        if model_path and os.path.exists(model_path):
            return model_path

        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        local_weights = models_dir / self.MODEL_FILENAME

        if local_weights.exists() and local_weights.stat().st_size > 1000:
            return str(local_weights)

        print(f"[INFO] Model weights missing in '{models_dir}'. Attempting auto-download...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for url in self.DOWNLOAD_URLS:
            try:
                print(f"[INFO] Fetching weights from: {url}")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=40) as response, open(local_weights, "wb") as out_file:
                    out_file.write(response.read())
                print(f"[SUCCESS] Saved weights to: {local_weights}")
                return str(local_weights)
            except Exception as exc:
                print(f"[WARNING] Download attempt failed for {url}: {exc}")

        raise RuntimeError(
            f"Failed to auto-download model. Please manually download '{self.MODEL_FILENAME}' "
            f"and place it inside '{models_dir}'."
        )

    def _preprocess_lighting(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply adaptive contrast enhancement (CLAHE) on the luminance channel."""
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def segment_nails(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segment all visible nails from the input image.

        Args:
            image_bgr: Input BGR image (H, W, 3).

        Returns:
            accumulated_mask: Unified uint8 binary mask of all nails (H, W).
            individual_masks: List of uint8 binary masks for each detected nail.
        """
        h, w = image_bgr.shape[:2]
        accumulated_mask = np.zeros((h, w), dtype=np.uint8)
        individual_masks: List[np.ndarray] = []

        # Enhance lighting contrast before inference
        enhanced_image = self._preprocess_lighting(image_bgr)
        results = self.model(enhanced_image, conf=self.conf_threshold, verbose=False)

        if not results or results[0].masks is None:
            print("[WARNING] No nails detected in input image.")
            return accumulated_mask, individual_masks

        masks_tensor = results[0].masks.data

        for mask_t in masks_tensor:
            mask_np = (mask_t.cpu().numpy() * 255).astype(np.uint8)

            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

            # Gentle morphological closing to preserve sharp nail tips
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

            individual_masks.append(mask_np)
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask_np)

        return accumulated_mask, individual_masks
