"""YOLO-based AI segmentation pipeline for nail detection."""
import os
import urllib.request
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


class YOLONailSegmenter:
    """Handles deep-learning based nail segmentation using YOLOv8."""

    MODEL_URL = "https://hf-mirror.com/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"
    FALLBACK_URL = "https://huggingface.co/mnemic/nails_seg_yolov8/resolve/main/nails_seg_s_yolov8_v1.pt"

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        """
        Initialize the YOLO nail segmenter.

        Args:
            model_path: Local path to weights. If None, resolves to default directory.
            conf_threshold: Minimum confidence score for detection.
        """
        self.conf_threshold = conf_threshold
        if model_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            self.model_path = os.path.join(models_dir, "nails_seg_s_yolov8_v1.pt")
        else:
            self.model_path = model_path

        self._ensure_model_exists()
        print(f"[INFO] Loading YOLO nail segmentation weights: {self.model_path}")
        self.model = YOLO(self.model_path)

    def _ensure_model_exists(self) -> None:
        """Download model weights automatically if missing."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
            return

        print(f"[INFO] Downloading YOLO weights to {self.model_path}...")
        for url in [self.MODEL_URL, self.FALLBACK_URL]:
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [("User-Agent", "Mozilla/5.0")]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, self.model_path)
                if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000:
                    print("[INFO] Model weights downloaded successfully.")
                    return
            except Exception as e:
                print(f"[WARN] Failed downloading from {url}: {e}")

        raise RuntimeError("Could not download YOLO weights from primary or fallback mirrors.")

    def segment_nails(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extract binary nail segmentation mask from input image.

        Args:
            image_bgr: Input image in BGR format.

        Returns:
            Binary mask (uint8) with values 0 or 255.
        """
        h, w = image_bgr.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )

        if not results or len(results) == 0:
            return combined_mask

        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return combined_mask

        masks_tensor = result.masks.data.cpu().numpy()
        for single_mask in masks_tensor:
            resized_mask = cv2.resize(
                single_mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

        return combined_mask

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Standard alias for segment_nails."""
        return self.segment_nails(image_bgr)

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Standard alias for segment_nails."""
        return self.segment_nails(image_bgr)


# Backward-compatibility alias
AINailSegmenter = YOLONailSegmenter
