import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Optional, List, Dict, Any

class NailSegmenter:
    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        if model_path and os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        else:
            print("[INFO] ONNX model not found. Running in heuristic ROI-masking fallback mode.")

    def _generate_fallback_mask(self, nail_info: Dict[str, Any], full_shape: tuple) -> np.ndarray:
        """تولید ماسک بیضی مایل بر اساس زاویه انگشت برای تست بدون مدل"""
        mask = np.zeros(full_shape[:2], dtype=np.uint8)
        tip = np.array(nail_info["tip"])
        dip = np.array(nail_info["dip"])
        
        center = ((tip + dip) / 2).astype(int)
        vec = tip - dip
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        length = int(np.linalg.norm(vec) * 0.8)
        width = int(length * 0.6)

        cv2.ellipse(mask, (int(center[0]), int(center[1])), (length // 2, width // 2),
                    angle, 0, 360, 255, -1)
        return mask

    def segment_nails(self, image_bgr: np.ndarray, detected_nails: List[Dict[str, Any]]) -> np.ndarray:
        """خروجی نهایی: ماسک باینری (۰ و ۲۵۵) به اندازه تصویر اصلی"""
        h, w, _ = image_bgr.shape
        full_mask = np.zeros((h, w), dtype=np.uint8)

        for nail in detected_nails:
            if self.session is None:
                mask_roi = self._generate_fallback_mask(nail, image_bgr.shape)
                full_mask = cv2.bitwise_or(full_mask, mask_roi)
            else:
                x1, y1, x2, y2 = nail["roi_box"]
                roi = image_bgr[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                input_tensor = cv2.resize(roi, (256, 256))
                input_tensor = input_tensor.astype(np.float32) / 255.0
                input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]
                
                outputs = self.session.run(None, {self.input_name: input_tensor})
                seg_out = outputs[0][0][0]
                seg_mask = (seg_out > 0.5).astype(np.uint8) * 255
                seg_mask_resized = cv2.resize(seg_mask, (x2 - x1, y2 - y1))
                
                full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], seg_mask_resized)

        return full_mask
