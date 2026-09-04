import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Dict, Any
import numpy as np
import os

class HandDetector:
    def __init__(self, model_path: str = "hand_landmarker.task", max_num_hands: int = 2):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"فایل مدل {model_path} یافت نشد. لطفاً دستور دانلود مدل را اجرا کنید."
            )
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.FINGERTIP_INDICES = {
            "THUMB": (4, 3),
            "INDEX": (8, 7),
            "MIDDLE": (12, 11),
            "RING": (16, 15),
            "PINKY": (20, 19)
        }

    def detect_fingertips(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.detector.detect(mp_image)
        detected_nails = []

        if not detection_result.hand_landmarks:
            return detected_nails

        for hand_landmarks in detection_result.hand_landmarks:
            for finger_name, (tip_idx, dip_idx) in self.FINGERTIP_INDICES.items():
                tip = hand_landmarks[tip_idx]
                dip = hand_landmarks[dip_idx]

                tip_px = (int(tip.x * w), int(tip.y * h))
                dip_px = (int(dip.x * w), int(dip.y * h))

                dist = np.linalg.norm(np.array(tip_px) - np.array(dip_px))
                box_radius = max(int(dist * 0.9), 20)

                x1 = max(0, tip_px[0] - box_radius)
                y1 = max(0, tip_px[1] - box_radius)
                x2 = min(w, tip_px[0] + box_radius)
                y2 = min(h, tip_px[1] + box_radius)

                detected_nails.append({
                    "finger": finger_name,
                    "tip": tip_px,
                    "dip": dip_px,
                    "roi_box": (x1, y1, x2, y2)
                })
        return detected_nails
