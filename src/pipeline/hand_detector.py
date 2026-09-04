import cv2
import mediapipe as mp
from typing import List, Dict, Any
import numpy as np

class HandDetector:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        # اندیس لندمارک‌های نوک انگشتان و مفاصل قبلی آن‌ها
        self.FINGERTIP_INDICES = {
            "THUMB": (4, 3),
            "INDEX": (8, 7),
            "MIDDLE": (12, 11),
            "RING": (16, 15),
            "PINKY": (20, 19)
        }

    def detect_fingertips(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """تشخیص نوک انگشتان و تخمین کادر برش (ROI) اطراف ناخن"""
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        detected_nails = []
        if not results.multi_hand_landmarks:
            return detected_nails

        for hand_landmarks in results.multi_hand_landmarks:
            for finger_name, (tip_idx, dip_idx) in self.FINGERTIP_INDICES.items():
                tip = hand_landmarks.landmark[tip_idx]
                dip = hand_landmarks.landmark[dip_idx]

                tip_px = (int(tip.x * w), int(tip.y * h))
                dip_px = (int(dip.x * w), int(dip.y * h))

                # تخمین اندازه ناخن بر اساس فاصله مفصل تا نوک انگشت
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
