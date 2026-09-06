"""Hand landmark detection and fingertip region-of-interest extraction using MediaPipe."""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class HandDetector:
    """Detects hand landmarks and isolates fingertip regions for nail processing."""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

        # Indices for 5 fingertips and their preceding DIP joints
        # Format: (Tip landmark index, DIP landmark index)
        self.finger_indices = [
            (4, 3),    # Thumb
            (8, 7),    # Index
            (12, 11),  # Middle
            (16, 15),  # Ring
            (20, 19),  # Pinky
        ]

    def detect(self, image_bgr: np.ndarray):
        """Process BGR image and return raw hand detection landmarks."""
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        return None

    def get_fingertip_rois(
        self,
        image_bgr: np.ndarray,
        multi_landmarks: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes and local pixel coordinates for all detected fingertips.
        """
        if not multi_landmarks:
            return []

        h, w = image_bgr.shape[:2]
        rois = []

        for hand_landmarks in multi_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            for tip_idx, dip_idx in self.finger_indices:
                tip_pt = pts[tip_idx]
                dip_pt = pts[dip_idx]

                # Estimate fingertip size from distance between DIP and TIP
                seg_length = np.hypot(tip_pt[0] - dip_pt[0], tip_pt[1] - dip_pt[1])
                roi_radius = int(max(20, seg_length * 1.25))

                # Compute ROI bounding box
                center_x = tip_pt[0]
                center_y = tip_pt[1]

                x1 = max(0, center_x - roi_radius)
                y1 = max(0, center_y - roi_radius)
                x2 = min(w, center_x + roi_radius)
                y2 = min(h, center_y + roi_radius)

                roi_crop = image_bgr[y1:y2, x1:x2]
                if roi_crop.size == 0:
                    continue

                # Local coordinates relative to the cropped ROI
                tip_local = (tip_pt[0] - x1, tip_pt[1] - y1)
                dip_local = (dip_pt[0] - x1, dip_pt[1] - y1)

                rois.append({
                    "roi": roi_crop,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "tip_local": tip_local,
                    "dip_local": dip_local,
                    "tip_global": tip_pt,
                    "dip_global": dip_pt,
                })

        return rois
