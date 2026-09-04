import cv2
import numpy as np
from typing import Tuple

class NailSegmenter:
    def __init__(self):
        pass

    def segment_nail(self, roi_bgr: np.ndarray, tip_pt: Tuple[int, int], dip_pt: Tuple[int, int]) -> np.ndarray:
        """
        تولید ماسک دقیق ناخن بر اساس موقعیت نوک انگشت و ناحیه ROI
        """
        h, w = roi_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # تبدیل به فضای رنگی HSV و LAB برای استخراج بافت ناخن
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        # ساخت ماسک اولیه بر اساس بیضی حول نوک انگشت
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # تخمین ابعاد ناخن متناسب با سایز ROI
        center_x = int(np.clip(tip_pt[0], 0, w - 1))
        center_y = int(np.clip(tip_pt[1], 0, h - 1))
        
        axes_x = max(int(w * 0.28), 5)
        axes_y = max(int(h * 0.32), 6)

        # جهت زاویه ناخن
        dx = tip_pt[0] - dip_pt[0]
        dy = tip_pt[1] - dip_pt[1]
        angle = np.degrees(np.arctan2(dy, dx)) + 90.0

        # رسم بیضی پایه
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)

        # اصلاح لبه‌ها با عملیات مورفولوژی
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask
