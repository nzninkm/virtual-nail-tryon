import cv2
import numpy as np
from typing import Tuple

class NailSegmenter:
    def __init__(self):
        pass

    def segment_nail(self, roi_bgr: np.ndarray, tip_pt: Tuple[int, int], dip_pt: Tuple[int, int]) -> np.ndarray:
        """
        استخراج دقیق و تطبیقی مرزهای صدف ناخن با تصحیح مرکز و تفکیک بافت
        """
        h, w = roi_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # ۱. محاسبه بردار رشد و جهت انگشت
        dx = float(tip_pt[0] - dip_pt[0])
        dy = float(tip_pt[1] - dip_pt[1])
        dist = np.hypot(dx, dy)

        if dist < 1e-4:
            unit_dx, unit_dy = 0.0, -1.0
        else:
            unit_dx, unit_dy = dx / dist, dy / dist

        # ۲. تصحیح موقعیت مرکز: صدف ناخن کمی عقب‌تر از نوک گوشتی انگشت قرار دارد
        nail_offset = dist * 0.22
        nail_cx = tip_pt[0] - unit_dx * nail_offset
        nail_cy = tip_pt[1] - unit_dy * nail_offset

        # ابعاد واقعی‌تر صدف ناخن (کشیده‌تر در راستای رشد)
        length = max(int(dist * 0.75), 10)  # طول در راستای انگشت
        width = max(int(dist * 0.52), 8)    # عرض ناخن

        angle_deg = np.degrees(np.arctan2(unit_dy, unit_dx))

        # ۳. ساخت ماسک هندسی پایه به شکل صدف ناخن (مستطیل با گوشه‌های گرد)
        base_mask = np.zeros((h, w), dtype=np.uint8)
        rect = ((nail_cx, nail_cy), (length, width), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.fillPoly(base_mask, [box], 255)

        # گرد کردن انتهای صدف ناخن متناسب با ریشه
        cv2.ellipse(
            base_mask,
            (int(nail_cx), int(nail_cy)),
            (int(length * 0.52), int(width * 0.5)),
            angle_deg,
            0,
            360,
            255,
            -1
        )

        # ۴. پردازش تطبیقی گرادیان و رنگ برای فیت شدن روی لبه‌های واقعی ناخن
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        contrast_gray = clahe.apply(gray)

        # استخراج لبه‌ها و گرادیان محلی
        blurred = cv2.GaussianBlur(contrast_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ۵. ترکیب ماسک اولیه با ویژگی‌های ساختاری ROI
        refined_mask = cv2.bitwise_and(base_mask, base_mask, mask=thresh)

        # اگر سگمنتیشن محلی خیلی کوچک شد، از نسخه مورفولوژی شده پایه استفاده می‌کنیم
        if cv2.countNonZero(refined_mask) < (cv2.countNonZero(base_mask) * 0.45):
            refined_mask = base_mask

        # ۶. اعمال فیلتر مورفولوژی برای صاف کردن حاشیه‌ها
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        refined_mask = cv2.GaussianBlur(refined_mask, (3, 3), 0)
        _, refined_mask = cv2.threshold(refined_mask, 127, 255, cv2.THRESH_BINARY)

        return refined_mask
