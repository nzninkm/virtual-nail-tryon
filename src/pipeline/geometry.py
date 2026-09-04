import cv2
import numpy as np
from typing import Tuple, Optional

class NailGeometry:
    @staticmethod
    def extract_nail_quad(mask: np.ndarray, tip_pt: Tuple[int, int], dip_pt: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        استخراج ۴ نقطه کلیدی ناخن به ترتیب:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 20:
            return None

        # برازش حداقل مستطیل چرخان روی ماسک
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        # محاسبه بردار رشد ناخن (از سمت مفصل به سمت نوک انگشت)
        growth_vec = np.array(tip_pt, dtype=np.float32) - np.array(dip_pt, dtype=np.float32)
        norm = np.linalg.norm(growth_vec)
        if norm > 1e-6:
            growth_vec /= norm
        else:
            growth_vec = np.array([0.0, -1.0], dtype=np.float32)

        # پروجکشن نقاط روی بردار رشد برای تفکیک ریشه و نوک
        projections = np.dot(box, growth_vec)
        sorted_idx = np.argsort(projections)

        root_pts = box[sorted_idx[:2]]   # نقاط نزدیک ریشه (Cuticle)
        tip_pts = box[sorted_idx[2:]]    # نقاط نزدیک نوک (Tip)

        # بردار عمود برای تفکیک چپ و راست
        perp_vec = np.array([-growth_vec[1], growth_vec[0]], dtype=np.float32)

        tip_left = tip_pts[0] if np.dot(tip_pts[0], perp_vec) < np.dot(tip_pts[1], perp_vec) else tip_pts[1]
        tip_right = tip_pts[1] if np.dot(tip_pts[0], perp_vec) < np.dot(tip_pts[1], perp_vec) else tip_pts[0]

        root_left = root_pts[0] if np.dot(root_pts[0], perp_vec) < np.dot(root_pts[1], perp_vec) else root_pts[1]
        root_right = root_pts[1] if np.dot(root_pts[0], perp_vec) < np.dot(root_pts[1], perp_vec) else root_pts[0]

        return np.array([tip_left, tip_right, root_right, root_left], dtype=np.float32)
