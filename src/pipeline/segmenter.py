import cv2
import numpy as np
from typing import Tuple

class NailSegmenter:
    def __init__(self):
        pass

    def segment_nail(self, roi_bgr: np.ndarray, tip_pt: Tuple[int, int], dip_pt: Tuple[int, int]) -> np.ndarray:
        """
        Extract precise nail contour using adaptive GrabCut segmentation guided by finger orientation.
        """
        h, w = roi_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # 1. Compute finger orientation vector
        dx = float(tip_pt[0] - dip_pt[0])
        dy = float(tip_pt[1] - dip_pt[1])
        dist = np.hypot(dx, dy)

        if dist < 1e-4:
            unit_dx, unit_dy = 0.0, -1.0
        else:
            unit_dx, unit_dy = dx / dist, dy / dist

        # 2. Offset nail bed center from the flesh fingertip towards the cuticle
        nail_offset = dist * 0.28
        nail_cx = tip_pt[0] - unit_dx * nail_offset
        nail_cy = tip_pt[1] - unit_dy * nail_offset

        # 3. Calculate dynamic nail dimensions based on distance
        nail_length = max(int(dist * 0.72), 12)
        nail_width = max(int(dist * 0.48), 10)
        angle_deg = np.degrees(np.arctan2(unit_dy, unit_dx))

        # 4. Initialize GrabCut mask
        # cv2.GC_BGD = 0, cv2.GC_FGD = 1, cv2.GC_PR_BGD = 2, cv2.GC_PR_FGD = 3
        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

        # Probable foreground: oriented bounding box around nail bed
        rect = ((nail_cx, nail_cy), (nail_length * 1.15, nail_width * 1.15), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.fillPoly(gc_mask, [box], cv2.GC_PR_FGD)

        # Definite foreground: compact core of the nail plate
        core_rect = ((nail_cx, nail_cy), (nail_length * 0.55, nail_width * 0.55), angle_deg)
        core_box = cv2.boxPoints(core_rect)
        core_box = np.intp(core_box)
        cv2.fillPoly(gc_mask, [core_box], cv2.GC_FGD)

        # 5. Run GrabCut optimization
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                roi_bgr,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                iterCount=3,
                mode=cv2.GC_INIT_WITH_MASK
            )
            final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            # Fallback to oriented geometric box if GrabCut encounters degenerate bounds
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(final_mask, [box], 255)

        # 6. Morphological refinement to eliminate stray edge noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 7. Extract the primary connected component closest to nail center
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt = None
            min_dist = float("inf")
            for cnt in contours:
                if cv2.contourArea(cnt) < 25:
                    continue
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    d = np.hypot(cx - nail_cx, cy - nail_cy)
                    if d < min_dist:
                        min_dist = d
                        best_cnt = cnt

            if best_cnt is not None:
                isolated_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(isolated_mask, [best_cnt], -1, 255, -1)
                final_mask = isolated_mask

        return final_mask
