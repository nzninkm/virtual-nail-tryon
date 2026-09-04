"""Robust and universal nail segmentation module adapting to any hand pose and skin tone."""
import cv2
import numpy as np
from typing import Tuple, Optional


class NailSegmenter:
    """Universal nail segmentation combining directional ROI alignment, adaptive color analysis, and iterative GrabCut."""

    def __init__(self):
        pass

    def _align_and_crop(
        self,
        roi_bgr: np.ndarray,
        tip: np.ndarray,
        dip: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float], float]:
        """Align finger axis vertically to normalize perspective variations across different hand poses."""
        finger_vec = tip - dip
        finger_len = float(np.linalg.norm(finger_vec))

        if finger_len < 2.0:
            finger_vec = np.array([0.0, -1.0], dtype=np.float32)
            finger_len = float(max(roi_bgr.shape[0], 20))

        # Finger orientation angle
        angle_rad = np.arctan2(finger_vec[1], finger_vec[0])
        angle_deg = float(np.degrees(angle_rad))

        # Rotation matrix to align finger along vertical Y axis (pointing up)
        center = (float(tip[0]), float(tip[1]))
        rot_angle = angle_deg + 90.0
        rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)

        h, w = roi_bgr.shape[:2]
        aligned = cv2.warpAffine(
            roi_bgr,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        return aligned, rot_mat, rot_angle, center, finger_len

    def segment_nail(
        self,
        roi_bgr: np.ndarray,
        tip_pt: Tuple[int, int],
        dip_pt: Tuple[int, int],
    ) -> np.ndarray:
        """
        Segment fingernail boundary adaptively across diverse lighting, shapes, and angles.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        h, w = roi_bgr.shape[:2]
        tip = np.asarray(tip_pt, dtype=np.float32)
        dip = np.asarray(dip_pt, dtype=np.float32)

        # 1. Normalize orientation
        aligned_bgr, rot_mat, rot_angle, center, finger_len = self._align_and_crop(
            roi_bgr, tip, dip
        )

        # Tip position in aligned coordinate space
        aligned_tip = np.array([center[0], center[1]], dtype=np.float32)

        # 2. Define adaptive nail search bounds in aligned space
        # Covers natural short nails to extended acrylic/fake nails
        half_w = max(5.0, finger_len * 0.28)
        top_offset = finger_len * 0.18    # Space beyond fingertip for long nails
        bottom_offset = finger_len * 0.45 # Space towards DIP joint for nail base

        x_min = int(np.clip(round(aligned_tip[0] - half_w), 0, w - 1))
        x_max = int(np.clip(round(aligned_tip[0] + half_w), 0, w - 1))
        y_min = int(np.clip(round(aligned_tip[1] - top_offset), 0, h - 1))
        y_max = int(np.clip(round(aligned_tip[1] + bottom_offset), 0, h - 1))

        if x_max <= x_min or y_max <= y_min:
            return np.zeros((h, w), dtype=np.uint8)

        # 3. Multi-color space & contrast extraction
        lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

        # Sample local skin baseline below cuticle area
        skin_y1 = int(np.clip(round(aligned_tip[1] + finger_len * 0.40), 0, h - 1))
        skin_y2 = int(np.clip(round(aligned_tip[1] + finger_len * 0.70), 0, h - 1))
        skin_patch = lab[skin_y1:skin_y2, x_min:x_max]

        if skin_patch.size > 0:
            skin_mean = np.mean(skin_patch, axis=(0, 1))
            diff_map = np.linalg.norm(lab.astype(np.float32) - skin_mean, axis=2)
        else:
            diff_map = np.zeros((h, w), dtype=np.float32)

        # 4. Construct adaptive GrabCut probabilistic matrix
        gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

        # Search window defined as probable background
        gc_mask[y_min:y_max, x_min:x_max] = cv2.GC_PR_BGD

        # Define high-confidence probable foreground based on skin differentiation & brightness
        search_diff = diff_map[y_min:y_max, x_min:x_max]
        if search_diff.size > 0:
            diff_thresh = np.percentile(search_diff, 40)
            candidate_fg = (diff_map > diff_thresh) & (gc_mask == cv2.GC_PR_BGD)
            gc_mask[candidate_fg] = cv2.GC_PR_FGD

        # High confidence core prior (center of the expected nail)
        core_w = max(2.0, half_w * 0.45)
        core_y_top = int(np.clip(round(aligned_tip[1] - finger_len * 0.05), 0, h - 1))
        core_y_bot = int(np.clip(round(aligned_tip[1] + finger_len * 0.25), 0, h - 1))
        core_x_l = int(np.clip(round(aligned_tip[0] - core_w), 0, w - 1))
        core_x_r = int(np.clip(round(aligned_tip[0] + core_w), 0, w - 1))

        if core_x_r > core_x_l and core_y_bot > core_y_top:
            gc_mask[core_y_top:core_y_bot, core_x_l:core_x_r] = cv2.GC_FGD

        # 5. Run GrabCut
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                aligned_bgr,
                gc_mask,
                None,
                bgd_model,
                fgd_model,
                iterCount=5,
                mode=cv2.GC_INIT_WITH_MASK,
            )
            raw_seg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
        except cv2.error:
            raw_seg = np.zeros((h, w), dtype=np.uint8)
            raw_seg[core_y_top:core_y_bot, core_x_l:core_x_r] = 255

        # Restrict extraction strictly within allowable bounding box
        bounded_seg = np.zeros_like(raw_seg)
        bounded_seg[y_min:y_max, x_min:x_max] = raw_seg[y_min:y_max, x_min:x_max]

        # 6. Extract dominant continuous contour
        contours, _ = cv2.findContours(
            bounded_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        aligned_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            # Pick contour closest to the expected nail centroid
            target_centroid = np.array([aligned_tip[0], aligned_tip[1] + finger_len * 0.10])
            best_cnt = None
            min_dist = float("inf")

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (finger_len * finger_len * 0.04):
                    continue
                m = cv2.moments(cnt)
                if m["m00"] == 0:
                    continue
                centroid = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
                dist = float(np.linalg.norm(centroid - target_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_cnt = cnt

            if best_cnt is not None:
                cv2.drawContours(aligned_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
            else:
                # Fallback to morphological ellipse
                cv2.ellipse(
                    aligned_mask,
                    (int(round(target_centroid[0])), int(round(target_centroid[1]))),
                    (int(round(half_w * 0.75)), int(round(finger_len * 0.22))),
                    0,
                    0,
                    360,
                    255,
                    -1,
                )
        else:
            # Fallback ellipse
            cv2.ellipse(
                aligned_mask,
                (int(round(aligned_tip[0])), int(round(aligned_tip[1] + finger_len * 0.10))),
                (int(round(half_w * 0.75)), int(round(finger_len * 0.22))),
                0,
                0,
                360,
                255,
                -1,
            )

        # 7. Rotate mask back to original coordinate system
        inv_rot_mat = cv2.getRotationMatrix2D(center, -rot_angle, 1.0)
        final_mask = cv2.warpAffine(
            aligned_mask,
            inv_rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 8. Boundary smoothing and edge cleanup
        k_size = max(3, int(round(finger_len * 0.04)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 1.2)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

        return final_mask
