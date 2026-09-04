import cv2
import numpy as np
from typing import Tuple

class NailRenderer:
    @staticmethod
    def hex_to_bgr(hex_code: str) -> Tuple[int, int, int]:
        hex_code = hex_code.lstrip('#')
        if len(hex_code) != 6:
            return (99, 30, 233)
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return (b, g, r)

    @staticmethod
    def blend_solid_color(roi_bgr: np.ndarray, mask: np.ndarray, target_bgr: Tuple[int, int, int]) -> np.ndarray:
        """
        ترکیب رنگ یکدست در فضای رنگی LAB با حفظ بافت و هایلایت
        """
        smooth_mask = cv2.GaussianBlur(mask, (5, 5), 1.5).astype(np.float32) / 255.0
        smooth_mask = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)

        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        color_block = np.full_like(roi_bgr, target_bgr, dtype=np.uint8)
        target_lab = cv2.cvtColor(color_block, cv2.COLOR_BGR2LAB).astype(np.float32)

        # استخراج هایلایت طبیعی از کانال L
        l_chan = roi_lab[:, :, 0]
        highlights = np.clip((l_chan - 180) * 1.5, 0, 75)

        blended_lab = roi_lab.copy()
        blended_lab[:, :, 0] = np.clip(roi_lab[:, :, 0] * 0.9 + highlights, 0, 255)
        blended_lab[:, :, 1] = target_lab[:, :, 1]
        blended_lab[:, :, 2] = target_lab[:, :, 2]

        blended_bgr = cv2.cvtColor(blended_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        output = roi_bgr.astype(np.float32) * (1.0 - smooth_mask) + blended_bgr * smooth_mask
        return np.clip(output, 0, 255).astype(np.uint8)

    @staticmethod
    def blend_pattern(roi_bgr: np.ndarray, mask: np.ndarray, warped_pattern: np.ndarray) -> np.ndarray:
        """
        ترکیب طرح/طراحی ناخن با مدهای نوری سایه‌گذاری و درخشش براق (Specular)
        """
        smooth_mask = cv2.GaussianBlur(mask, (5, 5), 1.5).astype(np.float32) / 255.0
        smooth_mask = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)

        if warped_pattern.shape[2] == 4:
            pattern_alpha = (warped_pattern[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            pattern_bgr = warped_pattern[:, :, :3].astype(np.float32)
            smooth_mask = smooth_mask * pattern_alpha
        else:
            pattern_bgr = warped_pattern.astype(np.float32)

        # ۱. استخراج سایه طبیعی زیرین
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shading = np.repeat(roi_gray[:, :, np.newaxis], 3, axis=2)

        # ۲. اعمال سایه روی طرح (Multiply Blending)
        shaded_pattern = pattern_bgr * (shading * 0.7 + 0.3)

        # ۳. استخراج و اعمال هایلایت‌های درخشان (Glossy / Specular Highlights)
        specular = np.maximum(0, roi_gray - 0.75) / 0.25
        specular_layer = np.repeat(specular[:, :, np.newaxis], 3, axis=2) * 190.0

        final_nail = np.clip(shaded_pattern + specular_layer, 0, 255)
        output = roi_bgr.astype(np.float32) * (1.0 - smooth_mask) + final_nail * smooth_mask
        return np.clip(output, 0, 255).astype(np.uint8)
