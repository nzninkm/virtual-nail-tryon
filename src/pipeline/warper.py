import cv2
import numpy as np

class NailWarper:
    @staticmethod
    def apply_cylindrical_curvature(pattern_img: np.ndarray, strength: float = 0.35) -> np.ndarray:
        """
        اعمال اعوجاج استوانه‌ای برای شبیه‌سازی برآمدگی و خمیدگی سطح ناخن
        """
        h, w = pattern_img.shape[:2]
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        theta = xx * (np.pi / 2.0) * strength
        mapped_x = np.sin(theta) / np.sin((np.pi / 2.0) * strength)

        map_x = ((mapped_x + 1.0) * 0.5 * (w - 1)).astype(np.float32)
        map_y = ((yy + 1.0) * 0.5 * (h - 1)).astype(np.float32)

        curved = cv2.remap(pattern_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return curved

    @staticmethod
    def warp_pattern_to_quad(pattern_img: np.ndarray, target_quad: np.ndarray, out_shape: tuple) -> np.ndarray:
        """
        مپ پرسپکتیو تصویر طرح به ۴ نقطه ناخن
        """
        ph, pw = pattern_img.shape[:2]
        src_pts = np.array([
            [0, 0],
            [pw - 1, 0],
            [pw - 1, ph - 1],
            [0, ph - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, target_quad)
        
        warped = cv2.warpPerspective(
            pattern_img,
            matrix,
            (out_shape[1], out_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0) if pattern_img.shape[2] == 4 else (0, 0, 0)
        )
        return warped
