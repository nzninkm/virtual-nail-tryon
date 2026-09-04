import cv2
import numpy as np

class NailRenderer:
    @staticmethod
    def hex_to_bgr(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def render_color(self, 
                     image_bgr: np.ndarray, 
                     mask: np.ndarray, 
                     hex_color: str = "#B71C1C", 
                     opacity: float = 0.85, 
                     glossiness: float = 0.4) -> np.ndarray:
        """اعمال رنگ در فضای رنگی Lab و حفظ بافت و هایلایت نوری"""
        if np.sum(mask) == 0:
            return image_bgr.copy()

        # ۱. نرم‌سازی مرزهای ماسک (Anti-aliasing)
        feathered_mask = cv2.GaussianBlur(mask, (7, 7), 0)
        alpha = (feathered_mask.astype(np.float32) / 255.0) * opacity
        alpha = np.expand_dims(alpha, axis=2)

        # ۲. تولید لایه رنگ هدف
        target_bgr = self.hex_to_bgr(hex_color)
        color_layer = np.full_like(image_bgr, target_bgr, dtype=np.uint8)

        # ۳. پردازش در فضای رنگی Lab
        img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        color_lab = cv2.cvtColor(color_layer, cv2.COLOR_BGR2Lab).astype(np.float32)

        rendered_lab = img_lab.copy()
        rendered_lab[:, :, 1] = color_lab[:, :, 1]
        rendered_lab[:, :, 2] = color_lab[:, :, 2]

        rendered_bgr = cv2.cvtColor(rendered_lab.astype(np.uint8), cv2.COLOR_Lab2BGR).astype(np.float32)

        # ۴. حفظ هایلایت‌ها و درخشش طبیعی ناخن
        gray_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(gray_orig, 200, 255, cv2.THRESH_BINARY)
        highlight_layer = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
        
        rendered_bgr += (highlight_layer * glossiness)
        rendered_bgr = np.clip(rendered_bgr, 0, 255)

        # ۵. ترکیب لایه‌ای با آلفا
        output = (alpha * rendered_bgr + (1.0 - alpha) * image_bgr.astype(np.float32)).astype(np.uint8)
        return output
