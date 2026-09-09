import cv2
import numpy as np

class NailPainter:
    """
    Advanced nail color overlay using alpha blending and 
    color preservation techniques.
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha # Transparency factor

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def paint(self, image, masks, color_hex):
        """
        Applies color to detected nail regions while preserving 
        original texture and lighting.
        """
        rgb_color = self.hex_to_rgb(color_hex)
        # Convert BGR to RGB for consistent processing
        bgr_color = rgb_color[::-1] 
        
        result = image.copy()
        
        # Check if masks is a dictionary (from some segmenters) or a direct numpy array
        if isinstance(masks, dict):
            combined_mask = masks.get('segmentation', None)
        else:
            combined_mask = masks

        if combined_mask is None:
            return image

        # Refine mask: Smoothing and Noise reduction
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask.astype(float), (5, 5), 0)

        # Ensure mask is 3-channel for element-wise operations
        mask_3ch = cv2.merge([combined_mask, combined_mask, combined_mask])
        
        # Color Overlay Logic:
        # Result = (Image * (1 - Alpha * Mask)) + (Color * Alpha * Mask)
        # This keeps the original shadows/texture visible under the color
        overlay = np.full(image.shape, bgr_color, dtype=np.uint8)
        
        # Normalized mask for blending (0.0 to 1.0)
        normalized_mask = mask_3ch if mask_3ch.max() <= 1.0 else mask_3ch / 255.0
        
        # Weighted blending
        result = (image * (1 - self.alpha * normalized_mask) + 
                  overlay * (self.alpha * normalized_mask)).astype(np.uint8)

        return result
