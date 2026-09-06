"""Photorealistic nail rendering engine with shadow preservation and specular highlight synthesis."""
import cv2
import numpy as np
from typing import Tuple


class NailRenderer:
    """Renders solid colors and textures onto nail plates while preserving natural lighting dynamics."""

    def __init__(self, feather_radius: int = 5):
        self.feather_radius = feather_radius

    def apply_solid_color(
        self,
        base_image: np.ndarray,
        nail_mask: np.ndarray,
        color_bgr: Tuple[int, int, int],
        finish: str = "glossy",
        blend_strength: float = 0.85,
    ) -> np.ndarray:
        """
        Apply realistic polish color onto the nail area using LAB luminance modulation and specular extraction.

        :param base_image: Original full BGR image.
        :param nail_mask: Binary or grayscale single-channel nail mask.
        :param color_bgr: Desired nail color in (B, G, R) format.
        :param finish: Finish type ('glossy', 'matte', 'metallic').
        :param blend_strength: Blending intensity (0.0 to 1.0).
        :return: Output BGR image with realistic nail polish.
        """
        if nail_mask is None or np.sum(nail_mask) == 0:
            return base_image.copy()

        output = base_image.copy()
        h, w = base_image.shape[:2]

        # 1. Soften mask boundaries (Anti-Aliasing / Feathering)
        ksize = self.feather_radius * 2 + 1
        soft_mask = cv2.GaussianBlur(nail_mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
        soft_mask_3c = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)

        # 2. Extract Luminance and Lighting structure from original nail
        lab_img = cv2.cvtColor(base_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        # Normalize luminance to serve as lighting multiplier [0.3 to 1.3]
        l_norm = l_channel.astype(np.float32) / 255.0
        shadow_map = np.clip(l_norm * 1.4, 0.2, 1.2)
        shadow_map_3c = np.repeat(shadow_map[:, :, np.newaxis], 3, axis=2)

        # 3. Create target solid color canvas
        color_canvas = np.full_like(base_image, color_bgr, dtype=np.uint8).astype(np.float32)

        # 4. Modulate color canvas with original shadows (Preserve 3D depth)
        shaded_color = color_canvas * shadow_map_3c

        # 5. Extract and amplify Specular Highlights (Environmental reflections)
        # Isolate brightest pixels within nail region
        gray_img = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray_img)

        # Extract top 15% brightest highlights
        _, highlight_thresh = cv2.threshold(enhanced_gray, 200, 255, cv2.THRESH_BINARY)
        highlight_mask = cv2.GaussianBlur(highlight_thresh.astype(np.float32) / 255.0, (7, 7), 0)
        highlight_mask = highlight_mask * soft_mask  # Constrain strictly to nail plate

        highlight_3c = np.repeat(highlight_mask[:, :, np.newaxis], 3, axis=2)

        # Apply finish characteristics
        if finish == "glossy":
            specular_boost = highlight_3c * 180.0
            composite = (shaded_color * blend_strength) + (base_image.astype(np.float32) * (1.0 - blend_strength))
            composite = np.clip(composite + specular_boost, 0, 255)
        elif finish == "metallic":
            metallic_sheen = (1.0 - np.abs(l_norm - 0.6)) * 40.0
            metallic_sheen_3c = np.repeat(metallic_sheen[:, :, np.newaxis], 3, axis=2)
            specular_boost = highlight_3c * 220.0
            composite = (shaded_color * blend_strength) + metallic_sheen_3c + specular_boost
            composite = np.clip(composite, 0, 255)
        elif finish == "matte":
            # Flatten highlights for a smooth diffused matte texture
            diffused_shade = cv2.GaussianBlur(shaded_color, (9, 9), 0)
            composite = (diffused_shade * blend_strength) + (base_image.astype(np.float32) * (1.0 - blend_strength))
            composite = np.clip(composite, 0, 255)
        else:
            composite = shaded_color

        # 6. Final Alpha Blending with anti-aliased edge transition
        final_result = (composite * soft_mask_3c) + (base_image.astype(np.float32) * (1.0 - soft_mask_3c))
        return np.clip(final_result, 0, 255).astype(np.uint8)
