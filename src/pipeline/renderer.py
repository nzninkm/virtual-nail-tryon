"""Photorealistic nail color rendering with luminance preservation and specular highlights."""
from typing import Tuple, Optional, Any
import cv2
import numpy as np


class NailRenderer:
    """Renders realistic nail polish with multiple finish types (glossy, matte, metallic)."""

    FINISH_PRESETS = {
        "glossy": {"highlight_boost": 1.4, "feather_radius": 5, "opacity": 0.88},
        "matte": {"highlight_boost": 0.3, "feather_radius": 3, "opacity": 0.92},
        "metallic": {"highlight_boost": 1.9, "feather_radius": 5, "opacity": 0.85},
    }

    def __init__(self, feather_radius: Optional[int] = None, default_opacity: Optional[float] = None):
        """
        Initialize the nail renderer with optional override defaults.

        Args:
            feather_radius: Optional default blur radius for mask feathering.
            default_opacity: Optional base alpha opacity for polish blending.
        """
        self.default_feather_radius = feather_radius
        self.default_opacity = default_opacity

    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Convert a hex color string (e.g. #B22222 or B22222) to BGR tuple."""
        cleaned_hex = hex_color.lstrip("#")
        if len(cleaned_hex) != 6:
            raise ValueError(f"Invalid hex color format: {hex_color}")
        r = int(cleaned_hex[0:2], 16)
        g = int(cleaned_hex[2:4], 16)
        b = int(cleaned_hex[4:6], 16)
        return (b, g, r)

    def render(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        hex_color: str = "#B22222",
        finish: str = "glossy",
    ) -> np.ndarray:
        """
        Apply photorealistic nail polish to the target mask area.

        Args:
            image_bgr: Original input image (H, W, 3) in uint8 BGR.
            mask: Binary nail mask (H, W) in uint8 (0 or 255).
            hex_color: Target nail polish color in hex string format.
            finish: Finish style preset ('glossy', 'matte', 'metallic').

        Returns:
            Rendered composite image in uint8 BGR.
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return image_bgr.copy()

        preset = self.FINISH_PRESETS.get(finish.lower(), self.FINISH_PRESETS["glossy"])
        feather_size = (
            self.default_feather_radius
            if self.default_feather_radius is not None
            else preset["feather_radius"]
        )
        opacity = (
            self.default_opacity
            if self.default_opacity is not None
            else preset["opacity"]
        )
        target_bgr = self.hex_to_bgr(hex_color)

        # 1. Soften mask boundaries (Anti-Aliasing / Feathering)
        if feather_size % 2 == 0:
            feather_size += 1
        soft_mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0).astype(np.float32) / 255.0
        soft_mask_3ch = np.repeat(soft_mask[:, :, np.newaxis], 3, axis=2)

        # 2. Extract luminance channel from original image in LAB space
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        norm_l = (l_channel.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # 3. Create target solid color layer and blend with base luminance (Multiply shading)
        color_layer = np.full_like(image_bgr, target_bgr, dtype=np.float32)
        shaded_color = color_layer * (norm_l * 0.8 + 0.2)

        # 4. Extract specular reflection highlights
        _, highlight_mask = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)
        highlight_soft = cv2.GaussianBlur(highlight_mask, (5, 5), 0).astype(np.float32) / 255.0
        highlight_layer = (highlight_soft * preset["highlight_boost"] * 255.0)[:, :, np.newaxis]
        highlight_layer = np.clip(highlight_layer, 0.0, 255.0)

        # 5. Composite metallic or glossy highlights using additive screen blending
        final_nail_layer = shaded_color + (highlight_layer * 0.6)
        final_nail_layer = np.clip(final_nail_layer, 0.0, 255.0)

        # 6. Alpha composite rendered nails onto the original image
        base_float = image_bgr.astype(np.float32)
        effective_alpha = soft_mask_3ch * opacity

        composited = (final_nail_layer * effective_alpha) + (base_float * (1.0 - effective_alpha))
        return np.clip(composited, 0, 255).astype(np.uint8)

    def apply_solid_color(
        self,
        *args: Any,
        base_image: Optional[np.ndarray] = None,
        image_bgr: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        color_bgr: Optional[Tuple[int, int, int]] = None,
        color: Optional[Tuple[int, int, int]] = None,
        hex_color: Optional[str] = None,
        finish: str = "glossy",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Legacy compatibility wrapper that handles various positional and keyword argument variations.
        """
        # Resolve target base image
        target_image = base_image if base_image is not None else (image_bgr if image_bgr is not None else image)
        if target_image is None and len(args) > 0:
            target_image = args[0]

        # Resolve target mask
        target_mask = mask
        if target_mask is None and len(args) > 1:
            target_mask = args[1]

        # Resolve color inputs
        target_color_bgr = color_bgr if color_bgr is not None else color
        if target_color_bgr is None and len(args) > 2 and isinstance(args[2], (tuple, list)):
            target_color_bgr = tuple(args[2])

        target_hex = hex_color
        if target_hex is None and len(args) > 2 and isinstance(args[2], str):
            target_hex = args[2]

        if target_hex is None and target_color_bgr is not None:
            b, g, r = target_color_bgr
            target_hex = f"#{r:02x}{g:02x}{b:02x}"
        elif target_hex is None:
            target_hex = "#B22222"

        # Resolve finish preset
        target_finish = finish
        if "finish" not in kwargs and len(args) > 3 and isinstance(args[3], str):
            target_finish = args[3]

        if target_image is None:
            raise ValueError("No input image was provided to apply_solid_color.")

        return self.render(
            image_bgr=target_image,
            mask=target_mask,
            hex_color=target_hex,
            finish=target_finish,
        )
