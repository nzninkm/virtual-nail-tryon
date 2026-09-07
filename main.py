"""Virtual Nail Try-On CLI application powered by YOLOv8 nail segmentation."""
import os
import argparse
import cv2
import numpy as np
from typing import Tuple

from src.pipeline.ai_segmenter import YOLONailSegmenter
from src.pipeline.renderer import NailRenderer


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert HEX color code (e.g. #B22222 or #FF1493) to BGR tuple."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) != 6:
        raise ValueError(f"Invalid hex color format: {hex_str}")
    rgb = tuple(int(hex_clean[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Nail Color Try-On Pipeline via YOLOv8")
    parser.add_argument("--image", type=str, required=True, help="Path to input hand image")
    parser.add_argument("--hex", type=str, default="#B22222", help="Hex color code (e.g. #B22222, #FF1493)")
    parser.add_argument("--finish", type=str, default="glossy", choices=["glossy", "matte", "metallic"], help="Polish finish style")
    parser.add_argument("--model", type=str, default="mnemic/nails_seg_yolov8", help="YOLOv8-seg model identifier or weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for nail detection")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output result")
    parser.add_argument("--save-mask", type=str, default=None, help="Optional path to save binary nail mask")
    parser.add_argument("--side-by-side", action="store_true", help="Save side-by-side before/after comparison")
    return parser.parse_args()


def main():
    """Run try-on pipeline with YOLO nail segmentation and photorealistic shader blend."""
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image not found: {args.image}")
        return

    orig_bgr = cv2.imread(args.image)
    if orig_bgr is None:
        print(f"[ERROR] Failed to read image: {args.image}")
        return

    print(f"[INFO] Processing: {args.image}")
    selected_bgr = hex_to_bgr(args.hex)
    print(f"[INFO] Selected Color (BGR): {selected_bgr} | Finish: {args.finish}")

    # 1. Initialize segmenter and renderer
    segmenter = YOLONailSegmenter(model_path=args.model, conf_threshold=args.conf)
    renderer = NailRenderer(feather_radius=3)

    # 2. Extract nail masks
    accumulated_mask, individual_masks = segmenter.segment_nails(orig_bgr)
    print(f"[INFO] Successfully segmented {len(individual_masks)} nails.")

    if len(individual_masks) == 0:
        print("[WARNING] Could not detect any nails to colorize.")
        return

    # Optionally save debug mask
    if args.save_mask:
        cv2.imwrite(args.save_mask, accumulated_mask)
        print(f"[INFO] Binary nail mask saved to: {args.save_mask}")

    # 3. Realistic Polish Rendering
    result_bgr = renderer.apply_solid_color(
        base_image=orig_bgr,
        nail_mask=accumulated_mask,
        color_bgr=selected_bgr,
        finish=args.finish,
        blend_strength=0.88,
    )

    # 4. Save result (Side-by-Side or single)
    if args.side_by_side:
        h = orig_bgr.shape[0]
        separator = np.full((h, 8, 3), (255, 255, 255), dtype=np.uint8)
        final_output = np.hstack([orig_bgr, separator, result_bgr])
    else:
        final_output = result_bgr

    cv2.imwrite(args.output, final_output)
    print(f"[SUCCESS] Pipeline completed successfully! Saved to: {args.output}")


if __name__ == "__main__":
    main()
