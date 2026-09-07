"""CLI entrypoint for virtual nail polish try-on pipeline."""
import argparse
import sys
import os
import cv2
import numpy as np

# Ensure src directory is in module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.segmenter import NailSegmenter
from src.pipeline.renderer import NailRenderer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Nail Polish Try-On Pipeline")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input hand image",
    )
    parser.add_argument(
        "--hex",
        type=str,
        default="#800020",
        help="Hex color string for nail polish (e.g. #800020 for burgundy)",
    )
    parser.add_argument(
        "--finish",
        type=str,
        default="glossy",
        choices=["glossy", "matte", "metallic"],
        help="Visual finish type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.jpg",
        help="Output file path",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Export side-by-side comparison with original image",
    )
    parser.add_argument(
        "--save-mask",
        type=str,
        default=None,
        help="Optional path to save generated segmentation mask",
    )
    return parser.parse_args()


def main():
    """Execute end-to-end segmentation and photorealistic rendering pipeline."""
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image not found: {args.image}")
        sys.exit(1)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"[ERROR] Failed to decode image: {args.image}")
        sys.exit(1)

    print(f"[INFO] Processing image: {args.image}")
    print(f"[INFO] Target color: {args.hex} | Finish: {args.finish}")

    segmenter = NailSegmenter()
    nail_mask = segmenter.segment(image_bgr)

    if nail_mask is None or np.count_nonzero(nail_mask) == 0:
        print("[WARN] No nails detected in image.")
    else:
        print(f"[INFO] Successfully segmented nail region.")

    if args.save_mask:
        cv2.imwrite(args.save_mask, nail_mask if nail_mask is not None else np.zeros_like(image_bgr[:, :, 0]))
        print(f"[INFO] Saved mask preview to: {args.save_mask}")

    renderer = NailRenderer()
    result_bgr = renderer.render(
        image_bgr=image_bgr,
        mask=nail_mask,
        hex_color=args.hex,
        finish=args.finish,
    )

    if args.side_by_side:
        final_output = np.hstack([image_bgr, result_bgr])
    else:
        final_output = result_bgr

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output, final_output)
    print(f"[SUCCESS] Pipeline completed successfully! Saved to: {args.output}")


if __name__ == "__main__":
    main()
