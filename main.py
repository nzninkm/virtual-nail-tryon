"""Virtual Nail Try-On command line entry point."""
import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

# Safe import for detector module across naming conventions
try:
    from src.pipeline.hand_detector import HandDetector
except ModuleNotFoundError:
    from src.pipeline.detector import HandDetector

from src.pipeline.geometry import NailGeometry
from src.pipeline.renderer import NailRenderer
from src.pipeline.segmenter import NailSegmenter
from src.pipeline.warper import NailWarper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual Nail Try-On: Recolor or apply nail art patterns to fingernails."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input hand image",
    )
    parser.add_argument(
        "--color",
        type=str,
        default=None,
        help="Target nail polish hex color (e.g. #E91E63)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Path to the nail art pattern texture image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.jpg",
        help="Path to save the resulting image (default: outputs/result.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hand_landmarker.task",
        help="Path to MediaPipe hand landmarker task model",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate segmentation mask overlays and landmarks to debug_mask.jpg",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image '{args.image}' not found.")
        sys.exit(1)

    # Set fallback default color if neither color nor pattern is specified
    if not args.color and not args.pattern:
        args.color = "#E91E63"

    pattern_img = None
    if args.pattern:
        if not os.path.exists(args.pattern):
            print(f"Error: Pattern texture '{args.pattern}' not found.")
            sys.exit(1)
        pattern_img = cv2.imread(args.pattern, cv2.IMREAD_UNCHANGED)
        if pattern_img is None:
            print(f"Error: Could not read pattern file '{args.pattern}'.")
            sys.exit(1)
        pattern_img = NailWarper.apply_cylindrical_curvature(pattern_img, strength=0.35)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: Could not read image '{args.image}'.")
        sys.exit(1)

    detector = HandDetector(model_path=args.model)
    segmenter = NailSegmenter()

    print("Detecting hand landmarks and fingertips...")
    nails = detector.detect_fingertips(image_bgr)
    print(f"Detected {len(nails)} nail candidates.")

    result = image_bgr.copy()
    debug_img = image_bgr.copy() if args.debug else None

    for nail in nails:
        x1, y1, x2, y2 = nail["roi_box"]
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        tip_local = (nail["tip"][0] - x1, nail["tip"][1] - y1)
        dip_local = (nail["dip"][0] - x1, nail["dip"][1] - y1)

        # Segment natural nail boundary
        mask = segmenter.segment_nail(roi, tip_local, dip_local)

        if args.debug and debug_img is not None:
            # Draw green contours around detected nail masks
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                offset_cnt = cnt + np.array([x1, y1])
                cv2.drawContours(debug_img, [offset_cnt], -1, (0, 255, 0), 2)
            # Red circle for fingertip, blue circle for DIP joint
            cv2.circle(debug_img, nail["tip"], 4, (0, 0, 255), -1)
            cv2.circle(debug_img, nail["dip"], 4, (255, 0, 0), -1)

        # Render either pattern or solid color
        if pattern_img is not None:
            quad = NailGeometry.extract_nail_quad(mask, tip_local, dip_local)
            if quad is not None:
                warped = NailWarper.warp_pattern_to_quad(pattern_img, quad, roi.shape[:2])
                roi_rendered = NailRenderer.blend_pattern(roi, mask, warped)
            else:
                fallback_bgr = (99, 30, 233)
                roi_rendered = NailRenderer.blend_solid_color(roi, mask, fallback_bgr)
        else:
            target_bgr = NailRenderer.hex_to_bgr(args.color)
            roi_rendered = NailRenderer.blend_solid_color(roi, mask, target_bgr)

        result[y1:y2, x1:x2] = roi_rendered

    # Ensure output directory exists before saving
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(args.output, result)
    print(f"Output saved to: {args.output}")

    if args.debug and debug_img is not None:
        debug_path = os.path.join(output_dir, "debug_mask.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug mask visualization saved to: {debug_path}")


if __name__ == "__main__":
    main()
