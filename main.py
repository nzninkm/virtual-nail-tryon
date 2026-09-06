"""Virtual Nail Try-On command line interface with robust error handling."""
import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from src.pipeline.hand_detector import HandDetector
from src.pipeline.segmenter import NailSegmenter
from src.pipeline.geometry import NailGeometry
from src.pipeline.warper import NailWarper
from src.pipeline.renderer import NailRenderer

segmenter = NailSegmenter(ai_model_path=args.ai_model)

def parse_args():
    """Parse CLI input arguments."""
    parser = argparse.ArgumentParser(
        description="Virtual Nail Try-On: Apply realistic solid colors or custom nail art patterns."
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
        help="Nail polish color in hex format (e.g. #E91E63)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Path to texture/pattern image for nail art",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.jpg",
        help="Path to save the rendered output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug mask and quad boundaries to outputs/debug_mask.jpg",
    )
        parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="Optional path to custom deep-learning nail segmentation model (.onnx)",
    )

    return parser.parse_args()


def load_image_safe(image_path: str) -> np.ndarray:
    """Safely load an image from disk with exception handling."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist at: '{image_path}'")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to decode image from: '{image_path}'. Ensure it is a valid image format.")

        return image
    except Exception as err:
        print(f"[ERROR] Failed to load image: {err}")
        return None


def main():
    """Main execution pipeline with exception boundaries."""
    args = parse_args()

    # 1. Safely load input hand image
    print(f"Loading input image: {args.image} ...")
    image_bgr = load_image_safe(args.image)
    if image_bgr is None:
        print("[WARNING] Skipping process: Hand image could not be loaded.")
        return

    # 2. Safely load and prepare pattern if provided
    pattern_img = None
    if args.pattern:
        print(f"Loading pattern image: {args.pattern} ...")
        pattern_raw = load_image_safe(args.pattern)
        if pattern_raw is not None:
            try:
                # Apply 3D cylindrical convexity to texture
                pattern_img = NailWarper.apply_cylindrical_curvature(pattern_raw, strength=0.35)
            except Exception as err:
                print(f"[ERROR] Failed to apply cylindrical curvature to pattern: {err}")
                pattern_img = None
        else:
            print("[WARNING] Pattern file could not be loaded. Falling back to default color.")

    if not args.color and pattern_img is None:
        args.color = "#E91E63"

    # 3. Detect fingertips and segment nails
    try:
        detector = HandDetector()
        segmenter = NailSegmenter()

        print("Detecting hand landmarks...")
        nails = detector.detect_fingertips(image_bgr)
        print(f"Found {len(nails)} nail candidates.")

        if not nails:
            print("[WARNING] No hand landmarks or fingertips were detected in the image.")
            return

    except Exception as err:
        print(f"[ERROR] Hand landmark detection failed: {err}")
        return

    result = image_bgr.copy()
    debug_img = image_bgr.copy() if args.debug else None

    # 4. Process each fingertip ROI safely
    for idx, nail in enumerate(nails, start=1):
        try:
            x1, y1, x2, y2 = nail["roi_box"]
            roi = result[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            tip_local = (nail["tip"][0] - x1, nail["tip"][1] - y1)
            dip_local = (nail["dip"][0] - x1, nail["dip"][1] - y1)

            # Segment natural nail mask
            mask = segmenter.segment_nail(roi, tip_local, dip_local)

            # Extract 4-point bounding quad
            quad = NailGeometry.extract_nail_quad(mask, tip_local, dip_local)

            # Render debug overlays if requested
            if args.debug and debug_img is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    offset_cnt = cnt + np.array([x1, y1])
                    cv2.drawContours(debug_img, [offset_cnt], -1, (0, 255, 0), 2)
                if quad is not None:
                    offset_quad = (quad + np.array([x1, y1])).astype(np.int32)
                    cv2.polylines(debug_img, [offset_quad], isClosed=True, color=(0, 255, 255), thickness=1)

            # Render pattern or solid color
            if pattern_img is not None and quad is not None:
                warped = NailWarper.warp_pattern_to_quad(pattern_img, quad, roi.shape[:2])
                roi_rendered = NailRenderer.blend_pattern(roi, mask, warped)
            else:
                target_bgr = NailRenderer.hex_to_bgr(args.color if args.color else "#E91E63")
                roi_rendered = NailRenderer.blend_solid_color(roi, mask, target_bgr)

            result[y1:y2, x1:x2] = roi_rendered

        except Exception as err:
            print(f"[WARNING] Error processing finger #{idx} ({nail.get('finger_name', 'unknown')}): {err}")
            continue

    # 5. Safely save results to disk
    try:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(args.output, result)
        print(f"[SUCCESS] Rendered output saved to: {args.output}")

        if args.debug and debug_img is not None:
            debug_path = os.path.join(out_dir, "debug_mask.jpg")
            cv2.imwrite(debug_path, debug_img)
            print(f"[SUCCESS] Debug overlay saved to: {debug_path}")

    except Exception as err:
        print(f"[ERROR] Failed to write output image to disk: {err}")


if __name__ == "__main__":
    main()
