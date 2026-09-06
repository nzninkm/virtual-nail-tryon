"""Virtual Nail Try-On CLI application for real-time nail color and art augmentation."""
import os
import argparse
import cv2
import numpy as np
from typing import Tuple, Optional

from src.pipeline.hand_detector import HandDetector
from src.pipeline.segmenter import NailSegmenter
from src.pipeline.renderer import NailRenderer
from src.pipeline.warper import NailWarper


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert HEX color code (e.g. #D22B2B or D22B2B) to BGR tuple."""
    hex_clean = hex_str.lstrip("#")
    if len(hex_clean) != 6:
        raise ValueError(f"Invalid hex color format: {hex_str}")
    rgb = tuple(int(hex_clean[i:i + 2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Nail Color & Art Try-On Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input hand image")
    parser.add_argument("--hex", type=str, default="#B22222", help="Hex color code (e.g. #FF1493, #800020)")
    parser.add_argument("--finish", type=str, default="glossy", choices=["glossy", "matte", "metallic"], help="Polish finish style")
    parser.add_argument("--pattern", type=str, default=None, help="Optional path to custom pattern/art image")
    parser.add_argument("--ai-model", type=str, default=None, help="Optional ONNX nail segmentation model path")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output result")
    parser.add_argument("--side-by-side", action="store_true", help="Save side-by-side before/after comparison")
    parser.add_argument("--debug", action="store_true", help="Display intermediate masks and landmarks")
    return parser.parse_args()


def main():
    """Run try-on pipeline with robust multi-finger detection and realistic lighting blend."""
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

    # Initialize modular pipeline components
    hand_detector = HandDetector()
    segmenter = NailSegmenter(ai_model_path=args.ai_model)
    renderer = NailRenderer(feather_radius=4)
    warper = NailWarper()

    # Step 1: Detect hand landmarks
    landmarks = hand_detector.detect(orig_bgr)
    if landmarks is None:
        print("[WARNING] No hand detected. Attempting full-frame segmentation fallback.")
        landmarks = []

    # Step 2: Extract fingertips & segment nail plates
    h, w = orig_bgr.shape[:2]
    accumulated_mask = np.zeros((h, w), dtype=np.uint8)

    fingertip_rois = hand_detector.get_fingertip_rois(orig_bgr, landmarks)

    for roi_data in fingertip_rois:
        roi = roi_data["roi"]
        x, y, rw, rh = roi_data["bbox"]
        tip_local = roi_data["tip_local"]
        dip_local = roi_data["dip_local"]

        # Run hybrid segmentation
        mask_local = segmenter.segment_nail(roi, tip_local, dip_local)
        accumulated_mask[y:y + rh, x:x + rw] = cv2.bitwise_or(
            accumulated_mask[y:y + rh, x:x + rw],
            mask_local
        )

    # Step 3: Realistic Polish Rendering
    result_bgr = renderer.apply_solid_color(
        base_image=orig_bgr,
        nail_mask=accumulated_mask,
        color_bgr=selected_bgr,
        finish=args.finish,
        blend_strength=0.88,
    )

    # Optional: Side-by-Side before/after visualization
    if args.side_by_side:
        separator = np.full((h, 8, 3), (255, 255, 255), dtype=np.uint8)
        final_output = np.hstack([orig_bgr, separator, result_bgr])
    else:
        final_output = result_bgr

    # Save output
    cv2.imwrite(args.output, final_output)
    print(f"[SUCCESS] Render completed successfully! Saved to: {args.output}")


if __name__ == "__main__":
    main()
