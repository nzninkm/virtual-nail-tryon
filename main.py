import argparse
import cv2
import os
from src.pipeline.hand_detector import HandDetector
from src.models.segmenter import NailSegmenter
from src.pipeline.renderer import NailRenderer

def run_tryon(image_path: str, color_hex: str, output_path: str, model_path: str = None):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    print("[1/4] Detecting hand landmarks and fingertip ROIs...")
    detector = HandDetector()
    fingertips = detector.detect_fingertips(image)
    print(f"      Found {len(fingertips)} potential fingernails.")

    print("[2/4] Segmenting fingernails...")
    segmenter = NailSegmenter(model_path=model_path)
    nail_mask = segmenter.segment_nails(image, fingertips)

    print(f"[3/4] Rendering color {color_hex} with lighting retention...")
    renderer = NailRenderer()
    result = renderer.render_color(image, nail_mask, hex_color=color_hex, opacity=0.85, glossiness=0.35)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"[4/4] Done! Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Nail Try-On Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input hand image")
    parser.add_argument("--color", type=str, default="#C2185B", help="Hex color code (e.g. #C2185B)")
    parser.add_argument("--output", type=str, default="outputs/result.jpg", help="Path to save result")
    parser.add_argument("--model", type=str, default=None, help="Path to segmentation ONNX model")
    
    args = parser.parse_args()
    run_tryon(args.image, args.color, args.output, args.model)
