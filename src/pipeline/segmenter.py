import cv2
import numpy as np
from typing import Tuple

class NailSegmenter:
    def __init__(self):
        pass

    def segment_nail(self, roi_bgr: np.ndarray, tip_pt: Tuple[int, int], dip_pt: Tuple[int, int]) -> np.ndarray:
        """
        Segment the exact natural nail boundaries using skin-differential color analysis
        and directional gradient edge clamping.
        """
        h, w = roi_bgr.shape[:2]
        if h < 10 or w < 10:
            return np.zeros((h, w), dtype=np.uint8)

        # 1. Compute directional axis of finger bone
        dx = float(tip_pt[0] - dip_pt[0])
        dy = float(tip_pt[1] - dip_pt[1])
        dist = np.hypot(dx, dy)

        if dist < 1e-4:
            unit_dx, unit_dy = 0.0, -1.0
        else:
            unit_dx, unit_dy = dx / dist, dy / dist

        # Perpendicular normal vector
        perp_dx, perp_dy = -unit_dy, unit_dx

        # 2. Estimate biological nail center and dynamic bounding dimensions
        nail_offset = dist * 0.30
        nail_cx = tip_pt[0] - unit_dx * nail_offset
        nail_cy = tip_pt[1] - unit_dy * nail_offset

        half_len = dist * 0.42
        half_width = dist * 0.28

        # 3. Create a directional search corridor along finger axis
        corridor_mask = np.zeros((h, w), dtype=np.uint8)
        c_p1 = (int(nail_cx - unit_dx * half_len - perp_dx * half_width), int(nail_cy - unit_dy * half_len - perp_dy * half_width))
        c_p2 = (int(nail_cx + unit_dx * half_len - perp_dx * half_width), int(nail_cy + unit_dy * half_len - perp_dy * half_width))
        c_p3 = (int(nail_cx + unit_dx * half_len + perp_dx * half_width), int(nail_cy + unit_dy * half_len + perp_dy * half_width))
        c_p4 = (int(nail_cx - unit_dx * half_len + perp_dx * half_width), int(nail_cy - unit_dy * half_len + perp_dy * half_width))
        cv2.fillConvexPoly(corridor_mask, np.array([c_p1, c_p2, c_p3, c_p4], dtype=np.int32), 255)

        # 4. Multi-color space nail tissue extraction
        # YCrCb: Cr channel separates skin pigments; Lab: B channel highlights keratin differences
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)

        cr = ycrcb[:, :, 1]
        b_channel = lab[:, :, 2]
        l_channel = lab[:, :, 0]

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        enhanced_l = clahe.apply(l_channel)

        # 5. Differential map inside corridor
        diff_score = (cr.astype(np.float32) * 0.4) + (enhanced_l.astype(np.float32) * 0.6) - (b_channel.astype(np.float32) * 0.3)
        diff_score = np.clip(diff_score, 0, 255).astype(np.uint8)

        # Directional gradient thresholding
        masked_diff = cv2.bitwise_and(diff_score, diff_score, mask=corridor_mask)
        valid_pixels = masked_diff[corridor_mask > 0]

        if valid_pixels.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        mean_val = np.mean(valid_pixels)
        std_val = np.std(valid_pixels)
        thresh_val = mean_val - 0.25 * std_val

        _, binary = cv2.threshold(masked_diff, thresh_val, 255, cv2.THRESH_BINARY)

        # 6. Directional morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 7. Select the optimal contour nearest to the biological nail center
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best_mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    d = np.hypot(cx - nail_cx, cy - nail_cy)
                    candidates.append((cnt, d, area))

            if candidates:
                # Rank by distance to calculated nail center and sensible area
                candidates.sort(key=lambda item: item[1] / np.sqrt(item[2]))
, 3))
        clean_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 7. Select the optimal contour nearest to the biological nail center
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best_mask = np.zeros((h, w), dtype=np.uint8)

        if contours:
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = m["m10"] / m["m00"]
                    cy = m["m01"] / m["m00"]
                    d = np.hypot(cx - nail_cx, cy - nail_cy)
                    candidates.append((cnt, d, area))

            if candidates:
                # Rank by distance to calculated nail center and sensible area
                candidates.sort(key=lambda item: item[1] / np.sqrt(item[2]))
                best_cnt = candidates[0][0]
                hull = cv2.convexHull(best_cnt)
                cv2.drawContours(best_mask, [hull], -1, 255, -1)
            else:
                cv2.fillConvexPoly(best_mask, np.array([c_p1, c_p2, c_p3, c_p4], dtype=np.int32), 255)
        else:
            cv2.fillConvexPoly(best_mask, np.array([c_p1, c_p2, c_p3, c_p4], dtype=np.int3argument("--color", type=str, default=None, help="Target nail polish hex color (e.g. #E91E63)")
    parser.add_argument("--pattern", type=str, default=None, help="Nail art pattern texture path")
    parser.add_argument("--output", type=str, default="outputs/result.jpg", help="Output result image path")
    parser.add_argument("--model", type=str, default="hand_landmarker.task", help="Path to MediaPipe landmarker task model")
    parser.add_argument("--debug", action="store_true", help="Save intermediate segmentation mask overlays for inspection")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} not found.")
        return

    if not args.color and not args.pattern:
        args.color = "#E91E63"

    pattern_img = None
    if args.pattern:
        if not os.path.exists(args.pattern):
            print(f"Error: Pattern texture {args.pattern} not found.")
            return
        pattern_img = cv2.imread(args.pattern, cv2.IMREAD_UNCHANGED)
        pattern_img = NailWarper.apply_cylindrical_curvature(pattern_img, strength=0.35)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print("Error: Could not read image.")
        return

    detector = HandDetector(model_path=args.model)
    segmenter = NailSegmenter()

    print("Detecting fingertips and landmarks...")
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

        # Segment nail boundary
        mask = segmenter.segment_nail(roi, tip_local, dip_local)

        if args.debug and debug_img is not None:
            # Draw contours of segmented mask in bright green
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                offset_cnt = cnt + np.array([x1, y1])
                cv2.drawContours(debug_img, [offset_cnt], -1, (0, 255, 0), 2)
            cv2.circle(debug_img, nail["tip"], 4, (0, 0, 255), -1)
            cv2.circle(debug_img, nail["dip"], 4, (255, 0, 0), -1)

        if pattern_img is not None:
            quad = NailGeometry.extract_nail_quad(mask, tip_local, dip_local)
            if quad is not None:
                warped = NailWarper.warp_pattern_to_quad(pattern_img, quad, roi.shape[:2])
                roi_rendered = NailRenderer.blend_pattern(roi, mask, warped)
            else:
                target_bgr = (99, 30, 233)
                roi_rendered = NailRenderer.blend_solid_color(roi, mask, target_bgr)
        else:
            target_bgr = NailRenderer.hex_to_bgr(args.color)
            roi_rendered = NailRenderer.blend_solid_color(roi, mask, target_bgr)

        result[y1:y2, x1:x2] = roi_rendered

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"Output saved to: {args.output}")

    if args.debug and debug_img is not None:
        debug_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "debug_mask.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug mask visualization saved to: {debug_path}")

if __name__ == "__main__":
    main()
