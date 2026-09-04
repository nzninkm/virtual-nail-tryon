import os
import cv2
import argparse
import numpy as np

from src.pipeline.hand_detector import HandDetector
from src.pipeline.segmenter import NailSegmenter
from src.pipeline.geometry import NailGeometry
from src.pipeline.warper import NailWarper
from src.pipeline.renderer import NailRenderer

def parse_args():
    parser = argparse.ArgumentParser(description="Virtual Nail Try-On (Solid Color & Art Pattern)")
    parser.add_argument("--image", type=str, required=True, help="مسیر تصویر دست ورودی")
    parser.add_argument("--color", type=str, default=None, help="کد رنگ لاک به صورت HEX مانند #E91E63")
    parser.add_argument("--pattern", type=str, default=None, help="مسیر تصویر طرح ناخن")
    parser.add_argument("--output", type=str, default="outputs/result.jpg", help="مسیر ذخیره تصویر خروجی")
    parser.add_argument("--model", type=str, default="hand_landmarker.task", help="مسیر فایل مدل MediaPipe")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"خطا: تصویر ورودی {args.image} یافت نشد.")
        return

    if not args.color and not args.pattern:
        print("اطلاعیه: رنگ مشخص نشد، از رنگ پیش‌فرض صورتی (#E91E63) استفاده می‌شود.")
        args.color = "#E91E63"

    pattern_img = None
    if args.pattern:
        if not os.path.exists(args.pattern):
            print(f"خطا: فایل طرح {args.pattern} یافت نشد.")
            return
        pattern_img = cv2.imread(args.pattern, cv2.IMREAD_UNCHANGED)
        pattern_img = NailWarper.apply_cylindrical_curvature(pattern_img, strength=0.35)

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print("خطا: خواندن تصویر ناموفق بود.")
        return

    detector = HandDetector(model_path=args.model)
    segmenter = NailSegmenter()

    print("در حال پردازش لندمارک‌های دست...")
    nails = detector.detect_fingertips(image_bgr)
    print(f"تعداد {len(nails)} ناخن شناسایی شد.")

    result = image_bgr.copy()

    for nail in nails:
        x1, y1, x2, y2 = nail["roi_box"]
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        tip_local = (nail["tip"][0] - x1, nail["tip"][1] - y1)
        dip_local = (nail["dip"][0] - x1, nail["dip"][1] - y1)

        # استخراج ماسک دقیق ناخن
        mask = segmenter.segment_nail(roi, tip_local, dip_local)

        if pattern_img is not None:
            # استخراج جهت و ۴ نقطه کلیدی
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
    print(f"عملیات با موفقیت انجام شد. نتیجه در: {args.output}")

if __name__ == "__main__":
    main()
