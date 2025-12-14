import cv2
import time
import argparse
from cv_pipeline import BallCVPipeline, HSVRange

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 webcam veya video yolu")
    ap.add_argument("--tracker", default="CSRT", choices=["CSRT", "KCF", "NONE"])
    ap.add_argument("--template", default=None, help="assets/ball_template.png (opsiyonel)")
    ap.add_argument("--trail", type=int, default=64)
    ap.add_argument("--min_radius", type=int, default=6)
    ap.add_argument("--max_radius", type=int, default=120)

    # hsv_calibrate.py çıktını buraya yapıştır
    ap.add_argument("--hsv_lower", default="(5,120,80)")
    ap.add_argument("--hsv_upper", default="(25,255,255)")
    args = ap.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {source}")

    lower = eval(args.hsv_lower)
    upper = eval(args.hsv_upper)

    pipeline = BallCVPipeline(
        hsv_range=HSVRange(lower=lower, upper=upper),
        trail_len=args.trail,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        tracker_type=args.tracker,
        template_path=args.template,
    )

    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis, det = pipeline.process(frame)

        now = time.time()
        dt = now - prev
        prev = now
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 1:
            video_fps = 30.0


        hud = f"VideoFPS: {video_fps:.1f} | tracker={args.tracker}"
        if det is not None:
            hud += f" | center={det.center} r={det.radius:.1f}"
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        cv2.imshow("Basketbol Topu Takibi (Klasik CV)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
