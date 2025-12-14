import cv2
import argparse

def nothing(_): pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 webcam veya video yolu")
    args = ap.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {source}")

    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 420, 350)

    # Başlangıç: turuncu için makul bir aralık (ortama göre değişir)
    cv2.createTrackbar("H_min", "Trackbars", 5, 179, nothing)
    cv2.createTrackbar("H_max", "Trackbars", 25, 179, nothing)
    cv2.createTrackbar("S_min", "Trackbars", 120, 255, nothing)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Trackbars", 80, 255, nothing)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("H_min", "Trackbars")
        h_max = cv2.getTrackbarPos("H_max", "Trackbars")
        s_min = cv2.getTrackbarPos("S_min", "Trackbars")
        s_max = cv2.getTrackbarPos("S_max", "Trackbars")
        v_min = cv2.getTrackbarPos("V_min", "Trackbars")
        v_max = cv2.getTrackbarPos("V_max", "Trackbars")

        lower = (h_min, s_min, v_min)
        upper = (h_max, s_max, v_max)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        vis = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.putText(frame, f"LOW={lower}  UP={upper}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)
        cv2.imshow("segmented", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            print("Kopyala-yapistir HSV araligi:")
            print(f"LOWER = {lower}")
            print(f"UPPER = {upper}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
