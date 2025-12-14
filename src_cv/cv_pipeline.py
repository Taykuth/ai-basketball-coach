from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class Detection:
    center: tuple[int, int]
    radius: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float


class BallCVPipeline:
    def __init__(
        self,
        hsv_range: HSVRange,
        trail_len: int = 64,
        min_radius: int = 6,
        max_radius: int = 120,
        tracker_type: str = "CSRT",  # "CSRT" | "KCF" | "NONE"
        template_path: str | None = None,
        template_min_score: float = 0.55,
    ):
        self.hsv_range = hsv_range
        self.trail = deque(maxlen=trail_len)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.tracker_type = tracker_type.upper()
        self.template_min_score = template_min_score

        self.tracker = None
        self.tracker_active = False

        # (Opsiyonel) Template
        self.template = None
        if template_path:
            t = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if t is None:
                raise RuntimeError(f"Template okunamadı: {template_path}")
            self.template = t

        # Koyu top / gece videosu için: hareket tabanlı maskeleme
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False
        )

        # Oyuncu gibi büyük blobları elemek + top yakınlık kısıtı
        self.max_blob_area_ratio = 0.03  # frame alanının %3'ünden büyük blob -> ele
        self.max_step_px = 120           # önceki top konumuna aşırı uzaksa -> ele

    # -------------------- Tracking --------------------
    def _make_tracker(self):
        if self.tracker_type == "NONE":
            return None

        if self.tracker_type == "CSRT":
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()

        if self.tracker_type == "KCF":
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                return cv2.legacy.TrackerKCF_create()

        raise RuntimeError(f"Tracker desteklenmiyor: {self.tracker_type}")

    def _init_tracker(self, frame_bgr: np.ndarray, det: Detection):
        if self.tracker_type == "NONE":
            return
        self.tracker = self._make_tracker()
        x, y, w, h = det.bbox
        self.tracker_active = bool(self.tracker.init(frame_bgr, (x, y, w, h)))

    def _update_tracker(self, frame_bgr: np.ndarray) -> Detection | None:
        if not self.tracker_active or self.tracker is None:
            return None

        ok, box = self.tracker.update(frame_bgr)
        if not ok:
            self.tracker_active = False
            return None

        x, y, w, h = [int(v) for v in box]
        cx, cy = x + w // 2, y + h // 2
        radius = 0.5 * max(w, h)
        return Detection(center=(cx, cy), radius=float(radius), bbox=(x, y, w, h), score=0.0)

    # -------------------- HSV Detection (fallback) --------------------
    def _hsv_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_range.lower, self.hsv_range.upper)

        # yumuşak morfoloji (top küçükken silinmesin)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask

    def _detect_by_contour_hsv(self, frame_bgr: np.ndarray) -> Detection | None:
        mask = self._hsv_mask(frame_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Detection | None = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 5:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue

            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < 0.55:
                continue

            x_i, y_i, w_i, h_i = cv2.boundingRect(c)
            score = float(circularity) * float(radius)

            d = Detection(
                center=(int(x), int(y)),
                radius=float(radius),
                bbox=(x_i, y_i, w_i, h_i),
                score=score,
            )

            if best is None or d.score > best.score:
                best = d

        return best

    # -------------------- Motion Detection (primary) --------------------
    def _motion_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        fg = self.bg.apply(frame_bgr)

        fg = cv2.GaussianBlur(fg, (5, 5), 0)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        return fg

    def _detect_by_motion(self, frame_bgr: np.ndarray) -> Detection | None:
        fg = self._motion_mask(frame_bgr)
        h, w = fg.shape[:2]
        frame_area = float(h * w)
        max_blob_area = self.max_blob_area_ratio * frame_area

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        prev = self.trail[0] if len(self.trail) > 0 else None
        candidates: list[Detection] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 30:
                continue
            if area > max_blob_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue

            circularity = 4 * np.pi * area / (perim * perim)
            if circularity < 0.45:
                continue

            x_i, y_i, w_i, h_i = cv2.boundingRect(c)
            if (w_i * h_i) > max_blob_area:
                continue

            cx, cy = int(x), int(y)

            dist = 0.0
            if prev is not None:
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > self.max_step_px:
                    continue

            score = float(circularity) * float(radius) - 0.01 * float(dist)

            candidates.append(
                Detection(
                    center=(cx, cy),
                    radius=float(radius),
                    bbox=(x_i, y_i, w_i, h_i),
                    score=score,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda d: d.score, reverse=True)
        return candidates[0]

    # -------------------- Template (optional) --------------------
    def _detect_by_template(self, frame_bgr: np.ndarray) -> Detection | None:
        if self.template is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.template_min_score:
            return None

        th, tw = self.template.shape[:2]
        x, y = max_loc
        cx, cy = x + tw // 2, y + th // 2
        radius = 0.5 * max(tw, th)

        return Detection(
            center=(cx, cy),
            radius=float(radius),
            bbox=(x, y, tw, th),
            score=float(max_val),
        )

    # -------------------- Pipeline --------------------
    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, Detection | None]:
        # 1) Öncelik: hareket tabanlı tespit
        det = self._detect_by_motion(frame_bgr)

        # 2) Yoksa HSV ile dene (fallback)
        if det is None:
            det = self._detect_by_contour_hsv(frame_bgr)

        # 3) Yoksa template dene (opsiyonel)
        if det is None:
            det = self._detect_by_template(frame_bgr)

        # 4) Hâlâ yoksa tracker ile devam et; tespit geldiyse tracker başlat
        if det is None:
            det = self._update_tracker(frame_bgr)
        else:
            if self.tracker_type != "NONE" and not self.tracker_active:
                self._init_tracker(frame_bgr, det)

        # 5) Son güvenlik: radius dışındaysa iptal
        if det is not None and (det.radius < self.min_radius or det.radius > self.max_radius):
            det = None

        # 6) Trail (None -> çizgiyi koparır)
        if det is not None:
            self.trail.appendleft(det.center)
        else:
            self.trail.appendleft(None)

        vis = frame_bgr.copy()

        # Debug: motion mask görmek istersen aç
        # fg = self._motion_mask(frame_bgr)
        # cv2.imshow("motion_mask", fg)

        if det is not None:
            cv2.circle(vis, det.center, int(det.radius), (0, 255, 255), 2)
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)

        for i in range(1, len(self.trail)):
            if self.trail[i - 1] is None or self.trail[i] is None:
                continue
            thickness = int(np.sqrt(self.trail.maxlen / float(i + 1)) * 2)
            cv2.line(vis, self.trail[i - 1], self.trail[i], (0, 0, 255), thickness)

        return vis, det
