import time
from collections import deque
import numpy as np

class StepDetector:
    """
    Geliştirilmiş adım sayıcı:
      1. Dinamik eşik (oyuncu boyuna göre, dikey leg length)
      2. Minimum zaman aralığı (debounce)
      3. Konum yumuşatma (hareketli ortalama)
      4. Zemin temelli dikey hareket kontrolü
    """

    def __init__(
        self,
        rel_thresh: float = 0.15,   # bacak uzunluğunun % kaçı hareket sayılır
        min_interval: float = 0.3,  # iki adım arası minimum saniye
        smooth_len: int = 3,        # yumuşatma için kaç kare
        ground_tol: float = 0.03    # zemin toleransı (frame_height yüzdesi)
    ):
        self.rel_thresh = rel_thresh
        self.min_interval = min_interval
        self.smooth_len = smooth_len
        self.ground_tol = ground_tol

        self.prev_left = None
        self.prev_right = None
        self.steps = 0
        self.last_step_time = 0.0

        self.left_hist = deque(maxlen=smooth_len)
        self.right_hist = deque(maxlen=smooth_len)

        self.ground_y = None

    def _avg_point(self, hist):
        if not hist:
            return None
        xs, ys = zip(*hist)
        return (np.mean(xs), np.mean(ys))

    def update(self, people, frame_height: int = None):
        if not people:
            return self.steps

        person = people[0]
        la = person.keypoints.get("left_ankle")
        ra = person.keypoints.get("right_ankle")
        lh = person.keypoints.get("left_hip")
        if not la or not ra or not lh:
            return self.steps

        if self.ground_y is None:
            self.ground_y = max(la.y, ra.y)

        self.left_hist.append((la.x, la.y))
        self.right_hist.append((ra.x, ra.y))

        left_pt = self._avg_point(self.left_hist)
        right_pt = self._avg_point(self.right_hist)

        if self.prev_left is None:
            self.prev_left, self.prev_right = left_pt, right_pt
            return self.steps

        # Dikey bacak uzunluğu ile dinamik threshold
        leg_len = abs(lh.y - la.y)
        move_threshold = max(leg_len * self.rel_thresh, 5.0)

        left_move = np.hypot(left_pt[0] - self.prev_left[0], left_pt[1] - self.prev_left[1])
        right_move = np.hypot(right_pt[0] - self.prev_right[0], right_pt[1] - self.prev_right[1])

        now = time.time()
        if now - self.last_step_time > self.min_interval:
            # Ayak yerden kalktı mı kontrolü
            ground_limit = (frame_height or self.ground_y) * self.ground_tol
            left_air = abs(la.y - self.ground_y) > ground_limit
            right_air = abs(ra.y - self.ground_y) > ground_limit

            if left_move > move_threshold and right_move < move_threshold and left_air:
                self.steps += 1
                self.last_step_time = now
                print(f"Step detected with LEFT foot. Total: {self.steps}")
            elif right_move > move_threshold and left_move < move_threshold and right_air:
                self.steps += 1
                self.last_step_time = now
                print(f"Step detected with RIGHT foot. Total: {self.steps}")

        self.prev_left, self.prev_right = left_pt, right_pt
        return self.steps
