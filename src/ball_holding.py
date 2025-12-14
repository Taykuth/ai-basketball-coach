import time
import numpy as np

class BallHoldingDetector:
    def __init__(self, hold_duration: float = 0.85, hold_threshold: float = 300):
        """
        hold_duration: topu elde tutma süresi (saniye)
        hold_threshold: el ile top arasındaki max mesafe (pixel)
        """
        self.hold_duration = hold_duration
        self.hold_threshold = hold_threshold
        self.hold_start_time = None
        self.is_holding = False

    def update(self, people, balls):
        if not people or not balls:
            self.hold_start_time = None
            self.is_holding = False
            return self.is_holding

        player = people[0]   # ilk oyuncu
        ball = balls[0]      # ilk top

        lw = player.keypoints.get("left_wrist")
        rw = player.keypoints.get("right_wrist")
        if lw is None or rw is None:
            return self.is_holding

        bx, by = ball.center

        left_distance = np.hypot(bx - lw.x, by - lw.y)
        right_distance = np.hypot(bx - rw.x, by - rw.y)

        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration and not self.is_holding:
                print("The ball is being held.")
                self.is_holding = True
        else:
            self.hold_start_time = None
            self.is_holding = False

        return self.is_holding
