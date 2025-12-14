import time

class DribbleCounter:
    """
    Dribble sayıcı: top yukarı çıkıp sonra aşağı inince 1 dribble sayar.
    Threshold frame yüksekliğine göre ayarlanır.
    """
    def __init__(self, frame_height: int = 720, rel_thresh: float = 0.02, min_interval: float = 0.2):
        self.prev_y = None
        self.state = "down"  # "up" veya "down"
        self.count = 0
        self.threshold = frame_height * rel_thresh  # piksel olarak dinamik
        self.last_time = 0.0
        self.min_interval = min_interval

    def update(self, y_center: float):
        now = time.time()
        if self.prev_y is None:
            self.prev_y = y_center
            return self.count

        delta = y_center - self.prev_y

        # Debounce
        if now - self.last_time < self.min_interval:
            self.prev_y = y_center
            return self.count

        # Top yukarı çıkıyor
        if delta < -self.threshold and self.state == "down":
            self.state = "up"
            self.last_time = now
            self.count += 1
            print(f"✅ Dribble detected! Total = {self.count}")

        # Top aşağı iniyor
        elif delta > self.threshold and self.state == "up":
            self.state = "down"
            self.last_time = now

        self.prev_y = y_center
        return self.count
