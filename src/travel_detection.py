class TravelDetector:
    def __init__(self):
        self.travel_detected = False

    def check(self, steps: int, is_holding: bool):
        """
        steps: o ana kadar yapılan adım sayısı
        is_holding: BallHoldingDetector’dan gelen bilgi
        """
        if is_holding and steps > 2:
            self.travel_detected = True
        else:
            self.travel_detected = False

        return self.travel_detected
