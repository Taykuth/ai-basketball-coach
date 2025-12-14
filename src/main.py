"""
AI Basketball Coach — Step 1
Goal of this step:
- Single, clean realtime pipeline that reads frames (webcam or video),
  runs YOLO pose + ball detection once per frame (with optional frame-skip),
  and returns a **normalized state** that future modules (dribble, holding,
  double-dribble, step, travel) will consume.
- Lightweight visualizer overlays (pose + ball boxes + basic HUD).

Next step (Step 2): plug in a Dribble Counter module using this state.
"""

from __future__ import annotations
from dribble_counter import DribbleCounter
import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from ball_holding import BallHoldingDetector
from step_detection import StepDetector
from travel_detection import TravelDetector
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Config (edit as needed)
# -------------------------------
CONFIG = {
    "pose_model": "yolov8s-pose.pt",          # make sure present or let ultralytics download
    "ball_model": "basketballModel.pt",       # your custom ball model path
    "conf_pose": 0.5,
    "conf_ball": 0.65,
    "device": None,                            # e.g. "cuda" or "cpu" or None (auto)
    "frame_skip": 1,                           # process every Nth frame (1 = every frame)
    "buffer_size": 30,                         # keep last N FrameState objects
}

# COCO keypoint index mapping used by YOLOv8 pose (17 keypoints)
KEYPOINT_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# -------------------------------
# Data structures
# -------------------------------
@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

@dataclass
class Person:
    keypoints: Dict[str, Keypoint]  # e.g. {"left_ankle": Keypoint(...), ...}
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

@dataclass
class Ball:
    center: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    conf: float
    area: float

@dataclass
class FrameState:
    timestamp: float
    frame_idx: int
    width: int
    height: int
    people: List[Person] = field(default_factory=list)
    balls: List[Ball] = field(default_factory=list)

# -------------------------------
# Models wrapper
# -------------------------------
class Models:
    def __init__(self, pose_path: str, ball_path: str, device: Optional[str] = None,
                 conf_pose: float = 0.5, conf_ball: float = 0.65) -> None:
        self.pose = YOLO(pose_path)
        self.ball = YOLO(r"C:\Users\osman\ai-basketball-coach\models\basketballModel.pt")
        self.device = device
        self.conf_pose = conf_pose
        self.conf_ball = conf_ball

    def infer_pose(self, frame: np.ndarray):
        return self.pose.predict(source=frame, conf=self.conf_pose, verbose=False, device=self.device)

    def infer_ball(self, frame: np.ndarray):
        return self.ball.predict(source=frame, conf=self.conf_ball, verbose=False, device=self.device)

# -------------------------------
# Detection & normalization
# -------------------------------

def parse_people(pose_res):
    people = []
    res = pose_res[0]
    kps = res.keypoints  # ultralytics Keypoints object
    boxes = res.boxes    # person boxes (if any)

    if kps is None:
        return people

    kp_np = kps.data.cpu().numpy()  # (num_persons, 17, 3)
    num_persons = kp_np.shape[0]

    box_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None

    for i in range(num_persons):
        kp_dict: Dict[str, Keypoint] = {}
        for name, idx in KEYPOINT_INDEX.items():
            x, y, c = kp_np[i, idx]
            kp_dict[name] = Keypoint(float(x), float(y), float(c))

        bbox = None
        if box_xyxy is not None and i < len(box_xyxy):
            x1, y1, x2, y2 = box_xyxy[i].astype(int)
            bbox = (x1, y1, x2, y2)

        people.append(Person(keypoints=kp_dict, bbox=bbox))
    return people



    res = pose_results[0]
    kps = res.keypoints  # ultralytics Keypoints object
    boxes = res.boxes    # person boxes (if any)

    if kps is None or kps.numpy() is None:
        return people

    kp_np = kps.numpy()  # shape: (num_persons, 17, 3) => x,y,conf
    num_persons = kp_np.shape[0]

    # Some frames may have no boxes; handle gracefully
    box_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else None

    for i in range(num_persons):
        kp_dict: Dict[str, Keypoint] = {}
        for name, idx in KEYPOINT_INDEX.items():
            x, y, c = kp_np[i, idx]
            kp_dict[name] = Keypoint(float(x), float(y), float(c))

        bbox = None
        if box_xyxy is not None and i < len(box_xyxy):
            x1, y1, x2, y2 = box_xyxy[i].astype(int)
            bbox = (x1, y1, x2, y2)

        people.append(Person(keypoints=kp_dict, bbox=bbox))
    return people


def parse_balls(ball_results) -> List[Ball]:
    balls: List[Ball] = []
    if not ball_results:
        return balls

    # There may be multiple result objects (one per augment or batch); iterate all
    for res in ball_results:
        if res.boxes is None or res.boxes.xyxy is None:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = float(max(0, (x2 - x1)) * max(0, (y2 - y1)))
            balls.append(Ball(center=(cx, cy), bbox=(x1, y1, x2, y2), conf=float(conf), area=area))
    return balls


# -------------------------------
# Visualizer
# -------------------------------

import cv2
import numpy as np
import numpy as np

class Visualizer:
    def __init__(self, draw_pose_overlay: bool = True) -> None:
        self.draw_pose_overlay = draw_pose_overlay

    def overlay(
        self,
        frame: np.ndarray,
        pose_results,
        state,  # FrameState
        dribbler_count: int = 0,
        holding: int = 0,
        steps: int = 0,
        travel: bool = False
    ) -> np.ndarray:

        out = frame.copy()

        # 1️⃣ Pose skeleton overlay
        if self.draw_pose_overlay and pose_results:
            pose_overlay = pose_results[0].plot()
            out = cv2.addWeighted(out, 0.5, pose_overlay, 0.5, 0)

        # 2️⃣ Ball overlay (insan gibi)
        for i, b in enumerate(state.balls):
            x1, y1, x2, y2 = b.bbox
            cx, cy = b.center

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Center keypoint (insan skeleton gibi)
            cv2.circle(out, (int(cx), int(cy)), 6, (0, 255, 0), -1)
            # Confidence text
            cv2.putText(
                out,
                f"Ball {i}: {b.conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # 3️⃣ HUD
        hud_lines = [
            f"Frame: {state.frame_idx}",
            f"People: {len(state.people)}",
            f"Balls: {len(state.balls)}"
        ]

        if state.people:
            avg_person_conf = np.mean([kp.conf for p in state.people for kp in p.keypoints.values()])
            hud_lines.append(f"Person avg: {avg_person_conf*100:.0f}%")
        if state.balls:
            avg_ball_conf = np.mean([b.conf for b in state.balls])
            hud_lines.append(f"Ball avg: {avg_ball_conf*100:.0f}%")

        hud_lines.append(f"Dribbles: {dribbler_count}")
        hud_lines.append(f"Holding: {holding}")
        hud_lines.append(f"Steps: {steps}")
        if travel:
            hud_lines.append("TRAVEL!")

        start_y = int(out.shape[0] * 0.05)
        line_spacing = 30
        for i, line in enumerate(hud_lines):
            y_pos = start_y + i * line_spacing
            # Outline
            cv2.putText(out, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            # İç yazı
            cv2.putText(out, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        return out




# -------------------------------
# Camera helper
# -------------------------------

def open_source(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        # try a couple of fallback indices if webcam
        if isinstance(source, int):
            for idx in range(3):
                cap2 = cv2.VideoCapture(idx)
                if cap2.isOpened():
                    return cap2
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


# -------------------------------
# Main loop
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="AI Basketball Coach — Step 1 (Unified Skeleton)")
    ap.add_argument("--source", default="0", help="Webcam index or video file path (default: 0)")
    ap.add_argument("--frame-skip", type=int, default=CONFIG["frame_skip"], help="Process every Nth frame")
    ap.add_argument("--device", default=CONFIG["device"], help="cuda/cpu/None")
    ap.add_argument("--pose", default=CONFIG["pose_model"], help="Pose model path")
    ap.add_argument("--ball", default=CONFIG["ball_model"], help="Ball model path")
    ap.add_argument("--conf-pose", type=float, default=CONFIG["conf_pose"], help="Pose confidence")
    ap.add_argument("--conf-ball", type=float, default=CONFIG["conf_ball"], help="Ball confidence")
    args = ap.parse_args()

    # Load models
    print(f"Loading models… pose={args.pose}  ball={args.ball}  device={args.device}")
    models = Models(args.pose, args.ball, device=args.device,
                    conf_pose=args.conf_pose, conf_ball=args.conf_ball)

    # Video source
    cap = open_source(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    print(f"Video opened: {args.source}  {width}x{height} @ {fps:.1f} fps")

    viz = Visualizer(draw_pose_overlay=True)
    buffer: deque[FrameState] = deque(maxlen=CONFIG["buffer_size"])  # store last N states
    dribbler = DribbleCounter()
    holder = BallHoldingDetector()
    step_detector = StepDetector()
    
    travel_detector = TravelDetector()
    try:
        import tkinter as tk
        _root = tk.Tk()
        screen_width = _root.winfo_screenwidth()
        screen_height = _root.winfo_screenheight()
        _root.destroy()
    except Exception:
        # tkinter çalışmazsa makul varsayılan
        screen_width, screen_height = 1920, 1080

    t0 = time.time()
    frame_idx = 0

    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # frame skip
        if args.frame_skip > 1 and (frame_idx % args.frame_skip) != 0:
            cv2.namedWindow("AI Coach", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("AI Coach", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("AI Coach", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        # Inference
        pose_res = models.infer_pose(frame)
        ball_res = models.infer_ball(frame)
        print("DEBUG – raw ball_res boxes:",
             [len(r.boxes) if hasattr(r, "boxes") else "no boxes"
             for r in ball_res])
        # Normalize
        people = parse_people(pose_res)
        balls = parse_balls(ball_res)
        print("DEBUG – parsed balls:", len(balls))
        # Holding ve step hesaplama
        holding = holder.update(people, balls)
        steps = step_detector.update(people)

        # 🔴 TRAVEL DETECTION BURAYA EKLENECEK
        travel = travel_detector.check(steps, holding)

        # Dribble counter update
        if balls:
            y_center = balls[0].center[1]   # sadece ilk topu alıyoruz
            dribble_count = dribbler.update(y_center)
        else:
            dribble_count = dribbler.count

        state = FrameState(
            timestamp=time.time(),
            frame_idx=frame_idx,
            width=frame.shape[1],
            height=frame.shape[0],
            people=people,
            balls=balls,
        )
        buffer.append(state)

        # HUD text (for debug)
        hud = f"Frame {frame_idx} | buffer {len(buffer)} | Dribbles: {dribble_count} | Holding: {holding} | Steps: {steps}"
        if travel:   # 👈 TRAVEL varsa HUD’a ekle
            hud += " | TRAVEL!"

       # Visualize
        out = viz.overlay(
            frame=frame,
            pose_results=pose_res,
            state=state,
            dribbler_count=dribble_count,
            holding=holding,
            steps=steps,
            travel=travel
            
)



        # İstersen sahnede kırmızı tint uyarısı da ekleyebilirsin:
        if travel:
            red_tint = np.full_like(out, (0, 0, 255), dtype=np.uint8)
            out = cv2.addWeighted(out, 0.7, red_tint, 0.3, 0)
            cv2.putText(out, "TRAVEL VIOLATION!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

       
        out_resized = cv2.resize(out, (screen_width, screen_height))

        cv2.namedWindow("AI Coach", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("AI Coach", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("AI Coach", out_resized)   # 👈 burada frame değil out_resized kullan




        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
       


    cap.release()
    cv2.destroyAllWindows()
    dt = time.time() - t0
    if frame_idx:
        print(f"Done. Processed {frame_idx} frames in {dt:.2f}s ({frame_idx/max(dt,1e-6):.1f} fps incl. skip)")


if __name__ == "__main__":
    main()
