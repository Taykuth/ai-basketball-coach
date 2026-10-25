# AI Basketball Coach

Python prototype that watches a basketball video (or a webcam) and counts dribbles, detects when the player is holding the ball, counts steps and flags a possible travel. Built together with my friend Osman.

## How it works

The main pipeline in `src/` runs two YOLO models on every frame:

| Step | How it is done | File |
| --- | --- | --- |
| Player pose | YOLOv8 pose model, 17 COCO keypoints | `src/main.py` |
| Ball | A YOLO ball detection model (`basketballModel.pt`, not included) | `src/main.py` |
| Dribble count | Ball centre moves up and then down by more than 2% of the frame height, with a 0.2 s debounce | `src/dribble_counter.py` |
| Holding | Ball within 300 px of either wrist for more than 0.85 s | `src/ball_holding.py` |
| Steps | Ankle movement larger than 15% of the hip-to-ankle length, smoothed over 3 frames, at least 0.3 s apart | `src/step_detection.py` |
| Travel | Holding the ball and more than 2 steps | `src/travel_detection.py` |

`src_cv/` is a second, model-free experiment for the ball only: HSV colour mask and contours, a motion mask and optional template matching, followed by an OpenCV CSRT or KCF tracker. `src_cv/hsv_calibrate.py` helps pick the HSV range for a given video.

A similar open-source project, [ayushpai/AI-Basketball-Referee](https://github.com/ayushpai/AI-Basketball-Referee), uses the same idea and distributes a YOLO basketball model under the same file name.

## Run

```bash
pip install -r requirements.txt

# YOLO pose + ball model
python src/main.py --source path/to/video.mp4 --pose yolov8s-pose.pt --ball models/basketballModel.pt

# webcam
python src/main.py --source 0 --ball models/basketballModel.pt

# model-free ball tracking
python src_cv/hsv_calibrate.py --source path/to/video.mp4
python src_cv/main_cv.py --source path/to/video.mp4 --hsv_lower "(5,120,80)" --hsv_upper "(25,255,255)"
```

Ultralytics downloads `yolov8s-pose.pt` automatically on first use. The ball model and videos are not in this repository: put your ball model in `models/` and use your own footage. Press `q` to quit the window.

## Limitations

- **No measured accuracy.** Dribble, step and travel counts have not been compared against hand-labelled videos, so there are no precision or recall numbers yet.
- **Simplified travel rule.** "More than 2 steps while holding" ignores the gather step and the pivot foot, so it will flag legal moves.
- **Wall-clock timing.** The debounce and holding thresholds use `time.time()`, not the video's frame timestamps. On a slow machine the same video gives different results.
- **Fixed pixel threshold.** The 300 px holding distance does not scale with resolution or distance to the camera.
- **One player, one ball.** Only the first detected person and ball are used.

## Next steps

1. Label 5–10 short clips by hand and measure dribble and step counting against them.
2. Switch timing to frame index ÷ FPS.
3. Scale the holding distance by the player's body size.
