# AI Basketball Coach

---

## 📌 English Version

### Overview
**AI Basketball Coach** is a Python-based real-time basketball analysis system. It tracks players and the ball using YOLO pose detection and a custom ball detection model. The system can detect dribbles, ball holding, steps, and travel violations.

### Features
- **Player Pose Detection (YOLOv8 Pose)**
- **Ball Detection (Custom YOLOv8 model)**
- **Dribble Counter:** Counts dribbles based on ball movement
- **Ball Holding Detector:** Detects when the player is holding the ball
- **Step Detector:** Monitors player steps to track travel violations
- **Travel Detector:** Flags traveling violations when steps exceed limits while holding the ball
- **Visualizer:** Overlays pose, ball, and HUD info on frames

### Installation
```bash
git clone https://github.com/yourusername/ai-basketball-coach.git
cd ai-basketball-coach
pip install -r requirements.txt

Usage
python src/main.py --source 0 --pose models/yolov8s-pose.pt --ball models/basketballModel.pt

Project Structure

ai-basketball-coach/
│
├─ src/
│  ├─ main.py
│  ├─ ball_holding.py
│  ├─ dribble_counter.py
│  ├─ step_detection.py
│  ├─ travel_detection.py
│  └─ extract_frame.py
│
├─ models/
│  ├─ basketballModel.pt
│  ├─ yolov8n-pose.pt
│  └─ yolov8s-pose.pt
│
├─ outputs/
│  └─ predict/
│
└─ frame.jpg

Notes

Tested on Python 3.11+ and OpenCV 4.x

Models must be downloaded or trained before running the code

Press q to quit the visualization

--------------
# AI Basketball Coach

## 📌 Genel Bakış
**AI Basketball Coach**, gerçek zamanlı basketbol analiz sistemi sunan bir Python projesidir. Oyuncu ve topu YOLO poz tespiti ve özel top tespiti modeli ile takip eder. Sistem driplingleri, top tutmayı, adımları ve travel ihlallerini algılar.

## Özellikler
- **Oyuncu Poz Tespiti (YOLOv8 Pose)**
- **Top Tespiti (Özel YOLOv8 modeli)**
- **Dribble Sayıcı:** Top hareketine göre driplingleri sayar
- **Top Tutma Algılayıcı:** Oyuncunun topu tutup tutmadığını algılar
- **Adım Algılayıcı:** Oyuncu adımlarını sayar ve travel ihlallerini izler
- **Travel Algılayıcı:** Top tutulurken adım sayısı limit aşarsa uyarı verir
- **Görselleştirici:** Karelere poz, top ve HUD bilgilerini bindirir

## Kurulum
```bash
git clone https://github.com/yourusername/ai-basketball-coach.git
cd ai-basketball-coach
pip install -r requirements.txt
