import cv2

input_path = "videos/source.mp4"
output_path = "videos/sample.mp4"

start_sec = 605    # 00:10:05
duration_sec = 780 # 13 dk

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError("Video açılamadı")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

start_frame = int(start_sec * fps)
end_frame = int((start_sec + duration_sec) * fps)

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_id >= start_frame and frame_id <= end_frame:
        out.write(frame)

    frame_id += 1
    if frame_id > end_frame:
        break

cap.release()
out.release()
print("Kesilen video kaydedildi:", output_path)
