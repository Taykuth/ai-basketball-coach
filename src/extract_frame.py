import cv2

cap = cv2.VideoCapture(r"C:\Users\osman\Downloads\segment_2.mp4")
ret, frame = cap.read()
cv2.imwrite("frame.jpg", frame)
cap.release()
print("frame.jpg kaydedildi")
