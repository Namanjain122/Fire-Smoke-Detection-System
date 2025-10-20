import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Try external camera (MXBRIO) first
camera_index = 1  # often external cameras are 1, but adjust if needed
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # force DirectShow

if not cap.isOpened():
    print("❌ Could not open MXBRIO, falling back to default camera")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # fallback

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    results = model(frame, conf=0.3)
    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Live Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
