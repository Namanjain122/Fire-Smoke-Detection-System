import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Open external camera (MXBRIO) or fallback
camera_index = 1
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Could not open MXBRIO, falling back to default camera")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Video Writer variables
recording = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
no_detection_counter = 0
max_no_detection_frames = 20  # stop recording after ~20 frames without detection

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    results = model(frame, conf=0.3)
    annotated_frame = results[0].plot()

    # Check if fire/smoke detected
    detected = False
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id].lower()
        if "fire" in class_name or "smoke" in class_name:
            detected = True
            break

    # Start recording if fire/smoke detected
    if detected:
        no_detection_counter = 0
        if not recording:
            filename = f"fire_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            out = cv2.VideoWriter(filename, fourcc, 20.0,
                                  (int(cap.get(3)), int(cap.get(4))))
            print(f"🔴 Recording started: {filename}")
            recording = True
    else:
        if recording:
            no_detection_counter += 1
            if no_detection_counter > max_no_detection_frames:
                recording = False
                out.release()
                print("✅ Recording stopped and saved")

    # Write frame if recording
    if recording and out is not None:
        out.write(frame)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Live Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if recording and out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
