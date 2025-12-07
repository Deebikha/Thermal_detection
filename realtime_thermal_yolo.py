import cv2
from ultralytics import YOLO
import numpy as np
import csv
import time

# -------------------------------
# Step 1: Load your trained YOLO model
# -------------------------------
model = YOLO("runs/detect/thermal_detection_run6/weights/best.pt")
class_names = model.names  # dictionary of class labels

# -------------------------------
# Step 2: Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Get frame width and height for saving video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
out = cv2.VideoWriter(
    'thermal_detection_output.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (width, height)
)

# Initialize CSV for logging predictions
csv_file = open("thermal_predictions.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])

frame_count = 0

# -------------------------------
# Step 3: Start processing frames
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Step 3a: Apply thermal effect (grayscale + color map)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal_frame = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    # Step 3b: YOLO prediction
    results = model.predict(thermal_frame, verbose=False, conf=0.3)

    # Step 3c: Draw boxes and labels & log to CSV & terminal
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = class_names[int(cls)]
            color = (0, 255, 0)
            cv2.rectangle(thermal_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                thermal_frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Print in terminal
            print(f"[{timestamp}] Frame {frame_count}: {label} {conf:.2f} Box: {x1},{y1},{x2},{y2}")

            # Save to CSV
            csv_writer.writerow([timestamp, label, conf.item(), x1, y1, x2, y2])

    # Step 3d: Display the frame
    cv2.imshow("Thermal YOLO Detection", thermal_frame)

    # Step 3e: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Step 4: Release resources
# -------------------------------
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
