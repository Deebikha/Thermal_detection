from ultralytics import YOLO
import csv
import os

# Load your trained YOLOv8 model
model = YOLO("runs/detect/thermal_detection_run5/weights/best.pt")

# Get class names
class_names = model.names  # {0: "person", 1: "bicycle", ...}

# Folder for saving all predictions
output_folder = "runs/detect/final"
os.makedirs(output_folder, exist_ok=True)

# Run predictions
results = model.predict(
    source="test_images",
    save=True,
    project="runs/detect",
    name="final",
    exist_ok=True,
    verbose=True
)

# Save all predictions in CSV with labels
csv_file = os.path.join(output_folder, "predictions.csv")
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image", "class_id", "label", "confidence", "x1", "y1", "x2", "y2"])
    
    for r in results:
        image_name = os.path.basename(r.path)
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            label = class_names[int(cls)]
            writer.writerow([image_name, int(cls), label, float(conf), x1, y1, x2, y2])

# Print predictions in terminal with labels
for r in results:
    print(f"\nImage: {r.path}")
    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        label = class_names[int(cls)]
        print(f"Class: {int(cls)} ({label}), Conf: {conf:.2f}, BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

print(f"\nAnnotated images saved in: {output_folder}")
print(f"CSV predictions saved in: {csv_file}")
