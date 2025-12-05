from ultralytics import YOLO
import torch

print("PyTorch:", torch.__version__)

# Use yolov8n pretrained as starting point (smallest for CPU)
model = YOLO('yolov8n.pt')

# Train (CPU friendly)
model.train(
    data='dataset.yaml',   # path to dataset.yaml you created
    epochs=30,             # reduce on CPU if needed
    imgsz=416,
    batch=4,               # decrease if OOM, try 2
    device='cpu',
    name='thermal_detection_run',
    workers=2
)
