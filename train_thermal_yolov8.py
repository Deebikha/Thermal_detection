# train_thermal_yolov8.py
from thermal_model.thermal_yolov8 import ThermalYOLOv8
from ultralytics import YOLO
import torch

def main():
    # -----------------------------
    # Initialize Thermal YOLOv8
    # -----------------------------
    model_path = 'yolov8n.pt'  # pretrained weights
    num_classes = 5
    in_channels = 1  # thermal input
    
    # Initialize your thermal-aware YOLOv8
    detector = ThermalYOLOv8(model_path=model_path, in_channels=in_channels, num_classes=num_classes)
    
    # -----------------------------
    # Prepare model for training
    # -----------------------------
    # Access underlying YOLO model for .train()
    yolo_model: YOLO = detector.model
    
    # -----------------------------
    # Training configuration
    # -----------------------------
    yolo_model.train(
        data='dataset.yaml',      # your dataset yaml
        epochs=30,                # number of epochs
        imgsz=416,                # image size
        batch=4,                  # batch size
        device='cuda' if torch.cuda.is_available() else 'cpu',
        name='thermal_detection_run',
        workers=4
    )

if __name__ == "__main__":
    main()
