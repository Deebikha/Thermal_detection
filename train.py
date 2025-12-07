from thermal_model.thermal_yolov8 import ThermalYOLOv8

def main():
    model = ThermalYOLOv8(model_path='yolov8n.pt', in_channels=1, num_classes=5)
    
    # Train on your dataset
    model.model.train(
        data='dataset.yaml',
        epochs=30,
        imgsz=416,
        batch=4,
        device='cpu',
        name='thermal_detection_run',
        workers=2
    )

if __name__ == "__main__":
    main()
