# thermal_model/thermal_yolov8_full.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from .modules import ThermalAwareConv, C2fThermal

# -----------------------------
# Thermal YOLOv8 Backbone
# -----------------------------
class ThermalYOLOv8Backbone(nn.Module):
    """YOLOv8 backbone with thermal-aware convolutions"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.stem = ThermalAwareConv(in_channels, 64, 3, 2)
        self.stage1_conv = ThermalAwareConv(64, 128, 3, 2)
        self.stage1_c2f = C2fThermal(128, 128, n=3)
        self.stage2_conv = ThermalAwareConv(128, 256, 3, 2)
        self.stage2_c2f = C2fThermal(256, 256, n=3)
        self.stage3_conv = ThermalAwareConv(256, 512, 3, 2)
        self.stage3_c2f = C2fThermal(512, 512, n=3)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1_conv(x)
        x = self.stage1_c2f(x)
        x = self.stage2_conv(x)
        x = self.stage2_c2f(x)
        x = self.stage3_conv(x)
        x = self.stage3_c2f(x)
        return x

# -----------------------------
# Thermal Preprocessing
# -----------------------------
class ThermalPreprocessor:
    """Thermal image preprocessing pipeline"""
    def adaptive_noise_reduction(self, img):
        gaussian = cv2.GaussianBlur(img, (5, 5), 0)
        median = cv2.medianBlur(img, 5)
        filtered = cv2.addWeighted(gaussian.astype(np.float32), 0.7,
                                   median.astype(np.float32), 0.3, 0)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def temperature_normalization(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        if len(img.shape) == 2:
            normalized = clahe.apply(img.astype(np.uint8))
        else:
            normalized = img
        return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    def edge_enhancement(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f_img = img.astype(np.float32)
        sobelx = cv2.Sobel(f_img, cv2.CV_32F, 1, 0, 3)
        sobely = cv2.Sobel(f_img, cv2.CV_32F, 0, 1, 3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
        enhanced = cv2.addWeighted(f_img, 0.7, gradient, 0.3, 0)
        return enhanced.astype(np.uint8)

    def morphological_processing(self, img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = self.adaptive_noise_reduction(img)
        img = self.temperature_normalization(img)
        img = self.edge_enhancement(img)
        img = self.morphological_processing(img)
        return img

# -----------------------------
# Multi-spectral Fusion
# -----------------------------
class MultiSpectralFusion:
    def __init__(self):
        self.weights = {'thermal':0.5, 'nir':0.25, 'visible':0.25}

    def adaptive_fusion(self, thermal, nir=None, visible=None):
        h, w = thermal.shape[:2]
        if len(thermal.shape)==2: thermal=cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
        if nir is not None and len(nir.shape)==2: nir=cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR)
        if visible is not None and len(visible.shape)==2: visible=cv2.cvtColor(visible, cv2.COLOR_GRAY2BGR)
        nir = cv2.resize(nir,(w,h)) if nir is not None else np.zeros_like(thermal)
        visible = cv2.resize(visible,(w,h)) if visible is not None else np.zeros_like(thermal)

        t_norm = cv2.normalize(thermal,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
        n_norm = cv2.normalize(nir,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
        v_norm = cv2.normalize(visible,None,0,1,cv2.NORM_MINMAX).astype(np.float32)

        fused = self.weights['thermal']*t_norm + self.weights['nir']*n_norm + self.weights['visible']*v_norm
        return np.clip(fused*255,0,255).astype(np.uint8)

# -----------------------------
# Thermal YOLOv8 with preprocessing
# -----------------------------
class ThermalYOLOv8:
    """Fully integrated thermal YOLOv8"""
    def __init__(self, model_path='yolov8n.pt', in_channels=1, num_classes=5):
        self.model = YOLO(model_path)
        self.backbone = ThermalYOLOv8Backbone(in_channels)
        self.model.model.backbone = self.backbone
        self.preprocessor = ThermalPreprocessor()
        self.fusion = MultiSpectralFusion()

        # Update first conv for thermal input
        self.model.model.backbone.stem.conv = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)

        # Update class info
        self.model.model.nc = num_classes
        self.model.model.names = {i: f'class{i}' for i in range(num_classes)}

    def process_frame(self, thermal_img, nir_img=None, visible_img=None):
        # Preprocess and fuse
        thermal_img = self.preprocessor.preprocess(thermal_img)
        fused_img = self.fusion.adaptive_fusion(thermal_img, nir_img, visible_img)

        # YOLO inference
        results = self.model(fused_img, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': self.model.names[int(box.cls[0])]
                })
        annotated = self.draw_detections(fused_img.copy(), detections)
        return annotated, detections

    def draw_detections(self, image, detections):
        for det in detections:
            bbox = det['bbox'].astype(int)
            cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            cv2.putText(image,f"{det['class_name']}:{det['confidence']:.2f}",(bbox[0],bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        return image

# -----------------------------
# Utility: simulate thermal from RGB
# -----------------------------
def simulate_thermal_from_rgb(rgb_img):
    gray=cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    thermal=cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    noise=np.random.normal(0,10,thermal.shape).astype(np.float32)
    return np.clip(thermal.astype(np.float32)+noise,0,255).astype(np.uint8)

# -----------------------------
# Demo
# -----------------------------
def main():
    detector = ThermalYOLOv8('yolov8n.pt', in_channels=1, num_classes=5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        thermal_img = simulate_thermal_from_rgb(frame)
        annotated, detections = detector.process_frame(thermal_img, None, frame)

        cv2.imshow('Thermal Detection', annotated)
        cv2.imshow('Simulated Thermal', thermal_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
