# thermal_yolov8.py
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
# Thermal YOLOv8 Wrapper
# -----------------------------
class ThermalYOLOv8(nn.Module):
    """YOLOv8 with a thermal-aware backbone"""
    def __init__(self, model_path='yolov8n.pt', in_channels=1, num_classes=5):
        super().__init__()
        # Load standard YOLOv8
        self.model = YOLO(model_path)
        
        # Create thermal backbone
        self.backbone = ThermalYOLOv8Backbone(in_channels)

        # Replace YOLOv8 backbone with thermal backbone
        self.model.model.backbone = self.backbone

        # Update the first conv layer if necessary
        if isinstance(self.model.model.backbone.stem.conv, nn.Conv2d):
            self.model.model.backbone.stem.conv = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=2, padding=1
            )

        # Update class info
        self.model.model.nc = num_classes
        self.model.model.names = {i: f'class{i}' for i in range(num_classes)}

    def forward(self, x):
        return self.model(x)
