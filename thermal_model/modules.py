# thermal_model/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalAttention(nn.Module):
    """Thermal attention mechanism"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class ThermalAwareConv(nn.Module):
    """Convolution with thermal attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.att = ThermalAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.att(x)

class C2fThermal(nn.Module):
    """Modified C2f block with thermal awareness"""
    def __init__(self, c1, c2, n=1, shortcut=False):
        super().__init__()
        self.c = int(c2 * 0.5)
        self.cv1 = ThermalAwareConv(c1, 2 * self.c, 1)
        self.cv2 = ThermalAwareConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(ThermalAwareConv(self.c, self.c, 3) for _ in range(n))
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
