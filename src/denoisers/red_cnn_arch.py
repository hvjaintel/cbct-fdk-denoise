"""RED-CNN architecture (Chen et al., low-dose CT) for train/export.

Slightly adapted with padding so spatial size is preserved — convenient for
OpenVINO tiled inference in this demo. Not a clinical model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """Residual Encoder–Decoder CNN (image-domain CT denoiser)."""

    def __init__(self, channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2)

        self.tconv1 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.tconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.tconv3 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.tconv4 = nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=1, padding=2)
        self.tconv5 = nn.ConvTranspose2d(channels, 1, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))

        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        # Residual to input; no final ReLU so the net cannot collapse to all-zeros
        out = out + residual_1
        return out
