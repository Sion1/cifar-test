"""ResNet-34 adapted for CIFAR-10 (32×32 inputs).

Differences vs torchvision's ImageNet ResNet-34:
  • first conv: 3×3 stride 1 (vs 7×7 stride 2) — preserves resolution on small images
  • initial maxpool removed
  • final FC: 10 classes
The penultimate feature (after global-avg-pool, before FC) is exposed for
downstream t-SNE visualization, and a hook on the last conv layer is provided
for Grad-CAM.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """ResNet for 32×32 CIFAR. layers=[3,4,6,3] gives ResNet-34."""

    def __init__(self, block, num_blocks, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc     = nn.Linear(512 * block.expansion, num_classes)
        self.feature_dim = 512 * block.expansion
        # Buffer for grad-cam: stash last-layer feature map + grad
        self._cam_feat = None
        self._cam_grad = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        """Returns penultimate feature (after GAP, before FC). Shape: (B, 512)."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feat = self.layer4(out)        # (B, 512, H, W)
        # Grad-CAM hooks: keep this feat alive on backward
        if feat.requires_grad:
            feat.register_hook(self._save_cam_grad)
        self._cam_feat = feat
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 512)
        return pooled, feat

    def _save_cam_grad(self, grad):
        self._cam_grad = grad

    def forward(self, x):
        pooled, _ = self.forward_features(x)
        return self.fc(pooled)


def build_resnet34(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
