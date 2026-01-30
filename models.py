#!/usr/bin/env python3
"""
Model architectures for SAM3 linear probing
"""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Simple linear classifier for segmentation"""
    def __init__(self, feature_dim=256, num_classes=2):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] feature maps from SAM3
        Returns:
            logits: [B, num_classes, H, W] per-pixel class logits
        """
        logits = self.classifier(features)
        return logits
