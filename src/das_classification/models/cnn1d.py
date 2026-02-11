# src/das_classification/models/cnn1d.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional 

import torch 
import torch.nn as nn
import torch.nn.functional as F 


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    num_classes: int
    base_width: int = 64
    dropout: float = 0.1


class ConvBlock(nn.Module):
    def __init__(
        self, cin: int, cout:int, k: int=9, stride: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x 


class DASConvClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        w = cfg.base_width 

        self.stem = nn.Sequential(
            ConvBlock(cfg.in_channels, w, k=9, stride=2, dropout=cfg.dropout),
            ConvBlock(w, 2*w, k=9, stride=2, dropout=cfg.dropout),
            ConvBlock(2*w, 4*w, k=9, stride=2, dropout=cfg.dropout),
            ConvBlock(4*w, 4*w, k=9, stride=2, dropout=cfg.dropout)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*w, 4*w),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4*w, cfg.num_classes)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(x)
        x = self.pool(x)
        return x
