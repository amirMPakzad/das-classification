# src/das_classification/models/cnn1d.py

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    num_classes: int
    base_width: int = 64
    dropout: float = 0.2
    # optional: small stochastic depth for regularization
    drop_path: float = 0.05


class DropPath(nn.Module):
    """Stochastic Depth (per-sample)."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SE1D(nn.Module):
    """Squeeze-and-Excitation for 1D conv features."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ResBlock1D(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        k: int = 9,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        pad = ((k - 1) // 2) * dilation

        self.conv1 = nn.Conv1d(cin, cout, kernel_size=k, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)

        self.conv2 = nn.Conv1d(cout, cout, kernel_size=k, stride=1, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)

        self.se = SE1D(cout) if use_se else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.dp = DropPath(drop_path)

        if cin != cout or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(cout),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)
        out = self.dp(out)

        out = out + identity
        out = F.gelu(out)
        return out


class DASConvClassifier(nn.Module):
    """
    Stronger 1D CNN:
    - stem downsampling
    - residual stages with dilations + SE
    - global avg pooling
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        w = cfg.base_width

        # Stem: light downsampling + feature lift
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, w, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(w),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Stages: increase width + downsample, then dilated blocks for larger receptive field
        dp = cfg.drop_path
        self.stage1 = nn.Sequential(
            ResBlock1D(w, w, k=9, stride=1, dilation=1, dropout=cfg.dropout * 0.5, drop_path=dp * 0.2),
            ResBlock1D(w, w, k=9, stride=1, dilation=2, dropout=cfg.dropout * 0.5, drop_path=dp * 0.4),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(w, 2*w, k=9, stride=2, dilation=1, dropout=cfg.dropout, drop_path=dp * 0.6),
            ResBlock1D(2*w, 2*w, k=9, stride=1, dilation=2, dropout=cfg.dropout, drop_path=dp * 0.8),
        )
        self.stage3 = nn.Sequential(
            ResBlock1D(2*w, 4*w, k=9, stride=2, dilation=1, dropout=cfg.dropout, drop_path=dp),
            ResBlock1D(4*w, 4*w, k=9, stride=1, dilation=3, dropout=cfg.dropout, drop_path=dp),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Head: BN + Dropout tends to stabilize
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(4*w),
            nn.Dropout(cfg.dropout),
            nn.Linear(4*w, 2*w),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2*w, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.head(x)
        return x



