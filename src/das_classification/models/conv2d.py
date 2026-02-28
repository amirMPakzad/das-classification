import torch
import torch.nn as nn


class DasConv2dModel(nn.Module):
    """
    Regularized CNN for DAS 3D spectrum input.

    Input batch:
      x: (B, C, T, F)

    Feeding modes:
      - mode="T_as_in_channels": (B, C, T, F) -> (B, T, C, F)
            => pass in_channels = T
      - mode="1ch": (B, C, T, F) -> (B, 1, C*T, F)
            => pass in_channels = 1

    Key anti-overfit changes vs your previous model:
      - Dropout2d on feature maps
      - Bottleneck 1x1 conv (forces compact representation)
      - GlobalAvgPool2d instead of huge Flatten -> FC (big overfit source)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        mode: str = "T_as_in_channels",   # "T_as_in_channels" or "1ch"
        pool: int = 4,
        base_ch: int = 32,                # lower = more regularization
        bottleneck_ch: int = 64,          # compact embedding
        dropout2d: float = 0.15,          # feature-map dropout
        dropout: float = 0.30,            # classifier dropout
        negative_slope: float = 0.01,
        norm: str = "bn",                 # "bn" or "gn"
        gn_groups: int = 8,
    ):
        super().__init__()

        if mode not in {"T_as_in_channels", "1ch"}:
            raise ValueError('mode must be "T_as_in_channels" or "1ch"')
        if mode == "1ch" and in_channels != 1:
            raise ValueError("For mode='1ch', in_channels must be 1.")
        self.mode = mode

        def Norm2d(ch: int) -> nn.Module:
            if norm == "bn":
                return nn.BatchNorm2d(ch)
            if norm == "gn":
                g = min(gn_groups, ch)
                while ch % g != 0 and g > 1:
                    g -= 1
                return nn.GroupNorm(g, ch)
            raise ValueError("norm must be 'bn' or 'gn'")

        act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        c1 = base_ch
        c2 = base_ch * 4  # keep it moderate (paper-ish but smaller than 256*big flatten)
        c3 = base_ch * 8

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            Norm2d(c1),
            act,
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            Norm2d(c2),
            act,
            nn.Dropout2d(p=dropout2d),
            nn.MaxPool2d(kernel_size=pool),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            Norm2d(c3),
            act,
            nn.Dropout2d(p=dropout2d),
            nn.MaxPool2d(kernel_size=pool),
        )

        # Bottleneck: forces compact representation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c3, bottleneck_ch, kernel_size=1, bias=False),
            Norm2d(bottleneck_ch),
            act,
        )

        # Global pooling removes dependence on (C,F) sizes and kills huge FC overfit
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(bottleneck_ch, num_classes),
        )

    def _make_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as (B,C,T,F), got {tuple(x.shape)}")

        if self.mode == "T_as_in_channels":
            # (B,C,T,F) -> (B,T,C,F)  => in_channels MUST equal T
            return x.permute(0, 2, 1, 3).contiguous()

        # mode == "1ch"
        b, c, t, f = x.shape
        return x.reshape(b, 1, c * t, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._make_input(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.bottleneck(x)
        x = self.gap(x)
        return self.classifier(x)


def infer_in_channels_from_meta(meta: dict, mode: str) -> int:
    """
    meta["shape"] is [N, C, T, F]
    """
    if "shape" not in meta or len(meta["shape"]) != 4:
        raise ValueError("meta must contain shape=[N,C,T,F].")
    _, C, T, _ = meta["shape"]
    if mode == "T_as_in_channels":
        return int(T)
    if mode == "1ch":
        return 1
    raise ValueError("mode must be 'T_as_in_channels' or '1ch'")


if __name__ == "__main__":
    B, C, T, F = 8, 32, 4, 2048
    x = torch.randn(B, C, T, F)
    meta = {"shape": [1000, C, T, F]}

    mode = "T_as_in_channels"
    in_ch = infer_in_channels_from_meta(meta, mode)
    model = DasConv2dModel(in_channels=in_ch, num_classes=9, mode=mode, base_ch=32, bottleneck_ch=64)
    print(model(x).shape)  # (8, 9)