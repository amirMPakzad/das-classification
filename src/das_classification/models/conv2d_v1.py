import torch
import torch.nn as nn


class DasConv2dModel(nn.Module):
    """
    Paper-style simple CNN (non-lazy, explicit in_channels):

      Conv(64)  -> LeakyReLU -> MaxPool(pool)
      Conv(256) -> LeakyReLU -> MaxPool(pool)
      Flatten
      Dense(fc_hidden=1024) -> ReLU -> Dropout
      Dense(num_classes)

    Expected dataset batch:
      x: (B, C, T, F)

    Feeding to Conv2d:
      - mode="T_as_in_channels": (B, C, T, F) -> (B, T, C, F)
            => you MUST pass in_channels = T
      - mode="1ch": (B, C, T, F) -> (B, 1, C*T, F)
            => you MUST pass in_channels = 1
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        mode: str = "T_as_in_channels",   # "T_as_in_channels" or "1ch"
        pool: int = 4,
        fc_hidden: int = 1024,
        dropout: float = 0.3,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if mode not in {"T_as_in_channels", "1ch"}:
            raise ValueError('mode must be "T_as_in_channels" or "1ch"')
        if mode == "1ch" and in_channels != 1:
            raise ValueError("For mode='1ch', in_channels must be 1 (Conv2d input channels).")
        self.mode = mode

        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=pool)

        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=pool)

        # Use LazyLinear ONLY for fc1 input dim (safe: optimizer sees params; LazyLinear has params)
        # If you prefer fully-static, replace with AdaptiveAvgPool2d and fixed Linear.
        self.fc1 = nn.LazyLinear(fc_hidden)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def _make_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as (B,C,T,F), got {tuple(x.shape)}")

        if self.mode == "T_as_in_channels":
            # (B,C,T,F) -> (B,T,C,F)
            return x.permute(0, 2, 1, 3).contiguous()

        # mode == "1ch"
        b, c, t, f = x.shape
        return x.reshape(b, 1, c * t, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._make_input(x)

        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


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
    # Example
    B, C, T, F = 8, 32, 8, 2048
    x = torch.randn(B, C, T, F)

    # Example meta:
    meta = {"shape": [1000, C, T, F]}

    mode = "T_as_in_channels"
    in_channels = infer_in_channels_from_meta(meta, mode)
    model = DasConv2dModel(in_channels=in_channels, num_classes=9, mode=mode, pool=4, fc_hidden=1024)
    print(model(x).shape)

    mode2 = "1ch"
    in_channels2 = infer_in_channels_from_meta(meta, mode2)
    model2 = DasConv2dModel(in_channels=in_channels2, num_classes=9, mode=mode2, pool=4, fc_hidden=1024)
    print(model2(x).shape)