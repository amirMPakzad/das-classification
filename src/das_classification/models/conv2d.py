import torch
import torch.nn as nn


class DasConv2dModel(nn.Module):
    """:
      Conv(64) -> LeakyReLU -> MaxPool
      Conv(256)-> LeakyReLU -> MaxPool
      Flatten
      Dense(1024) -> ReLU (or LeakyReLU) -> Dropout
      Dense(num_classes)

    Input is 3D magnitude spectrum: (channel × time_window × frequency)

    Expected input tensor:
      x: (B, C, T, F)

    How it is fed to Conv2d:
      - mode="T_as_in_channels": treat time-windows as input channels to Conv2d
            (B, C, T, F) -> (B, T, C, F)
        This is usually the most natural for your representation.

      - mode="1ch": flatten (C,T) into height, use a single input channel
            (B, C, T, F) -> (B, 1, C*T, F)
        This can work too, but changes inductive bias.

    Use logits output with CrossEntropyLoss (do NOT apply softmax in forward).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        mode: str = "T_as_in_channels",   # "T_as_in_channels" or "1ch"
        pool: int = 4,                    # "pool size = 4" as described
        fc_hidden: int = 1024,            # dense layer size (your paragraph: 1024)
        dropout: float = 0.3,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if mode not in {"T_as_in_channels", "1ch"}:
            raise ValueError('mode must be "T_as_in_channels" or "1ch"')
        self.mode = mode
        self.pool = pool

        # We don't hardcode in_channels because it depends on T (or 1).
        # We'll build conv layers lazily on first forward if needed.
        self._built = False

        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # Placeholders; created once we know in_channels and flatten size.
        self.conv1 = None
        self.conv2 = None
        self.pool1 = nn.MaxPool2d(kernel_size=pool)
        self.pool2 = nn.MaxPool2d(kernel_size=pool)

        self.fc1 = None
        self.fc2 = None
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes

    def _make_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as (B,C,T,F), got {tuple(x.shape)}")
        if self.mode == "T_as_in_channels":
            # (B,C,T,F) -> (B,T,C,F)
            return x.permute(0, 2, 1, 3).contiguous()
        else:
            # (B,C,T,F) -> (B,1,C*T,F)
            b, c, t, f = x.shape
            return x.reshape(b, 1, c * t, f)

    def _lazy_build(self, x2d: torch.Tensor) -> None:
        # x2d is (B, in_ch, H, W)
        in_ch = int(x2d.shape[1])

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)

        # determine flatten dim by running a tiny forward on shape only
        with torch.no_grad():
            y = self.pool1(self.act(self.conv1(x2d)))
            y = self.pool2(self.act(self.conv2(y)))
            flat_dim = int(y.shape[1] * y.shape[2] * y.shape[3])

        self.fc1 = nn.Linear(flat_dim, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.num_classes)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d = self._make_input(x)

        if not self._built:
            self._lazy_build(x2d)

        x2d = self.pool1(self.act(self.conv1(x2d)))
        x2d = self.pool2(self.act(self.conv2(x2d)))

        x2d = torch.flatten(x2d, 1)
        x2d = self.dropout(torch.relu(self.fc1(x2d)))
        logits = self.fc2(x2d)
        return logits


if __name__ == "__main__":
    # Example: your memmap batches are (B,C,T,F)
    B, C, T, F = 8, 32, 8, 2048
    x = torch.randn(B, C, T, F)

    # Recommended: time windows as in_channels
    model = DasConv2dModel(num_classes=9, mode="T_as_in_channels", pool=4, fc_hidden=1024, dropout=0.3)
    logits = model(x)
    print("logits:", logits.shape)  # (8, 9)

    # Alternative: single-channel image with height=C*T
    model2 = DasConv2dModel(num_classes=9, mode="1ch", pool=4, fc_hidden=1024, dropout=0.3)
    logits2 = model2(x)
    print("logits2:", logits2.shape)  # (8, 9)