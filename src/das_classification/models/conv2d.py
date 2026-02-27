import torch
import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int =3,s=1,p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class DasConv2dModel(nn.Module):
    """
       Expected input:
         - default layout: (B, C, T, F)
         - optional layout: (B, C, F, T) via input_layout="CFT"
    """

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            base_channels: int = 64,
            dropout: float = 0.3,
            input_layout: str = "CTF",  # "CTF" -> (C,T,F), "CFT" -> (C,F,T)
    ):
        super().__init__()
        if input_layout not in {"CTF", "CFT"}:
            raise ValueError("input_layout must be 'CTF' or 'CFT'")

        self.input_layout = input_layout
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            ConvBNAct(in_channels, c1, k=3, s=1, p=1),
            ConvBNAct(c1, c1, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=(1, 2)),

            ConvBNAct(c1, c2, k=3, s=1, p=1),
            ConvBNAct(c2, c2, k=3, s=1, p=1),
            nn.MaxPool2d(kernel_size=(2, 2)),

            ConvBNAct(c2, c3, k=3, s=1, p=1),
            ConvBNAct(c3, c3, k=3, s=1, p=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=c3, out_features=num_classes),
        )

    def _normalize_layout(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,C,T,F or B,C,F,T), got shape {tuple(x.shape)}")
        if self.input_layout == "CFT":
            x = x.transpose(-1, -2)  # (B,C,F,T) -> (B,C,T,F)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_layout(x)
        x = self.features(x)
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    # quick sanity check
    model = DasConv2dModel(in_channels=32, num_classes=9, input_layout="CTF")
    dummy = torch.randn(8, 32, 4, 2048)
    logits = model(dummy)
    print("logits shape:", logits.shape)  # (8, 9)
