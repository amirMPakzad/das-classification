# src/das_classification/train/test.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from das_classification.data.constants import LABELS
from .metrics import confusion_matrix, per_class_accuracy


@dataclass 
class TestResult:
    loss: float
    acc : float 
    cm : torch.Tensor 


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    return epoch


@torch.no_grad()
def test_loop(
    model: nn.Module, 
    loader: DataLoader,
    device: str,
    num_classes: int,
)-> TestResult:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch in loader:
        x, y = batch 
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += float(loss.item())
        pred = torch.argmax(logits, dim=-1)
        total_acc += float((pred == y).float().mean().item())

        cm_total += confusion_matrix(logits, y, num_classes=num_classes)
        n+=1 

    if n == 0:
        return TestResult(loss=float("nan"), acc=float("nan"), cm=cm_total)

    return TestResult(loss=total_loss / n, acc=total_acc / n, cm=cm_total)


def format_cm(cm: torch.Tensor, labels: List[str]) -> str:
    """
    Pretty print confusion matrix as text (small K). Rows=true, cols=pred.
    """
    K = cm.shape[0]
    colw = max(6, max(len(s) for s in labels) + 2)
    header = " " * colw + "".join(f"{labels[j]:>{colw}}" for j in range(K))
    lines = [header]
    for i in range(K):
        row = "".join(f"{int(cm[i, j]):>{colw}d}" for j in range(K))
        lines.append(f"{labels[i]:>{colw}}{row}")
    return "\n".join(lines)


def save_report_txt(
    out_path: str,
    loss: float,
    acc: float, 
    cm: torch.Tensor,
    labels: List[str]
) -> None:
    out = []
    out.append(f"test_loss: {loss:.6f}")
    out.append(f"test_acc: {acc:.6f}")

    per_class_acc = per_class_accuracy(cm)
    out.append("\n per class accuracy")
    for name, v in zip(labels, per_class_acc.tolist()):
        out.append(f"  {name:>14s}: {v:.4f}" if v == v else f"  {name:>14s}: nan")

    out.append("\nconfusion matrix (rows=true, cols=pred):")
    out.append(format_cm(cm, labels))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")
