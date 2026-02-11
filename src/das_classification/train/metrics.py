# src/das_classification/train/metrics.py

from __future__ import annotations 
import torch 


@torch.no_grad()
def accuracy(logits: torch.Tensor, y:torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def confusion_matrix(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Returns (k, k) where rows=True class, cols = pred class
    """
    pred = torch.argmax(logits, dim=-1,).view(-1)
    y = y.view(-1)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device="cpu")
    for t, p in zip(y.cpu(), pred.cpu()):
        cm[int(t), int(p)] += 1
    return cm 


@torch.no_grad()
def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.to(torch.float32)
    denom = cm.sum(dim=1)
    correct = cm.diag()
    acc = correct / torch.clamp(denom, min=1.0)
    acc = torch.where(denom > 0, acc, torch.tensor(float('nan')))
    return acc


@torch.no_grad()
def percision_recall_f1_from_cm(
    cm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
    """
    cm: (K,K) rows=true, cols=pred
    Returns: precision(K,), recall(K,), f1(K,)
    """
    cm = cm.to(torch.float32)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp 

    precision = tp / torch.clamp(tp+fp, min=1.0)
    recall = tp / torch.clamp(tp+fp, min=1.0)
    f1 = (2 * precision * recall) / torch.clamp(precision + recall, min = 1e-8)

    has_true = (cm.sum(dim=1) > 0)
    precision = torch.where(has_true, precision, torch.tensor(float("nan")))
    recall = torch.where(has_true, recall, torch.tensor(float("nan")))
    f1 = torch.where(has_true, f1, torch.tensor(float("nan")))
    return precision, recall, f1


@torch.no_grad()
def macro_f1(f1_per_class: torch.Tensor) -> float:
    """
    average over classes, ignoring nan
    """
    mask = torch.isfinite(f1_per_class)
    if mask.sum() == 0:
        return float("nan")
    return float(f1_per_class[mask].mean().item())

