# src/das_classification/train/eval.py

from __future__ import annotations
from dataclasses import dataclass


import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader 

from .metrics import accuracy 

@dataclass
class EvalResult:
    loss: float
    acc: float 


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0 
    n = 0 

    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        #x = x.unsqueeze(1)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += float(loss.item())
        total_acc += float(accuracy(logits, y))
        n += 1
    
    if n == 0:
        return EvalResult(loss=float("nan"), acc=float("nan"))
    
    return EvalResult(loss=total_loss / n, acc = total_acc / n)
