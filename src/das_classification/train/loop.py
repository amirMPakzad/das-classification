# src/das_classification/train/loop.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .eval import evaluate
from .metrics import accuracy
from .history import JsonlHistoryWriter


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    save_dir: str = "runs"


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    cfg: TrainConfig,
    class_weights: Optional[torch.Tensor] = None
) -> None:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = JsonlHistoryWriter(str(save_dir / "history.jsonl"))

    best_val = float("inf")
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y = batch 
            x = x.to(device)
            y = y.to(device)

            x = x.unsqueeze(1)

            logits = model(x)
            loss = F.cross_entropy(
                logits, 
                y,
                weight=(class_weights.to(device) if class_weights is not None else None),
                label_smoothing=0.05,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()

            acc = accuracy(logits.detach(), y)

            running_loss += float(loss.item())
            running_acc += float(acc)
            n_batches += 1

            if (global_step % cfg.log_interval) == 0:
                print(
                    f"epoch {epoch:03d} step {global_step:06d} | "
                    f"train loss {loss.item():.4f} acc {acc:.3f}"
                )
                history.log({
                    "type": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float(loss.item()),
                    "train_acc": float(acc),
                })

            global_step += 1

        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        # val
        if val_loader is not None:
            val = evaluate(model, val_loader, device)
            val_loss = float(val.loss)
            val_acc = float(val.acc)
            print(f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            print(f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f}")

        # epoch-level history record
        history.log({
            "type": "epoch",
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # checkpoints
        ckpt_path = save_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

        if val_loader is not None and val_loss < best_val:
            best_val = val_loss
            best_path = save_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": best_val}, best_path)
            print(f"  saved best -> {best_path}")
