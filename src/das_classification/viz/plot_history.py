# src/das_classification/viz/plot_history.py

from __future__ import annotations

import json 
from pathlib import Path 
from typing import Dict, List, Tuple, Optional


import matplotlib.pyplot as plt 


def load_epoch_history(history_path: str) -> Dict[str, List[float]]:
    p = Path(history_path)

    if not p.exists():
        raise FileNotFoundError("file not found")
    
    epochs = []
    train_loss, val_loss = [], []
    train_acc, val_acc = [], [] 

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "epoch":
                continue
            epochs.append(int(rec["epoch"]))
            train_loss.append(float(rec.get("train_loss", float("nan"))))
            val_loss.append(float(rec.get("val_loss", float("nan"))))
            train_acc.append(float(rec.get("train_acc", float("nan"))))
            val_acc.append(float(rec.get("val_acc", float("nan"))))

    return {
        "epochs" : epochs,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "train_acc": train_acc
    }


def plot_history(
    history_path: str,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    series = load_epoch_history(history_path)
    epochs = series["epoch"]

    # Loss plot
    plt.figure()
    plt.plot(epochs, series["train_loss"], label="train_loss")
    plt.plot(epochs, series["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.legend()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(Path(save_dir) / "loss.png"), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    # Acc plot
    plt.figure()
    plt.plot(epochs, series["train_acc"], label="train_acc")
    plt.plot(epochs, series["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy")
    plt.legend()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(str(Path(save_dir) / "acc.png"), dpi=150)

    if show:
        plt.show()
    else:
        plt.close()