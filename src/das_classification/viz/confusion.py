# src/das_classification/viz/confusion.py

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def save_confusion_matrix_png(
    cm,
    labels: List[str],
    out_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """
    cm: (K,K) tensor or numpy
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cm = cm.detach().cpu().numpy() if hasattr(cm, "detach") else np.asarray(cm)
    K = cm.shape[0]

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_show = cm / row_sum
    else:
        cm_show = cm

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_show, aspect="auto", origin="upper")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(K), labels, rotation=45, ha="right")
    plt.yticks(range(K), labels)

    for i in range(K):
        for j in range(K):
            val = cm_show[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
