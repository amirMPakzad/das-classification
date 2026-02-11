# src/das_classification/train/imbalance.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass
class ImbalanceCfg:
    enabled: bool = False
    method: str = "weights"  # "weights" | "sampler" | "both"
    power: float = 1.0       # weight ~ 1/(count^power)
    min_count: int = 1
    max_weight: float = 100.0


def _get_label(sample) -> int:
    y = sample.y if hasattr(sample, "y") else sample[1]
    if isinstance(y, torch.Tensor):
        return int(y.item())
    return int(y)


@torch.no_grad()
def count_labels(ds: Dataset, indices: Optional[Sequence[int]] = None, num_classes: int = 0) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError("num_classes must be > 0")

    counts = torch.zeros((num_classes,), dtype=torch.long)
    rng = range(len(ds)) if indices is None else indices
    for i in rng:
        y = _get_label(ds[i])
        counts[y] += 1
    return counts


@torch.no_grad()
def make_class_weights(counts: torch.Tensor, cfg: ImbalanceCfg) -> torch.Tensor:
    counts_f = counts.to(torch.float32).clamp(min=float(cfg.min_count))
    w = 1.0 / torch.pow(counts_f, float(cfg.power))
    w = torch.clamp(w, max=float(cfg.max_weight))
    w = w / torch.clamp(w.mean(), min=1e-8)  # normalize mean=1
    return w


@torch.no_grad()
def make_weighted_sampler_from_dataset(
    ds: Dataset,
    class_weights: torch.Tensor,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    ds should be the dataset you pass to DataLoader (e.g., a Subset).
    Creates per-sample weights aligned with ds indexing [0..len(ds)-1].
    """
    n = len(ds)
    sw = torch.empty((n,), dtype=torch.float32)
    for i in range(n):
        y = _get_label(ds[i])
        sw[i] = float(class_weights[y])

    if num_samples is None:
        num_samples = n

    return WeightedRandomSampler(weights=sw, num_samples=int(num_samples), replacement=True)
