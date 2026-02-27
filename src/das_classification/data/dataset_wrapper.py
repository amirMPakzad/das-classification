import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import DASDataset


def _to_set(values: Optional[Iterable[str]]) -> Set[str]:
    if values is None:
        return set()
    return {str(v) for v in values}


class DASDatasetWrapper(Dataset):
    """
    Wrapper around DASDataset to drop classes and relabel targets.

    Example:
      ds = DASRelabeledDataset(
          root="fct",
          split="train",
          drop_classes=["regular", "construction"],
      )

    New label IDs are contiguous [0..K-1] following original class order
    with dropped classes removed.
    """

    def __init__(
        self,
        root: str,
        split: str,
        drop_classes: Optional[Sequence[str]] = None,
    ):
        self.base = DASDataset(root=root, split=split)
        self.split = split

        self.meta = self.base.meta

        dropped = _to_set(drop_classes)
        original_classes = list(self.base.classes)

        self.classes: List[str] = [c for c in original_classes if c not in dropped]
        if not self.classes:
            raise ValueError("All classes were dropped; nothing remains.")

        old_name_to_id = {name: int(self.base.class_to_idx[name]) for name in original_classes}
        keep_old_ids = [old_name_to_id[name] for name in self.classes]
        self.old_to_new: Dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(keep_old_ids)}

        self.class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: name for i, name in enumerate(self.classes)}

        # Keep only indices whose original label belongs to kept classes
        y_all = self.base.y  # memmap (N,)
        old_split_indices = self.base.indices.astype(np.int64)
        old_labels = np.array(y_all[old_split_indices], dtype=np.int64)
        keep_mask = np.isin(old_labels, np.array(keep_old_ids, dtype=np.int64))

        self.indices = old_split_indices[keep_mask]
        self._old_labels_for_kept = old_labels[keep_mask]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        x = torch.from_numpy(np.array(self.base.X[j], copy=False))  # (C, T, F)

        old_y = int(self._old_labels_for_kept[i])
        new_y = int(self.old_to_new[old_y])
        y = torch.tensor(new_y, dtype=torch.long)
        return x, y






if __name__ == "__main__":
    ROOT = "data/main"
    DROP = ["regular", "construction"]

    train_ds = DASDatasetWrapper(root=ROOT, split="train", drop_classes=DROP)
    val_ds = DASDatasetWrapper(root=ROOT, split="val", drop_classes=DROP)
    test_ds = DASDatasetWrapper(root=ROOT, split="test", drop_classes=DROP)
    print("new classes:", train_ds.classes)
    print("class_to_idx:", train_ds.class_to_idx)
    print("sizes:", len(train_ds), len(val_ds), len(test_ds))

