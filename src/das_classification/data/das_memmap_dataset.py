import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DASMemmapDataset(Dataset):
    """
    Reads X/y memmaps + index list produced by build_memmap_dataset().
    Zero-copy reads (memmap) + per-sample indexing.
    """

    def __init__(self, root: str, split: str):
        """
        root: cfg.dataset.root (folder containing meta.json, X_*.memmap, y_*.memmap, idx_*.npy)
        split: "train" | "val" | "test"
        """
        self.root = os.path.abspath(root)

        meta_path = os.path.join(self.root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {self.root}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        x_path = os.path.join(self.root, self.meta["memmap_X"])
        y_path = os.path.join(self.root, self.meta["memmap_y"])

        n_total = int(self.meta["n_total"])
        feat_len = int(self.meta["feature_len"])

        self.X = np.memmap(x_path, mode="r", dtype=np.float32, shape=(n_total, feat_len))
        self.y = np.memmap(y_path, mode="r", dtype=np.int32, shape=(n_total,))

        idx_path = os.path.join(self.root, f"idx_{split}.npy")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Split index file not found: {idx_path}")
        self.indices = np.load(idx_path).astype(np.int64)

        self.split = split
        self.num_classes = len(self.meta["labels"])
        self.classes = list(self.meta["labels"])  # ordered class names
        self.class_to_idx = dict(self.meta["label_to_int"])
        self.idx_to_class = {int(k): v for k, v in self.meta["int_to_label"].items()}

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        x = torch.from_numpy(np.array(self.X[j], copy=False))  # (feature_len,)
        y = torch.tensor(int(self.y[j]), dtype=torch.long)
        return x, y


if __name__ == "__main__":
    root = "out_dir"
    train_ds = DASMemmapDataset(root, "train")
    val_ds = DASMemmapDataset(root, "val")
    test_ds = DASMemmapDataset(root, "test")

    print(train_ds.num_classes)
    print(train_ds.classes)
    print(len(train_ds))