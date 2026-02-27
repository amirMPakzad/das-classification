import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class DASDataset(Dataset):
    """
    Reads memmap produced by build_memmap_cnn3d():
      X: (N, C, T, F)
      y: (N,)
      idx_{split}.npy: indices
    """

    def __init__(self, root: str, split: str):
        self.root = os.path.abspath(root)

        meta_path = os.path.join(self.root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {self.root}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.classes = list(self.meta["labels"])
        self.class_to_idx = dict(self.meta["label_to_int"])
        self.idx_to_class = {int(k): v for k, v in self.meta["int_to_label"].items()}

        x_path = os.path.join(self.root, self.meta["memmap_X"])
        y_path = os.path.join(self.root, self.meta["memmap_y"])

        N, C, T, F = map(int, self.meta["shape"])

        self.X = np.memmap(x_path, mode="r", dtype=np.float32, shape=(N, C, T, F))
        self.y = np.memmap(y_path, mode="r", dtype=np.int32, shape=(N,))

        idx_path = os.path.join(self.root, f"idx_{split}.npy")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Split file not found: {idx_path}")

        self.indices = np.load(idx_path).astype(np.int64)
        self.split = split

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        # memmap slice is already contiguous enough; np.array(copy=False) avoids extra copy
        x = torch.from_numpy(np.array(self.X[j], copy=False))  # (C, T, F)
        y = torch.tensor(int(self.y[j]), dtype=torch.long)
        return x, y


if __name__ == "__main__":
    train_ds = DASDataset("fct", "train")
    val_ds = DASDataset("fct", "val")
    test_ds = DASDataset("fct", "test")

    print(len(train_ds))
    print(len(val_ds))
    print(len(test_ds))