import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DASMemmapDataset(Dataset):
    def __init__(self, memmap_dir: str, split: str, transform=None, target_transform=None):
        assert split in {"train", "val", "test"}
        self.memmap_dir = Path(memmap_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        meta_path = self.memmap_dir / f"{split}_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.N = int(self.meta["total_n"])
        self.F = int(self.meta["F"])
        self.x_path = Path(self.meta["x_path"])
        self.y_path = Path(self.meta["y_path"])

        # mapping
        self.class_names_by_id = self.meta["class_names_by_id"]
        self.class_to_id = self.meta["class_to_id"]
        self.id_to_class = {int(k): v for k, v in self.meta["id_to_class"].items()}

        self._X = None
        self._Y = None

    def __len__(self):
        return self.N

    def _ensure_open(self):
        if self._X is None or self._Y is None:
            self._X = np.memmap(self.x_path, mode="r", dtype=np.float32, shape=(self.N, self.F))
            self._Y = np.memmap(self.y_path, mode="r", dtype=np.int64, shape=(self.N,))

    def class_name(self, y: int) -> str:
        return self.class_names_by_id[int(y)]  # y=0 -> class_names_by_id[0]

    def __getitem__(self, idx):
        self._ensure_open()
        x = torch.from_numpy(np.array(self._X[idx], copy=False))
        y = int(self._Y[idx])

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, torch.tensor(y, dtype=torch.long)



