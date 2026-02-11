import os
import h5py
import torch
from torch.utils.data import Dataset

class DASDataset(Dataset):
    """
    Reads per-class HDF5 files. Each file contains dataset 'x' with shape (N, F).
    y is returned as class index (0..C-1) based on file order (sorted by class name).
    """
    def __init__(self, h5_dir: str, classes = None):
        self.h5_dir = os.path.abspath(h5_dir)

        if not classes:
            raise ValueError(
                "classes must be provided explicitly (e.g. from config) "
                "to keep label indices stable."
            )
        
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.paths = [os.path.join(self.h5_dir, f"{c}.h5") for c in self.classes]
        for p in self.paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing class file: {p}")

        # Build index mapping: global idx -> (file_i, local_idx)
        self.offsets = []
        total = 0
        self.lengths = []
        for p in self.paths:
            with h5py.File(p, "r") as hf:
                n = hf["x"].shape[0]
            self.offsets.append(total)
            self.lengths.append(n)
            total += n
        self.total = total

        self._handles = [None] * len(self.paths)

    def __len__(self):
        return self.total

    def _ensure_open(self, i):
        if self._handles[i] is None:
            self._handles[i] = h5py.File(self.paths[i], "r")

    def __getitem__(self, idx):
        # find which file
        # (linear scan is fine for few classes; if many, we can binary search)
        file_i = 0
        while file_i + 1 < len(self.offsets) and idx >= self.offsets[file_i + 1]:
            file_i += 1
        local_idx = idx - self.offsets[file_i]

        self._ensure_open(file_i)
        x = self._handles[file_i]["x"][local_idx]  # numpy -> h5py returns array-like
        y = file_i

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __del__(self):
        for h in self._handles:
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass
