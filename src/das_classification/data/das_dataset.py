import torch
import os
import json
from torch.utils.data import Dataset


def build_class_mapping(root_dir: str):
    class_names = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx


class DASDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None, target_transform=None):
        assert split in {"train", "val", "test"}, "split must be one of train, val or test"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.class_names, self.class_to_idx = build_class_mapping(root_dir)

        # هر آیتم دیتاست = (path, sample_idx)
        self.index = self._build_index()

        # cache برای جلوگیری از load تکراری فایل
        self._cache_path = None
        self._cache_payload = None

    def _scan_files(self):
        files = []
        prefix = f"{self.split}_"  # خیلی مهم: train_ / val_ / test_
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for dirpath, _, filenames in os.walk(class_dir):
                for fn in filenames:
                    if fn.endswith(".pt") and fn.startswith(prefix):
                        files.append(os.path.join(dirpath, fn))

        if not files:
            raise RuntimeError(f"Found no .pt files for split='{self.split}' under root='{self.root_dir}'")
        return files

    def _build_index(self):
        index = []
        for path in self._scan_files():
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict) or "y" not in payload or "x" not in payload:
                raise RuntimeError(f"Bad payload in {path} (expected dict with keys x,y,...)")

            n = len(payload["y"])
            for i in range(n):
                index.append((path, i))
        return index

    def __len__(self):
        return len(self.index)

    def _load_cached(self, path):
        if self._cache_path != path:
            self._cache_path = path
            self._cache_payload = torch.load(path, map_location="cpu")
        return self._cache_payload

    def __getitem__(self, idx):
        path, i = self.index[idx]
        try:
            payload = self._load_cached(path)
            x = payload["x"][i]
            y = payload["y"][i]
            return x, y
        except Exception as e:
            raise RuntimeError(f"Dataset crash at idx={idx} path={path} local_i={i}") from e

    def save_mapping(self, out_path: str):
        obj = {"class_names_sorted": self.class_names, "class_to_idx": self.class_to_idx}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    root = "processed_splits"
    train_ds = DASDataset(root, "train")
    val_ds = DASDataset(root, "val")
    test_ds = DASDataset(root, "test")
    train_ds.save_mapping(os.path.join(root, "class_mapping.json"))
    print(len(train_ds))
    print(len(val_ds))
    print(len(test_ds))