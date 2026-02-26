import os
import json
import torch
from torch.utils.data import Dataset


def build_class_mapping(root_dir: str):
    class_names = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_idx


def scan_split_files(root_dir: str, class_names, split: str):
    files = []
    prefix = f"{split}_"
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        for dirpath, _, filenames in os.walk(class_dir):
            for fn in filenames:
                if fn.endswith(".pt") and fn.startswith(prefix):
                    files.append(os.path.join(dirpath, fn))
    if not files:
        raise RuntimeError(f"No .pt files for split='{split}' under root='{root_dir}'")
    files.sort()
    return files


@torch.no_grad()
def precompute_index(root_dir: str, split: str, out_dir: str = None, map_location="cpu"):
    """
    Builds an index list [(path, i), ...] for a given split by loading each .pt once,
    and saves it to disk.

    Saves:
      - index_{split}.pt  (torch file containing list of tuples)
      - meta_{split}.json (basic metadata)
    """
    if out_dir is None:
        out_dir = root_dir
    os.makedirs(out_dir, exist_ok=True)

    class_names, class_to_idx = build_class_mapping(root_dir)
    files = scan_split_files(root_dir, class_names, split)

    index = []
    total_samples = 0

    for path in files:
        payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, dict) or "y" not in payload or "x" not in payload:
            raise RuntimeError(f"Bad payload in {path} (expected dict with keys x,y,...)")

        n = len(payload["y"])
        total_samples += n
        # extend index with (path, sample_idx)
        index.extend([(path, i) for i in range(n)])

    index_path = os.path.join(out_dir, f"index_{split}.pt")
    meta_path = os.path.join(out_dir, f"meta_{split}.json")

    torch.save(index, index_path)

    meta = {
        "root_dir": root_dir,
        "split": split,
        "num_files": len(files),
        "num_samples": total_samples,
        "class_names_sorted": class_names,
        "class_to_idx": class_to_idx,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return index_path, meta_path


class DASDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        transform=None,
        target_transform=None,
        index_dir: str = None,
        map_location="cpu",
    ):
        assert split in {"train", "val", "test"}, "split must be one of train, val or test"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.map_location = map_location

        self.class_names, self.class_to_idx = build_class_mapping(root_dir)

        if index_dir is None:
            index_dir = root_dir

        self.index_path = os.path.join(index_dir, f"index_{split}.pt")
        self.meta_path = os.path.join(index_dir, f"meta_{split}.json")

        # Load precomputed index (or build once if missing)
        if not os.path.exists(self.index_path):
            precompute_index(root_dir=root_dir, split=split, out_dir=index_dir, map_location=map_location)

        self.index = torch.load(self.index_path, map_location="cpu")

        # Per-process cache (each worker has its own instance)
        self._cache_path = None
        self._cache_payload = None

    def __len__(self):
        return len(self.index)

    def _load_cached(self, path):
        if self._cache_path != path:
            self._cache_path = path
            self._cache_payload = torch.load(path, map_location=self.map_location)
        return self._cache_payload

    def __getitem__(self, idx):
        path, i = self.index[idx]
        payload = self._load_cached(path)

        x = payload["x"][i]
        y = payload["y"][i]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


if __name__ == "__main__":
    root = "../../../data/processed"     # adjust
    out = os.path.join(root, "_index")  # keep indices separate (recommended)
    for sp in ["train", "val", "test"]:
        idx_path, meta_path = precompute_index(root_dir=root, split=sp, out_dir=out)
        print(sp, "->", idx_path, meta_path)


    train_ds = DASDataset(root, "train", index_dir=out)
    print("len(train):", len(train_ds))