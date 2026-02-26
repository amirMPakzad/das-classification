import os
import json
from pathlib import Path

import numpy as np
import torch


def iter_pt_files(root: str, split: str):
    root = Path(root)
    prefix = f"{split}_"
    for p in root.rglob("*.pt"):
        name = p.name
        if name.startswith(prefix) and "_chunk" not in name:  # اگر chunked داری و میخوای از chunk استفاده کنی، این شرط رو بردار
            yield p

def class_name_from_path(root: Path, file_path: Path) -> str:
    # class name = اولین پوشه بعد از root
    rel = file_path.relative_to(root)
    return rel.parts[0]  # مثلا processed/regular/regularshit/train_x.pt -> "regular"


def build_split_memmap(root: str, out_dir: str, split: str):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(iter_pt_files(root, split)))
    if not files:
        raise RuntimeError(f"No .pt files found for split={split} under root={root}")

    # --- Pass 1: total N, F + build label<->class mapping ---
    total_n = 0
    F = None

    label_to_class = {}  # int -> str
    class_to_label = {}  # str -> int

    for p in files:
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            continue

        x = payload["x"]
        y = payload["y"]
        if not (torch.is_tensor(x) and torch.is_tensor(y)):
            continue
        if x.ndim != 2 or y.ndim != 1:
            continue

        n, f = x.shape
        if F is None:
            F = int(f)
        elif int(f) != F:
            raise RuntimeError(f"Feature dim mismatch in {p}: got {f}, expected {F}")

        if y.shape[0] != n:
            raise RuntimeError(f"Length mismatch in {p}: x={n}, y={y.shape[0]}")

        # mapping from folder name
        cls_name = class_name_from_path(root, p)

        # y باید در یک فایل، فقط یک کلاس داشته باشد
        uniq = torch.unique(y)
        if uniq.numel() != 1:
            raise RuntimeError(f"Mixed labels in one file {p}: unique={uniq.tolist()}")

        label = int(uniq.item())

        # consistency checks
        if label in label_to_class and label_to_class[label] != cls_name:
            raise RuntimeError(f"Label {label} mapped to both {label_to_class[label]} and {cls_name}")

        if cls_name in class_to_label and class_to_label[cls_name] != label:
            raise RuntimeError(f"Class {cls_name} mapped to both {class_to_label[cls_name]} and {label}")

        label_to_class[label] = cls_name
        class_to_label[cls_name] = label

        total_n += int(n)

    if F is None or total_n == 0:
        raise RuntimeError(f"No valid payloads found for split={split}")

    # make class_names_by_id (index == y)
    max_id = max(label_to_class.keys())
    class_names_by_id = [None] * (max_id + 1)
    for k, v in label_to_class.items():
        class_names_by_id[k] = v

    if any(v is None for v in class_names_by_id):
        missing = [i for i, v in enumerate(class_names_by_id) if v is None]
        raise RuntimeError(f"Missing class names for label ids: {missing}")

    # --- Output paths ---
    x_path = out_dir / f"{split}_x.mmap"
    y_path = out_dir / f"{split}_y.mmap"
    meta_path = out_dir / f"{split}_meta.json"

    # --- Create memmaps ---
    X = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(total_n, F))
    Y = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(total_n,))

    # --- Pass 2: write ---
    cursor = 0
    for p in files:
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            continue

        x = payload["x"]
        y = payload["y"]
        if not (torch.is_tensor(x) and torch.is_tensor(y) and x.ndim == 2 and y.ndim == 1):
            continue

        n, f = x.shape
        if int(f) != F or y.shape[0] != n:
            continue

        x_np = x.detach().cpu().to(torch.float32).numpy()
        y_np = y.detach().cpu().to(torch.int64).numpy()

        X[cursor:cursor+n, :] = x_np
        Y[cursor:cursor+n] = y_np
        cursor += int(n)

    X.flush()
    Y.flush()

    if cursor != total_n:
        raise RuntimeError(f"Write mismatch for split={split}: expected {total_n}, wrote {cursor}")

    meta = {
        "split": split,
        "total_n": int(total_n),
        "F": int(F),
        "x_path": str(x_path.resolve()),
        "y_path": str(y_path.resolve()),
        "class_names_by_id": class_names_by_id,                 # index == y
        "id_to_class": {str(k): v for k, v in label_to_class.items()},
        "class_to_id": {k: int(v) for k, v in class_to_label.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{split}] done: N={total_n}, F={F}, classes={len(class_names_by_id)}")


if __name__ == "__main__":
    ROOT = "../data/processed"     # مسیر فایل‌های pt
    OUT  = "../data/memmap_out"    # مسیر خروجی memmap
    for split in ["train", "val", "test"]:
        build_split_memmap(ROOT, OUT, split)