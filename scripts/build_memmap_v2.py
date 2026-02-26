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
        # اگر chunked داری و میخوای از chunk استفاده کنی، این شرط رو بردار
        if name.startswith(prefix) and "_chunk" not in name:
            yield p


def class_name_from_path(root: Path, file_path: Path) -> str:
    # class name = اولین پوشه بعد از root
    rel = file_path.relative_to(root)
    return rel.parts[0]  # مثلا processed/regular/... -> "regular"


def build_split_memmap(root: str, out_dir: str, split: str):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(iter_pt_files(root, split)))
    if not files:
        raise RuntimeError(f"No .pt files found for split={split} under root={root}")

    # ---------- Pass 1: total N, F + collect (old_label <-> class) ----------
    total_n = 0
    F = None

    label_to_class = {}  # old int -> str
    class_to_label = {}  # str -> old int

    valid_files = []

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

        cls_name = class_name_from_path(root, p)

        uniq = torch.unique(y)
        if uniq.numel() != 1:
            raise RuntimeError(f"Mixed labels in one file {p}: unique={uniq.tolist()}")

        old_label = int(uniq.item())

        # consistency checks
        if old_label in label_to_class and label_to_class[old_label] != cls_name:
            raise RuntimeError(
                f"Label {old_label} mapped to both {label_to_class[old_label]} and {cls_name}"
            )
        if cls_name in class_to_label and class_to_label[cls_name] != old_label:
            raise RuntimeError(
                f"Class {cls_name} mapped to both {class_to_label[cls_name]} and {old_label}"
            )

        label_to_class[old_label] = cls_name
        class_to_label[cls_name] = old_label

        total_n += int(n)
        valid_files.append(p)

    if F is None or total_n == 0 or not label_to_class:
        raise RuntimeError(f"No valid payloads found for split={split}")

    # ---------- Build remap: old labels -> new contiguous labels ----------
    present_old_labels = sorted(label_to_class.keys())  # e.g. [0,1,3,5]
    old_to_new = {old: new for new, old in enumerate(present_old_labels)}  # -> [0..K-1]
    new_to_old = {new: old for old, new in old_to_new.items()}

    # class_names_by_new_id (index == new_y)
    class_names_by_id = [label_to_class[new_to_old[i]] for i in range(len(present_old_labels))]

    # ---------- Output paths ----------
    x_path = out_dir / f"{split}_x.mmap"
    y_path = out_dir / f"{split}_y.mmap"
    meta_path = out_dir / f"{split}_meta.json"

    # ---------- Create memmaps ----------
    X = np.memmap(x_path, mode="w+", dtype=np.float32, shape=(total_n, F))
    Y = np.memmap(y_path, mode="w+", dtype=np.int64, shape=(total_n,))

    # ---------- Pass 2: write with remapped labels ----------
    cursor = 0
    for p in valid_files:
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

        # convert
        x_np = x.detach().cpu().to(torch.float32).numpy()

        # remap y: old -> new
        old_label = int(torch.unique(y).item())
        new_label = old_to_new.get(old_label, None)
        if new_label is None:
            raise RuntimeError(f"Found label {old_label} not present in old_to_new map. File: {p}")

        # چون کل فایل یک لیبل است، سریع‌ترین حالت:
        y_np = np.full((n,), new_label, dtype=np.int64)

        X[cursor:cursor + n, :] = x_np
        Y[cursor:cursor + n] = y_np
        cursor += int(n)

    X.flush()
    Y.flush()

    if cursor != total_n:
        raise RuntimeError(f"Write mismatch for split={split}: expected {total_n}, wrote {cursor}")

    # also build class_to_new_id (folder name -> new id)
    class_to_new_id = {label_to_class[old]: int(new) for old, new in old_to_new.items()}

    meta = {
        "split": split,
        "total_n": int(total_n),
        "F": int(F),
        "x_path": str(x_path.resolve()),
        "y_path": str(y_path.resolve()),

        # NEW contiguous label space:
        "num_classes": int(len(class_names_by_id)),
        "class_names_by_id": class_names_by_id,  # index == new y

        # mapping info:
        "old_label_ids_present": [int(x) for x in present_old_labels],
        "old_to_new": {str(k): int(v) for k, v in old_to_new.items()},
        "new_to_old": {str(k): int(v) for k, v in new_to_old.items()},

        # convenience:
        "class_to_id": class_to_new_id,  # class name -> NEW id
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[{split}] done: N={total_n}, F={F}, "
        f"old_labels={present_old_labels} -> new_labels=0..{len(class_names_by_id)-1}, "
        f"classes={len(class_names_by_id)}"
    )


if __name__ == "__main__":
    ROOT = "../data/processed"     # مسیر فایل‌های pt
    OUT  = "../data/memmap_out"    # مسیر خروجی memmap
    for split in ["train", "val", "test"]:
        build_split_memmap(ROOT, OUT, split)