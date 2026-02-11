import json
from pathlib import Path

def ensure_splits(ds, splits_dir: Path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    train_p = splits_dir / "train.json"
    val_p   = splits_dir / "val.json"
    test_p  = splits_dir / "test.json"

    if train_p.exists() and val_p.exists() and test_p.exists():
        return  # already there

    splits_dir.mkdir(parents=True, exist_ok=True)

    # --- blocked split per class ---
    import numpy as np
    num_classes = len(ds.classes)
    per_class = [[] for _ in range(num_classes)]
    for i in range(len(ds)):
        _, y = ds[i]
        per_class[int(y)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for c in range(num_classes):
        idxs = np.array(per_class[c], dtype=np.int64)

        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        tr = idxs[:n_train]
        va = idxs[n_train:n_train + n_val]
        te = idxs[n_train + n_val:]

        train_idx.extend(tr.tolist())
        val_idx.extend(va.tolist())
        test_idx.extend(te.tolist())

    rng = np.random.default_rng(seed)
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    train_p.write_text(json.dumps(train_idx), encoding="utf-8")
    val_p.write_text(json.dumps(val_idx), encoding="utf-8")
    test_p.write_text(json.dumps(test_idx), encoding="utf-8")
