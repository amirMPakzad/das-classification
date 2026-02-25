# src/das-classification/data/splits.py

import json
from pathlib import Path
import json
from pathlib import Path
import numpy as np
import h5py


def ensure_splits(ds, splits_dir: Path,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    seed=42, min_channels_per_split=1):
    train_p = splits_dir / "train.json"
    val_p   = splits_dir / "val.json"
    test_p  = splits_dir / "test.json"

    if train_p.exists() and val_p.exists() and test_p.exists():
        return

    splits_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx = [], [], []
    rng_global = np.random.default_rng(seed)

   
    for file_i, path in enumerate(ds.paths):
        with h5py.File(path, "r") as hf:
            ch = np.array(hf["ch"][:], dtype=np.int32)  # (N,)
        uniq = np.unique(ch)

        rng = np.random.default_rng(seed + 1000 * file_i)
        rng.shuffle(uniq)

        n_ch = len(uniq)
        n_tr = max(min_channels_per_split, int(n_ch * train_ratio))
        n_va = max(min_channels_per_split, int(n_ch * val_ratio))
    
        if n_tr + n_va >= n_ch:
            n_tr = max(1, n_ch - 2)
            n_va = 1

        ch_tr = set(uniq[:n_tr].tolist())
        ch_va = set(uniq[n_tr:n_tr + n_va].tolist())
        ch_te = set(uniq[n_tr + n_va:].tolist())

        offset = ds.offsets[file_i]
        N = ds.lengths[file_i]

        # local indices 0..N-1
        local = np.arange(N, dtype=np.int64)
        global_idx = offset + local

        mask_tr = np.isin(ch, list(ch_tr))
        mask_va = np.isin(ch, list(ch_va))
        mask_te = np.isin(ch, list(ch_te))

        train_idx.extend(global_idx[mask_tr].tolist())
        val_idx.extend(global_idx[mask_va].tolist())
        test_idx.extend(global_idx[mask_te].tolist())

    # shuffle 
    rng_global.shuffle(train_idx)
    rng_global.shuffle(val_idx)
    rng_global.shuffle(test_idx)

    train_p.write_text(json.dumps(train_idx), encoding="utf-8")
    val_p.write_text(json.dumps(val_idx), encoding="utf-8")
    test_p.write_text(json.dumps(test_idx), encoding="utf-8")





def ensure_splits_old(ds, splits_dir: Path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
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
