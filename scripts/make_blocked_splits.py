import json
from pathlib import Path
import numpy as np

from das_classification.data.das_dataset import DASDataset


def main(
    out_dir: str = "../data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    h5_data_dir: str = "../data/processed",
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9

    ds = DASDataset(h5_data_dir)
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
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        tr = idxs[:n_train]
        va = idxs[n_train:n_train + n_val]
        te = idxs[n_train + n_val:]

        train_idx.extend(tr.tolist())
        val_idx.extend(va.tolist())
        test_idx.extend(te.tolist())

    rng = np.random.default_rng(42)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, arr):
        with open(out / f"{name}.json", "w") as f:
            json.dump([int(x) for x in arr], f)

    dump("train", train_idx)
    dump("val", val_idx)
    dump("test", test_idx)

    print("classes:", ds.classes)
    print("sizes:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})

if __name__ == "__main__":
    main()
