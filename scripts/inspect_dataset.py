import h5py
import numpy as np
import sys
from pathlib import Path


def inspect_h5(path: Path):
    print(f"\n========== Inspecting {path.name} ==========")

    with h5py.File(path, "r") as f:
        # Required datasets
        if "x" not in f or "y" not in f:
            print("Missing required datasets 'x' or 'y'")
            return

        x = f["x"]
        y = f["y"]

        print("Shape of x:", x.shape)
        print("Shape of y:", y.shape)
        print("Feature dimension:", x.shape[1])
        print("Dtype x:", x.dtype)
        print("Dtype y:", y.dtype)

        # Number of samples
        n = x.shape[0]
        print("Number of samples:", n)

        # Class distribution
        labels = y[:]
        unique, counts = np.unique(labels, return_counts=True)
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")

        # Optional metadata
        print("\nFile attributes:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_dataset.py train.h5 val.h5 test.h5")
        sys.exit(0)

    for arg in sys.argv[1:]:
        inspect_h5(Path(arg))