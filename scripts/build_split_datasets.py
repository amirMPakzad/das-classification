from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Tuple

import h5py
import numpy as np


# -----------------------------
# Chunk 1: Optional FFT feature function
# -----------------------------
# We try to reuse your existing fft() from data_loader.
# If not available, we fall back to a simple magnitude spectrum.



def fft_features(w: np.ndarray) -> np.ndarray:
    """
    Fallback FFT features:
    - input w: shape (fsize,) float32
    - output: shape (F,) float32
    """
    x = np.fft.rfft(w)
    mag = np.abs(x).astype(np.float32)
    # log scaling similar to what you used for noise check
    mag = np.log10(mag + 1.0)
    return mag


def compute_features(window_1d: np.ndarray, sample_len: int | None) -> np.ndarray:
    feat = fft_features(window_1d)

    if sample_len is not None:
        feat = feat[:sample_len]
    return feat


def spec_cmp_1d(x_spec: np.ndarray) -> bool:
    split = int(len(x_spec) * 0.1)
    return (np.mean(x_spec[:split]) - np.mean(x_spec[split:])) > 0.05


def window_noise_ok(window_1d: np.ndarray, keep_if_regular: bool) -> bool:
    if keep_if_regular:
        return True
    x = np.fft.rfft(window_1d)[1:]
    x = np.log10(np.abs(x) + 1.0)
    return spec_cmp_1d(x)


def load_split_samples(split_dir: Path, split_name: str) -> np.ndarray:
    p = split_dir / f"{split_name}_samples.npy"
    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected {p} shape (N,3), got {arr.shape}")
    return arr.astype(np.int64)


def iter_split_windows(
    raw_h5_path: Path,
    split_samples: np.ndarray,
    fsize: int,
    shift: int,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    with h5py.File(raw_h5_path, "r") as f:
        data = f["Acquisition"]["Raw[0]"]["RawData"]  # (time, channels)

        T = int(data.shape[0])
        C = int(data.shape[1])

        for row in split_samples:
            pos = int(row[0])
            ch = int(row[1])

            if ch < 0 or ch >= C:
                continue

            start = pos * shift
            end = start + fsize
            if end > T or start < 0:
                continue

            w = data[start:end, ch].astype(np.float32)
            yield pos, ch, w

def build_split_h5(
    raw_h5_path: Path,
    split_dir: Path,
    out_path: Path,
    split_name: str,
    class_id: int,
    fsize: int,
    shift: int,
    sample_len: int | None,
    drop_noise: bool,
    keep_if_regular: bool,
    max_samples: int | None,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    split_samples = load_split_samples(split_dir, split_name)


    if max_samples is not None and split_samples.shape[0] > max_samples:
        idx = rng.choice(split_samples.shape[0], size=max_samples, replace=False)
        split_samples = split_samples[idx]


    dummy = np.zeros((fsize,), dtype=np.float32)
    dummy_feat = compute_features(dummy, sample_len)
    feat_dim = int(dummy_feat.shape[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)


    valid_meta = []  # list of (pos, ch)
    for pos, ch, w in iter_split_windows(raw_h5_path, split_samples, fsize, shift):
        if drop_noise:
            if not window_noise_ok(w, keep_if_regular=keep_if_regular):
                continue
        valid_meta.append((pos, ch))

    n = len(valid_meta)
    if n == 0:
        raise RuntimeError(f"No valid samples for split={split_name} after filtering.")

    # Write output HDF5
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as hf:
        # Datasets
        x_ds = hf.create_dataset(
            "x",
            shape=(n, feat_dim),
            dtype="float32",
            chunks=(min(1024, n), feat_dim),
            compression="gzip",
            compression_opts=4,
        )
        y_ds = hf.create_dataset("y", shape=(n,), dtype="int64")
        pos_ds = hf.create_dataset("pos", shape=(n,), dtype="int32")
        ch_ds = hf.create_dataset("ch", shape=(n,), dtype="int32")
        src_ds = hf.create_dataset("src", shape=(n,), dtype=str_dt)

        # Save config metadata for reproducibility
        hf.attrs["raw_h5"] = str(raw_h5_path.name)
        hf.attrs["split_name"] = split_name
        hf.attrs["class_id"] = int(class_id)
        hf.attrs["fsize"] = int(fsize)
        hf.attrs["shift"] = int(shift)
        hf.attrs["sample_len"] = int(sample_len) if sample_len is not None else -1
        hf.attrs["drop_noise"] = bool(drop_noise)

        # Pass B: fill
        # Re-open raw H5 so we can iterate again and write features
        with h5py.File(raw_h5_path, "r") as f:
            data = f["Acquisition"]["Raw[0]"]["RawData"]
            T = int(data.shape[0])

            src_name = raw_h5_path.name
            out_i = 0
            for pos, ch in valid_meta:
                start = pos * shift
                end = start + fsize
                if end > T:
                    continue

                w = data[start:end, ch].astype(np.float32)

                # (safety) apply same filter again (should match pass A)
                if drop_noise and not window_noise_ok(w, keep_if_regular=keep_if_regular):
                    continue

                feat = compute_features(w, sample_len)  # (feat_dim,)
                if feat.shape[0] != feat_dim:
                    # If your fft() ever returns variable size, hard-fail early
                    raise RuntimeError(f"Feature dim mismatch: got {feat.shape[0]} expected {feat_dim}")

                x_ds[out_i] = feat
                y_ds[out_i] = class_id
                pos_ds[out_i] = pos
                ch_ds[out_i] = ch
                src_ds[out_i] = src_name
                out_i += 1

        # If something got skipped on pass B (shouldn't often), shrink is hard in HDF5.
        # We enforce consistency by error if mismatch is big.
        if out_i != n:
            raise RuntimeError(f"Wrote {out_i} samples but expected {n}. Filtering mismatch?")

    print(f"[OK] wrote split={split_name} n={n} -> {out_path}")


# -----------------------------
# Chunk 6: Optional check against split_summary.json
# -----------------------------
def warn_if_mismatch(split_dir: Path, fsize: int, shift: int) -> None:
    summ_path = split_dir / "split_summary.json"
    if not summ_path.exists():
        return
    try:
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        cfg = summ.get("config", {})
        sf = int(cfg.get("fsize", -1))
        sh = int(cfg.get("shift", -1))
        if sf != -1 and sh != -1 and (sf != fsize or sh != shift):
            print(
                f"[WARN] split_summary.json says fsize={sf}, shift={sh} "
                f"but you asked for fsize={fsize}, shift={shift}. "
                f"These MUST match to preserve leakage guarantees."
            )
    except Exception:
        pass


# -----------------------------
# Chunk 7: CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train/val/test HDF5 datasets from leakage-safe split npy + raw H5"
    )
    p.add_argument("--raw-h5", type=Path, required=True, help="Path to reconstruction .h5")
    p.add_argument("--split-dir", type=Path, required=True, help="Directory containing train/val/test_samples.npy")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for train.h5/val.h5/test.h5")

    p.add_argument("--fsize", type=int, default=4096, help="Window size (raw samples). MUST match split config.")
    p.add_argument("--shift", type=int, default=2048, help="Hop size (raw samples). MUST match split config.")
    p.add_argument("--sample-len", type=int, default=2048, help="Feature length cap (after FFT). Use 0 to disable.")
    p.add_argument("--drop-noise", action="store_true", help="Apply spectral noise filtering")
    p.add_argument(
        "--keep-if-regular",
        action="store_true",
        help="If set, noise filter always keeps samples (useful for regular baseline).",
    )

    p.add_argument("--class-id", type=int, required=True, help="Class label id for this recording (event classification)")
    p.add_argument("--max-train", type=int, default=0, help="Cap train samples (0 = no cap)")
    p.add_argument("--max-val", type=int, default=0, help="Cap val samples (0 = no cap)")
    p.add_argument("--max-test", type=int, default=0, help="Cap test samples (0 = no cap)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sample_len = None if args.sample_len == 0 else int(args.sample_len)
    max_train = None if args.max_train == 0 else int(args.max_train)
    max_val = None if args.max_val == 0 else int(args.max_val)
    max_test = None if args.max_test == 0 else int(args.max_test)

    warn_if_mismatch(args.split_dir, fsize=args.fsize, shift=args.shift)

    # Build each split file
    build_split_h5(
        raw_h5_path=args.raw_h5,
        split_dir=args.split_dir,
        out_path=args.out_dir / "train.h5",
        split_name="train",
        class_id=args.class_id,
        fsize=args.fsize,
        shift=args.shift,
        sample_len=sample_len,
        drop_noise=args.drop_noise,
        keep_if_regular=args.keep_if_regular,
        max_samples=max_train,
        seed=args.seed,
    )

    build_split_h5(
        raw_h5_path=args.raw_h5,
        split_dir=args.split_dir,
        out_path=args.out_dir / "val.h5",
        split_name="val",
        class_id=args.class_id,
        fsize=args.fsize,
        shift=args.shift,
        sample_len=sample_len,
        drop_noise=args.drop_noise,
        keep_if_regular=args.keep_if_regular,
        max_samples=max_val,
        seed=args.seed,
    )

    build_split_h5(
        raw_h5_path=args.raw_h5,
        split_dir=args.split_dir,
        out_path=args.out_dir / "test.h5",
        split_name="test",
        class_id=args.class_id,
        fsize=args.fsize,
        shift=args.shift,
        sample_len=sample_len,
        drop_noise=args.drop_noise,
        keep_if_regular=args.keep_if_regular,
        max_samples=max_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()