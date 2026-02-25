from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ChronoSplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    fsize: int = 8192
    shift: int = 2048

    # Optional: explicitly set purge in "position steps" (window-index steps).
    # If None, we infer it from fsize/shift so that windows cannot overlap in raw samples.
    purge_positions: Optional[int] = None

    # Optional: treat time as grouped blocks of fixed size (e.g. 16 windows per block)
    # and split on block boundaries (still chronological).
    # If block_size=1, it's a simple chronological split on individual positions.
    block_size: int = 16

    # If True, we only consider time positions that have ANY event (bmp[t,:].any()).
    # This is often what you want in "per-class positives-only" pipelines.
    positives_only: bool = True

    def validate(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"train+val+test must sum to 1.0, got {s}")
        if self.fsize <= 0 or self.shift <= 0:
            raise ValueError("fsize and shift must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")


def infer_purge_positions(fsize: int, shift: int) -> int:
    """
    Two windows at indices i and j overlap in raw samples if:
        |i - j| * shift < fsize
    To guarantee non-overlap across splits, we 'purge' +/-K around boundaries, where:
        K = ceil(fsize/shift) - 1

    Example: fsize=8192, shift=2048 => fsize/shift=4 => K=3
    Then any index difference <=3 would overlap in raw samples.
    """
    return max(0, math.ceil(fsize / shift) - 1)


def _positions_to_blocks(pos: np.ndarray, block_size: int) -> np.ndarray:
    """
    Map each position index to a block id: block_id = pos // block_size
    """
    return (pos // block_size).astype(np.int64)


def chronological_block_split_positions(
    bitmap: np.ndarray,
    cfg: ChronoSplitConfig,
) -> Dict[str, np.ndarray]:
    """
    Returns:
        {
          "train_pos": array([...]),
          "val_pos":   array([...]),
          "test_pos":  array([...]),
        }
    All arrays are sorted unique window indices (time positions).
    """
    cfg.validate()

    if bitmap.ndim != 2:
        raise ValueError(f"Expected bitmap [time_pos, channel], got shape {bitmap.shape}")

    n_pos = bitmap.shape[0]
    all_positions = np.arange(n_pos, dtype=np.int64)

    # Optionally restrict to positions that actually contain positives for this class
    if cfg.positives_only:
        # Keep only time indices where at least one channel is True
        mask = bitmap.any(axis=1)
        all_positions = all_positions[mask]

    if all_positions.size == 0:
        return {
            "train_pos": np.empty((0,), dtype=np.int64),
            "val_pos":   np.empty((0,), dtype=np.int64),
            "test_pos":  np.empty((0,), dtype=np.int64),
        }

    # --- Chronological splitting happens at the BLOCK level ---
    # We will create block IDs for the candidate positions and split blocks in time order.
    block_ids = _positions_to_blocks(all_positions, cfg.block_size)
    unique_blocks = np.unique(block_ids)  # already sorted (chronological)

    nb = unique_blocks.size
    # Compute counts of blocks per split (rounded, with safety to avoid negatives)
    n_train = int(round(nb * cfg.train_ratio))
    n_val = int(round(nb * cfg.val_ratio))
    n_test = nb - n_train - n_val
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0

    # Split blocks chronologically: earliest blocks -> train, then val, then test
    train_blocks = unique_blocks[:n_train]
    val_blocks   = unique_blocks[n_train:n_train + n_val]
    test_blocks  = unique_blocks[n_train + n_val:]

    def take_blocks(block_set: np.ndarray) -> np.ndarray:
        if block_set.size == 0:
            return np.empty((0,), dtype=np.int64)
        # Select positions whose block_id is in block_set
        # np.isin is fine here because sizes are modest (hundreds to thousands)
        keep = np.isin(block_ids, block_set)
        return np.sort(all_positions[keep])

    split_positions = {
        "train_pos": take_blocks(train_blocks),
        "val_pos":   take_blocks(val_blocks),
        "test_pos":  take_blocks(test_blocks),
    }
    return split_positions


def purge_boundaries(
    split_positions: Dict[str, np.ndarray],
    purge_positions: int,
) -> Dict[str, np.ndarray]:
    """
    Removes any position from a split if it lies within +/- purge_positions
    of any position in another split.

    This is a strict rule that ensures there is a "gap" around boundaries
    so raw windows cannot overlap (when purge_positions is chosen correctly).
    """
    if purge_positions <= 0:
        return split_positions

    # Convert to sets for fast membership
    sets = {k: set(v.tolist()) for k, v in split_positions.items()}
    keys = list(split_positions.keys())

    def conflicts(pos: int, own_key: str) -> bool:
        # If any neighbor within +/-purge_positions exists in another split -> conflict
        for other_key in keys:
            if other_key == own_key:
                continue
            other = sets[other_key]
            for d in range(-purge_positions, purge_positions + 1):
                if (pos + d) in other:
                    return True
        return False

    purged = {}
    for key, arr in split_positions.items():
        purged[key] = np.asarray([p for p in arr.tolist() if not conflicts(p, key)], dtype=np.int64)
    return purged


def verify_no_raw_overlap(split_positions: Dict[str, np.ndarray], fsize: int, shift: int) -> None:
    """
    Verifies:
      1) no exact overlap of positions across splits
      2) no raw-sample overlap given window extraction [p*shift, p*shift+fsize)

    Raises RuntimeError if a violation is found.
    """
    keys = ["train_pos", "val_pos", "test_pos"]
    arrs = {k: np.sort(split_positions[k]) for k in keys}
    sets = {k: set(arrs[k].tolist()) for k in keys}

    # 1) Exact overlap
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = sets[a].intersection(sets[b])
            if inter:
                raise RuntimeError(f"Position overlap between {a} and {b}: {len(inter)}")

    # 2) Raw overlap check
    min_sep = math.ceil(fsize / shift)  # need |i-j| >= min_sep to avoid overlap
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pa, pb = arrs[a], arrs[b]
            if pa.size == 0 or pb.size == 0:
                continue

            # For each p in smaller array, check nearest neighbor distance in the other array
            # (O(n log n) with searchsorted)
            if pa.size > pb.size:
                pa, pb = pb, pa  # iterate smaller

            for p in pa:
                idx = np.searchsorted(pb, p)
                neighbors = []
                if idx < pb.size:
                    neighbors.append(abs(int(pb[idx]) - int(p)))
                if idx > 0:
                    neighbors.append(abs(int(pb[idx - 1]) - int(p)))
                if neighbors and min(neighbors) < min_sep:
                    raise RuntimeError(
                        f"Raw overlap risk between splits: positions too close (min_sep={min_sep})"
                    )


def chronological_block_split_plus_purge(
    bitmap: np.ndarray,
    cfg: ChronoSplitConfig,
) -> Dict[str, np.ndarray]:
    """
    Full pipeline:
      1) chronological block split on time positions
      2) infer purge_positions (if None)
      3) purge around boundaries
      4) verify no raw overlap

    Returns split_positions dict with keys train_pos/val_pos/test_pos.
    """
    split_positions = chronological_block_split_positions(bitmap, cfg)

    purge = cfg.purge_positions
    if purge is None:
        purge = infer_purge_positions(cfg.fsize, cfg.shift)

    split_positions = purge_boundaries(split_positions, purge_positions=purge)
    verify_no_raw_overlap(split_positions, fsize=cfg.fsize, shift=cfg.shift)
    return split_positions


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    bitmap = np.load("running/running_2023-04-17T122413+0100.npy")  # shape [time_windows, channels], bool
    cfg = ChronoSplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                            fsize=8192, shift=2048,
                            block_size=16,
                             positives_only=True)  # recommended for per-class positives-only
    splits = chronological_block_split_plus_purge(bitmap, cfg)
    print({k: v.size for k, v in splits.items()})
    pass