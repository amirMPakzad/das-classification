from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class WeightedChronoSplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    fsize: int = 8192
    shift: int = 2048

    # If None -> inferred from fsize/shift
    purge_positions: Optional[int] = None

    # Keep contiguous blocks of time windows together
    block_size: int = 16

    # If True: only consider positions that have at least one True in bitmap
    # This matches your "positives-only per class" dataset building.
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
    Two windows i and j overlap in raw samples if |i-j|*shift < fsize.
    Purge by K=ceil(fsize/shift)-1 ensures different splits cannot overlap in raw.
    """
    return max(0, math.ceil(fsize / shift) - 1)


def purge_boundaries(split_positions: Dict[str, np.ndarray], purge_positions: int) -> Dict[str, np.ndarray]:
    """
    Remove any position in a split if it is within +/- purge_positions of a position in another split.
    This creates a strict "gap" that prevents raw-window overlap across splits.
    """
    if purge_positions <= 0:
        return split_positions

    keys = ["train_pos", "val_pos", "test_pos"]
    sets = {k: set(split_positions[k].tolist()) for k in keys}

    def conflict(pos: int, own: str) -> bool:
        for k in keys:
            if k == own:
                continue
            other = sets[k]
            for d in range(-purge_positions, purge_positions + 1):
                if (pos + d) in other:
                    return True
        return False

    out = {}
    for k in keys:
        arr = split_positions[k]
        out[k] = np.asarray([p for p in arr.tolist() if not conflict(p, k)], dtype=np.int64)
    return out


def verify_no_raw_overlap(split_positions: Dict[str, np.ndarray], fsize: int, shift: int) -> None:
    """
    Verifies no exact position overlap AND no raw-window overlap.
    Window at p is [p*shift, p*shift+fsize).
    """
    keys = ["train_pos", "val_pos", "test_pos"]
    arrs = {k: np.sort(split_positions[k]) for k in keys}
    sets = {k: set(arrs[k].tolist()) for k in keys}

    # 1) exact overlap
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = sets[keys[i]].intersection(sets[keys[j]])
            if inter:
                raise RuntimeError(f"Position overlap between {keys[i]} and {keys[j]}: {len(inter)}")

    # 2) raw overlap
    min_sep = math.ceil(fsize / shift)  # need |i-j| >= min_sep
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pa, pb = arrs[a], arrs[b]
            if pa.size == 0 or pb.size == 0:
                continue
            # iterate smaller
            if pa.size > pb.size:
                pa, pb = pb, pa
            for p in pa:
                idx = np.searchsorted(pb, p)
                neighbors = []
                if idx < pb.size:
                    neighbors.append(abs(int(pb[idx]) - int(p)))
                if idx > 0:
                    neighbors.append(abs(int(pb[idx - 1]) - int(p)))
                if neighbors and min(neighbors) < min_sep:
                    raise RuntimeError(
                        f"Raw overlap risk between splits: positions too close (<{min_sep})"
                    )


def _block_ranges(n_pos: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays (block_starts, block_ends) for blocks:
      [0:block_size), [block_size:2*block_size), ...
    end is exclusive.
    """
    starts = np.arange(0, n_pos, block_size, dtype=np.int64)
    ends = np.minimum(starts + block_size, n_pos).astype(np.int64)
    return starts, ends


def weighted_chronological_block_split_positions(
    bitmap: np.ndarray,
    cfg: WeightedChronoSplitConfig,
) -> Dict[str, np.ndarray]:
    """
    Chronological split where boundary selection is based on EVENT MASS:
      weight[pos] = number of True channels at this time-window index
    We split blocks so that sum(weights) approximately matches train/val/test ratios.

    Returns dict of sorted window indices for train/val/test.
    """
    cfg.validate()
    if bitmap.ndim != 2:
        raise ValueError(f"Expected bitmap [time_pos, channel], got {bitmap.shape}")

    n_pos, _ = bitmap.shape

    # weight per time-window index: how many channels are active
    w = bitmap.sum(axis=1).astype(np.int64)  # shape [n_pos]

    # Optional: restrict positions considered to "positives-only"
    # Important: we still split at the BLOCK level chronologically; we just ignore zero-weight positions.
    if cfg.positives_only:
        # Keep weights as-is, but later we will only output positions with w>0
        pass

    # Build blocks and compute block weights
    b_starts, b_ends = _block_ranges(n_pos, cfg.block_size)

    block_w = np.array([int(w[s:e].sum()) for s, e in zip(b_starts, b_ends)], dtype=np.int64)

    # If positives_only, blocks with zero weight effectively don't contribute; that's OK.
    total_w = int(block_w.sum())

    # Edge case: no positives at all
    if total_w == 0:
        return {
            "train_pos": np.empty((0,), dtype=np.int64),
            "val_pos": np.empty((0,), dtype=np.int64),
            "test_pos": np.empty((0,), dtype=np.int64),
        }

    # Compute target cumulative weights for boundaries
    tgt_train = cfg.train_ratio * total_w
    tgt_val = (cfg.train_ratio + cfg.val_ratio) * total_w

    # Walk blocks in chronological order and pick boundary blocks
    cum = 0
    train_end_block = len(block_w)  # exclusive index
    val_end_block = len(block_w)

    for i, bw in enumerate(block_w):
        cum += int(bw)
        if train_end_block == len(block_w) and cum >= tgt_train:
            train_end_block = i + 1  # split AFTER this block
        if val_end_block == len(block_w) and cum >= tgt_val:
            val_end_block = i + 1
            break

    # Safety: ensure ordering
    train_end_block = max(0, min(train_end_block, len(block_w)))
    val_end_block = max(train_end_block, min(val_end_block, len(block_w)))

    # Convert block ranges to position indices
    def blocks_to_positions(b0: int, b1: int) -> np.ndarray:
        if b1 <= b0:
            return np.empty((0,), dtype=np.int64)
        s = int(b_starts[b0])
        e = int(b_ends[b1 - 1])  # end of last included block
        pos = np.arange(s, e, dtype=np.int64)
        if cfg.positives_only:
            pos = pos[w[pos] > 0]
        return pos

    train_pos = blocks_to_positions(0, train_end_block)
    val_pos   = blocks_to_positions(train_end_block, val_end_block)
    test_pos  = blocks_to_positions(val_end_block, len(block_w))

    return {
        "train_pos": np.sort(train_pos),
        "val_pos":   np.sort(val_pos),
        "test_pos":  np.sort(test_pos),
    }


def weighted_chrono_block_split_plus_purge(
    bitmap: np.ndarray,
    cfg: WeightedChronoSplitConfig,
) -> Dict[str, np.ndarray]:
    """
    Full pipeline:
      1) weighted chronological block split (balanced by sum of True cells)
      2) purge around split boundaries (to prevent raw overlap)
      3) verify no raw overlap
    """
    split_positions = weighted_chronological_block_split_positions(bitmap, cfg)

    purge = cfg.purge_positions
    if purge is None:
        purge = infer_purge_positions(cfg.fsize, cfg.shift)

    split_positions = purge_boundaries(split_positions, purge_positions=purge)
    verify_no_raw_overlap(split_positions, fsize=cfg.fsize, shift=cfg.shift)
    return split_positions


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # bmp = np.load("running_2023-04-17T122413+0100.npy")  # bool [time_pos, channel]
    # cfg = WeightedChronoSplitConfig(
    #     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    #     fsize=8192, shift=2048,
    #     block_size=16,
    #     positives_only=True,   # your per-class positives-only workflow
    #     purge_positions=None,  # infer (will become 3 for 8192/2048)
    # )
    # splits = weighted_chrono_block_split_plus_purge(bmp, cfg)
    # print({k: v.size for k, v in splits.items()})
    pass