from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from split_chrological import chronological_block_split_plus_purge, ChronoSplitConfig
from weighted_chrono_split import WeightedChronoSplitConfig, weighted_chronological_block_split_positions


@dataclass
class SplitConfig:
    class_name: str
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    fsize: int = 8192
    shift: int = 2048
    purge_positions: int | None = None
    group_positions: int = 16
    seed: int = 42
    include_all_negatives: bool = False
    negative_to_positive_ratio: float | None = None

    def validate(self) -> None:
        s = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"train+val+test must be 1.0, got {s}")
        if self.shift <= 0 or self.fsize <= 0:
            raise ValueError("fsize and shift must be positive")
        if self.group_positions <= 0:
            raise ValueError("group_positions must be positive")
        if self.include_all_negatives and self.negative_to_positive_ratio is not None:
            raise ValueError(
                "Set either include_all_negatives=True OR negative_to_positive_ratio, not both"
            )


def infer_purge_positions(fsize: int, shift: int) -> int:
    # Two windows overlap if |pos_i - pos_j| * shift < fsize.
    # Purging by this many position steps around split boundaries prevents shared raw samples.
    return max(0, math.ceil(fsize / shift) - 1)


def load_bitmap(npy_path: Path) -> np.ndarray:
    bmp = np.load(npy_path)
    if bmp.ndim != 2:
        raise ValueError(f"Expected 2D bitmap [position, channel], got shape {bmp.shape}")
    # Treat any non-zero as event.
    return bmp != 0


def build_position_groups(n_positions: int, group_positions: int) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for start in range(0, n_positions, group_positions):
        end = min(n_positions, start + group_positions)
        groups.append(np.arange(start, end, dtype=np.int64))
    return groups


def assign_groups(
    groups: List[np.ndarray], train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(groups), dtype=np.int64)
    rng.shuffle(order)

    n = len(order)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    # Keep sizes non-negative after rounding.
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = 0

    train_ids = order[:n_train]
    val_ids = order[n_train : n_train + n_val]
    test_ids = order[n_train + n_val :]

    def flatten(ids: np.ndarray) -> np.ndarray:
        if len(ids) == 0:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate([groups[i] for i in ids], axis=0)

    return {
        "train_pos": np.sort(flatten(train_ids)),
        "val_pos": np.sort(flatten(val_ids)),
        "test_pos": np.sort(flatten(test_ids)),
    }

def assign_groups_stratified(
    groups: List[np.ndarray],
    bitmap: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Determine which groups contain ANY event
    group_has_event = np.array([bool(bitmap[g, :].any()) for g in groups], dtype=bool)

    pos_groups = np.where(group_has_event)[0]
    neg_groups = np.where(~group_has_event)[0]

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    def split_ids(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(ids)
        if n == 0:
            return ids[:0], ids[:0], ids[:0]

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_val = max(0, n_val + n_test)
            n_test = 0

        # Ensure at least 1 group in val/test if possible (for positives)
        # (only if there are enough groups)
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            # Re-balance if we exceeded n
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break

        a = ids[:n_train]
        b = ids[n_train:n_train + n_val]
        c = ids[n_train + n_val:n_train + n_val + n_test]
        return a, b, c

    pos_tr, pos_va, pos_te = split_ids(pos_groups)
    neg_tr, neg_va, neg_te = split_ids(neg_groups)

    def flatten(group_ids: np.ndarray) -> np.ndarray:
        if len(group_ids) == 0:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate([groups[i] for i in group_ids], axis=0)

    train_pos = np.sort(np.concatenate([flatten(pos_tr), flatten(neg_tr)]))
    val_pos   = np.sort(np.concatenate([flatten(pos_va), flatten(neg_va)]))
    test_pos  = np.sort(np.concatenate([flatten(pos_te), flatten(neg_te)]))

    return {"train_pos": train_pos, "val_pos": val_pos, "test_pos": test_pos}


def purge_boundaries(
    split_positions: Dict[str, np.ndarray], purge_positions: int
) -> Dict[str, np.ndarray]:
    if purge_positions <= 0:
        return split_positions

    all_sets = {k: set(v.tolist()) for k, v in split_positions.items()}

    def conflict(pos: int, own_key: str) -> bool:
        for other_key, other_set in all_sets.items():
            if other_key == own_key:
                continue
            for d in range(-purge_positions, purge_positions + 1):
                if (pos + d) in other_set:
                    return True
        return False

    purged: Dict[str, np.ndarray] = {}
    for key, arr in split_positions.items():
        kept = [p for p in arr.tolist() if not conflict(p, key)]
        purged[key] = np.asarray(kept, dtype=np.int64)
    return purged


def build_samples(
    bitmap: np.ndarray,
    split_positions: Dict[str, np.ndarray],
    include_all_negatives: bool,
    negative_to_positive_ratio: float | None,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples: Dict[str, np.ndarray] = {}

    for split_name, pos_arr in split_positions.items():
        split_key = split_name.replace("_pos", "")
        if pos_arr.size == 0:
            samples[split_key] = np.empty((0, 3), dtype=np.int64)
            continue

        sub = bitmap[pos_arr, :]  # [pos, ch] bool
        pos_idx, ch_idx = np.where(sub)
        abs_pos = pos_arr[pos_idx]

        pos_samples = np.stack(
            [abs_pos, ch_idx.astype(np.int64), np.ones_like(abs_pos, dtype=np.int64)], axis=1
        )

        if include_all_negatives:
            npos, nch = sub.shape
            all_pos = np.repeat(pos_arr, nch)
            all_ch = np.tile(np.arange(nch, dtype=np.int64), npos)
            labels = sub.reshape(-1).astype(np.int64)
            samples[split_key] = np.stack([all_pos, all_ch, labels], axis=1)
            continue

        if negative_to_positive_ratio is None:
            samples[split_key] = pos_samples
            continue

        neg_pos_idx, neg_ch_idx = np.where(~sub)
        n_pos = pos_samples.shape[0]
        n_neg_target = int(round(n_pos * negative_to_positive_ratio))
        n_neg_target = min(n_neg_target, neg_pos_idx.shape[0])

        if n_neg_target > 0:
            pick = rng.choice(neg_pos_idx.shape[0], size=n_neg_target, replace=False)
            neg_abs_pos = pos_arr[neg_pos_idx[pick]]
            neg_samples = np.stack(
                [
                    neg_abs_pos,
                    neg_ch_idx[pick].astype(np.int64),
                    np.zeros(n_neg_target, dtype=np.int64),
                ],
                axis=1,
            )
            merged = np.concatenate([pos_samples, neg_samples], axis=0)
            rng.shuffle(merged)
            samples[split_key] = merged
        else:
            samples[split_key] = pos_samples

    return samples


def verify_no_leakage(split_positions: Dict[str, np.ndarray], fsize: int, shift: int) -> None:
    keys = ["train_pos", "val_pos", "test_pos"]
    sets = {k: set(split_positions[k].tolist()) for k in keys}

    # 1) Exact position overlap check
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = sets[a].intersection(sets[b])
            if inter:
                raise RuntimeError(f"Position leakage between {a} and {b}: {len(inter)} overlaps")

    # 2) Raw-sample overlap check for windowed extraction
    # Window at position p covers [p*shift, p*shift+fsize).
    min_sep = math.ceil(fsize / shift)
    arrs = {k: np.sort(split_positions[k]) for k in keys}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pa, pb = arrs[a], arrs[b]
            if pa.size == 0 or pb.size == 0:
                continue

            for p in pa:
                idx = np.searchsorted(pb, p)
                neighbors = []
                if idx < pb.size:
                    neighbors.append(abs(int(pb[idx]) - int(p)))
                if idx > 0:
                    neighbors.append(abs(int(pb[idx - 1]) - int(p)))
                if neighbors and min(neighbors) < min_sep:
                    raise RuntimeError(
                        f"Temporal leakage between {a} and {b}: windows can overlap in raw samples"
                    )

def summarize(
    bitmap: np.ndarray,
    samples: Dict[str, np.ndarray],
    split_positions: Dict[str, np.ndarray],
) -> Dict:
    """
    Enhanced summary:
      - num_positions: how many time-window indices (pos) are assigned to the split
      - sum_true: how many (pos, ch) are True inside those positions (event mass)
      - avg_true_per_pos: sum_true / num_positions  (event density per time-window)
      - channels_per_pos_stats: stats of True-count per pos inside the split
      - num_samples/num_positive/num_negative: derived from your constructed samples
    """

    def channels_per_pos_stats(pos_arr: np.ndarray) -> Dict[str, float | int]:
        if pos_arr.size == 0:
            return {
                "median": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "max": 0,
            }
        # count of active channels at each selected time position
        cpp = bitmap[pos_arr, :].sum(axis=1).astype(np.int64)  # shape [num_positions]
        # If you used positives_only=True, cpp should all be >0, but not guaranteed.
        return {
            "median": float(np.median(cpp)),
            "p90": float(np.percentile(cpp, 90)),
            "p95": float(np.percentile(cpp, 95)),
            "max": int(cpp.max()),
        }

    def split_stats(name: str) -> Dict[str, int | float | Dict[str, float | int]]:
        pos_arr = split_positions[name + "_pos"]
        rows = samples[name]

        # sample-based counts (from your dataset builder)
        pos_count = int((rows[:, 2] == 1).sum()) if rows.size else 0
        neg_count = int((rows[:, 2] == 0).sum()) if rows.size else 0
        num_samples = int(rows.shape[0]) if rows.size else 0

        # bitmap-based mass (independent of how you sampled negatives)
        # sum_true = number of True cells (pos,ch) in bitmap restricted to pos_arr
        sum_true = int(bitmap[pos_arr, :].sum()) if pos_arr.size else 0
        num_positions = int(pos_arr.size)
        avg_true_per_pos = float(sum_true / num_positions) if num_positions > 0 else 0.0

        return {
            "num_positions": num_positions,
            "sum_true": sum_true,
            "avg_true_per_pos": avg_true_per_pos,
            "channels_per_pos_stats": channels_per_pos_stats(pos_arr),
            "num_samples": num_samples,
            "num_positive": pos_count,
            "num_negative": neg_count,
        }

    return {
        "bitmap_shape": [int(bitmap.shape[0]), int(bitmap.shape[1])],
        "train": split_stats("train"),
        "val": split_stats("val"),
        "test": split_stats("test"),
    }


def run(config: SplitConfig, npy_path: Path, out_dir: Path) -> None:
    config.validate()
    recording_name = npy_path.stem.split("_")[0]
    output_dir = out_dir / config.class_name
    output_dir = output_dir / recording_name
    output_dir.mkdir(parents=True, exist_ok=True)

    bitmap = load_bitmap(npy_path)
    n_positions = bitmap.shape[0]

    groups = build_position_groups(n_positions=n_positions, group_positions=config.group_positions)
    split_positions = assign_groups_stratified(
        bitmap = bitmap,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        groups=groups,
        seed=42,
    )

    purge = config.purge_positions
    if purge is None:
        purge = infer_purge_positions(config.fsize, config.shift)

    split_positions = purge_boundaries(split_positions, purge_positions=purge)
    verify_no_leakage(split_positions, fsize=config.fsize, shift=config.shift)

    samples = build_samples(
        bitmap=bitmap,
        split_positions=split_positions,
        include_all_negatives=config.include_all_negatives,
        negative_to_positive_ratio=config.negative_to_positive_ratio,
        seed=config.seed,
    )

    np.save(output_dir / "train_samples.npy", samples["train"])
    np.save(output_dir / "val_samples.npy", samples["val"])
    np.save(output_dir / "test_samples.npy", samples["test"])

    np.save(output_dir / "train_positions.npy", split_positions["train_pos"])
    np.save(output_dir / "val_positions.npy", split_positions["val_pos"])
    np.save(output_dir / "test_positions.npy", split_positions["test_pos"])

    stats = summarize(bitmap, samples, split_positions)
    stats["config"] = {
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "fsize": config.fsize,
        "shift": config.shift,
        "purge_positions": purge,
        "group_positions": config.group_positions,
        "seed": config.seed,
        "include_all_negatives": config.include_all_negatives,
        "negative_to_positive_ratio": config.negative_to_positive_ratio,
    }

    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe DAS train/val/test split builder")
    p.add_argument("--npy", type=Path, required=True, help="Path to event bitmap .npy [position, channel]")
    p.add_argument("--class-name", type=str, required=True, help="Class name")
    p.add_argument("--out", type=Path, default=Path("splits"), help="Output directory")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio")
    p.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.1, help="Test ratio")
    p.add_argument("--fsize", type=int, default=8192, help="Window length in raw samples")
    p.add_argument("--shift", type=int, default=2048, help="Window shift in raw samples")
    p.add_argument(
        "--purge-positions",
        type=int,
        default=None,
        help="Extra exclusion around split boundaries in position units. Default derives from fsize/shift",
    )
    p.add_argument(
        "--group-positions",
        type=int,
        default=16,
        help="Contiguous position block size kept together before splitting",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--negative-to-positive-ratio",
        type=float,
        default=False,
        help="If set, sample negatives to this ratio vs positives. If unset, include all negatives.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SplitConfig(
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        fsize=args.fsize,
        shift=args.shift,
        purge_positions=args.purge_positions,
        group_positions=args.group_positions,
        seed=args.seed,
        include_all_negatives=args.negative_to_positive_ratio is None,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        class_name = args.class_name
    )
    run(cfg, npy_path=args.npy, out_dir=args.out)


if __name__ == "__main__":
    main()

