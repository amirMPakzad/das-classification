# src/das_classification/config.py 

from __future__ import annotations 

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from das_classification.train.imbalance import ImbalanceCfg


import yaml 
from das_classification.data.constants import LABELS, WIN, HOP 

@dataclass(frozen=True)
class SplitCfg:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


@dataclass(frozen=True)
class DatasetCfg:
    root: str
    classes: list[str] = None
    split: SplitCfg = SplitCfg()



@dataclass(frozen=True)
class TrainCfg:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    imbalance: ImbalanceCfg = field(default_factory=ImbalanceCfg)



@dataclass(frozen=True)
class RunCfg:
    base_dir: str = "runs"
    name: str = "baseline"
    splits_dir: str = "data/interim/splits"
    device: str = ""  # auto if empty
    seed: int = 42
    deterministic: bool = True


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetCfg
    train: TrainCfg
    run: RunCfg


def _get(d: Dict[str, Any], key: str, default: Any = None):
    return d[key] if key in d else default


def load_config(path: str) -> AppConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ds_raw = raw.get("dataset", {})
    tr_raw = raw.get("training", raw.get("train", {}))  # allow both keys
    run_raw = raw.get("run", {})

    ds_split_raw = ds_raw.get("split", {}) or {}
    split = SplitCfg(
        train=float(_get(ds_split_raw, "train", 0.8)),
        val=float(_get(ds_split_raw, "val", 0.1)),
        test=float(_get(ds_split_raw, "test", 0.1)),
        )
    
    dataset = DatasetCfg(
        root=_get(ds_raw, "root", "data/raw/DAS-dataset/data"),
        classes =_get(ds_raw, "classes", None),
        split =split,
    )

    
    imb_raw = tr_raw.get("imbalance", {})
    imbalance = ImbalanceCfg(
        enabled=bool(_get(imb_raw, "enabled", False)),
        method=str(_get(imb_raw, "method", "weights")),
        power=float(_get(imb_raw, "power", 1.0)),
        min_count=int(_get(imb_raw, "min_count", 1)),
        max_weight=float(_get(imb_raw, "max_weight", 100.0)),
    )


    train = TrainCfg(
        batch_size=int(_get(tr_raw, "batch_size", 8)),
        num_workers=int(_get(tr_raw, "num_workers", 4)),
        epochs=int(_get(tr_raw, "epochs", 20)),
        lr=float(_get(tr_raw, "lr", 1e-3)),
        weight_decay=float(_get(tr_raw, "weight_decay", 1e-4)),
        grad_clip=float(_get(tr_raw, "grad_clip", 1.0)),
        log_interval=int(_get(tr_raw, "log_interval", 20)),
        imbalance=imbalance,
    )

    run = RunCfg(
        base_dir=_get(run_raw, "base_dir", "runs"),
        name=_get(run_raw, "name", "baseline"),
        splits_dir=_get(run_raw, "splits_dir", "data/interim/splits"),
        device=_get(run_raw, "device", ""),
        seed=int(_get(run_raw, "seed", 42)),
        deterministic=bool(_get(run_raw, "deterministic", True)),
    )


    cfg = AppConfig(dataset=dataset, train=train, run=run)

    if "window" in raw:
        w = raw["window"]
        if int(w.get("length", WIN)) != WIN or int(w.get("hop", HOP)) != HOP:
            raise ValueError(f"Config window must match dataset spec: WIN={WIN}, HOP={HOP}")

    if "labels" in raw:
        if list(raw["labels"]) != list(LABELS):
            raise ValueError("Config labels must match constants.LABELS (order matters).")


    return cfg