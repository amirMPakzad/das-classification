# src/das_classification/utils/logging.py 

from __future__ import annotations

import json 
import os 
from dataclasses import asdict, is_dataclass
from datetime import datetime 
from pathlib import Path 
from typing import Any, Dict, Optional

try:
    from rich.console import Console 
    from rich.logging import RichHandler
    import logging 
except Exception:
    Console = None 
    RichHandler = None 
    import logging 


def make_run_dir(base_dir:str, name:str="run") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir 


def setup_logger(run_dir: Path, level: str = "INFO") -> "logging.Logger":
    logger = logging.getLogger("das_event")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False 


    if RichHandler is not None:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)
        logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # File handler
    fh = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    return logger


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def save_config(run_dir: Path, cfg: Any) -> None:
    """
    Save cfg (dict or dataclass or nested) into run_dir/config.json
    """
    payload = _to_jsonable(cfg)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

