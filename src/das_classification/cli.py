# src/das_classification/cli.py

import json
from pathlib import Path

import typer
import torch
from torch.utils.data import DataLoader, Subset

from das_classification.data.das_dataset import DASDataset
from das_classification.config import load_config
from das_classification.models.cnn1d import DASConvClassifier, ModelConfig
from das_classification.utils.seed import seed_everything, SeedConfig
from das_classification.train.imbalance import count_labels, make_class_weights, make_weighted_sampler_from_dataset
from das_classification.utils.logging import setup_logger, make_run_dir
from das_classification.train.loop import train_loop, TrainConfig
from das_classification.train.test import test_loop, load_checkpoint, save_report_txt
from das_classification.viz.confusion import save_confusion_matrix_png
from das_classification.viz.plot_history import plot_history
from das_classification.data.splits import ensure_splits


app = typer.Typer(no_args_is_help=True)

@app.command()
def sanity(
    config: str = typer.Option(..., help="Path to an app config YAML"),
    batch_size: int = 8,
    num_workers: int = 0,
    keep_negatives: bool = False,
):
    cfg = load_config(config)

    ds = DASDataset(h5_dir=cfg.dataset.root)
    typer.echo(f"Samples: {len(ds)}")

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    batch = next(iter(dl))
    typer.echo(f"x: {batch.x.shape} | y: {batch.y.shape}")



@app.command()  
def train(
    config: str = typer.Option(..., help="Path to an app config YAML")
):
    cfg = load_config(config)
    run_dir = make_run_dir(cfg.run.base_dir, name=cfg.run.name)
    logger = setup_logger(run_dir)

    #seed 
    seed_everything(SeedConfig(seed=cfg.run.seed, deterministic=cfg.run.deterministic))
    


    ds = DASDataset(cfg.dataset.root, cfg.dataset.classes)
    print("classes:", ds.classes)
    print("len:", len(ds))


    splits_dir = Path(cfg.run.splits_dir)

    ensure_splits(
        ds,
        splits_dir,
        train_ratio=cfg.dataset.split.train,
        val_ratio=cfg.dataset.split.val,
        test_ratio=cfg.dataset.split.test,
        seed=cfg.run.seed,
    )

    train_idx = json.loads((splits_dir/"train.json").read_text())
    val_idx   = json.loads((splits_dir/"val.json").read_text())


    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    class_weights = None
    sampler = None

    num_classes = len(ds.classes)

    if cfg.train.imbalance.enabled:
        counts = count_labels(ds, indices=train_idx, num_classes=num_classes)
        cw = make_class_weights(counts, cfg.train.imbalance)

        logger.info(f"class counts: {counts.tolist()}")
        logger.info(f"class weights: {[round(float(x),3) for x in cw.tolist()]}")

        if cfg.train.imbalance.method in ("weights", "both"):
            class_weights = cw.to(torch.float32)

        if cfg.train.imbalance.method in ("sampler", "both"):
            sampler = make_weighted_sampler_from_dataset(train_ds, cw)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,               
        num_workers=cfg.train.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    x0, _ = ds[0]
    in_channels = int(x0.shape[0])       
    num_classes = len(ds.classes)

    model = DASConvClassifier(ModelConfig(in_channels=in_channels, num_classes=num_classes))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    cfg_train = TrainConfig(
        epochs=cfg.train.epochs, 
        lr=cfg.train.lr, 
        weight_decay=cfg.train.weight_decay, 
        grad_clip=cfg.train.grad_clip,
        log_interval=cfg.train.log_interval,
        save_dir=str(run_dir)
        )

    train_loop(model, train_loader, val_loader, device, cfg_train,
                class_weights=class_weights)

    
@app.command()
def test(
    config: str = typer.Option(..., help="Path to an app config YAML"),
    ckpt: str = typer.Option("", help="Checkpoint path. If empty, uses <run_dir>/best.pt"),
    run_dir: str = typer.Option("", help="Run directory that contains best.pt/last.pt"),
):
    cfg = load_config(config)
    seed_everything(SeedConfig(cfg.run.seed, deterministic=cfg.run.deterministic))

    ds = DASDataset(cfg.dataset.root, cfg.dataset.classes)
    run_dir = make_run_dir(cfg.run.base_dir, name=cfg.run.name)
    logger = setup_logger(run_dir)
    logger.info(f"classes: {ds.classes}")
    logger.info(f"len: {len(ds)}")

    # --- load test split ---
    splits_dir = Path(cfg.run.splits_dir)
    with open(splits_dir / "test.json", "r") as f:
        test_idx = json.load(f)

    test_ds = Subset(ds, test_idx)
    loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    # --- model shape ---
    x0, _ = ds[0]  # dataset returns (x, y)
    in_channels = int(x0.shape[0])
    num_classes = len(ds.classes)
    labels = list(ds.classes)  

    model = DASConvClassifier(
        ModelConfig(in_channels=in_channels, num_classes=num_classes)
    )

    device = cfg.run.device if getattr(cfg.run, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- resolve checkpoint path & output dir ---
    if ckpt:
        ckpt_path = Path(ckpt).expanduser().resolve()
        out_dir = ckpt_path.parent
    else:
        if not run_dir:
            raise typer.BadParameter("Provide --run-dir or --ckpt")
        run_dir_path = Path(run_dir).expanduser().resolve()
        ckpt_path = run_dir_path / "best.pt"
        out_dir = run_dir_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # --- load checkpoint ---
    epoch = load_checkpoint(model, str(ckpt_path), device=device)

    # --- test ---
    model.eval()
    with torch.inference_mode():
        res = test_loop(model, loader, device=device, num_classes=num_classes)

    # --- save reports ---
    report_path = str(out_dir / "test_report.txt")
    save_report_txt(report_path, res.loss, res.acc, res.cm, labels)

    logger.info(f"Loaded epoch: {epoch}")
    logger.info(f"test loss: {res.loss:.6f} | test acc: {res.acc:.4f}")
    logger.info(f"Saved report: {report_path}")

    save_confusion_matrix_png(
        res.cm,
        labels,
        str(out_dir / "confusion_test.png"),
        normalize=True,
        title="Test Confusion Matrix (row-normalized)",
    )



@app.command
def plot_history_cmd(
    run_dir: str = typer.Option(..., help="Run directory that contains history.jsonl"),
    save_dir: str = typer.Option("", help="Optional: directory to save loss.png/acc.png"),
    no_show: bool = typer.Option(False, help="Do not open GUI windows"),
):
    hist_path = str(Path(run_dir) / "history.jsonl")
    plot_history(
        history_path=hist_path,
        save_dir=(save_dir if save_dir else None),
        show=(not no_show)
    )
    print(f"Plotted history from: {hist_path}")
    if save_dir:
        print(f"Saved figures to: {save_dir}")



if __name__ == "__main__":
    app()
