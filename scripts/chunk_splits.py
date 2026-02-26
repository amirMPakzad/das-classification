import os
from pathlib import Path
import torch

def chunk_one_file(src_path: Path, dst_dir: Path, chunk_size: int):
    payload = torch.load(src_path, map_location="cpu")
    if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
        raise RuntimeError(f"Bad payload in {src_path}")

    x = payload["x"]
    y = payload["y"]
    n = x.shape[0]
    if not (hasattr(y, "shape") and y.ndim > 0 and y.shape[0] == n):
        # اگر y شکلش متفاوت بود، واضح fail کن
        raise RuntimeError(f"y shape mismatch in {src_path}: x={n}, y={getattr(y,'shape',None)}")

    keys = list(payload.keys())
    dst_dir.mkdir(parents=True, exist_ok=True)

    base = src_path.stem  # مثلا train_foo_0001
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = {}
        for k in keys:
            v = payload[k]
            # اگر per-sample است، slice؛ اگر متا/اسکالر است، همان را کپی
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                chunk[k] = v[start:end].contiguous()
            elif isinstance(v, (list, tuple)) and len(v) == n:
                chunk[k] = v[start:end]
            else:
                chunk[k] = v

        out_name = f"{base}_chunk{start:07d}_{end:07d}.pt"
        torch.save(chunk, dst_dir / out_name)

def chunk_dataset_tree(src_root: str, dst_root: str, chunk_size: int = 1024):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for src_path in src_root.rglob("*.pt"):
        fn = src_path.name
        if not (fn.startswith("train_") or fn.startswith("val_") or fn.startswith("test_")):
            continue

        rel = src_path.relative_to(src_root)      # مثلا classA/train_x.pt
        dst_dir = (dst_root / rel.parent)         # همان ساختار پوشه‌ها
        chunk_one_file(src_path, dst_dir, chunk_size)

if __name__ == "__main__":
    dst_root = "../data/processed_chunked"
    src_root = "../data/processed"
    os.makedirs(dst_root, exist_ok=True)
    chunk_dataset_tree(
        src_root=src_root,
        dst_root=dst_root,
        chunk_size=1024,
    )
    print("Done.")