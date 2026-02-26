import torch
from pathlib import Path

def chunk_xy_tree(src_root: str, dst_root: str, chunk_size: int = 1024):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_out = 0

    for src_path in src_root.rglob("*.pt"):
        name = src_path.name
        if "_chunk" in name:
            continue

        payload = torch.load(src_path, map_location="cpu")
        if not isinstance(payload, dict) or "x" not in payload or "y" not in payload:
            continue

        x = payload["x"]
        y = payload["y"]

        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise RuntimeError(f"Bad types in {src_path}: x/y must be tensors")

        if x.ndim < 1 or y.ndim != 1:
            raise RuntimeError(f"Bad shapes in {src_path}: x={tuple(x.shape)}, y={tuple(y.shape)}")

        n = x.shape[0]
        if y.shape[0] != n:
            raise RuntimeError(f"Length mismatch in {src_path}: x={n}, y={y.shape[0]}")


        rel = src_path.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        base = src_path.stem
        n_files += 1

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            out_payload = {
                "x": x[start:end].contiguous(),
                "y": y[start:end].contiguous(),
            }
            out_path = out_dir / f"{base}_chunk{start:07d}_{end:07d}.pt"
            torch.save(out_payload, out_path)
            n_out += 1

    print(f"Done. input_files={n_files}, output_chunks={n_out}")

if __name__ == "__main__":
    chunk_xy_tree(
        src_root="../data/processed",
        dst_root="../data/processed_xy_chunked",
        chunk_size=1024,  # 512/1024/2048 بسته به سرعت و RAM
    )