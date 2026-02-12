import os
from glob import glob
import h5py
import numpy as np
from data_loader import fft  

def spec_cmp_1d(x_spec: np.ndarray) -> bool:
    split = int(len(x_spec) * 0.1)
    return (np.mean(x_spec[:split]) - np.mean(x_spec[split:])) > 0.05

def window_noise_ok(window_1d: np.ndarray, keep_if_regular: bool) -> bool:
    if keep_if_regular:
        return True
    x = np.fft.rfft(window_1d)[1:]
    x = np.log10(np.abs(x) + 1.0)
    return spec_cmp_1d(x)

def iter_windows(label_dir: str, fsize: int, shift: int, decimate: int):
    for ds in glob(os.path.join(label_dir, "*.h5")):
        with h5py.File(ds, "r") as f:
            data = f["Acquisition"]["Raw[0]"]["RawData"]
            bmp = np.load(ds[:-2] + "npy")

            for pos, ch in np.transpose(np.where(bmp))[::decimate]:
                start = pos * shift
                end = start + fsize
                if end > data.shape[0]:
                    continue
                yield data[start:end, ch].astype(np.float32)

def build_one_class(
    data_root: str,
    label: str,
    out_dir: str,
    sample_len: int,
    fsize: int,
    shift: int,
    drop_noise: bool,
    decimate: int,
):
    label_dir = os.path.join(data_root, label)
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.h5")

    
    n = 0
    for w in iter_windows(label_dir, fsize, shift, decimate):
        if drop_noise:
            ok = window_noise_ok(w, keep_if_regular=(label == "regular"))
            if not ok:
                continue
        n += 1

    if n == 0:
        raise RuntimeError(f"No samples for label={label} after filtering.")

  
    dummy = np.zeros((1, fsize), dtype=np.float32)
    feat = fft(dummy)  
    feat_dim = feat.shape[1]
    if sample_len:
        feat_dim = min(sample_len, feat_dim)

   
    with h5py.File(out_path, "w") as hf:
        x_ds = hf.create_dataset(
            "x", shape=(n, feat_dim), dtype="float32",
            chunks=(min(1024, n), feat_dim)
        )
        hf.attrs["label"] = label
        hf.attrs["fsize"] = fsize
        hf.attrs["shift"] = shift
        hf.attrs["sample_len"] = sample_len
        hf.attrs["drop_noise"] = int(drop_noise)
        hf.attrs["decimate"] = int(decimate)

        idx = 0
        for w in iter_windows(label_dir, fsize, shift, decimate):
            if drop_noise:
                ok = window_noise_ok(w, keep_if_regular=(label == "regular"))
                if not ok:
                    continue
            x = fft(w.reshape(1, -1))
            if sample_len:
                x = x[:, :sample_len]
            x_ds[idx] = x[0].astype(np.float32)
            idx += 1

        assert idx == n, (idx, n)

    print(f"[OK] {label}: wrote {n} samples -> {out_path}")
    return out_path

if __name__ == "__main__":
   
    data_root = os.path.abspath("../data")          
    out_dir   = os.path.abspath("../data/processed")   

    
    labels = ["regular", "construction", "manipulation"]

    # decimation per class
    decim = {"regular":10}

    for label in labels:
        build_one_class(
            data_root=data_root,
            label=label,
            out_dir=out_dir,
            sample_len=2048,
            fsize=8192,
            shift=2048,
            drop_noise=True,              
            decimate=decim.get(label, 1),
        )
