# src/das_classification/viz/quicklook.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from das_event.data.h5io import read_h5
from das_event.data.constants import WIN, HOP


@dataclass(frozen=True)
class QuicklookInputs:
    h5_path: str
    npy_path: Optional[str] = None


def _infer_sidecar_npy(h5_path: str) -> Optional[str]:
    stem, _ = os.path.splitext(h5_path)
    npy = stem + ".npy"
    return npy if os.path.exists(npy) else None


def _load_bitmap(npy_path: str) -> np.ndarray:
    bm = np.load(npy_path)
    bm = np.asarray(bm)

    # اگر چندبعدی بود، روی محور کانال/فضا collapse می‌کنیم تا 1D شود
    if bm.ndim > 1:
        # فرض: یکی از محورها زمان/پنجره است. معمولاً collapse با max خوب جواب می‌دهد.
        # اگر جهت اشتباه بود، با bm.T هم تست می‌کنیم.
        bm1 = bm.max(axis=0)
        if bm1.ndim != 1:
            bm1 = bm.max(axis=-1)
        bm = bm1

    bm = (bm > 0).astype(np.uint8)
    return bm


def _make_window_starts(T: int, win: int = WIN, hop: int = HOP) -> np.ndarray:
    if T < win:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(0, T - win + 1, hop, dtype=np.int64)


def _downsample_2d(x: np.ndarray, max_t: int = 4000, max_c: int = 400) -> np.ndarray:
    """
    x: (T, C)
    خروجی: (T', C') برای نمایش سریع
    """
    T, C = x.shape
    step_t = max(1, T // max_t)
    step_c = max(1, C // max_c)
    return x[::step_t, ::step_c]


def quicklook(
    h5_path: str,
    npy_path: Optional[str] = None,
    *,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    t0: int = 0,
    seconds: Optional[float] = None,
    fs: Optional[float] = None,
) -> None:
    """
    رسم یک نگاه سریع از:
      - heatmap سیگنال DAS (|x| یا RMS ساده) در بازه انتخابی
      - bitmap رویداد (اگر وجود داشته باشد)
      - خطوط مرزی windowing با WIN/HOP (اختیاری)

    پارامترهای زمان:
      - t0: شروع به نمونه
      - اگر fs و seconds بدهی، طول بازه = seconds * fs
    """
    if npy_path is None:
        npy_path = _infer_sidecar_npy(h5_path)

    rec = read_h5(h5_path)
    x = rec.x  # (T, C)
    T, C = x.shape

    if fs is not None and seconds is not None:
        L = int(round(seconds * fs))
    else:
        L = min(T - t0, 200_000)  # پیش‌فرض: حداکثر 200k نمونه برای نمایش
    t1 = min(T, t0 + max(1, L))

    x_seg = x[t0:t1]  # (L, C)

    # یک نمای تصویری پایدار: قدرمطلق/انرژی
    # برای کاهش outlierها: clip بر اساس percentiles
    img = np.abs(x_seg)
    lo, hi = np.percentile(img, 2.0), np.percentile(img, 98.0)
    img = np.clip(img, lo, hi)

    img_ds = _downsample_2d(img)

    bitmap = None
    if npy_path is not None and os.path.exists(npy_path):
        bitmap = _load_bitmap(npy_path)

    # شکل کلی: اگر bitmap داریم، 2 ردیف؛ اگر نداریم، 1 ردیف
    nrows = 2 if bitmap is not None else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6 if nrows == 1 else 8), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    # --- Plot 1: signal heatmap ---
    ax0 = axes[0]
    ax0.imshow(img_ds.T, aspect="auto", origin="lower")
    ax0.set_ylabel("Channel (downsampled)")
    ax0.set_xlabel("Time (downsampled)")
    ax0.set_title(title or os.path.basename(h5_path))

    # window grid (نمایشی): فقط اگر بازه خیلی بزرگ نباشه
    # ما روی axis downsample شده خط می‌کشیم، پس باید scale کنیم
    starts = _make_window_starts(T)
    if starts.size > 0:
        # فقط windowهایی که در بازه [t0,t1) می‌افتند
        starts_in = starts[(starts >= t0) & (starts < t1)]
        if starts_in.size > 0:
            # تبدیل به مختصات downsample شده
            step_t = max(1, (t1 - t0) // 4000)
            xs = (starts_in - t0) // step_t
            # اگر خیلی زیاد شد، یکی در میان/چندتا یکی
            if xs.size > 200:
                xs = xs[:: max(1, xs.size // 200)]
            for xline in xs:
                ax0.axvline(x=xline, linewidth=0.6)

    # --- Plot 2: bitmap + window index range ---
    if bitmap is not None:
        ax1 = axes[1]
        # تعداد پنجره‌ها برای کل فایل و نگاشت بازه انتخابی به window index
        full_starts = _make_window_starts(T)
        nW = full_starts.size

        if nW == 0:
            ax1.text(0.5, 0.5, "No windows (T < WIN).", ha="center", va="center")
        else:
            # بازه زمانی انتخابی را به محدوده window index نگاشت می‌کنیم
            # پنجره k شروعش = k*HOP
            w0 = int(max(0, (t0) // HOP))
            w1 = int(min(nW, (t1 - 1) // HOP + 1))

            bm = bitmap
            if bm.shape[0] != nW:
                # align محافظه‌کارانه
                m = min(bm.shape[0], nW)
                bm = bm[:m]
                nW = m
                w0 = min(w0, nW)
                w1 = min(w1, nW)

            bm_seg = bm[w0:w1]

            ax1.step(np.arange(w0, w1), bm_seg, where="mid")
            ax1.set_ylim(-0.1, 1.1)
            ax1.set_ylabel("bitmap")
            ax1.set_xlabel("window index")
            ax1.set_title(f"Bitmap (windows): [{w0}, {w1}) | WIN={WIN}, HOP={HOP}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
