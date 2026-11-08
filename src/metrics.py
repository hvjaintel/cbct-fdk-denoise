"""Simple image-quality and timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def timed(store: dict, key: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = round((time.perf_counter() - t0) * 1000.0, 2)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    err = mse(a, b)
    if err <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(err))


def snr_estimate(volume: np.ndarray) -> float:
    """Rough SNR proxy using mean / std in a central ROI."""
    n = volume.shape[0]
    c0, c1 = n // 4, 3 * n // 4
    roi = volume[c0:c1, c0:c1, c0:c1]
    std = float(roi.std())
    if std < 1e-8:
        return 0.0
    return float(roi.mean() / std)
