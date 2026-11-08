"""Classical volume denoising fallback (commercially permissive stack)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def classical_denoise(volume: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Mild 3D denoise: Gaussian + light median blend.

    Tuned for the synthetic low-dose demo — not a clinical algorithm.
    """
    vol = volume.astype(np.float32)
    sigma = 0.6 + 0.6 * float(strength)
    smooth = gaussian_filter(vol, sigma=sigma)
    med = median_filter(vol, size=3)
    out = 0.65 * smooth + 0.35 * med
    # Preserve edges roughly via residual mix
    residual = vol - out
    out = out + 0.15 * residual
    return np.clip(out, 0.0, 1.0).astype(np.float32)
