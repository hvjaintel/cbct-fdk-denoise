"""Feldkamp–Davis–Kress (FDK) cone-beam reconstruction (NumPy reference)."""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy.ndimage import map_coordinates

from .config import GeometryConfig


def _ramp_filter(n: int, pixel_size: float) -> np.ndarray:
    """1D ramp (Ram-Lak) filter in frequency domain."""
    freqs = fftfreq(n, d=pixel_size)
    return np.abs(freqs).astype(np.float64)


def filter_projections(projections: np.ndarray, geometry: GeometryConfig) -> np.ndarray:
    """Apply cosine weighting + ramp filter along detector rows (u)."""
    n_angles, nv, nu = projections.shape
    sod, sdd = geometry.sod, geometry.sdd
    det_half = (nu - 1) / 2.0
    uu = (np.arange(nu, dtype=np.float64) - det_half) * geometry.pixel_size
    vv = (np.arange(nv, dtype=np.float64) - det_half) * geometry.pixel_size
    U, V = np.meshgrid(uu, vv, indexing="xy")

    # Cosine weighting for cone-beam geometry
    weight = sod / np.sqrt(sod**2 + U**2 + V**2)
    weighted = projections.astype(np.float64) * weight[None, ...]

    ramp = _ramp_filter(nu, geometry.pixel_size)
    filtered = np.empty_like(weighted)
    for a in range(n_angles):
        for v in range(nv):
            row = weighted[a, v]
            filt = np.real(ifft(fft(row) * ramp))
            filtered[a, v] = filt
    # Scale by pixel size (approximate continuous filter)
    filtered *= geometry.pixel_size
    return filtered.astype(np.float32)


def backproject(
    filtered: np.ndarray,
    angles: np.ndarray,
    geometry: GeometryConfig,
) -> np.ndarray:
    """Voxel-driven FDK backprojection."""
    n = geometry.volume_size
    nv = filtered.shape[1]
    nu = filtered.shape[2]
    sod, sdd = geometry.sod, geometry.sdd
    half = (n - 1) / 2.0
    det_u_half = (nu - 1) / 2.0
    det_v_half = (nv - 1) / 2.0

    zz, yy, xx = np.meshgrid(
        (np.arange(n) - half) * geometry.voxel_size,
        (np.arange(n) - half) * geometry.voxel_size,
        (np.arange(n) - half) * geometry.voxel_size,
        indexing="ij",
    )
    volume = np.zeros((n, n, n), dtype=np.float64)
    d_theta = float(2.0 * np.pi / max(len(angles), 1))

    for i, ang in enumerate(angles.astype(np.float64)):
        c, s = np.cos(ang), np.sin(ang)
        # Rotate voxel coords into angle-0 frame where source is at (0,0,-sod)
        x_rot = c * xx + s * zz
        z_rot = -s * xx + c * zz
        y = yy

        # Distance weight
        denom = (sod + z_rot) ** 2
        denom = np.maximum(denom, 1e-6)
        w = (sod**2) / denom

        # Project onto detector
        u = (sdd * x_rot) / (sod + z_rot)
        v = (sdd * y) / (sod + z_rot)
        iu = u / geometry.pixel_size + det_u_half
        iv = v / geometry.pixel_size + det_v_half

        vals = map_coordinates(filtered[i], [iv, iu], order=1, mode="constant", cval=0.0)
        volume += w * vals

    volume *= d_theta / 2.0
    return volume.astype(np.float32)


def reconstruct_fdk(
    projections: np.ndarray,
    angles: np.ndarray,
    geometry: GeometryConfig,
) -> np.ndarray:
    """Full FDK pipeline: filter + backproject, then normalize to [0, 1]."""
    filtered = filter_projections(projections, geometry)
    volume = backproject(filtered, angles, geometry)
    # Soft normalize for display / denoise (robust to outliers)
    lo, hi = np.percentile(volume, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return volume
