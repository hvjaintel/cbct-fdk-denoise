"""Synthetic 3D phantom for CBCT demos (no patient data)."""

from __future__ import annotations

import numpy as np


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    rz, ry, rx = radii
    return ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def shepp_logan_3d(size: int = 64) -> np.ndarray:
    """Low-resolution 3D Shepp-Logan–style phantom in [0, 1]."""
    vol = np.zeros((size, size, size), dtype=np.float32)
    c = (size - 1) / 2.0
    s = size / 2.0

    ellipsoids = [
        # (center_z,y,x as fractions of half-size), radii fractions, intensity
        ((0.0, 0.0, 0.0), (0.69, 0.92, 0.90), 1.0),
        ((0.0, -0.0184, 0.0), (0.6624, 0.874, 0.880), -0.8),
        ((0.0, 0.0, 0.22), (0.11, 0.31, 0.16), -0.2),
        ((0.0, 0.0, -0.22), (0.16, 0.41, 0.21), -0.2),
        ((0.0, 0.35, 0.0), (0.21, 0.25, 0.25), 0.1),
        ((0.0, 0.1, 0.0), (0.046, 0.046, 0.046), 0.1),
        ((0.0, -0.1, 0.0), (0.046, 0.046, 0.046), 0.1),
        ((0.06, -0.605, -0.08), (0.046, 0.023, 0.023), 0.1),
        ((0.06, -0.605, 0.06), (0.023, 0.023, 0.046), 0.1),
        ((-0.08, -0.605, 0.0), (0.023, 0.046, 0.023), 0.1),
    ]

    for (cz, cy, cx), (rz, ry, rx), amp in ellipsoids:
        mask = _ellipsoid_mask(
            vol.shape,
            (c + cz * s, c + cy * s, c + cx * s),
            (max(rz * s, 1.0), max(ry * s, 1.0), max(rx * s, 1.0)),
        )
        vol[mask] += np.float32(amp)

    vol = np.clip(vol, 0.0, None)
    peak = float(vol.max()) or 1.0
    return (vol / peak).astype(np.float32)
