"""Simple cone-beam forward projector for the synthetic demo."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .config import GeometryConfig


def _rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def forward_project(volume: np.ndarray, geometry: GeometryConfig, seed: int = 42, noise_sigma: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a volume into cone-beam sinogram [n_angles, det_v, det_u].

    Uses line integrals along source→detector rays with trilinear sampling.
    """
    n = geometry.volume_size
    nd = geometry.detector_size
    angles = np.linspace(0.0, 2.0 * np.pi, geometry.n_angles, endpoint=False, dtype=np.float64)
    proj = np.zeros((geometry.n_angles, nd, nd), dtype=np.float32)

    # Volume coordinates centered at origin
    half = (n - 1) / 2.0
    coords_1d = (np.arange(n, dtype=np.float64) - half) * geometry.voxel_size

    # Detector plane local coords
    det_half = (nd - 1) / 2.0
    uu = (np.arange(nd, dtype=np.float64) - det_half) * geometry.pixel_size
    vv = (np.arange(nd, dtype=np.float64) - det_half) * geometry.pixel_size
    U, V = np.meshgrid(uu, vv, indexing="xy")

    n_samples = int(np.ceil(np.sqrt(3) * n)) + 4
    sod, sdd = geometry.sod, geometry.sdd

    for i, ang in enumerate(angles):
        R = _rotation_matrix_y(ang)
        source = R @ np.array([0.0, 0.0, -sod], dtype=np.float64)
        # Detector center and axes in world coordinates
        det_center = R @ np.array([0.0, 0.0, sdd - sod], dtype=np.float64)
        u_axis = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_axis = R @ np.array([0.0, 1.0, 0.0], dtype=np.float64)

        det_pts = (
            det_center[None, None, :]
            + U[..., None] * u_axis
            + V[..., None] * v_axis
        )  # [v,u,3]

        direction = det_pts - source  # [v,u,3]
        # Sample along each ray through the object bounding box
        ts = np.linspace(0.15, 0.85, n_samples, dtype=np.float64)
        samples = source + ts[:, None, None, None] * direction[None, ...]  # [t,v,u,3]

        # Map world xyz → volume index (z,y,x)
        # World: x horizontal, y vertical, z along optical axis at angle 0
        x = samples[..., 0]
        y = samples[..., 1]
        z = samples[..., 2]
        ix = x / geometry.voxel_size + half
        iy = y / geometry.voxel_size + half
        iz = z / geometry.voxel_size + half

        vals = map_coordinates(volume, [iz, iy, ix], order=1, mode="constant", cval=0.0)
        # Approximate physical path length per sample
        step = np.linalg.norm(direction, axis=-1) / n_samples
        line = (vals * step[None, ...]).sum(axis=0).astype(np.float32)
        proj[i] = line

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_sigma, size=proj.shape).astype(np.float32)
        proj = np.clip(proj + noise, 0.0, None)

    return proj, angles.astype(np.float32)
