"""Tetrahedral 3D-CGR occupancy voxels (visit-count grids, then smoothed)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

ALPHAS = (0.3, 0.5, 0.7, 0.9)
VOXEL_SIZE = 20
GAUSSIAN_SIGMA = 0.45
PCA_N = 202
RESAMPLE_MODES = ("arc", "idx")
VERTICES = np.array(
    [
        [1.0, 1.0, 1.0],  # A
        [-1.0, -1.0, 1.0],  # C
        [-1.0, 1.0, -1.0],  # G
        [1.0, -1.0, -1.0],  # T
    ],
    dtype=np.float64,
)
_BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def clean_seq(seq: str) -> str:
    return str(seq).upper().replace("U", "T").replace(" ", "")


def extract_windows(sequence: str, n_seg: int, n_pts: int) -> list[str]:
    seq = clean_seq(sequence)
    if n_seg <= 1:
        return [seq[:n_pts] if len(seq) > n_pts else seq]
    if len(seq) <= n_pts:
        return [seq] * n_seg
    segments = []
    max_start = len(seq) - n_pts
    for k in range(n_seg):
        start = int(round(k * max_start / max(n_seg - 1, 1)))
        segments.append(seq[start : start + n_pts])
    return segments


def cgr_walk(sequence: str, alpha: float) -> np.ndarray:
    seq = clean_seq(sequence)
    if not seq:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.zeros((len(seq), 3), dtype=np.float64)
    cur = np.zeros(3, dtype=np.float64)
    a = float(alpha)
    for i, ch in enumerate(seq):
        idx = _BASE.get(ch, -1)
        if idx >= 0:
            cur = (cur + VERTICES[idx]) * a
        pts[i] = cur
    if len(pts) > 1:
        mean = pts.mean(axis=0, keepdims=True)
        std = pts.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        pts = (pts - mean) / std
    return pts


def _idx_resample(points: np.ndarray, n_pts: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return np.zeros((n_pts, 3), dtype=np.float64)
    if len(p) == 1:
        return np.tile(p[:1], (n_pts, 1))
    t = np.linspace(0.0, len(p) - 1, n_pts)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(p) - 1)
    w = (t - i0)[:, None]
    return p[i0] * (1.0 - w) + p[i1] * w


def _arc_resample(points: np.ndarray, n_pts: int) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        base = p[:1] if len(p) else np.zeros((1, 3))
        return np.tile(base, (n_pts, 1))
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-12:
        return np.tile(p[0], (n_pts, 1))
    targets = np.linspace(0.0, total, n_pts)
    out = np.zeros((n_pts, 3), dtype=np.float64)
    j = 0
    for i, c in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] < c:
            j += 1
        if j + 1 >= len(cum):
            out[i] = p[-1]
            continue
        u = (c - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
        out[i] = p[j] * (1.0 - u) + p[j + 1] * u
    return out


def resample(points: np.ndarray, n_pts: int, mode: str) -> np.ndarray:
    if mode == "idx":
        return _idx_resample(points, n_pts)
    return _arc_resample(points, n_pts)


def points_to_density(points: np.ndarray, voxel_size: int = VOXEL_SIZE, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    grid = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 0:
        return grid
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0
    q = 2.0 * (p - lo) / (span + 1e-12) - 1.0
    idx = np.floor((q + 1.0) / 2.0 * (voxel_size - 1)).astype(int)
    idx = np.clip(idx, 0, voxel_size - 1)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    dens = np.log1p(grid)
    dens = gaussian_filter(dens, sigma=float(sigma))
    mx = float(dens.max())
    if mx > 0:
        dens = dens / mx
    return dens.astype(np.float32)


def encode_one(sequence: str, n_seg: int, n_pts: int) -> np.ndarray:
    windows = extract_windows(sequence, n_seg, n_pts)
    blocks = []
    for win in windows:
        for alpha in ALPHAS:
            cloud = cgr_walk(win, alpha)
            for mode in RESAMPLE_MODES:
                pts = resample(cloud, n_pts, mode)
                blocks.append(points_to_density(pts).ravel())
    return np.concatenate(blocks, axis=0)


def encode_voxel_matrix(
    sequences: Sequence[str],
    *,
    n_seg: int,
    n_pts: int,
    pca: PCA | None = None,
    fit_pca: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, PCA | None]:
    n = len(sequences)
    rows = []
    for i, seq in enumerate(sequences):
        rows.append(encode_one(seq, n_seg, n_pts))
        if verbose and ((i + 1) % 50 == 0 or i + 1 == n):
            print(f"  3D-CGR voxels {i + 1}/{n}", flush=True)
    X = np.stack(rows, axis=0)
    if fit_pca:
        n_comp = min(PCA_N, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X = pca.fit_transform(X)
        return X.astype(np.float32), pca
    if pca is not None:
        return pca.transform(X).astype(np.float32), pca
    return X.astype(np.float32), None
