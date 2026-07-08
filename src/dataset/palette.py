"""Quantification de palette unifiée par cycle.

Convention :
  - Index 0 = transparence (RGB conventionnel (0,0,0), jamais utilisé pour du contenu)
  - Indices 1..15 = couleurs du sprite

Détection de transparence : alpha=0 sur le canal RGBA des frames d'entrée.

Stratégie :
  1. Collecter toutes les couleurs RGB des pixels NON transparents sur toutes les
     frames du cycle.
  2. Si <= 15 couleurs uniques → mapping direct.
  3. Sinon → k-means à 15 clusters dans l'espace RGB.
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans

from .types import RawCycle

MAX_COLORS = 15  # hors transparent


def _gather_opaque_pixels(frames: list[np.ndarray]) -> np.ndarray:
    """Concatène les pixels opaques (alpha > 0) de toutes les frames -> (M, 3) uint8."""
    parts = []
    for f in frames:
        mask = f[:, :, 3] > 0
        parts.append(f[mask][:, :3])
    if not parts:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(parts, axis=0)


def build_palette(cycle: RawCycle, rng: np.random.Generator | None = None) -> np.ndarray:
    """Construit la palette (16, 3) uint8 d'un cycle.

    Index 0 = (0,0,0) sentinelle transparente. Index 1..15 = couleurs.
    Padding au-delà du nombre réel de couleurs avec (0,0,0).
    """
    pixels = _gather_opaque_pixels(cycle.frames)
    palette = np.zeros((16, 3), dtype=np.uint8)
    if pixels.shape[0] == 0:
        return palette

    uniq = np.unique(pixels, axis=0)
    if uniq.shape[0] <= MAX_COLORS:
        palette[1:1 + uniq.shape[0]] = uniq
        return palette

    # k-means sur les pixels (pondère naturellement par fréquence)
    seed = 0 if rng is None else int(rng.integers(0, 2**31))
    km = KMeans(n_clusters=MAX_COLORS, n_init=4, random_state=seed)
    # Sous-échantillonnage si trop de pixels (k-means lent)
    if pixels.shape[0] > 20000:
        idx = np.random.default_rng(seed).choice(pixels.shape[0], 20000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels
    km.fit(sample)
    centers = np.clip(np.round(km.cluster_centers_), 0, 255).astype(np.uint8)
    palette[1:1 + MAX_COLORS] = centers
    return palette


def quantize_frame(frame_rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Mappe chaque pixel d'une frame RGBA -> index palette uint8 (H, W).

    Pixels transparents (alpha=0) -> index 0.
    Pixels opaques -> nearest neighbor euclidien sur la palette[1:].
    """
    H, W = frame_rgba.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)
    alpha = frame_rgba[:, :, 3]
    opaque_mask = alpha > 0
    if not opaque_mask.any():
        return out

    rgb = frame_rgba[:, :, :3][opaque_mask].astype(np.int32)  # (M, 3)
    pal = palette[1:].astype(np.int32)  # (15, 3)
    # distances (M, 15)
    d = np.sum((rgb[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    nn = np.argmin(d, axis=1).astype(np.uint8) + 1  # +1 car palette[0] est transparent
    out[opaque_mask] = nn
    return out
