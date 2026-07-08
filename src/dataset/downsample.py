"""Downsampling mode-par-bloc pour préserver la palette indexée.

Sprites natifs en 32x32 (HGSS), 64x64 (Emerald combat), 96x96 (Emerald animated).
Ratios entiers 1:1, 2:1, 3:1 vers 32x32.

Mode-par-bloc : pour chaque bloc kxk, la valeur de sortie = index le plus fréquent
parmi les pixels NON-transparents du bloc. Si tous transparents, on sort transparent.
"""
from __future__ import annotations
import numpy as np

TARGET = 32


def downsample_indices(idx: np.ndarray) -> np.ndarray:
    """idx : array (H, W) uint8, valeurs 0..15. H == W et H multiple de 32.
    Retourne (32, 32) uint8 par mode-par-bloc.
    """
    H, W = idx.shape
    assert H == W, f"non-carré {idx.shape}"
    if H == TARGET:
        return idx.copy()
    if H % TARGET != 0:
        raise ValueError(f"taille {H} non multiple de {TARGET}")
    k = H // TARGET  # facteur de réduction

    # Reshape en blocs : (32, k, 32, k) -> (32, 32, k*k)
    blocks = idx.reshape(TARGET, k, TARGET, k).transpose(0, 2, 1, 3).reshape(TARGET, TARGET, k * k)

    # Pour chaque bloc, mode parmi les pixels non-transparents (idx > 0).
    out = np.zeros((TARGET, TARGET), dtype=np.uint8)
    # Vectorisé : on calcule la fréquence de chaque index 0..15 par bloc via bincount
    # via une approche par one-hot peu coûteuse pour 16 classes.
    one_hot = np.eye(16, dtype=np.int32)[blocks]  # (32, 32, k*k, 16)
    counts = one_hot.sum(axis=2)  # (32, 32, 16)
    # Force la classe 0 à -1 pour qu'elle ne gagne que si elle est seule
    opaque_counts = counts.copy()
    opaque_counts[:, :, 0] = 0
    # argmax sur 1..15 ; si tout est nul -> on garde 0 (transparent)
    has_opaque = opaque_counts.sum(axis=2) > 0
    out_opaque = np.argmax(opaque_counts, axis=2).astype(np.uint8)
    out[has_opaque] = out_opaque[has_opaque]
    return out
