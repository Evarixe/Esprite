"""Downsampling mode-par-bloc pour préserver la palette indexée.

Sprites natifs en 32x32 (HGSS), 64x64 (Emerald combat), 96x96 (Emerald animated).
Ratios entiers 1:1, 2:1, 3:1 vers 32x32.

Mode-par-bloc : pour chaque bloc kxk, la valeur de sortie = index le plus fréquent
parmi les pixels NON-transparents du bloc. Si tous transparents, on sort transparent.

Detection de phase (`detect_phase`) : outil d'INGESTION par source. Une source peut
etre du "super-pixel" (art upscale par un facteur entier), parfois avec un offset
d'1+ px dans la planche. Dans ce cas il existe une phase (dy,dx) ou chaque bloc kxk
est UNIFORME (variance ~0) -> le downscale kxk a cette phase est SANS PERTE. Sinon la
source est du pixel-art natif haute def et le kxk perd reellement du detail (a montrer
en planche avant integration). A brancher dans chaque parser de source, cf le probe LPC
qui a revele "natif 64px, aucune phase lossless" -> perte reelle assumee.
"""
from __future__ import annotations
import numpy as np

TARGET = 32
LOSSLESS_EPS = 0.5   # variance intra-bloc en-dessous = source super-pixel alignee (downscale sans perte)


def _block_variance(arr: np.ndarray, k: int, dy: int, dx: int) -> float:
    """Variance moyenne intra-bloc kxk a la phase (dy,dx). Marche sur un array 2D
    (indices) ou 3D (H,W,C, ex RGBA). ~0 => blocs uniformes => alignement super-pixel."""
    a = arr[dy:, dx:]
    h = (a.shape[0] // k) * k
    w = (a.shape[1] // k) * k
    if h == 0 or w == 0:
        return float("inf")
    a = a[:h, :w].astype(np.float32)
    if a.ndim == 2:
        b = a.reshape(h // k, k, w // k, k)
        return float(b.var(axis=(1, 3)).mean())
    b = a.reshape(h // k, k, w // k, k, a.shape[2])
    return float(b.var(axis=(1, 3)).mean())


def detect_phase(arr: np.ndarray, k: int, eps: float = LOSSLESS_EPS):
    """Cherche la phase (dy,dx) dans [0,k)^2 qui minimise la variance intra-bloc kxk.

    Retourne (phase, variance, lossless). `lossless=True` (variance < eps) => la source
    est un upscale entier xk aligne a cette phase : downscaler kxk n'y perd rien (1 pixel
    par bloc suffit). `lossless=False` => detail natif reel sous le bloc -> perte a montrer.
    Passe un seul array representatif (ex une frame nette) ou empile-en plusieurs en (N*H,W)."""
    if k <= 1:
        return (0, 0), 0.0, True
    best_phase, best_var = (0, 0), float("inf")
    for dy in range(k):
        for dx in range(k):
            v = _block_variance(arr, k, dy, dx)
            if v < best_var:
                best_phase, best_var = (dy, dx), v
    return best_phase, best_var, best_var < eps


def downscale_auto(idx: np.ndarray, k: int):
    """Downscale METICULEUX : detecte la phase et ne l'applique QUE si lossless.

    Regle apprise a la dure : une phase (dy,dx) != (0,0) rogne l'image de dy/dx px.
    Ne l'appliquer que sur une source super-pixel avec bordure (lossless=True, la
    variance ~0 garantit qu'on ne perd rien). Sinon -> phase (0,0), on garde les dims
    exactes et on assume la perte du natif haute def (a montrer en planche).

    NB : la phase se detecte/applique idealement au niveau de la PLANCHE avant
    l'extraction des cellules (sinon on rogne chaque cellule). Retourne (out, info)
    avec info = (phase_utilisee, variance, lossless)."""
    phase, var, lossless = detect_phase(idx, k)
    use = phase if lossless else (0, 0)
    return mode_downscale(idx, k, phase=use), (use, var, lossless)


def mode_downscale(idx: np.ndarray, k: int, phase: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Mode-par-bloc kxk sur indices palette (0=transparent, ne gagne que seul), a une
    phase donnee. Generalise `downsample_indices` (k quelconque + offset). Sortie de
    taille (floor((H-dy)/k), floor((W-dx)/k)). Utilise detect_phase pour aligner la phase
    sur la grille native d'une source upscalee."""
    dy, dx = phase
    a = idx[dy:, dx:]
    h = (a.shape[0] // k) * k
    w = (a.shape[1] // k) * k
    a = a[:h, :w]
    oh, ow = h // k, w // k
    if k == 1:
        return a.copy()
    blocks = a.reshape(oh, k, ow, k).transpose(0, 2, 1, 3).reshape(oh, ow, k * k)
    counts = np.eye(16, dtype=np.int32)[blocks].sum(axis=2)   # (oh, ow, 16)
    counts[:, :, 0] = 0                                        # transparent ne gagne que seul
    out = np.zeros((oh, ow), dtype=np.uint8)
    has_opaque = counts.sum(axis=2) > 0
    out[has_opaque] = np.argmax(counts, axis=2).astype(np.uint8)[has_opaque]
    return out


def downsample_indices(idx: np.ndarray) -> np.ndarray:
    """idx : array (H, W) uint8, valeurs 0..15. H == W et H multiple de 32.
    Retourne (32, 32) uint8 par mode-par-bloc (phase 0). Cas canonique du pipeline ;
    pour une source upscalee avec offset, detecter la phase (detect_phase) puis appeler
    mode_downscale(idx, k, phase) directement.
    """
    H, W = idx.shape
    assert H == W, f"non-carré {idx.shape}"
    if H == TARGET:
        return idx.copy()
    if H % TARGET != 0:
        raise ValueError(f"taille {H} non multiple de {TARGET}")
    return mode_downscale(idx, H // TARGET, phase=(0, 0))
