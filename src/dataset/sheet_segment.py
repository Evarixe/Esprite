"""Segmentation deterministe de sprite-sheets a fond transparent (connected-components).

Outil de PLOMBERIE : isole chaque frame d'un sheet via scipy.ndimage (bounding-boxes),
sans aucune analyse visuelle. Le groupage en cycles et le tagging action/direction sont
faits par un HUMAIN via le labeler (cf sheet_labeler / arene), pas ici.

Pipeline :
  mask alpha>0 -> (dilatation optionnelle pour recoller epee/bouclier au corps)
  -> ndi.label -> bounding-boxes -> filtre par taille (ecarte bruit + cartouche credit)
  -> ordre de lecture (par rangees, gauche->droite).

Usage :
    from dataset.sheet_segment import segment_sheet, cluster_rows, extract_frame
    boxes = segment_sheet("runs/.../link_cap.png")
    rows = cluster_rows(boxes)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from PIL import Image


@dataclass(frozen=True)
class Box:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def load_rgba(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def foreground_mask(sheet: np.ndarray, bg_tol: int = 16) -> np.ndarray:
    """Masque booleen du foreground. Fond transparent (alpha=0) -> alpha>0. Sinon
    (sheet opaque RGB) -> pixels != couleur de FOND, estimee comme la couleur la plus
    frequente (mode) et non le coin : le coin est peu fiable (cf sheets TSR ou le coin
    differe du fond). Les sprites sont epars, donc le mode = le fond."""
    alpha = sheet[..., 3]
    if alpha.max() > 0 and alpha.min() == 0:
        return alpha > 0
    flat = sheet[..., :3].reshape(-1, 3)
    uniq, cnt = np.unique(flat, axis=0, return_counts=True)
    bg = uniq[cnt.argmax()].astype(int)
    return np.any(np.abs(sheet[..., :3].astype(int) - bg) > bg_tol, axis=2)


def segment_sheet(path: str | Path, *, dilate: int = 2, min_px: int = 8,
                  max_px: int = 40, bg_tol: int = 16) -> list[Box]:
    """Retourne les bounding-boxes des sprites d'un sheet, en ordre de lecture.

    dilate : iterations de dilatation binaire avant labeling (recolle les parties
             disjointes d'un sprite ; 0 pour desactiver). max_px ecarte le cartouche
             credit et les fusions multi-frames ; min_px ecarte le bruit."""
    sheet = load_rgba(path)
    mask = foreground_mask(sheet, bg_tol)
    if dilate > 0:
        mask_d = ndi.binary_dilation(mask, iterations=dilate)
    else:
        mask_d = mask
    lab, n = ndi.label(mask_d)
    boxes: list[Box] = []
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        # reserre la boite sur le masque NON dilate (bornes pixel exactes)
        y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
        sub = mask[y0:y1, x0:x1]
        if not sub.any():
            continue
        rr = np.where(sub.any(axis=1))[0]
        cc = np.where(sub.any(axis=0))[0]
        b = Box(y0 + int(rr[0]), x0 + int(cc[0]), y0 + int(rr[-1]) + 1, x0 + int(cc[-1]) + 1)
        if min_px <= b.h <= max_px and min_px <= b.w <= max_px:
            boxes.append(b)
    return reading_order(boxes)


def reading_order(boxes: list[Box], row_tol: float = 0.5) -> list[Box]:
    """Trie en ordre de lecture : par rangees (regroupees si leurs y se recouvrent),
    puis gauche->droite dans chaque rangee."""
    return [b for row in cluster_rows(boxes, row_tol) for b in row]


def cluster_rows(boxes: list[Box], row_tol: float = 0.5) -> list[list[Box]]:
    """Regroupe les boites en rangees : deux boites sont sur la meme rangee si leurs
    intervalles verticaux se recouvrent d'au moins row_tol * min(hauteurs). Rangees
    triees haut->bas, boites triees gauche->droite dans chaque rangee."""
    if not boxes:
        return []
    rows: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        placed = False
        for row in rows:
            ry0 = min(x.y0 for x in row)
            ry1 = max(x.y1 for x in row)
            overlap = min(b.y1, ry1) - max(b.y0, ry0)
            if overlap >= row_tol * min(b.h, ry1 - ry0):
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    rows.sort(key=lambda row: min(x.y0 for x in row))
    for row in rows:
        row.sort(key=lambda b: b.x0)
    return rows


def extract_frame(sheet: np.ndarray, box: Box) -> np.ndarray:
    """Sous-image RGBA (h, w, 4) reserree sur la boite."""
    return sheet[box.y0:box.y1, box.x0:box.x1].copy()
