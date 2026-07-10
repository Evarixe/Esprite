"""Ingestion des manifestes du labeler (sprite-sheets TSR) -> RawCycle.

Deterministe : lit un manifeste JSON (data/labels/<feuille>.json) + la sheet, extrait
chaque frame, l'ancre (anti-jitter) et la pad en 32x32 RGBA. Le miroir L<->R (global
`mirror_lr`, ou per-cycle pour l'ancien format) genere la direction opposee par flip.

Ancrage anti-jitter :
- VERTICAL : bande commune du cycle [min y0, max y1], calee en bas (pieds stables),
  chaque frame a son offset relatif -> mouvement vertical preserve.
- HORIZONTAL : si pas de grille regulier -> alignement par cellule (retire la foulee
  k*p, corps fixe, membres libres) ; sinon -> centroide des pixels opaques (stable).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from .types import RawCycle

TARGET = 32
_LR = {"left": "right", "right": "left"}


def _paste(canvas: np.ndarray, frame: np.ndarray, y: int, x: int):
    """Colle frame (RGBA) dans canvas 32x32 a (y,x), avec clipping."""
    h, w = frame.shape[:2]
    y, x = int(round(y)), int(round(x))
    dy0, dy1 = max(0, y), min(TARGET, y + h)
    dx0, dx1 = max(0, x), min(TARGET, x + w)
    if dy1 <= dy0 or dx1 <= dx0:
        return
    canvas[dy0:dy1, dx0:dx1] = frame[dy0 - y:dy1 - y, dx0 - x:dx1 - x]


def _centroid_x(frame: np.ndarray) -> float:
    col_mass = (frame[..., 3] > 0).sum(axis=0).astype(float)
    tot = col_mass.sum()
    if tot == 0:
        return frame.shape[1] / 2.0
    return float((np.arange(frame.shape[1]) * col_mass).sum() / tot)


def anchor_cycle(sheet: np.ndarray, boxes: list[list[int]]) -> list[np.ndarray]:
    """Extrait + ancre les frames d'un cycle -> liste de 32x32 RGBA."""
    n = len(boxes)
    y0 = np.array([b[0] for b in boxes]); x0 = np.array([b[1] for b in boxes])
    y1 = np.array([b[2] for b in boxes]); x1 = np.array([b[3] for b in boxes])
    vtop, vbot = int(y0.min()), int(y1.max())
    vh = min(TARGET, vbot - vtop)
    wmed = float(np.median(x1 - x0))
    base_pad = (TARGET - wmed) / 2.0

    # horizontal : alignement par cellule si pas regulier, sinon centroide
    cx = (x0 + x1) / 2.0
    xanchor = None
    if n >= 3:
        gaps = np.diff(cx)
        p = float(np.median(gaps))
        if p > 0 and (gaps.max() - gaps.min()) <= 0.25 * p:
            d = x0 - np.arange(n) * p
            ref = float(np.median(d))
            xanchor = base_pad + (d - ref)
    frames = [sheet[b[0]:b[2], b[1]:b[3]] for b in boxes]
    if xanchor is None:                                   # fallback centroide -> centre a 16
        xanchor = np.array([TARGET / 2.0 - _centroid_x(f) for f in frames])

    out = []
    for k, f in enumerate(frames):
        canvas = np.zeros((TARGET, TARGET, 4), np.uint8)
        cy = (TARGET - vh) + (int(y0[k]) - vtop)
        _paste(canvas, f, cy, xanchor[k])
        out.append(canvas)
    return out


def parse_tsr_labeled(manifest_path: str | Path, source: str = "tsr",
                      root: Path | None = None):
    """Yield des RawCycle depuis un manifeste du labeler. `source` = tag de source
    (ex 'tsr_zelda_minishcap'). `root` : base pour resoudre le chemin de sheet."""
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    sheet_path = Path(str(m["sheet"]).replace("\\", "/"))
    if root is not None and not sheet_path.is_absolute():
        sheet_path = root / sheet_path
    from PIL import Image
    sheet = np.array(Image.open(sheet_path).convert("RGBA"))
    character = manifest_path.stem
    mirror_lr = bool(m.get("mirror_lr", False))

    for cyc in m.get("cycles", []):
        action, direction = cyc["action"], cyc["direction"]
        frames = anchor_cycle(sheet, cyc["boxes"])
        meta = {"source": source, "action": action, "direction": direction,
                "character": character, "pokemon_id": None}
        yield RawCycle(frames=frames, metadata=dict(meta))
        # miroir : cycle L/R -> direction opposee (flip horizontal) si mirror_lr (ou
        # per-cycle pour l'ancien format de manifeste).
        do_mirror = (mirror_lr or cyc.get("mirror")) and direction in _LR
        if do_mirror:
            flipped = [f[:, ::-1].copy() for f in frames]
            mmeta = dict(meta); mmeta["direction"] = _LR[direction]
            yield RawCycle(frames=flipped, metadata=mmeta)
