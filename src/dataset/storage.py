"""Sauvegarde / chargement du dataset traité.

Format :
  - dataset.npz : cycles (N, MAX_FRAMES, 32, 32) uint8, lengths (N,) uint8, palettes (N, 16, 3) uint8
  - dataset.meta.json : liste de N dicts métadonnées (alignée à l'axe 0)
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from .types import ProcessedCycle


def save_dataset(cycles: list[ProcessedCycle], out_dir: Path, max_frames: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(cycles)
    cycles_arr = np.zeros((N, max_frames, 32, 32), dtype=np.uint8)
    lengths = np.zeros((N,), dtype=np.uint8)
    palettes = np.zeros((N, 16, 3), dtype=np.uint8)
    meta = []
    for i, c in enumerate(cycles):
        L = c.indices.shape[0]
        if L > max_frames:
            raise ValueError(f"cycle {i} a {L} frames > max_frames={max_frames}")
        cycles_arr[i, :L] = c.indices
        lengths[i] = L
        palettes[i] = c.palette_rgb
        meta.append(c.metadata)
    np.savez_compressed(out_dir / "dataset.npz",
                        cycles=cycles_arr, lengths=lengths, palettes=palettes)
    (out_dir / "dataset.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_dataset(in_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    data = np.load(in_dir / "dataset.npz")
    meta = json.loads((in_dir / "dataset.meta.json").read_text(encoding="utf-8"))
    return data["cycles"], data["lengths"], data["palettes"], meta
