"""Audit visuel : génère une grille PNG d'échantillons aléatoires pour vérif humaine.

Sortie : data/audit_sample.png — grille de N cycles, chaque ligne = un cycle,
colonnes = frames du cycle (avec padding visuel pour cycles courts).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset.storage import load_dataset


def render_cycle(indices: np.ndarray, length: int, palette: np.ndarray) -> np.ndarray:
    """indices (max_frames, 32, 32) -> (32, length*32, 4) RGBA"""
    rgba_palette = np.concatenate([palette, np.full((16, 1), 255, dtype=np.uint8)], axis=1)
    rgba_palette[0] = (0, 0, 0, 0)  # index 0 = transparent
    frames_rgba = rgba_palette[indices[:length]]  # (length, 32, 32, 4)
    return np.concatenate(list(frames_rgba), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/audit_sample.png"))
    ap.add_argument("--n", type=int, default=64, help="nombre de cycles à échantillonner")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-source", action="store_true",
                    help="échantillonne équitablement par source")
    args = ap.parse_args()

    cycles, lengths, palettes, meta = load_dataset(args.data)
    rng = np.random.default_rng(args.seed)

    if args.per_source:
        by_src: dict[str, list[int]] = {}
        for i, m in enumerate(meta):
            by_src.setdefault(m["source"], []).append(i)
        per = max(1, args.n // len(by_src))
        idxs = []
        for src, lst in by_src.items():
            lst = np.array(lst)
            rng.shuffle(lst)
            idxs.extend(lst[:per].tolist())
        idxs = np.array(idxs)
    else:
        idxs = rng.choice(len(cycles), args.n, replace=False)

    max_frames = cycles.shape[1]
    rows = []
    bg = np.full((32, max_frames * 32, 4), 60, dtype=np.uint8)
    bg[:, :, 3] = 255  # gris sombre opaque
    for i in idxs:
        row = bg.copy()
        rendered = render_cycle(cycles[i], int(lengths[i]), palettes[i])
        row[:, :rendered.shape[1]] = rendered
        rows.append(row)

    grid = np.concatenate(rows, axis=0)  # (N*32, max_frames*32, 4)
    Image.fromarray(grid, mode="RGBA").save(args.out)
    print(f"saved {args.out} ({grid.shape[1]}x{grid.shape[0]}, {len(idxs)} cycles)")

    # Aussi, stats résumé
    from collections import Counter
    src_counts = Counter(m["source"] for m in meta)
    action_counts = Counter(m["action"] for m in meta)
    len_counts = Counter(int(l) for l in lengths)
    print("sources:", dict(src_counts))
    print("actions:", dict(action_counts))
    print("lengths:", dict(sorted(len_counts.items())))


if __name__ == "__main__":
    main()
