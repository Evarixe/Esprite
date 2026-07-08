"""Échantillonne plusieurs cycles depuis un checkpoint et rend une grille PNG.

Layout par ligne (1 cycle) :
  [ref?]  [gen frame 0]  [gen frame 1]  ...  [gen frame N-1]   (+ slots vides pour alignement)

Le nombre de frames générées est libre — le modèle peut produire moins ou plus que
demandé. Les frames invalides (structure cassée) sont marquées par un overlay rouge
hachuré.

Légende status :
  valid          → pas d'overlay
  short          → overlay rouge clair  (le modèle a coupé avant 1024 pixels)
  no_separator   → overlay orange       (1024 pixels OK, mais pas de FRAME_SEP/SEQ_END derrière)
  unknown_token  → overlay rouge foncé  (token hors-vocab attendu, frame coupée)

Sortie : PNG + un résumé texte sur stdout (cycle, n_frames généré, statuses, stop_reason).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
import time
from genmodel.model import SpriteTransformer
from genmodel.sample import generate, GenRequest, build_prefix
from genmodel.graph_sampler import GraphSampler, GraphSamplerPool
from genmodel.vocab import PIXELS_PER_FRAME, MAX_FRAMES


SPRITE = 32
PAD = 2
BG = (60, 60, 60, 255)


def render_palette_image(indices: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """indices (H, W) uint8 0..15, palette_rgb (16, 3) uint8 -> (H, W, 4) RGBA."""
    rgba = np.concatenate([palette_rgb, np.full((16, 1), 255, dtype=np.uint8)], axis=1)
    rgba[0] = (0, 0, 0, 0)
    return rgba[indices]


def flatten_on_bg(rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite une frame RGBA sur le fond BG → RGBA opaque. La transparence
    (index 0) devient le gris du fond au lieu de blanc-viewer : une frame entièrement
    transparente se lit alors comme une case unie (artefact 'frame vide' visible)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgba[..., :3].astype(np.float32) * a + np.array(BG[:3], np.float32) * (1 - a)).astype(np.uint8)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def make_empty_slot() -> np.ndarray:
    """Slot vide (pas de frame ici) — gris uni, légèrement plus foncé que le BG."""
    img = np.zeros((SPRITE, SPRITE, 4), dtype=np.uint8)
    img[..., :3] = (30, 32, 38)
    img[..., 3] = 255
    return img


_STATUS_COLOR = {
    "short":         (200, 70, 70, 110),    # rouge clair semi-transparent
    "no_separator":  (220, 140, 60, 100),   # orange
    "unknown_token": (140, 30, 30, 130),    # rouge foncé
}


def add_status_overlay(img: np.ndarray, status: str) -> np.ndarray:
    """Si status != 'valid', dessine un cadre coloré + 3 diagonales hachurées."""
    if status == "valid":
        return img
    out = img.copy()
    color = _STATUS_COLOR.get(status, (200, 70, 70, 110))
    # Cadre 1px
    out[0, :] = color; out[-1, :] = color
    out[:, 0] = color; out[:, -1] = color
    # Diagonales hachurées (pas de PIL.draw pour éviter l'overhead, on fait à la main)
    for k in range(0, SPRITE, 8):
        for d in range(SPRITE):
            x = (k + d) % SPRITE
            y = d
            # mix alpha
            a = color[3] / 255.0
            out[y, x, :3] = (out[y, x, :3] * (1 - a) + np.array(color[:3]) * a).astype(np.uint8)
    return out


def build_grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    max_cols = max(len(r) for r in rows)
    cell = SPRITE + PAD
    W = max_cols * cell + PAD
    H = len(rows) * cell + PAD
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., :3] = BG[:3]
    out[..., 3] = BG[3]
    empty = make_empty_slot()
    for r, row in enumerate(rows):
        for c in range(max_cols):
            y0 = PAD + r * cell
            x0 = PAD + c * cell
            img = row[c] if c < len(row) else empty
            out[y0:y0 + SPRITE, x0:x0 + SPRITE] = img
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8, help="Nombre de cycles à échantillonner")
    ap.add_argument("--frames", type=int, default=2, help="Frames cibles (cap haut côté sécurité)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--upscale", type=int, default=4, help="Agrandissement nearest-neighbor du PNG final")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpriteTransformer().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    # Cast bf16 statique pour activer le path CUDA Graphs dans generate()
    if device == "cuda":
        model = model.to(torch.bfloat16)

    data = np.load(args.data / "dataset.npz")
    cycles_all = data["cycles"]
    palettes_all = data["palettes"]
    meta = json.loads((args.data / "dataset.meta.json").read_text(encoding="utf-8"))
    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))
    val_idxs = splits["val"]

    from genmodel.vocab import SOURCE_ACTION_MAP

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(val_idxs, args.n, replace=False)

    # Pool de buckets réutilisé pour tous les cycles : démarre petit, promeut
    # dynamiquement selon la longueur réellement générée. Captures lazy + cachées.
    pool = None
    if device == "cuda":
        pool = GraphSamplerPool(model)
        print(f"GraphSamplerPool buckets: {pool.buckets}")

    rows = []
    timings = {}
    per_cycle_times = []
    print(f"=== sampling {args.n} cycles, target frames={args.frames}, T={args.temperature} ===")
    for k, ci in enumerate(chosen):
        m = meta[ci]
        ref_frame = cycles_all[ci, 0] if not args.no_ref else None
        req = GenRequest(
            action=SOURCE_ACTION_MAP[m["action"]],
            direction=m.get("direction"),
            frames=args.frames,
            reference=ref_frame,
            temperature=args.temperature,
            seed=args.seed + k,
        )
        t0 = time.perf_counter()
        result = generate(model, req, device=device, pool=pool, timings=timings)
        cycle_time = time.perf_counter() - t0
        per_cycle_times.append(cycle_time)
        palette = palettes_all[ci]

        row = []
        if ref_frame is not None:
            row.append(flatten_on_bg(render_palette_image(ref_frame, palette)))
        for fr in result.frames:
            base = flatten_on_bg(render_palette_image(fr.indices, palette))
            row.append(add_status_overlay(base, fr.status))
        rows.append(row)

        statuses = ", ".join(f"{fr.status}" for fr in result.frames) or "(no frames)"
        print(f"  cycle {ci:>4} (poke_id={m.get('pokemon_id')}, dir={m.get('direction')}, "
              f"act={m['action']}): {len(result.frames)} frames "
              f"[{result.n_valid} valid / {result.n_invalid} invalid] "
              f"stop={result.stop_reason} | {statuses}")

    grid = build_grid(rows)
    img = Image.fromarray(grid, mode="RGBA")
    if args.upscale > 1:
        img = img.resize((grid.shape[1] * args.upscale, grid.shape[0] * args.upscale), Image.NEAREST)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"\nsaved {args.out} ({img.width}x{img.height}, upscale ×{args.upscale})")

    # ===== Breakdown timings =====
    total = sum(per_cycle_times)
    n = len(per_cycle_times)
    print(f"\n=== Timing breakdown ({n} cycles) ===")
    print(f"  Total wall              : {total:7.2f}s  ({total/n:.2f}s / cycle)")
    print(f"    capture (1 fois)      : {timings.get('capture', 0):7.2f}s")
    print(f"    prefix forward (eager): {timings.get('prefix_forward', 0):7.2f}s")
    print(f"    setup + load prefix   : {timings.get('setup_prefix', 0):7.2f}s")
    print(f"    sample loop           : {timings.get('sample_loop', 0):7.2f}s")
    n_tok = timings.get('n_tokens_emitted', 0)
    if n_tok > 0 and timings.get('sample_loop'):
        print(f"      {n_tok} tokens émis @ {timings['sample_loop']/n_tok*1000:.2f} ms/token")
    print(f"  Per-cycle times: {[f'{t:.2f}s' for t in per_cycle_times]}")


if __name__ == "__main__":
    main()
