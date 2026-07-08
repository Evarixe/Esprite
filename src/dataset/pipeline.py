"""Orchestrateur : RawCycle -> ProcessedCycle (quantif palette + downsample 32x32)."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
from tqdm import tqdm

from .types import RawCycle, ProcessedCycle
from .palette import build_palette, quantize_frame
from .downsample import downsample_indices, TARGET
from .sources import iter_all_sources


def process_cycle(cycle: RawCycle, rng: np.random.Generator | None = None) -> ProcessedCycle:
    palette = build_palette(cycle, rng=rng)
    out_frames = []
    for frame in cycle.frames:
        q = quantize_frame(frame, palette)            # (H, W) en indices natifs
        d = downsample_indices(q)                     # (32, 32)
        out_frames.append(d)
    indices = np.stack(out_frames, axis=0)             # (n_frames, 32, 32)
    return ProcessedCycle(indices=indices, palette_rgb=palette, metadata=dict(cycle.metadata))


def run_pipeline(assets_root: Path, max_anim_frames: int = 16, seed: int = 0) -> list[ProcessedCycle]:
    rng = np.random.default_rng(seed)
    out: list[ProcessedCycle] = []
    for raw in tqdm(iter_all_sources(assets_root, max_anim_frames=max_anim_frames),
                    desc="processing", unit="cycle"):
        try:
            out.append(process_cycle(raw, rng=rng))
        except Exception as e:
            print(f"[skip] {raw.metadata} -> {type(e).__name__}: {e}")
    return out
