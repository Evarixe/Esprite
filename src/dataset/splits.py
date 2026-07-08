"""Splits train/val/eval stratifiés par (action, source).

Tailles : 85% / 10% / 5%. Indices déterministes via seed.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import json
from pathlib import Path


def make_splits(meta: list[dict], seed: int = 0,
                ratios: tuple[float, float, float] = (0.85, 0.10, 0.05)) -> dict[str, list[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        key = (m.get("action"), m.get("source"))
        buckets[key].append(i)

    train, val, evl = [], [], []
    for key, idxs in buckets.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train.extend(idxs[:n_train].tolist())
        val.extend(idxs[n_train:n_train + n_val].tolist())
        evl.extend(idxs[n_train + n_val:].tolist())

    return {"train": sorted(train), "val": sorted(val), "eval": sorted(evl)}


def save_splits(splits: dict[str, list[int]], out_path: Path) -> None:
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
