"""Splits train/val stratifiés par (action, source).

Tailles : 85% / 15%. Indices déterministes via seed. Pas de set 'eval' separe : le DPO
best-of-2 se juge en arene sur du potentiellement overfit -> un test pristine ne donnerait
pas la comparaison propre qu'il suppose. On garde un SEUL hold-out (val) pour les
indicateurs, protege du DPO (campagnes tirees du TRAIN uniquement, cf dpo_campaign).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import json
from pathlib import Path


def make_splits(meta: list[dict], seed: int = 0,
                ratios: tuple[float, float] = (0.85, 0.15)) -> dict[str, list[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        key = (m.get("action"), m.get("source"))
        buckets[key].append(i)

    train, val = [], []
    for key, idxs in buckets.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_train = int(round(len(idxs) * ratios[0]))
        train.extend(idxs[:n_train].tolist())
        val.extend(idxs[n_train:].tolist())

    return {"train": sorted(train), "val": sorted(val)}


def save_splits(splits: dict[str, list[int]], out_path: Path) -> None:
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
