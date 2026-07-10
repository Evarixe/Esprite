"""Tableau d'etat du dataset : quantites par tag/action, nombre de frames et source,
ventilees par split. Pour voir OU sont les limites (strates rares, bucket long, actions
peu representees en val) et decider sur du reel plutot que sur des minimums arbitraires.

Usage :
    set PYTHONPATH=src
    uv run python tools/dataset_stats.py [--data data]
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from genmodel.vocab import SOURCE_ACTION_MAP


def token_action(raw: str) -> str:
    return SOURCE_ACTION_MAP.get(raw, raw)


def bucket(n: int) -> str:
    return "short(2)" if n <= 2 else "long(3-16)"


def _split_of(idx: int, splits: dict) -> str:
    for name, ids in splits.items():
        if idx in ids:
            return name
    return "?"


def table(title: str, rows: dict, split_names: list[str]):
    """rows : {key -> Counter(split->count)}. Affiche key + colonnes par split + total."""
    keys = sorted(rows, key=lambda k: -sum(rows[k].values()))
    w = max([len(str(k)) for k in keys] + [len(title)])
    hdr = f"{title:<{w}} | " + " | ".join(f"{s:>10}" for s in split_names) + " | " + f"{'TOTAL':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for k in keys:
        c = rows[k]
        tot = sum(c.values())
        line = f"{str(k):<{w}} | " + " | ".join(f"{c.get(s,0):>10}" for s in split_names) + " | " + f"{tot:>8}"
        print(line)
    tot_row = Counter()
    for c in rows.values():
        tot_row.update(c)
    print("-" * len(hdr))
    print(f"{'TOTAL':<{w}} | " + " | ".join(f"{tot_row.get(s,0):>10}" for s in split_names) + " | " + f"{sum(tot_row.values()):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()

    meta = json.loads((args.data / "dataset.meta.json").read_text(encoding="utf-8"))
    lengths = np.load(args.data / "dataset.npz")["lengths"]
    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))
    split_names = list(splits.keys())
    idx2split = {}
    for name, ids in splits.items():
        for i in ids:
            idx2split[i] = name

    by_action = defaultdict(Counter)
    by_nframes = defaultdict(Counter)
    by_source = defaultdict(Counter)
    by_action_bucket = defaultdict(Counter)   # (action, bucket) -> split counts
    for i, m in enumerate(meta):
        sp = idx2split.get(i, "?")
        a = token_action(m.get("action"))
        n = int(lengths[i])
        by_action[a][sp] += 1
        by_nframes[n][sp] += 1
        by_source[m.get("source")][sp] += 1
        by_action_bucket[f"{a} / {bucket(n)}"][sp] += 1

    print(f"Dataset : {len(meta)} cycles | splits " + " ".join(f"{s}={len(ids)}" for s, ids in splits.items()))
    table("ACTION (tag modele)", by_action, split_names)
    table("N_FRAMES", by_nframes, split_names)
    table("SOURCE", by_source, split_names)
    table("ACTION x BUCKET", by_action_bucket, split_names)

    # Alerte visibilite (informative, PAS de forcage) : actions dont le val est trop mince.
    print("\nLimites d'assessabilite (val faible = mesure de qualite fragile pour ce tag) :")
    for a in sorted(by_action, key=lambda k: by_action[k].get("val", 0)):
        v = by_action[a].get("val", 0)
        tr = by_action[a].get("train", 0)
        flag = "  <-- val mince" if v < 20 else ""
        print(f"  {a:<10} train={tr:>5}  val={v:>4}{flag}")


if __name__ == "__main__":
    main()
