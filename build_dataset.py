"""Point d'entrée : construit le dataset à partir de assets/ vers data/.

Usage :
    python build_dataset.py [--assets ASSETS_ROOT] [--out DATA_DIR] [--seed N]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset.pipeline import run_pipeline
from dataset.storage import save_dataset
from dataset.splits import make_splits, save_splits
from genmodel.attributes import build_identity_registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-anim-frames", type=int, default=16)
    args = ap.parse_args()

    print(f"[pipeline] assets={args.assets} out={args.out}")
    cycles = run_pipeline(args.assets, max_anim_frames=args.max_anim_frames, seed=args.seed)
    print(f"[pipeline] {len(cycles)} cycles processed")

    save_dataset(cycles, args.out)
    print(f"[pipeline] saved dataset.npz + dataset.meta.json")

    splits = make_splits([c.metadata for c in cycles], seed=args.seed)
    save_splits(splits, args.out / "splits.json")
    print(f"[pipeline] splits: train={len(splits['train'])} val={len(splits['val'])}")

    registry = build_identity_registry([c.metadata for c in cycles])
    (args.out / "identity_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"[pipeline] identity registry: {len(registry)} identites (dont __none__)")


if __name__ == "__main__":
    main()
