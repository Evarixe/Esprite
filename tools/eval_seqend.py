"""Approche B (CLI) — mesure l'écart de position de <SEQ_END> sur un checkpoint.

Génère un échantillon équilibré court/long depuis le val set, demande à chaque fois
la vraie longueur N du cycle, et rapporte à quel point le modèle respecte N (position
réelle de SEQ_END vs cible N×1025).

Usage :
    set PYTHONPATH=src
    uv run python eval_seqend.py --ckpt runs/gen_4000/last.pt --n-per-bucket 16
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from genmodel.model import SpriteTransformer
from genmodel.data import CyclesTokenDataset
from genmodel.eval_metrics import seqend_gap_eval, pick_eval_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-per-bucket", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--no-ref", action="store_true", help="évalue la génération sans référence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="écrit le résumé JSON ici")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpriteTransformer().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)

    ds = CyclesTokenDataset(args.data, split=args.split, ref_prob=0.0, augment_palette=False)
    idx = pick_eval_indices(ds, n_per_bucket=args.n_per_bucket, seed=args.seed)
    print(f"[eval_seqend] {len(idx)} cycles ({args.split}), with_ref={not args.no_ref}, "
          f"temp={args.temperature}")

    summary = seqend_gap_eval(model, ds, idx, device=device, temperature=args.temperature,
                              with_ref=not args.no_ref, seed=args.seed)

    print(f"\n=== SEQ_END gap — {args.ckpt} ===")
    print(f"  générations           : {summary.n_total}")
    print(f"  SEQ_END émis           : {summary.n_seq_end} ({summary.frac_seq_end:.1%})")
    print(f"  écart absolu moyen     : {summary.mean_abs_gap:.1f} tokens")
    print(f"  écart signé moyen      : {summary.mean_signed_gap:+.1f} (négatif=trop court)")
    print(f"  écart absolu médian    : {summary.median_abs_gap:.1f}")
    print(f"  exact (gap==0)         : {summary.frac_exact:.1%}")
    print(f"  --- par N demandé ---")
    for n, st in summary.per_n.items():
        print(f"    N={n:2d} | n={st['count']:3d} | seq_end={st['frac_seq_end']:.0%} | "
              f"|gap|={st['mean_abs_gap']:8.1f} | signed={st['mean_signed_gap']:+9.1f}")

    if args.out is not None:
        payload = {
            "ckpt": str(args.ckpt),
            "n_total": summary.n_total, "n_seq_end": summary.n_seq_end,
            "frac_seq_end": summary.frac_seq_end,
            "mean_abs_gap": summary.mean_abs_gap, "mean_signed_gap": summary.mean_signed_gap,
            "median_abs_gap": summary.median_abs_gap, "frac_exact": summary.frac_exact,
            "per_n": summary.per_n,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\n[eval_seqend] résumé écrit dans {args.out}")


if __name__ == "__main__":
    main()
