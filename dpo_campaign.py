"""Campagne DPO -- genere des paires SAME-MODEL best-of-2 pour l'arene (vote simple).

Chaque item est genere 2x (2 seeds) depuis LE MEME checkpoint -> deux pseudo-checkpoints
`seedA`/`seedB` : la logique de duel de l'arene marche telle quelle, et les paires
votees sont same-model (longueurs similaires -> biais de longueur nul).

Pool = dataset COMPLET (best-of-2 ne compare rien a la verite terrain -> pas de fuite).
Variete forcee : stratification (source, action, direction), sujets distincts, fraction
with/no-ref imposee, memoire anti-repetition inter-passes (`shown_ledger.json`).

Usage :
    set PYTHONPATH=src
    uv run python dpo_campaign.py --ckpt runs/gen_20000/last.pt \
        --name dpo_cycle01 --n-items 50 --no-ref-frac 0.5 --temperature 0.7 \
        --ledger runs/dpo_loop/shown_ledger.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from genmodel.model import SpriteTransformer
from genmodel.sample import generate, GenRequest
from genmodel.graph_sampler import GraphSamplerPool
from genmodel.vocab import SOURCE_ACTION_MAP, PIXELS_PER_FRAME
from arena.campaign import b64_frame
from monitor.heartbeat import HeartbeatWriter
from monitor.gpu import gpu_stats

SEED_A, SEED_B = "seedA", "seedB"


def subject_of(m: dict) -> str:
    """Cle de sujet pour la variete : pokemon ou personnage."""
    return f"pk{m['pokemon_id']}" if m.get("pokemon_id") is not None else f"ch:{m.get('character')}"


def pick_items(meta, lengths, pool_idx, n_items, no_ref_frac, seed, shown: set, max_frames):
    """Selection variete-forcee sur le pool complet : round-robin par strate
    (source, action, direction), 1 sujet distinct par item, anti-repetition via
    `shown`, fraction no-ref imposee et decorrelee des strates."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[int]] = {}
    for ci in pool_idx:
        if int(lengths[ci]) > max_frames:
            continue
        if ci in shown:
            continue
        m = meta[ci]
        key = (m["source"], m.get("action"), m.get("direction"))
        strata.setdefault(key, []).append(ci)
    for k in strata:
        strata[k] = list(rng.permutation(strata[k]))

    keys = sorted(strata.keys())
    rng.shuffle(keys)
    chosen, used_subjects = [], set()
    # round-robin sur les strates, 1 sujet distinct par item
    while len(chosen) < n_items and any(strata[k] for k in keys):
        for k in keys:
            if len(chosen) >= n_items:
                break
            while strata[k]:
                ci = strata[k].pop()
                subj = subject_of(meta[ci])
                if subj not in used_subjects:
                    used_subjects.add(subj); chosen.append(ci); break

    items = []
    for i, ci in enumerate(chosen):
        m = meta[ci]
        period = max(1, round(1.0 / no_ref_frac)) if no_ref_frac > 0 else 0
        with_ref = not (period and i % period == 0)
        items.append({
            "item_id": f"it{i:03d}", "cycle_idx": int(ci),
            "source": m["action"], "action": SOURCE_ACTION_MAP[m["action"]],
            "direction": m.get("direction"), "n_frames": int(lengths[ci]),
            "with_ref": bool(with_ref), "pokemon_id": m.get("pokemon_id"),
            "subject": subject_of(m),
        })
    return items


def gen_side(model, items, cycles_all, side_seed, temperature, out_dir, side_id, device, pool,
             hb=None, done0=0, total=0):
    gen_dir = out_dir / "gens" / side_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(items):
        ref = cycles_all[it["cycle_idx"], 0] if it["with_ref"] else None
        req = GenRequest(action=it["action"], direction=it["direction"], frames=it["n_frames"],
                         reference=ref, temperature=temperature, seed=side_seed + k)
        res = generate(model, req, device=device, pool=pool)
        exp = it["n_frames"] * (PIXELS_PER_FRAME + 1)
        act = len(res.raw_tokens) if res.stop_reason == "seq_end" else None
        (gen_dir / f"{it['item_id']}.json").write_text(json.dumps({
            "item_id": it["item_id"], "ckpt_id": side_id,
            "frames": [b64_frame(fr.indices) for fr in res.frames],
            "statuses": [fr.status for fr in res.frames],
            "n_pixels": [fr.n_pixels for fr in res.frames],
            "stop_reason": res.stop_reason, "n_valid": res.n_valid, "n_invalid": res.n_invalid,
            "n_raw_tokens": len(res.raw_tokens),
            "seqend_gap": (act - exp) if act is not None else None,
        }), encoding="utf-8")
        if hb is not None:
            hb.update(step=done0 + k + 1, total_steps=total, bucket="gen",
                      loss=0.0, top1=0.0, n_tokens=0, dt=0.0, lr=0.0,
                      gpu=gpu_stats(), force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--name", required=True, help="dossier runs/arena/<name> (si --out-dir absent)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Dossier des DONNEES de campagne (campaign.json, gens, votes). "
                         "Defaut runs/arena/<name>. En lignee : RUN/campaigns/cNN.")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Dossier LIGNEE ou battre heartbeat/config (dashboard unique). "
                         "Defaut = out-dir (standalone).")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--no-ref-frac", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--ledger", type=Path, default=None,
                    help="JSON des cycle_idx deja montres (anti-repetition inter-passes)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir if args.out_dir is not None else Path("runs") / "arena" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir if args.run_dir is not None else out_dir   # heartbeat/config lignee
    run_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data / "dataset.npz")
    cycles_all, lengths, palettes = data["cycles"], data["lengths"], data["palettes"]
    meta = json.loads((args.data / "dataset.meta.json").read_text(encoding="utf-8"))
    pool_idx = list(range(len(meta)))   # POOL COMPLET (tous splits)

    shown = set()
    if args.ledger and args.ledger.exists():
        shown = set(json.loads(args.ledger.read_text()))
    items = pick_items(meta, lengths, pool_idx, args.n_items, args.no_ref_frac,
                       args.seed, shown, args.max_frames)
    n_ref = sum(it["with_ref"] for it in items)
    print(f"[dpo-camp] {len(items)} items ({n_ref} with-ref / {len(items)-n_ref} no-ref), "
          f"{len(set(it['subject'] for it in items))} sujets distincts")

    for it in items:
        it["palette"] = palettes[it["cycle_idx"]].tolist()
        it["ref"] = b64_frame(cycles_all[it["cycle_idx"], 0])

    model = SpriteTransformer().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    model.eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)
    pool = GraphSamplerPool(model) if device == "cuda" else None

    # Monitoring de la generation (barre live " gen X/2N " dans le dashboard).
    total = 2 * len(items)
    (run_dir / "config.json").write_text(json.dumps(
        {"mode": "gen_campaign", "total_steps": total, "name": args.name,
         "ckpt": str(args.ckpt), "temperature": args.temperature}, indent=2))
    hb = HeartbeatWriter(run_dir / "heartbeat.json", min_interval_s=0.25)

    t0 = time.time()
    base = args.seed * 100_000
    gen_side(model, items, cycles_all, base,          args.temperature, out_dir, SEED_A, device, pool, hb, 0, total)
    gen_side(model, items, cycles_all, base + 50_000, args.temperature, out_dir, SEED_B, device, pool, hb, len(items), total)
    print(f"[dpo-camp] 2x{len(items)} generations en {time.time()-t0:.1f}s")

    campaign = {
        "name": args.name, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": args.temperature, "seed": args.seed, "dataset_anchor": False,
        "mode": "dpo_same_model", "ckpt": str(args.ckpt),
        "checkpoints": [{"id": SEED_A, "path": str(args.ckpt)}, {"id": SEED_B, "path": str(args.ckpt)}],
        "items": items,
    }
    (out_dir / "campaign.json").write_text(json.dumps(campaign, indent=1), encoding="utf-8")

    # ledger anti-repetition : ajoute les cycles montres
    if args.ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        merged = sorted(shown | {it["cycle_idx"] for it in items})
        args.ledger.write_text(json.dumps(merged))
        print(f"[dpo-camp] ledger maj : {len(merged)} cycles montres cumules")
    print(f"[dpo-camp] campagne : {out_dir}\n  eval : python -m arena.server --campaign {out_dir} --port 8766 --target-duels {len(items)}")


if __name__ == "__main__":
    main()
