"""Campagne de pré-génération pour l'arène d'éval humaine.

Principe : on fige un ensemble d'items d'évaluation (cycle du val set, tags, N,
with/without ref, seed, température) et on génère CHAQUE item sur CHAQUE
checkpoint avec exactement les mêmes conditions. L'éval (arena A/B, planches)
tourne ensuite sans GPU depuis les sorties stockées — comparable, reproductible,
et sans disputer la VRAM à un training en cours.

Le ground truth du dataset est stocké comme pseudo-checkpoint `__dataset__`
(ancre de calibration : % de préférence vs vraies données).

Sortie : runs/arena/<name>/
    campaign.json            # items (ref+palette incluses), checkpoints, params
    gens/<ckpt_id>/<item_id>.json   # frames b64 + statuses + méta de génération
    votes.jsonl              # créé/rempli par le serveur d'éval

Usage :
    set PYTHONPATH=src
    uv run python -m arena.campaign --name 20k_vs_15k ^
        --ckpt gen_15000=runs/gen_15000/last.pt --ckpt gen_20000=runs/gen_20000/last.pt ^
        --n-items 48 --no-ref-frac 0.33 --temperature 0.5 --seed 0

Un `--ckpt` supplémentaire sur une campagne EXISTANTE ajoute le checkpoint aux
gens sans toucher aux items (mêmes conditions garanties) : relancer avec --name
identique et uniquement les nouveaux --ckpt.
"""
from __future__ import annotations
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import torch

from genmodel.model import SpriteTransformer
from genmodel.sample import generate, GenRequest
from genmodel.graph_sampler import GraphSamplerPool
from genmodel.vocab import SOURCE_ACTION_MAP, PIXELS_PER_FRAME

DATASET_ID = "__dataset__"


def b64_frame(indices: np.ndarray) -> str:
    """(32, 32) uint8 -> base64 (1024 octets bruts)."""
    return base64.b64encode(indices.astype(np.uint8).tobytes()).decode("ascii")


def pick_items(meta: list[dict], lengths: np.ndarray, val_idxs: list[int],
               n_items: int, no_ref_frac: float, seed: int) -> list[dict]:
    """Sélection stratifiée par (action_tag, bucket court/long).

    Round-robin sur les strates pour équilibrer, puis assignation no-ref
    intercalée dans chaque strate (pas de corrélation strate <-> ref).
    """
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    for ci in val_idxs:
        tag = SOURCE_ACTION_MAP[meta[ci]["action"]]
        bucket = "long" if int(lengths[ci]) > 2 else "short"
        strata.setdefault((tag, bucket), []).append(ci)
    for k in strata:
        strata[k] = list(rng.permutation(strata[k]))

    keys = sorted(strata.keys())
    chosen: list[int] = []
    while len(chosen) < n_items and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(chosen) < n_items:
                chosen.append(strata[k].pop())

    items = []
    for i, ci in enumerate(chosen):
        m = meta[ci]
        # no-ref intercalé : 1 item sur round(1/frac), régulier dans l'ordre round-robin
        # (donc décorrélé des strates)
        if no_ref_frac <= 0:
            with_ref = True
        else:
            period = max(1, round(1.0 / no_ref_frac))
            with_ref = (i % period != 0)
        items.append({
            "item_id": f"it{i:03d}",
            "cycle_idx": int(ci),
            "source": m["action"],
            "action": SOURCE_ACTION_MAP[m["action"]],
            "direction": m.get("direction"),
            "n_frames": int(lengths[ci]),
            "with_ref": bool(with_ref),
            "seed": seed * 100_000 + i,
            "pokemon_id": m.get("pokemon_id"),
        })
    return items


def generate_for_checkpoint(ckpt_id: str, ckpt_path: Path, items: list[dict],
                            cycles_all: np.ndarray, temperature: float,
                            out_dir: Path, device: str) -> None:
    model = SpriteTransformer().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    if device == "cuda":
        model = model.to(torch.bfloat16)
    pool = GraphSamplerPool(model) if device == "cuda" else None

    gen_dir = out_dir / "gens" / ckpt_id
    gen_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        ci = it["cycle_idx"]
        ref = cycles_all[ci, 0] if it["with_ref"] else None
        req = GenRequest(
            action=it["action"], direction=it["direction"],
            frames=it["n_frames"], reference=ref,
            temperature=temperature, seed=it["seed"],
        )
        result = generate(model, req, device=device, pool=pool)
        expected_end = it["n_frames"] * (PIXELS_PER_FRAME + 1)
        actual_end = len(result.raw_tokens) if result.stop_reason == "seq_end" else None
        payload = {
            "item_id": it["item_id"], "ckpt_id": ckpt_id,
            "frames": [b64_frame(fr.indices) for fr in result.frames],
            "statuses": [fr.status for fr in result.frames],
            "n_pixels": [fr.n_pixels for fr in result.frames],
            "stop_reason": result.stop_reason,
            "n_valid": result.n_valid, "n_invalid": result.n_invalid,
            "n_raw_tokens": len(result.raw_tokens),
            "seqend_gap": (actual_end - expected_end) if actual_end is not None else None,
        }
        (gen_dir / f"{it['item_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8")
        print(f"  [{ckpt_id}] {k + 1}/{len(items)} {it['item_id']} "
              f"({it['action']}, N={it['n_frames']}, ref={it['with_ref']}): "
              f"{len(result.frames)} frames, {result.n_valid} valid, "
              f"stop={result.stop_reason}")
    dt = time.perf_counter() - t0
    print(f"  [{ckpt_id}] terminé en {dt:.1f}s ({dt / len(items):.2f}s / item)")
    del model, pool
    if device == "cuda":
        torch.cuda.empty_cache()


def write_dataset_gens(items: list[dict], cycles_all: np.ndarray,
                       lengths: np.ndarray, out_dir: Path) -> None:
    gen_dir = out_dir / "gens" / DATASET_ID
    gen_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        ci = it["cycle_idx"]
        n = int(lengths[ci])
        payload = {
            "item_id": it["item_id"], "ckpt_id": DATASET_ID,
            "frames": [b64_frame(cycles_all[ci, f]) for f in range(n)],
            "statuses": ["valid"] * n,
            "n_pixels": [PIXELS_PER_FRAME] * n,
            "stop_reason": "seq_end",
            "n_valid": n, "n_invalid": 0,
            "n_raw_tokens": n * (PIXELS_PER_FRAME + 1),
            "seqend_gap": 0,
        }
        (gen_dir / f"{it['item_id']}.json").write_text(
            json.dumps(payload), encoding="utf-8")


def parse_ckpt_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--ckpt attend id=path (ex: gen_20000=runs/gen_20000/last.pt), reçu: {s}")
    cid, p = s.split("=", 1)
    if cid == DATASET_ID:
        raise argparse.ArgumentTypeError(f"{DATASET_ID} est réservé au ground truth")
    return cid, Path(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Nom de campagne (dossier runs/arena/<name>)")
    ap.add_argument("--ckpt", action="append", required=True, type=parse_ckpt_arg,
                    metavar="ID=PATH", help="Checkpoint à générer (répétable)")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--n-items", type=int, default=48)
    ap.add_argument("--no-ref-frac", type=float, default=0.33,
                    help="Fraction d'items générés SANS référence")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dataset-anchor", action="store_true",
                    help="Ne pas inclure le ground truth comme pseudo-checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("runs") / "arena" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = out_dir / "campaign.json"

    data = np.load(args.data / "dataset.npz")
    cycles_all, lengths, palettes_all = data["cycles"], data["lengths"], data["palettes"]
    meta = json.loads((args.data / "dataset.meta.json").read_text(encoding="utf-8"))
    splits = json.loads((args.data / "splits.json").read_text(encoding="utf-8"))

    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        items = campaign["items"]
        print(f"[campaign] campagne existante '{args.name}' : {len(items)} items figés, "
              f"ajout de checkpoints uniquement")
    else:
        items = pick_items(meta, lengths, splits["val"], args.n_items,
                           args.no_ref_frac, args.seed)
        # ref + palette embarquées dans campaign.json (les gens restent légers)
        for it in items:
            ci = it["cycle_idx"]
            it["palette"] = palettes_all[ci].tolist()
            it["ref"] = b64_frame(cycles_all[ci, 0])
        campaign = {
            "name": args.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": args.temperature,
            "seed": args.seed,
            "dataset_anchor": not args.no_dataset_anchor,
            "checkpoints": [],
            "items": items,
        }
        n_ref = sum(1 for it in items if it["with_ref"])
        print(f"[campaign] {len(items)} items ({n_ref} with-ref, {len(items) - n_ref} no-ref), "
              f"T={args.temperature}, seed={args.seed}")

    known = {c["id"] for c in campaign["checkpoints"]}
    if campaign["dataset_anchor"] and DATASET_ID not in known:
        print(f"[campaign] ancre ground truth {DATASET_ID}")
        write_dataset_gens(items, cycles_all, lengths, out_dir)
        campaign["checkpoints"].append({"id": DATASET_ID, "path": None})
        known.add(DATASET_ID)

    for cid, cpath in args.ckpt:
        if cid in known:
            print(f"[campaign] {cid} déjà généré, skip")
            continue
        print(f"[campaign] génération {cid} <- {cpath}")
        generate_for_checkpoint(cid, cpath, items, cycles_all,
                                campaign["temperature"], out_dir, device)
        campaign["checkpoints"].append({"id": cid, "path": str(cpath)})
        known.add(cid)
        # sauvegarde incrémentale : un crash ne perd pas les checkpoints déjà générés
        campaign_path.write_text(json.dumps(campaign, indent=1), encoding="utf-8")

    campaign_path.write_text(json.dumps(campaign, indent=1), encoding="utf-8")
    print(f"[campaign] campagne écrite : {campaign_path}")
    print(f"[campaign] éval : uv run python -m arena.server --campaign {out_dir}")


if __name__ == "__main__":
    main()
