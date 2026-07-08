"""Prototype DPO : fine-tune le generateur sur les preferences d'arene.

Usage :
    set PYTHONPATH=src
    uv run python dpo_train.py --ckpt runs/gen_20000/last.pt \
        --campaign runs/arena/movement_20k --out runs/gen_20000_dpo \
        --beta 0.1 --lr 1e-6 --epochs 2

Ref (figee) = policy init = --ckpt. Log-probs de ref pre-calculees une fois.
Chaque paire : forward chosen+rejected dans la policy, loss DPO, backward, step.
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
from genmodel.dpo import extract_pairs, build_sequence, sequence_logprob, dpo_loss
from monitor.heartbeat import HeartbeatWriter
from monitor.control import init_control_file, poll_and_act
from monitor.gpu import gpu_stats


def load_model(ckpt: Path, device: str, train: bool):
    m = SpriteTransformer().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    if train:
        m.train()
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True, action="append",
                    help="Repetable : fenetre glissante = campagne courante + N precedentes. "
                         "Les paires de toutes les campagnes passees sont poolees.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-ckpt-threshold", type=int, default=4000)
    ap.add_argument("--max-frames", type=int, default=16, help="skip paires > N frames (VRAM)")
    ap.add_argument("--global-offset", type=int, default=0,
                    help="Step de lignee au demarrage. La phase DPO avance l'horloge : "
                         "le checkpoint sauve step = global_offset + steps_DPO, pour rester "
                         "sur le meme continuum que le SFT (cosine LR globale, consolidation).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # Fenetre glissante : pool des paires sur toutes les campagnes passees. Chaque
    # paire est taguee par l'index de sa campagne (item_id se recoupe entre campagnes).
    campaigns = []   # [(items_map, dir)]
    all_pairs = []   # [(camp_idx, Pair)]
    for cdir in args.campaign:
        votes = [json.loads(l) for l in (cdir / "votes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        items = {it["item_id"]: it for it in json.loads((cdir / "campaign.json").read_text(encoding="utf-8"))["items"]}
        ci = len(campaigns)
        campaigns.append((items, cdir))
        for p in extract_pairs(votes):
            all_pairs.append((ci, p))
    print(f"[dpo] fenetre {len(args.campaign)} campagne(s) -> {len(all_pairs)} paires")

    # cache de sequences par (camp_idx, item, ckpt) + skip paires trop longues (VRAM)
    seq_cache = {}
    def seq_for(ci, item_id, ckpt):
        key = (ci, item_id, ckpt)
        if key not in seq_cache:
            items, cdir = campaigns[ci]
            gen = json.loads((cdir / "gens" / ckpt / f"{item_id}.json").read_text(encoding="utf-8"))
            seq_cache[key] = build_sequence(items[item_id], gen)
        return seq_cache[key]

    kept = [(ci, p) for ci, p in all_pairs
            if max(seq_for(ci, p.item_id, p.chosen_ckpt)["n_frames"],
                   seq_for(ci, p.item_id, p.rejected_ckpt)["n_frames"]) <= args.max_frames]
    print(f"[dpo] {len(kept)} paires retenues (<= {args.max_frames} frames)")

    # --- Monitoring (meme dashboard que le training) ---
    total_steps = args.epochs * len(kept)
    (args.out / "config.json").write_text(json.dumps(
        {**vars(args), "total_steps": total_steps, "n_pairs": len(kept), "mode": "dpo"},
        default=str, indent=2))
    init_control_file(args.out)
    heartbeat = HeartbeatWriter(args.out / "heartbeat.json", min_interval_s=0.25)

    policy = load_model(args.ckpt, device, train=True)
    ref = load_model(args.ckpt, device, train=False)
    print(f"[dpo] policy+ref charges ({policy.n_params/1e6:.1f}M chacun)")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # --- Pre-calcul des log-probs de reference (une fois, figees) ---
    t0 = time.time()
    ref_logp = {}
    uniq = {(ci, p.item_id, c) for ci, p in kept for c in (p.chosen_ckpt, p.rejected_ckpt)}
    ref.set_grad_checkpoint(False)
    with torch.no_grad():
        for (ci, item_id, ckpt) in uniq:
            ref_logp[(ci, item_id, ckpt)] = sequence_logprob(ref, seq_for(ci, item_id, ckpt), device).item()
    print(f"[dpo] {len(uniq)} log-probs de ref pre-calculees en {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    # Lignee unifiee : on continue le history.json existant du dossier (rows SFT
    # deja presentes) et on ajoute les rows DPO au step GLOBAL -> courbe continue.
    hist_path = args.out / "history.json"
    if hist_path.exists():
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            hist = []
    else:
        hist = []
    step = 0
    stop = False
    for ep in range(args.epochs):
        if stop:
            break
        order = rng.permutation(len(kept))
        run_loss, run_acc, run_margin, nb = 0.0, 0, 0.0, 0
        for idx in order:
            ci, p = kept[idx]
            bc, br = seq_for(ci, p.item_id, p.chosen_ckpt), seq_for(ci, p.item_id, p.rejected_ckpt)
            long = max(bc["n_frames"], br["n_frames"]) * 1025 > args.grad_ckpt_threshold
            policy.set_grad_checkpoint(long)
            pc = sequence_logprob(policy, bc, device)
            pr = sequence_logprob(policy, br, device)
            rc = torch.tensor(ref_logp[(ci, p.item_id, p.chosen_ckpt)], device=device)
            rr = torch.tensor(ref_logp[(ci, p.item_id, p.rejected_ckpt)], device=device)
            loss, margin, logits = dpo_loss(pc, pr, rc, rr, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if device == "cuda":
                torch.cuda.empty_cache()
            run_loss += loss.item(); run_margin += margin.item()
            run_acc += int(logits.item() > 0); nb += 1
            step += 1
            g = gpu_stats() or {}
            heartbeat.update(step=step, total_steps=total_steps, bucket="dpo",
                             loss=loss.item(), top1=float(logits.item() > 0),
                             n_tokens=0, dt=0.0, lr=args.lr, gpu=g)
            if step % 25 == 0:
                aL, aA, aM = run_loss / nb, run_acc / nb, run_margin / nb
                print(f"[dpo] ep{ep} step {step}/{total_steps} | loss {aL:.4f} | "
                      f"acc(chosen prefere) {aA:.2f} | marge {aM:+.1f}", flush=True)
                hist.append({"step": args.global_offset + step, "mode": "dpo",
                             "train_loss_short": aL, "train_loss": aL,
                             "train_top1": aA, "train_top1_short": aA, "dpo_margin": aM,
                             "lr": args.lr, "epoch": args.global_offset + step,
                             "gpu_temp_c": g.get("temp_c"), "gpu_power_w": g.get("power_w"),
                             "gpu_mem_used_gb": g.get("mem_used_gb"), "gpu_util_pct": g.get("util_pct")})
                (args.out / "history.json").write_text(json.dumps(hist, indent=2))
                run_loss, run_acc, run_margin, nb = 0.0, 0, 0.0, 0
                if poll_and_act(args.out)["stop"]:
                    print("[dpo] stop demande via dashboard", flush=True); stop = True; break
        print(f"[dpo] === epoch {ep} terminee (step {step}) ===", flush=True)

    global_step = args.global_offset + step   # la phase DPO avance l'horloge de lignee
    torch.save({"model": policy.state_dict(), "step": global_step,
                "dpo": vars(args), "dpo_steps": step, "global_offset": args.global_offset},
               args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2))   # lignee unifiee
    print(f"[dpo] sauve {args.out/'last.pt'} -- step lignee {global_step} "
          f"(= offset {args.global_offset} + {step} DPO)", flush=True)


if __name__ == "__main__":
    main()
