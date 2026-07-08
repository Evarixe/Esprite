"""Boucle d'entraînement du transformer génératif.

Usage :
    uv run python -m genmodel.train \
        --data data --out runs/gen_v1 \
        --total-steps 50000 --warmup 2000 \
        --lr 3e-4 --batch-short 32 --batch-long 2

Le training compte en STEPS (mises à jour de gradient), pas en epochs : les buckets
sont mélangés en flot continu et un step = un batch consommé d'un bucket donné.

Le monitor (dashboard) est intégré : à chaque `--monitor-every` steps, on dump
`history.json` et on poll `control.json` (pause / stop / checkpoint à la demande).
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CyclesTokenDataset, BucketBatchSampler, collate_pad, SHORT_BUCKET, LONG_BUCKET
from .model import SpriteTransformer
from .loss import next_token_ce, LossWeights, NEUTRAL
from monitor.control import init_control_file, poll_and_act
from monitor.heartbeat import HeartbeatWriter
from monitor.gpu import gpu_stats


def get_lr(step: int, total_steps: int, warmup: int, peak_lr: float, min_lr_frac: float = 0.05) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


def cycle(loader):
    """Itère sans fin sur un DataLoader."""
    while True:
        for batch in loader:
            yield batch


def thermal_pause(seconds: float, reason: str, heartbeat=None, step=0, total=0, lr=0.0):
    """Pause bloquante `seconds`, GPU au repos pour refroidir. Met à jour le
    heartbeat pendant l'attente pour que le dashboard reflète l'état 'paused' +
    la température qui descend. Retourne la durée réellement écoulée (à soustraire
    du wall-clock d'entraînement)."""
    import torch
    print(f"[pause] {reason} — refroidissement {seconds:.0f}s")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    t0 = time.time()
    end = t0 + seconds
    while time.time() < end:
        if heartbeat is not None:
            g = gpu_stats()
            heartbeat.update(step=step, total_steps=total, bucket="pause",
                             loss=float("nan"), top1=0.0, n_tokens=0,
                             dt=1.0, lr=lr, gpu=g, force=True)
            if g:
                remain = int(end - time.time())
                print(f"[pause] {remain:3d}s restants — GPU {g['temp_c']}C / {g['util_pct']}%")
        time.sleep(5)
    elapsed = time.time() - t0
    print(f"[pause] reprise après {elapsed:.0f}s")
    return elapsed


def wait_until_cool(target_c: int, cool_to: int, max_wait: float = 600,
                    heartbeat=None, step=0, total=0, lr=0.0):
    """Pause adaptative : attend que la GPU redescende sous `cool_to`.
    Plafonné à max_wait. Retourne la durée écoulée (0 si pas de stats GPU)."""
    import torch
    g = gpu_stats()
    if not g:
        return 0.0
    print(f"[pause] GPU {g['temp_c']}C >= {target_c}C — attente jusqu'à <{cool_to}C")
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    t0 = time.time()
    while True:
        g = gpu_stats()
        if not g or g["temp_c"] < cool_to or (time.time() - t0) > max_wait:
            break
        if heartbeat is not None:
            heartbeat.update(step=step, total_steps=total, bucket="pause",
                             loss=float("nan"), top1=0.0, n_tokens=0,
                             dt=1.0, lr=lr, gpu=g, force=True)
        print(f"[pause] GPU {g['temp_c']}C, cible <{cool_to}C")
        time.sleep(5)
    elapsed = time.time() - t0
    print(f"[pause] refroidi en {elapsed:.0f}s")
    return elapsed


def evaluate(model, loader, device, max_batches: int = 20) -> dict:
    model.eval()
    losses_by_bucket: dict[str, list[float]] = {SHORT_BUCKET: [], LONG_BUCKET: []}
    top1s_by_bucket: dict[str, list[float]] = {SHORT_BUCKET: [], LONG_BUCKET: []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            tokens = batch["tokens"].to(device, non_blocking=True)
            x_pos = batch["x_pos"].to(device, non_blocking=True)
            y_pos = batch["y_pos"].to(device, non_blocking=True)
            f_pos = batch["frame_pos"].to(device, non_blocking=True)
            roles = batch["roles"].to(device, non_blocking=True)
            attn  = batch["attn_mask"].to(device, non_blocking=True)
            lm    = batch["loss_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens, x_pos, y_pos, f_pos, roles, attn)
            loss, info = next_token_ce(logits.float(), tokens, roles, lm, weights=NEUTRAL)
            bk = batch["buckets"][0]
            losses_by_bucket[bk].append(loss.item())
            top1s_by_bucket[bk].append(info["top1"])
    model.train()
    out = {}
    all_losses, all_top1 = [], []
    for bk in (SHORT_BUCKET, LONG_BUCKET):
        if losses_by_bucket[bk]:
            out[f"val_loss_{bk}"] = sum(losses_by_bucket[bk]) / len(losses_by_bucket[bk])
            out[f"val_top1_{bk}"] = sum(top1s_by_bucket[bk]) / len(top1s_by_bucket[bk])
            all_losses.extend(losses_by_bucket[bk]); all_top1.extend(top1s_by_bucket[bk])
    if all_losses:
        out["val_loss"] = sum(all_losses) / len(all_losses)
        out["val_top1"] = sum(all_top1) / len(all_top1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total-steps", type=int, default=50000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch-short", type=int, default=32)
    ap.add_argument("--batch-long", type=int, default=2)
    ap.add_argument("--short-only", action="store_true",
                    help="Exclut le bucket long (séquences 16k tokens) du training")
    ap.add_argument("--long-only", action="store_true",
                    help="Entraîne UNIQUEMENT sur le bucket long. Évite la fragmentation"
                         " allocator de mélanger des shapes très différentes.")
    ap.add_argument("--no-palette-swap", action="store_true",
                    help="Désactive la palette swap au train (Option B : apprentissage"
                         " en palette canonique). Sans ref, le swap rend la couleur"
                         " non-résolvable et empoisonne le signal contenu.")
    ap.add_argument("--continuation", action="store_true",
                    help="Checkpoint intermediaire vise : LR plat a --lr, pas de warmup ni cosine.")
    ap.add_argument("--min-lr-frac", type=float, default=0.30,
                    help="Fraction du peak LR en fin de cosine (ignore si --continuation). "
                         "0.30 pour checkpoints intermediaires (finit chaud, ~9e-5 avec pic 3e-4, "
                         "le modele apprend encore en fin de run et enchaine sans re-ramper). "
                         "0.10 pour un run final unique (cf brief).")
    # Cosine GLOBALE (consciente de la lignée) : LR indexé sur le step CUMULÉ, pas
    # par-run → décroît en douceur à travers les runs enchaînés au lieu du LR plat.
    # Prend le pas sur --continuation. warmup auto-sauté si global-offset >> warmup.
    ap.add_argument("--global-offset", type=int, default=0,
                    help="Steps DÉJÀ écoulés dans la décroissance (0 au 1er run ; les runs "
                         "enchaînés passent l'offset cumulé). Pic --lr à offset=0.")
    ap.add_argument("--global-total", type=int, default=0,
                    help="Longueur TOTALE de la décroissance (pas une position absolue). "
                         ">0 active la cosine globale : pic --lr → --min-lr-frac·lr sur global-total steps.")
    # Poids de loss par classe de cible (cf loss.LossWeights). Défauts = point de
    # travail courant (proto local). Train uniquement ; val reste neutre (comparable).
    ap.add_argument("--w-color",   type=float, default=1.5,  help="pixel couleur (1-15) contenu")
    ap.add_argument("--w-transp",  type=float, default=1.0,  help="pixel transparent (0) contenu")
    ap.add_argument("--w-sep",     type=float, default=20.0, help="<FRAME_SEP>")
    ap.add_argument("--w-end",     type=float, default=40.0, help="<SEQ_END> (le compte)")
    ap.add_argument("--w-ref",     type=float, default=0.5,  help="pixels ref dé-masqués (0=masqué)")
    ap.add_argument("--w-ref-end", type=float, default=20.0, help="<REF_END>")
    ap.add_argument("--empty-cache-every", type=int, default=250,
                    help="torch.cuda.empty_cache() tous les N steps (0=off) — défragmente, "
                         "évite le paging WDDM quand le pic long-bucket colle au plafond VRAM.")
    ap.add_argument("--grad-ckpt-threshold", type=int, default=6000,
                    help="Gradient checkpointing si seq_len >= seuil (0 = jamais). VRAM basse sur bucket long.")
    ap.add_argument("--pause-every", type=int, default=0,
                    help="Pause thermique programmee tous les N steps (0 = jamais).")
    ap.add_argument("--pause-secs", type=float, default=120.0,
                    help="Duree d'une pause programmee (--pause-every).")
    ap.add_argument("--pause-temp-c", type=int, default=0,
                    help="Pause adaptative : si GPU depasse cette temp (C) en fin de step, "
                         "pause jusqu'a redescente sous --pause-cool-c. 0 = desactive.")
    ap.add_argument("--pause-cool-c", type=int, default=0,
                    help="Temp cible (C) de fin de pause adaptative. 0 = (pause-temp-c - 15).")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Chemin d'un checkpoint .pt à charger (model state_dict) avant entraînement")
    ap.add_argument("--append-history", action="store_true",
                    help="Lignee unifiee : charge history.json existant du --out et y ajoute "
                         "(au lieu d'ecraser). Rows stampees au step GLOBAL (history-base+local).")
    ap.add_argument("--history-base", type=int, default=0,
                    help="Horloge ABSOLUE de lignee au local step 0. Distinct de --global-offset "
                         "(position dans la decroissance LR). En lignee : = global_step courant.")
    ap.add_argument("--ref-prob", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--monitor-every", type=int, default=20,
                    help="Steps entre 2 dumps de history.json et 2 polls de control.json")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    init_control_file(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    train_ds = CyclesTokenDataset(args.data, split="train", ref_prob=args.ref_prob,
                                  augment_palette=not args.no_palette_swap)
    val_ds = CyclesTokenDataset(args.data, split="val", ref_prob=args.ref_prob,
                                augment_palette=False)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")
    print(f"[train] train buckets: {{k: len(v) for k,v in train_ds.buckets.items()}} ->",
          {k: len(v) for k, v in train_ds.buckets.items()})

    # --short-only / --long-only : on ne mélange jamais les 2 buckets dans le même
    # run, pour éviter que le caching allocator PyTorch garde des reservations
    # short ET long simultanément → fragmentation → OOM proche du plafond VRAM.
    assert not (args.short_only and args.long_only), "choisir l'un OU l'autre, pas les deux"
    if args.long_only:
        bsz = {LONG_BUCKET: args.batch_long}
        print("[train] long-only: bucket short exclu")
    elif args.short_only:
        bsz = {SHORT_BUCKET: args.batch_short}
        print("[train] short-only: bucket long exclu")
    else:
        bsz = {SHORT_BUCKET: args.batch_short, LONG_BUCKET: args.batch_long}
    train_sampler = BucketBatchSampler(train_ds, batch_sizes=bsz, shuffle=True, seed=args.seed)
    val_sampler = BucketBatchSampler(val_ds, batch_sizes=bsz, shuffle=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=collate_pad, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                            collate_fn=collate_pad, num_workers=args.num_workers,
                            pin_memory=(device == "cuda"))

    train_weights = LossWeights(
        w_color=args.w_color, w_transp=args.w_transp, w_sep=args.w_sep,
        w_end=args.w_end, w_ref=args.w_ref, w_ref_end=args.w_ref_end)
    print(f"[train] loss weights: {train_weights}")

    model = SpriteTransformer().to(device)
    print(f"[train] model params: {model.n_params/1e6:.2f}M")
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[train] resumed model from {args.resume} (step {state.get('step', '?')})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    heartbeat = HeartbeatWriter(args.out / "heartbeat.json", min_interval_s=0.25)

    # Lignee unifiee (--append-history) : on continue le history.json existant du
    # dossier au lieu de repartir d'une liste vide. Les rows sont stampees au step
    # GLOBAL (voir gstep ci-dessous) -> une seule courbe continue SFT+DPO.
    hist_path = args.out / "history.json"
    if args.append_history and hist_path.exists():
        try:
            history: list[dict] = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    else:
        history = []
    log_buffer = {"loss_sum_short": 0.0, "n_short": 0,
                  "loss_sum_long": 0.0, "n_long": 0,
                  "top1_sum_short": 0.0, "top1_sum_long": 0.0,
                  "top1_transp_sum": 0.0, "top1_opaque_sum": 0.0, "top1_struct_sum": 0.0,
                  "n_breakdown": 0,
                  "tokens_sum": 0, "wall_sum": 0.0}

    def lr_at(step: int) -> float:
        # Cosine globale (lignée) > continuation plate > cosine par-run.
        # global-offset = steps DÉJÀ écoulés dans la décroissance (0 au 1er run) ;
        # global-total = longueur TOTALE de la décroissance. warmup=0 → pic 3e-4 à
        # offset=0 (continu avec le LR plat d'avant), min à global-total. Les runs
        # enchaînés incrémentent l'offset → même courbe, sans saut.
        if args.global_total > 0:
            return get_lr(args.global_offset + step, args.global_total, 0,
                          args.lr, args.min_lr_frac)
        if args.continuation:
            return args.lr
        return get_lr(step, args.total_steps, args.warmup, args.lr, args.min_lr_frac)

    # Validation initiale (step 0) : ancre les courbes val à l'origine. Surtout utile
    # en continuation (val-every grand → la courbe orange n'aurait que 2-3 points) :
    # ce point vient du checkpoint de départ, avant tout pas d'entraînement.
    gstep = lambda local: args.history_base + local   # horloge absolue lignee (base 0 = standalone)
    init_val = evaluate(model, val_loader, device, max_batches=10)
    if init_val:
        history.append({"step": gstep(0), "lr": lr_at(0), "epoch": gstep(0), "mode": "sft", **init_val})
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"  [val @ step 0 (checkpoint départ)] " + " ".join(f"{k}={v:.3f}" for k, v in init_val.items()))

    train_iter = cycle(train_loader)
    t_start = time.time()
    last_log_t = t_start

    for step in range(args.total_steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(train_iter)
        bk = batch["buckets"][0]

        tokens = batch["tokens"].to(device, non_blocking=True)
        x_pos = batch["x_pos"].to(device, non_blocking=True)
        y_pos = batch["y_pos"].to(device, non_blocking=True)
        f_pos = batch["frame_pos"].to(device, non_blocking=True)
        roles = batch["roles"].to(device, non_blocking=True)
        attn  = batch["attn_mask"].to(device, non_blocking=True)
        lm    = batch["loss_mask"].to(device, non_blocking=True)

        # Gradient checkpointing seulement sur séquences longues (budget VRAM)
        model.set_grad_checkpoint(
            args.grad_ckpt_threshold > 0 and tokens.shape[1] >= args.grad_ckpt_threshold)
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens, x_pos, y_pos, f_pos, roles, attn)
        loss, info = next_token_ce(logits.float(), tokens, roles, lm, weights=train_weights)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0

        # Accumule pour le log moyenné
        log_buffer[f"loss_sum_{bk}"] += loss.item()
        log_buffer[f"n_{bk}"] += 1
        log_buffer[f"top1_sum_{bk}"] += info["top1"]
        log_buffer["top1_transp_sum"] += info["top1_transparent"]
        log_buffer["top1_opaque_sum"] += info["top1_opaque"]
        log_buffer["top1_struct_sum"] += info["top1_struct"]
        log_buffer["n_breakdown"] += 1
        log_buffer["tokens_sum"] += info["tokens_supervised"]
        log_buffer["wall_sum"] += dt

        # Heartbeat live (throttlé à 4 Hz côté writer)
        heartbeat.update(
            step=step + 1, total_steps=args.total_steps, bucket=bk,
            loss=loss.item(), top1=info["top1"], n_tokens=info["tokens_supervised"],
            dt=dt, lr=lr, gpu=gpu_stats(),
        )

        # Log périodique
        if (step + 1) % args.log_every == 0 or step == 0:
            now = time.time()
            short_mean_loss = (log_buffer["loss_sum_short"] / log_buffer["n_short"]) if log_buffer["n_short"] else None
            long_mean_loss  = (log_buffer["loss_sum_long"]  / log_buffer["n_long"])  if log_buffer["n_long"]  else None
            short_mean_top1 = (log_buffer["top1_sum_short"] / log_buffer["n_short"]) if log_buffer["n_short"] else None
            long_mean_top1  = (log_buffer["top1_sum_long"]  / log_buffer["n_long"])  if log_buffer["n_long"]  else None
            elapsed = now - t_start
            tps = log_buffer["tokens_sum"] / max(1e-6, log_buffer["wall_sum"])
            nb = max(1, log_buffer["n_breakdown"])
            g = gpu_stats() or {}   # snapshot GPU au moment du log → série temporelle thermique/puissance
            # Métriques allocator PyTorch : tranchent fuite (allocated↑) vs fragmentation
            # (reserved↑, allocated plat). NVML mem_used = reserved+contexte, masque la distinction.
            if device == "cuda":
                torch_alloc_gb = round(torch.cuda.memory_allocated() / 1024**3, 3)
                torch_reserved_gb = round(torch.cuda.memory_reserved() / 1024**3, 3)
            else:
                torch_alloc_gb = torch_reserved_gb = None
            row = {
                "step": gstep(step + 1),
                "mode": "sft",
                "lr": lr,
                "gpu_temp_c":      g.get("temp_c"),
                "gpu_power_w":     g.get("power_w"),
                "gpu_mem_used_gb": g.get("mem_used_gb"),
                "torch_alloc_gb":    torch_alloc_gb,
                "torch_reserved_gb": torch_reserved_gb,
                "gpu_util_pct":    g.get("util_pct"),
                "train_loss_short": short_mean_loss,
                "train_loss_long": long_mean_loss,
                "train_loss": short_mean_loss if short_mean_loss is not None else long_mean_loss,
                "train_top1_short": short_mean_top1,
                "train_top1_long": long_mean_top1,
                "train_top1": short_mean_top1 if short_mean_top1 is not None else long_mean_top1,
                "train_top1_transparent": log_buffer["top1_transp_sum"] / nb,
                "train_top1_opaque":      log_buffer["top1_opaque_sum"] / nb,
                "train_top1_struct":      log_buffer["top1_struct_sum"] / nb,
                "grad_norm": float(gnorm.item()),
                "tokens_per_sec": tps,
                "elapsed_s": elapsed,
                "time_s": now - last_log_t,
                "epoch": gstep(step),  # alias pour le dashboard (axe x)
            }
            history.append(row)
            print(f"[step {step+1:6d}/{args.total_steps}] "
                  f"loss(s)={short_mean_loss if short_mean_loss is None else f'{short_mean_loss:.3f}'} "
                  f"loss(l)={long_mean_loss if long_mean_loss is None else f'{long_mean_loss:.3f}'} "
                  f"top1(s)={short_mean_top1 if short_mean_top1 is None else f'{short_mean_top1:.3f}'} "
                  f"lr={lr:.2e} {tps/1000:.1f}k tok/s")
            last_log_t = now
            # Reset buffers
            log_buffer = {"loss_sum_short": 0.0, "n_short": 0,
                          "loss_sum_long": 0.0, "n_long": 0,
                          "top1_sum_short": 0.0, "top1_sum_long": 0.0,
                          "top1_transp_sum": 0.0, "top1_opaque_sum": 0.0, "top1_struct_sum": 0.0,
                          "n_breakdown": 0,
                          "tokens_sum": 0, "wall_sum": 0.0}

        # Défragmentation périodique : rend les segments cached au driver → garde la
        # VRAM loin du plafond, évite le paging WDDM (le pic long-bucket qui colle à
        # 31.4 GB et fait s'effondrer le débit). Coût : ré-allocation au batch suivant.
        if device == "cuda" and args.empty_cache_every > 0 and (step + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

        # Eval périodique
        if (step + 1) % args.val_every == 0:
            val_metrics = evaluate(model, val_loader, device, max_batches=10)
            history[-1].update(val_metrics)
            print(f"  [val @ step {step+1}] " + " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))

        # Dump history + poll control
        if (step + 1) % args.monitor_every == 0:
            (args.out / "history.json").write_text(json.dumps(history, indent=2))
            ctrl = poll_and_act(args.out)
            if ctrl["checkpoint_now"]:
                torch.save({"model": model.state_dict(), "step": gstep(step + 1),
                            "config": vars(args)},
                           args.out / f"manual_step{gstep(step+1):07d}.pt")
                print(f"[train] manual checkpoint @ step {step+1}")
            if ctrl["stop"]:
                print(f"[train] stop requested via dashboard — exiting at step {step+1}")
                break

        # Checkpoint régulier
        if (step + 1) % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "step": gstep(step + 1),
                        "config": vars(args)},
                       args.out / "last.pt")

        # Pauses thermiques (en fin de step, état déjà sauvegardable).
        # Le temps de pause est soustrait du wall-clock pour ne pas polluer tok/s.
        paused = 0.0
        if args.pause_temp_c > 0:
            g = gpu_stats()
            if g and g["temp_c"] >= args.pause_temp_c:
                cool_to = args.pause_cool_c if args.pause_cool_c > 0 else args.pause_temp_c - 15
                paused += wait_until_cool(args.pause_temp_c, cool_to, heartbeat=heartbeat,
                                          step=step + 1, total=args.total_steps, lr=lr)
        if args.pause_every > 0 and (step + 1) % args.pause_every == 0 and (step + 1) < args.total_steps:
            paused += thermal_pause(args.pause_secs, f"programmée @ step {step+1}",
                                    heartbeat=heartbeat, step=step + 1,
                                    total=args.total_steps, lr=lr)
        if paused > 0:
            t_start += paused      # exclut la pause du elapsed
            last_log_t += paused

    # Final
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"model": model.state_dict(), "step": gstep(step + 1),
                "config": vars(args)}, args.out / "last.pt")
    print(f"[train] done. last global step={gstep(step+1)} (local {step+1}), elapsed={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
