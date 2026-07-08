"""Boucle d'entraînement de l'encodeur contrastif.

Usage :
    python -m encoder.train --data data --out runs/encoder_v1 \
        --epochs 100 --batch-size 256 --lr 1e-3 --temperature 0.2
"""
from __future__ import annotations
import argparse
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data import ContrastivePairs
from .model import SpriteEncoder
from .loss import nt_xent
from monitor.control import init_control_file, poll_and_act


def evaluate(model, loader, device, temperature, max_batches=20):
    model.eval()
    losses, top1s, aligns = [], [], []
    with torch.no_grad():
        for i, (va, vb, _) in enumerate(loader):
            if i >= max_batches:
                break
            va = va.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                za = model(va); zb = model(vb)
                loss, m = nt_xent(za.float(), zb.float(), temperature)
            losses.append(loss.item()); top1s.append(m["top1"]); aligns.append(m["alignment"])
    model.train()
    return {
        "val_loss": sum(losses) / len(losses),
        "val_top1": sum(top1s) / len(top1s),
        "val_alignment": sum(aligns) / len(aligns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("runs/encoder_v1"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    init_control_file(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    train_ds = ContrastivePairs(args.data, split="train", augment=True)
    val_ds = ContrastivePairs(args.data, split="val", augment=False)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    model = SpriteEncoder(in_ch=16, feat_dim=args.feat_dim).to(device)
    print(f"[train] model params: {model.n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader),
                              eta_min=args.lr * 0.05)

    use_amp = device == "cuda" and not args.no_amp
    history = []
    best_val = float("inf")
    step = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        ep_loss = 0.0; ep_top1 = 0.0; n = 0
        for va, vb, _ in train_loader:
            va = va.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    za = model(va); zb = model(vb)
                    # NT-Xent en fp32 pour stabilité
                    loss, metrics = nt_xent(za.float(), zb.float(), args.temperature)
            else:
                za = model(va); zb = model(vb)
                loss, metrics = nt_xent(za, zb, args.temperature)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item(); ep_top1 += metrics["top1"]; n += 1
            step += 1

        train_loss = ep_loss / n
        train_top1 = ep_top1 / n
        val_metrics = evaluate(model, val_loader, device, args.temperature)
        lr_now = opt.param_groups[0]["lr"]
        dt = time.time() - t0

        row = {"epoch": epoch, "step": step, "lr": lr_now,
               "train_loss": train_loss, "train_top1": train_top1,
               **val_metrics, "time_s": dt}
        history.append(row)
        print(f"[ep {epoch:3d}] loss={train_loss:.4f} top1={train_top1:.3f} "
              f"val_loss={val_metrics['val_loss']:.4f} val_top1={val_metrics['val_top1']:.3f} "
              f"align={val_metrics['val_alignment']:.4f} lr={lr_now:.2e} ({dt:.1f}s)")

        # Save history every epoch (cheap)
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

        # Best checkpoint
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "val_loss": best_val,
                        "config": vars(args)},
                       args.out / "best.pt")

        # --- Contrôle dashboard (pause / stop / checkpoint à la demande) ---
        ctrl = poll_and_act(args.out)
        if ctrl["checkpoint_now"]:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_metrics["val_loss"], "config": vars(args)},
                       args.out / f"manual_ep{epoch:04d}.pt")
            print(f"[train] manual checkpoint saved (ep {epoch})")
        if ctrl["stop"]:
            print(f"[train] stop requested via dashboard — exiting after ep {epoch}")
            break

    # Final checkpoint
    torch.save({"model": model.state_dict(), "epoch": epoch,
                "config": vars(args)}, args.out / "last.pt")
    print(f"[train] done. best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
