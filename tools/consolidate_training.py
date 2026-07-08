"""Consolide la lignée d'entraînement complète (gen_200b → gen_2000b, 2000 steps)
en un fichier de données unique + un graphe d'évolution.

La lignée est reconstruite via le champ `resume` de chaque config.json. Les steps
locaux de chaque run sont décalés par un offset global cumulatif pour obtenir une
échelle 0→2000 continue. Les changements de régime (LR, palette swap) sont annotés.

Sorties :
  - runs/lineage_3000.json : history complète, 1 entrée par log, champ `global_step`
  - runs/lineage_3000.csv  : même chose en CSV plat
  - runs/lineage_3000.png  : graphe loss + opaque + struct + lr avec bandes de phase
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path("runs")

# Lignée explicite (ordre chronologique), reconstruite via resume.
# (nom, label_phase) — le label décrit le régime de CE run.
LINEAGE = [
    ("gen_200b",  "from scratch · LR 5e-4 · swap ON"),
    ("gen_500",   "LR 2e-4 · swap ON"),
    ("gen_1000",  "mixte · LR 1e-5 plat · swap ON"),
    ("gen_1300",  "LR 3e-4 · swap OFF"),
    ("gen_2000b", "LR 3e-4 · swap OFF"),
    ("gen_3000",  "mixte · LR 3e-4 · swap OFF · thermique"),
    ("gen_4000",  "dataset v2 (5499) · mixte · LR 3e-4 · swap OFF · thermique"),
    ("gen_5000",  "dataset 5771 (walk-2) · λ=20 · LR 3e-4 · 400W · expandable_segments"),
    ("gen_7500",  "λ=20 · +2500 steps · 400W · biais SEQ_END neutralisé"),
    ("gen_10000", "LossWeights complet (ref 0.5) · +2500 · 1er amorçage no-ref · val_top1 0.953"),
    ("gen_15000", "+5000 · WIP · val_top1 0.960 · no-ref progresse (contours) · mouvement à évaluer"),
]


def load_history(name: str) -> list[dict]:
    p = RUNS / name / "history.json"
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    consolidated: list[dict] = []
    phases: list[dict] = []  # pour les bandes du graphe
    offset = 0

    for name, label in LINEAGE:
        hist = load_history(name)
        cfg = json.loads((RUNS / name / "config.json").read_text(encoding="utf-8"))
        run_steps = cfg.get("total_steps", hist[-1]["step"])
        start_global = offset
        for row in hist:
            r = dict(row)
            r["run"] = name
            r["phase_label"] = label
            r["local_step"] = row["step"]
            r["global_step"] = offset + row["step"]
            r["swap_off"] = bool(cfg.get("no_palette_swap"))
            consolidated.append(r)
        phases.append({
            "run": name, "label": label,
            "start": start_global, "end": start_global + run_steps,
            "swap_off": bool(cfg.get("no_palette_swap")),
        })
        offset += run_steps

    total = offset
    print(f"[consolidate] {len(consolidated)} log points sur {total} steps cumulés")
    for ph in phases:
        print(f"  {ph['start']:5d}–{ph['end']:5d}  {ph['run']:10s}  {ph['label']}")

    # --- JSON ---
    (RUNS / "lineage_3000.json").write_text(
        json.dumps({"phases": phases, "history": consolidated}, indent=2), encoding="utf-8")

    # --- CSV ---
    cols = ["global_step", "run", "local_step", "swap_off", "lr",
            "train_loss_short", "train_loss_long", "train_top1_short",
            "train_top1_struct", "train_top1_transparent", "train_top1_opaque",
            "val_loss_short", "val_top1_short"]
    with open(RUNS / "lineage_3000.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in consolidated:
            w.writerow(r)

    # --- Graphe ---
    gs = [r["global_step"] for r in consolidated]
    loss_s = [r.get("train_loss_short") for r in consolidated]
    opaque = [r.get("train_top1_opaque") for r in consolidated]
    struct = [r.get("train_top1_struct") for r in consolidated]
    lr = [r.get("lr") for r in consolidated]
    # val points
    val_gs = [r["global_step"] for r in consolidated if r.get("val_loss_short") is not None]
    val_loss = [r["val_loss_short"] for r in consolidated if r.get("val_loss_short") is not None]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 2, 1]})

    # Bandes de phase (vert = swap OFF, gris = swap ON)
    for ax in (ax1, ax2, ax3):
        for ph in phases:
            color = "#1d3a1d" if ph["swap_off"] else "#2a2a30"
            ax.axvspan(ph["start"], ph["end"], color=color, alpha=0.35, zorder=0)

    # Lignes verticales aux transitions + labels en haut
    for ph in phases:
        ax1.axvline(ph["start"], color="#555", lw=0.6, ls="--", zorder=1)
        ax1.text(ph["start"] + (ph["end"] - ph["start"]) / 2, ax1.get_ylim()[1],
                 ph["label"], fontsize=7.5, ha="center", va="bottom",
                 color="#cfd3da", rotation=0)

    # Panel 1 : loss
    ax1.plot(gs, loss_s, color="#5aa1f0", lw=1.3, label="train_loss (short)")
    ax1.plot(val_gs, val_loss, "o-", color="#f0a050", lw=1.2, ms=5, label="val_loss (short)")
    ax1.set_ylabel("loss"); ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"wan_sprites — évolution entraînement génératif (lignée {total} steps)",
                  fontsize=12, color="#e0e4ea")
    ax1.grid(alpha=0.15)

    # Panel 2 : top1 par classe
    ax2.plot(gs, opaque, color="#e060a0", lw=1.4, label="opaque (couleurs)")
    ax2.plot(gs, struct, color="#60d0a0", lw=1.0, alpha=0.8, label="struct (FRAME_SEP/SEQ_END)")
    ax2.axhline(1/15, color="#888", ls=":", lw=1, label="hasard couleur (0.067)")
    ax2.set_ylabel("top-1 accuracy"); ax2.legend(loc="center right", fontsize=9)
    ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.15)
    # Annotation du déblocage
    ax2.annotate("swap OFF →\ncontenu décolle", xy=(1300, 0.13), xytext=(1450, 0.45),
                 fontsize=9, color="#e060a0",
                 arrowprops=dict(arrowstyle="->", color="#e060a0", lw=1.2))

    # Panel 3 : LR
    ax3.plot(gs, lr, color="#80e0a0", lw=1.2)
    ax3.set_ylabel("LR"); ax3.set_xlabel("global step")
    ax3.set_yscale("log"); ax3.grid(alpha=0.15)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0e1116")
    fig.patch.set_facecolor("#0e1116")
    for ax in (ax1, ax2, ax3):
        ax.tick_params(colors="#aab", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.yaxis.label.set_color("#cfd3da"); ax.xaxis.label.set_color("#cfd3da")

    plt.tight_layout()
    plt.savefig(RUNS / "lineage_3000.png", dpi=130, facecolor="#0e1116")
    print(f"[consolidate] écrit lineage_3000.json / .csv / .png")


if __name__ == "__main__":
    main()
