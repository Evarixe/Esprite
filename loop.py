"""Orchestrateur de la boucle human-in-the-loop (workflow unifie SFT + DPO).

Machine a etats re-invocable (pause humaine obligatoire au vote). Un seul continuum :
horloge globale + cosine LR partagees ; SFT et DPO sont des TYPES de step sur la meme
courbe (DPO lit lr_at(step)/10, beta=1 -- cf PROJECT.md "Boucle human-in-the-loop").

    set PYTHONPATH=src
    uv run python loop.py init --ckpt runs/gen_20000/last.pt --global-step 20000 --horizon 40000
    uv run python loop.py run        # phase SFT : entraine K steps, genere la campagne DPO, s'arrete
    #   -> lancer l'arene + voter (la commande est imprimee), puis :
    uv run python loop.py run        # phase vote : si quota atteint, lance le DPO, cycle suivant
    uv run python loop.py status

Phases : sft -> await_votes -> (dpo) -> sft -> ... -> done (global_step >= horizon).
Shell-out bloquant vers train.py / dpo_campaign.py / dpo_train.py. Les serveurs
arene/dashboard restent lances a cote (plus robuste sur Windows).
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from genmodel.train import get_lr

LOOP_DIR = Path("runs") / "dpo_loop"    # LA lignee : un seul run (history/last.pt/heartbeat uniques)
STATE = LOOP_DIR / "state.json"
LEDGER = LOOP_DIR / "shown_ledger.json"
SNAP_DIR = LOOP_DIR / "snapshots"       # jalons de lignee (rollback), un .pt par fin de phase
CAMP_DIR = LOOP_DIR / "campaigns"       # data de campagnes (campaign.json/gens/votes), pas des runs
PY = sys.executable   # meme interpreteur que celui qui lance loop.py (chemin absolu robuste)
ENV = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8"}


def sh(cmd: list):
    cmd = [str(c) for c in cmd]   # subprocess exige des str (on passe des int/float/Path)
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=ENV, check=True)


def load() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save(s: dict):
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def cosine_lr(s: dict, step: int) -> float:
    """LR SFT a un step global, sur la cosine [decay_start -> horizon]."""
    return get_lr(step - s["decay_start"], s["horizon"] - s["decay_start"], 0,
                  s["lr_peak"], s["min_lr_frac"])


def ckpt_step(p: Path) -> int:
    import torch
    return int(torch.load(p, map_location="cpu", weights_only=False).get("step", 0))


def snapshot(global_step: int):
    """Copie la tete last.pt en jalon de lignee snapshots/step_XXXXXXX.pt (rollback)."""
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    src = LOOP_DIR / "last.pt"
    if src.exists():
        dst = SNAP_DIR / f"step_{global_step:07d}.pt"
        shutil.copy2(src, dst)
        print(f"[loop] snapshot -> {dst}")


def cmd_init(a):
    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    s = {
        "cycle": 0, "global_step": a.global_step, "decay_start": a.global_step,
        "horizon": a.horizon, "lr_peak": a.lr_peak, "min_lr_frac": a.min_lr_frac,
        "ckpt": str(a.ckpt), "sft_steps": a.sft_steps, "duels": a.duels,
        "dpo_epochs": a.dpo_epochs, "dpo_beta": a.dpo_beta, "dpo_lr_div": a.dpo_lr_div,
        "window": a.window, "campaigns": [], "phase": "sft",
    }
    save(s)
    print(f"[loop] init : step {s['global_step']} -> horizon {s['horizon']}, "
          f"cycle SFT {s['sft_steps']} / DPO {s['duels']} duels. phase=sft")


def phase_sft(s: dict):
    c = s["cycle"]
    lr = cosine_lr(s, s["global_step"])
    off = s["global_step"] - s["decay_start"]
    tot = s["horizon"] - s["decay_start"]
    out = LOOP_DIR   # meme dossier lignee : history append, last.pt roulant
    print(f"[loop] cycle {c} -- SFT {s['sft_steps']} steps @ LR {lr:.2e} "
          f"(global {s['global_step']} -> {s['global_step']+s['sft_steps']})")
    sh([PY, "-m", "genmodel.train", "--out", out, "--resume", s["ckpt"], "--append-history",
        "--history-base", s["global_step"],   # horloge absolue (off = position decroissance LR)
        "--lr", s["lr_peak"], "--global-offset", off, "--global-total", tot, "--min-lr-frac", s["min_lr_frac"],
        "--total-steps", s["sft_steps"], "--no-palette-swap",
        "--batch-short", 16, "--batch-long", 1, "--num-workers", 0,
        "--grad-ckpt-threshold", 6000, "--empty-cache-every", 250,
        "--pause-temp-c", 82, "--pause-cool-c", 52,
        "--log-every", 25, "--monitor-every", 25, "--val-every", 250, "--ckpt-every", 100])
    # Lit le step GLOBAL reel du checkpoint (train.py le sauve) au lieu de crediter
    # sft_steps en aveugle : un arret anticipe (stop dashboard) n'avance l'horloge que
    # des steps reellement faits. Fallback += sft_steps si le ckpt n'a pas de step.
    done = ckpt_step(out / "last.pt")
    s["global_step"] = done if done > s["global_step"] else s["global_step"] + s["sft_steps"]
    s["ckpt"] = str(out / "last.pt")
    snapshot(s["global_step"])
    s["phase"] = "campaign"
    save(s)   # SFT persiste AVANT la campagne : jamais reperdu si la gen crashe
    print(f"[loop] SFT termine (global {s['global_step']}). phase=campaign -- relance `loop.py run`")


def phase_campaign(s: dict):
    c = s["cycle"]
    camp_dir = CAMP_DIR / f"c{c:02d}"   # DATA de campagne sous la lignee (pas un run separe)
    print(f"[loop] cycle {c} -- generation campagne DPO ({s['duels']} items x 2 seeds) -> {camp_dir}")
    sh([PY, "dpo_campaign.py", "--ckpt", s["ckpt"], "--name", f"dpoloop_c{c:02d}",
        "--out-dir", camp_dir, "--run-dir", LOOP_DIR,   # heartbeat/config battent dans la lignee
        "--n-items", s["duels"], "--no-ref-frac", 0.5, "--temperature", 0.7,
        "--seed", c, "--ledger", LEDGER])
    s["campaigns"] = (s["campaigns"] + [str(camp_dir)])[-s["window"]:]
    s["phase"] = "await_votes"
    save(s)
    print(f"\n[loop] >> VOTE - lance l'arene puis vote {s['duels']} duels :")
    print(f"    {PY} -m arena.server --campaign {camp_dir} --port 8766 --target-duels {s['duels']}")
    print(f"  puis :  uv run python loop.py run")


def phase_votes(s: dict):
    camp = Path(s["campaigns"][-1])
    votes = camp / "votes.jsonl"
    n = sum(1 for l in votes.read_text(encoding="utf-8").splitlines() if l.strip() and '"mode": "arena"' in l) if votes.exists() else 0
    if n < s["duels"]:
        print(f"[loop] vote en cours : {n}/{s['duels']} duels -- vote encore puis relance `loop.py run`")
        return
    c = s["cycle"]
    dpo_lr = cosine_lr(s, s["global_step"]) / s["dpo_lr_div"]
    out = LOOP_DIR   # meme dossier lignee : history append (mode dpo), last.pt roulant
    print(f"[loop] cycle {c} -- DPO @ LR {dpo_lr:.2e} (= cosine/{s['dpo_lr_div']}), beta={s['dpo_beta']}, "
          f"fenetre {len(s['campaigns'])} campagne(s)")
    camp_args = []
    for cd in s["campaigns"]:
        camp_args += ["--campaign", cd]
    sh([PY, "dpo_train.py", "--ckpt", s["ckpt"], *camp_args, "--out", out,
        "--beta", s["dpo_beta"], "--lr", dpo_lr, "--epochs", s["dpo_epochs"],
        "--max-frames", 16, "--global-offset", s["global_step"]])
    s["global_step"] = ckpt_step(out / "last.pt")   # DPO avance l'horloge
    s["ckpt"] = str(out / "last.pt")
    snapshot(s["global_step"])
    s["cycle"] += 1
    s["phase"] = "done" if s["global_step"] >= s["horizon"] else "sft"
    save(s)
    print(f"[loop] cycle {c} boucle. global_step={s['global_step']}, ckpt={s['ckpt']}, phase={s['phase']}")
    if s["phase"] == "done":
        print("[loop] [OK] horizon atteint -- boucle terminee.")


def cmd_run(a):
    s = load()
    if s["phase"] == "sft":
        phase_sft(s)
    elif s["phase"] == "campaign":
        phase_campaign(s)
    elif s["phase"] == "await_votes":
        phase_votes(s)
    else:
        print(f"[loop] phase={s['phase']} -- rien a faire.")


def cmd_status(a):
    s = load()
    print(json.dumps({k: s[k] for k in ("cycle", "global_step", "horizon", "phase", "ckpt")}, indent=2))
    print(f"LR SFT courant : {cosine_lr(s, s['global_step']):.2e} | LR DPO : {cosine_lr(s, s['global_step'])/s['dpo_lr_div']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--ckpt", type=Path, required=True)
    i.add_argument("--global-step", type=int, required=True)
    i.add_argument("--horizon", type=int, default=40000)
    i.add_argument("--lr-peak", type=float, default=3e-4)
    i.add_argument("--min-lr-frac", type=float, default=0.10)
    i.add_argument("--sft-steps", type=int, default=3000)
    i.add_argument("--duels", type=int, default=50)
    i.add_argument("--dpo-epochs", type=int, default=3)
    i.add_argument("--dpo-beta", type=float, default=1.0)
    i.add_argument("--dpo-lr-div", type=float, default=10.0)
    i.add_argument("--window", type=int, default=3)
    i.set_defaults(func=cmd_init)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("status").set_defaults(func=cmd_status)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
