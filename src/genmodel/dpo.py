"""DPO depuis les preferences d'arene -- fine-tuning du generateur sur l'oeil humain.

Principe (aligne brief : humain arbitre, PAS de reward model appris -> pas de Goodhart) :
les votes A>B de l'arene sont directement des paires (chosen, rejected) pour un meme
prompt. On fine-tune une copie de la politique (init = checkpoint courant) pour
augmenter la log-vraisemblance du chosen et diminuer celle du rejected, regularise
par une reference figee (le meme checkpoint) :

    L = -log ?( beta ? [ (log?(c) - log?_ref(c)) - (log?(r) - log?_ref(r)) ] )

log?(y) = somme des log-probs des tokens de CONTENU (frames generees + separateurs),
soit -(CE masquee) sommee sur la zone de generation. Les sequences chosen/rejected
sont reconstruites depuis les frames stockees de chaque generation (meme tokenizer
que l'entrainement), avec le prefixe (action/dir/N/ref) de l'item.
"""
from __future__ import annotations
import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .tokenize import encode_cycle, make_loss_mask
from .vocab import MAX_FRAMES, SPRITE_SIZE, ROLE

# Priorite d'axe pour deduire la preference quand le vote est par-axe (mouvement d'abord).
_AXIS_PRIORITY = ("movement", "cleanliness", "fidelity")


def _b64_to_frame(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(SPRITE_SIZE, SPRITE_SIZE)


def preference_of(vote: dict) -> str | None:
    """'left'/'right' (le cote prefere) ou None (tie/both_bad/ambigu)."""
    o = vote.get("outcome")
    if o in ("left", "right"):
        return o
    axes = vote.get("axes") or {}
    for ax in _AXIS_PRIORITY:
        if axes.get(ax) in ("left", "right"):
            return axes[ax]
    return None


@dataclass
class Pair:
    item_id: str
    chosen_ckpt: str
    rejected_ckpt: str


def extract_pairs(votes: list[dict]) -> list[Pair]:
    """Votes d'arene -> paires (chosen, rejected). Ignore ties/both_bad et l'ancre
    dataset comme CHOSEN n'a pas de sens a imiter (le modele ne peut pas devenir le
    dataset) -- mais dataset en rejected/chosen reste informatif. On garde tout sauf
    les paires impliquant deux fois le meme ckpt."""
    pairs = []
    for v in votes:
        if v.get("mode") != "arena":
            continue
        pref = preference_of(v)
        if pref is None:
            continue
        left, right = v.get("left"), v.get("right")
        if not left or not right or left == right:
            continue
        chosen, rejected = (left, right) if pref == "left" else (right, left)
        pairs.append(Pair(v["item_id"], chosen, rejected))
    return pairs


def build_sequence(item: dict, gen: dict) -> dict:
    """Reconstruit la sequence de tokens d'une generation stockee, prete pour le
    forward. Retourne dict de tenseurs (1, T) + loss_mask (1, T-1)."""
    frames_b64 = gen["frames"]
    n = max(1, min(MAX_FRAMES, len(frames_b64)))
    cyc = np.zeros((MAX_FRAMES, SPRITE_SIZE, SPRITE_SIZE), dtype=np.uint8)
    for i in range(n):
        cyc[i] = _b64_to_frame(frames_b64[i])
    ref = _b64_to_frame(item["ref"]) if item.get("with_ref") else None
    seq = encode_cycle(cyc, n, item["source"], item.get("direction"), ref)
    lm = make_loss_mask(seq)
    t = lambda a, dt: torch.from_numpy(a.astype(dt)).unsqueeze(0)
    return {
        "tokens": t(seq.tokens, np.int64),
        "x_pos": t(seq.x_pos, np.int64),
        "y_pos": t(seq.y_pos, np.int64),
        "frame_pos": t(seq.frame_pos, np.int64),
        "roles": t(seq.roles, np.int64),
        "loss_mask": t(lm, np.float32),
        "n_frames": n,
    }


def sequence_logprob(model, batch: dict, device: str, length_normalized: bool = True,
                     exclude_struct: bool = True) -> torch.Tensor:
    """log?(reponse|prompt) sur la zone de contenu.

    length_normalized=True -> MOYENNE par token (et non somme) : chosen/rejected ont
    des longueurs differentes -> la somme recompense les sequences longues (biais de
    longueur -> runaway) ET amplifie le gradient ? L. La moyenne rend la preference
    par-token, independante de la longueur.

    exclude_struct=True -> masque `FRAME_SEP`/`SEQ_END` (role CONTENT_SEP), le DPO ne
    porte que sur les PIXELS. Les decisions de compte de frames sont basse entropie
    (quasi-deterministes) : un micro-shift les flippe -> runaway. On GELE cette grammaire
    (durement acquise via l'approche A) et le DPO ne raffine que l'esthetique des pixels."""
    tok = batch["tokens"].to(device)
    x = batch["x_pos"].to(device); y = batch["y_pos"].to(device)
    fp = batch["frame_pos"].to(device); roles = batch["roles"].to(device)
    lm = batch["loss_mask"].to(device).clone()               # (1, T-1) contenu = pixels+sep
    if exclude_struct:
        lm = lm * (roles[:, 1:] == ROLE.CONTENT_PIXEL).to(lm.dtype)   # pixels seuls
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tok, x, y, fp, roles, attn_mask=None)  # (1, T, V)
    logits = logits[:, :-1].float()                           # (1, T-1, V)
    targets = tok[:, 1:]                                       # (1, T-1)
    logp = F.log_softmax(logits, dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    s = (tgt_logp * lm).sum()
    return s / lm.sum().clamp_min(1.0) if length_normalized else s


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta: float):
    """DPO scalaire pour une paire. policy/ref = log? sommes (chosen/rejected)."""
    pi_logratio = policy_c - policy_r
    ref_logratio = ref_c - ref_r
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)
    # marge (pour monitoring) : >0 = la politique prefere chosen davantage que la ref
    margin = (policy_c - policy_r).detach()
    return loss, margin, logits.detach()
