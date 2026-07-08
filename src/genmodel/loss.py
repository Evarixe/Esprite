"""Cross-entropy next-token avec pondération par classe de token-cible.

Étant donné :
  logits     : (B, T, V) du modèle (positions 0..T-1 prédisent token suivant)
  tokens     : (B, T)    séquence d'entrée
  roles      : (B, T)    rôle de chaque position (cf vocab.ROLE)
  loss_mask  : (B, T-1)  1 sur les positions de CONTENU à apprendre (exclut padding + préfixe)

On calcule la CE entre logits[:, :-1] et tokens[:, 1:], pondérée par un poids PAR
POSITION dérivé de (rôle de la cible, classe de la cible). Le poids unifie trois
leviers en un seul schéma (cf PROJECT.md « Respect du nombre de frames ») :

  - contenu transparent (index 0)      → w_transp   (down-weight = casse le pari-sûr)
  - contenu couleur (index 1..15)      → w_color    (up-weight  = pousse le dessin)
  - <FRAME_SEP>                        → w_sep       (grammaire « continue »)
  - <SEQ_END>                          → w_end       (le COMPTE — levier max)
  - pixels de la RÉFÉRENCE (dé-masqués)→ w_ref × {w_transp|w_color}  (reconstruction
                                          = apprend à dessiner de zéro, cf cold-start)
  - <REF_END>                          → w_ref_end

Le préfixe de conditionnement (action/dir/N/REF_START/GEN_START) et le padding
restent à poids 0. Loss normalisée par la somme des poids (échelle stable). Top-1
reste calculé sur le CONTENU non pondéré (loss_mask) pour rester comparable.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .vocab import FRAME_SEP, SEQ_END, REF_END, ROLE


@dataclass(frozen=True)
class LossWeights:
    """Coefficients de loss par classe de token-cible. Défauts = point de travail
    courant (proto local), pas neutres — voir PROJECT.md."""
    w_color: float = 1.5     # pixel couleur (1..15) en zone contenu
    w_transp: float = 1.0    # pixel transparent (0) en zone contenu
    w_sep: float = 20.0      # <FRAME_SEP>
    w_end: float = 40.0      # <SEQ_END>
    w_ref: float = 0.5       # multiplicateur global des pixels de ref (0 = ref masquée)
    w_ref_end: float = 20.0  # <REF_END>


NEUTRAL = LossWeights(w_color=1.0, w_transp=1.0, w_sep=1.0, w_end=1.0, w_ref=0.0, w_ref_end=0.0)


def next_token_ce(logits: torch.Tensor, tokens: torch.Tensor, roles: torch.Tensor,
                  loss_mask: torch.Tensor, weights: LossWeights = LossWeights()):
    B, T, V = logits.shape
    shift_logits  = logits[:, :-1].contiguous()        # (B, T-1, V)
    shift_targets = tokens[:, 1:].contiguous()         # (B, T-1)
    shift_roles   = roles[:, 1:].contiguous()          # (B, T-1) rôle de la cible
    mask = loss_mask.to(torch.float32)                 # (B, T-1) contenu, hors padding

    losses = F.cross_entropy(
        shift_logits.view(-1, V), shift_targets.view(-1), reduction="none",
    ).view(B, T - 1)

    # --- Poids par position ---
    w = torch.zeros_like(losses)
    is_transp = (shift_targets == 0)
    is_color  = (shift_targets >= 1) & (shift_targets <= 15)

    # Zone CONTENU : le loss_mask garantit contenu réel (exclut padding + préfixe).
    m = mask.bool()
    w = torch.where(m & is_transp, torch.full_like(w, weights.w_transp), w)
    w = torch.where(m & is_color,  torch.full_like(w, weights.w_color),  w)
    # Tokens structurels (par valeur, > 15 → jamais un pixel) : écrasent la valeur ci-dessus.
    w = torch.where(m & (shift_targets == FRAME_SEP), torch.full_like(w, weights.w_sep), w)
    w = torch.where(m & (shift_targets == SEQ_END),   torch.full_like(w, weights.w_end), w)

    # Zone RÉFÉRENCE (dé-masquée) : rôle PREFIX_PIXEL, jamais du padding (c'est le préfixe).
    if weights.w_ref > 0.0:
        ref = (shift_roles == ROLE.PREFIX_PIXEL)
        w = torch.where(ref & is_transp, torch.full_like(w, weights.w_ref * weights.w_transp), w)
        w = torch.where(ref & is_color,  torch.full_like(w, weights.w_ref * weights.w_color),  w)
    if weights.w_ref_end > 0.0:
        w = torch.where(shift_targets == REF_END, torch.full_like(w, weights.w_ref_end), w)

    weight_sum = w.sum().clamp_min(1.0)
    loss = (losses * w).sum() / weight_sum

    with torch.no_grad():
        mask_sum = mask.sum().clamp_min(1.0)   # positions de contenu (non pondéré)
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_targets).float()

        # Top-1 global et breakdown : sur le CONTENU uniquement (loss_mask), non pondéré.
        top1 = (correct * mask).sum() / mask_sum
        is_transparent = (shift_targets == 0).float() * mask
        is_opaque      = ((shift_targets > 0) & (shift_targets < 16)).float() * mask
        is_struct      = (shift_targets >= 16).float() * mask

        def safe_acc(target_mask):
            n = target_mask.sum().clamp_min(1.0)
            return (correct * target_mask).sum() / n, n

        top1_transp, n_transp = safe_acc(is_transparent)
        top1_opaque, n_opaque = safe_acc(is_opaque)
        top1_struct, n_struct = safe_acc(is_struct)

    return loss, {
        "top1": float(top1.item()),
        "top1_transparent": float(top1_transp.item()),
        "top1_opaque":      float(top1_opaque.item()),
        "top1_struct":      float(top1_struct.item()),
        "frac_transparent": float((n_transp / mask_sum).item()),
        "frac_opaque":      float((n_opaque / mask_sum).item()),
        "frac_struct":      float((n_struct / mask_sum).item()),
        "tokens_supervised": int(mask_sum.item()),
    }
