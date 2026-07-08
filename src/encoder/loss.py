"""NT-Xent loss (SimCLR-style InfoNCE).

Inputs :
  z_a, z_b : (B, D), déjà L2-normalisés.
  temperature τ.

On forme 2B vues, calcule la matrice de similarité 2B x 2B, masque la diagonale,
et pour chaque vue la positive est son partenaire (vue i dans batch a <-> vue i
dans batch b). Tous les autres = négatives.

Loss = - mean log( exp(sim_pos/τ) / sum_{k != i} exp(sim_ik/τ) ).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.2) -> tuple[torch.Tensor, dict]:
    B, D = z_a.shape
    z = torch.cat([z_a, z_b], dim=0)            # (2B, D)
    sim = (z @ z.t()) / temperature             # (2B, 2B)
    # Masque diagonale (un sample n'est pas sa propre négative ni positive)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Pour chaque ligne i (0..2B-1), la positive est :
    #   i + B si i < B, sinon i - B
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)

    loss = F.cross_entropy(sim, targets)

    # Métriques accessoires
    with torch.no_grad():
        pred = sim.argmax(dim=1)
        top1 = (pred == targets).float().mean().item()
        # Alignment : 1 - cos(z_a, z_b) moyen, plus bas = mieux
        alignment = (1 - (z_a * z_b).sum(dim=1)).mean().item()

    return loss, {"top1": top1, "alignment": alignment}
