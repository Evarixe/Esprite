"""Approche B — métrique d'écart de position de <SEQ_END> (évaluation, pas gradient).

La structure d'une séquence de génération est déterministe une fois N (nb de frames
demandé) fixé : après <GEN_START>, N frames occupent N×1024 pixels + (N−1) <FRAME_SEP>
+ 1 <SEQ_END>, soit exactement N×1025 tokens. La position attendue de <SEQ_END>,
mesurée en offset depuis <GEN_START>, vaut donc précisément N×1025.

On génère en demandant N frames, on repère la position réelle d'émission de <SEQ_END>
(= len(raw_tokens), puisque le flot émis inclut le <SEQ_END> terminal), et on calcule
l'écart signé `réel − N×1025` :
  - écart < 0 : le modèle s'arrête trop tôt (animation tronquée).
  - écart > 0 : déborde (frames en trop).
  - <SEQ_END> jamais émis (stop_reason ≠ 'seq_end') : runaway / censuré à droite, compté
    à part (`no_seq_end`), PAS agrégé dans l'écart moyen (sa vraie valeur est inconnue,
    bornée par le cap de sécurité).

Métrique non différentiable (échantillonnage autoregressif) → évaluation/monitoring
uniquement. L'approche A (reweighting de loss) est ce qui produit l'effet ; B le mesure.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch

from .vocab import SOURCE_ACTION_MAP, PIXELS_PER_FRAME
from .sample import GenRequest, generate
from .graph_sampler import GraphSamplerPool

_FRAME_STRIDE = PIXELS_PER_FRAME + 1   # 1025 = 1024 pixels + 1 token (séparateur/fin)


@dataclass
class GapResult:
    requested_n: int
    emitted_seq_end: bool
    real_pos: int | None        # offset de SEQ_END depuis GEN_START (None si non émis)
    expected_pos: int           # N × 1025
    gap: int | None             # real_pos − expected_pos (None si non émis)
    stop_reason: str
    n_emitted: int              # nb total de tokens émis (= longueur du flot)


@dataclass
class GapSummary:
    n_total: int = 0
    n_seq_end: int = 0                       # générations où SEQ_END a été émis
    mean_abs_gap: float = float("nan")       # moyenne |écart| sur les SEQ_END émis
    mean_signed_gap: float = float("nan")    # moyenne signée (biais tôt/tard)
    median_abs_gap: float = float("nan")
    frac_seq_end: float = float("nan")       # part des générations qui terminent proprement
    frac_exact: float = float("nan")         # part avec écart == 0 (parmi SEQ_END émis)
    per_n: dict = field(default_factory=dict)  # N -> {mean_abs_gap, frac_seq_end, count}
    results: list = field(default_factory=list)


def _summarize(results: list[GapResult]) -> GapSummary:
    s = GapSummary(n_total=len(results), results=results)
    if not results:
        return s
    emitted = [r for r in results if r.emitted_seq_end]
    s.n_seq_end = len(emitted)
    s.frac_seq_end = len(emitted) / len(results)
    if emitted:
        gaps = np.array([r.gap for r in emitted], dtype=np.float64)
        s.mean_abs_gap = float(np.abs(gaps).mean())
        s.mean_signed_gap = float(gaps.mean())
        s.median_abs_gap = float(np.median(np.abs(gaps)))
        s.frac_exact = float((gaps == 0).mean())
    # breakdown par N
    by_n: dict[int, list[GapResult]] = {}
    for r in results:
        by_n.setdefault(r.requested_n, []).append(r)
    for n, rs in sorted(by_n.items()):
        em = [r for r in rs if r.emitted_seq_end]
        s.per_n[n] = {
            "count": len(rs),
            "frac_seq_end": len(em) / len(rs),
            "mean_abs_gap": float(np.mean([abs(r.gap) for r in em])) if em else float("nan"),
            "mean_signed_gap": float(np.mean([r.gap for r in em])) if em else float("nan"),
        }
    return s


@torch.no_grad()
def seqend_gap_eval(model, dataset, local_indices, device: str = "cuda",
                    temperature: float = 0.3, with_ref: bool = True,
                    seed: int = 0, pool: GraphSamplerPool | None = None,
                    max_frames_safety: int = 16) -> GapSummary:
    """Évalue l'écart de position de SEQ_END sur un échantillon du dataset.

    Pour chaque cycle d'indice local `i` : on demande N = sa vraie longueur, avec
    sa frame 0 comme référence (si with_ref), et on mesure où le modèle place SEQ_END.

    Le modèle DOIT être en bf16 statique pour le path graph (sinon fallback dynamique,
    plus lent mais correct). `pool` réutilisable entre appels pour amortir captures.
    """
    own_pool = pool is None
    if own_pool and device == "cuda" and next(model.parameters()).dtype == torch.bfloat16:
        pool = GraphSamplerPool(model)

    results: list[GapResult] = []
    for k, i in enumerate(local_indices):
        ci = int(dataset.indices[int(i)])
        L = int(dataset.lengths[ci])
        if L > max_frames_safety:
            continue
        meta = dataset.meta[ci]
        action = SOURCE_ACTION_MAP[meta["action"]]
        ref = dataset.cycles[ci][0] if with_ref else None
        req = GenRequest(
            action=action,
            direction=meta.get("direction"),
            frames=L,
            reference=ref,
            temperature=temperature,
            seed=seed + k,
            max_frames_safety=max_frames_safety,
        )
        res = generate(model, req, device=device, pool=pool)
        emitted = res.stop_reason == "seq_end"
        real_pos = len(res.raw_tokens) if emitted else None
        expected = L * _FRAME_STRIDE
        gap = (real_pos - expected) if emitted else None
        results.append(GapResult(
            requested_n=L, emitted_seq_end=emitted, real_pos=real_pos,
            expected_pos=expected, gap=gap, stop_reason=res.stop_reason,
            n_emitted=len(res.raw_tokens),
        ))

    return _summarize(results)


def pick_eval_indices(dataset, n_per_bucket: int = 12, seed: int = 0) -> list[int]:
    """Échantillonne des indices locaux équilibrés entre bucket court et long, pour
    que la métrique couvre les deux régimes de longueur (sinon 96% de N=2 noie le long)."""
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for bk, locs in dataset.buckets.items():
        if not locs:
            continue
        take = min(n_per_bucket, len(locs))
        picked.extend(rng.choice(locs, size=take, replace=False).tolist())
    return picked
