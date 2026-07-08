"""Classement des checkpoints depuis les votes arena — Bradley-Terry + bootstrap.

Entrée : liste de duels (i, j, outcome) avec outcome = 1.0 (i gagne),
0.5 (égalité / both_bad), 0.0 (j gagne). Indices dans une liste d'ids.

Bradley-Terry par algorithme MM (Hunter 2004) : rapide, stable, sans dépendance
au-delà de numpy. Les égalités comptent 0.5 victoire de chaque côté (approximation
standard, suffisante pour un juge unique). Échelle affichée façon Elo :
rating = 1500 + 400·log10(p) avec moyenne géométrique des forces normalisée à 1.
"""
from __future__ import annotations
import numpy as np


def bradley_terry(n: int, duels: list[tuple[int, int, float]],
                  iters: int = 200, eps: float = 1e-9) -> np.ndarray:
    """Force p (n,) par MM. duels: (i, j, outcome_i)."""
    wins = np.zeros(n)                 # victoires (pondérées) par joueur
    games = np.zeros((n, n))           # nb de duels par paire
    for i, j, o in duels:
        wins[i] += o
        wins[j] += 1.0 - o
        games[i, j] += 1
        games[j, i] += 1
    p = np.ones(n)
    for _ in range(iters):
        denom = np.zeros(n)
        for i in range(n):
            nz = games[i] > 0
            denom[i] = np.sum(games[i, nz] / (p[i] + p[nz]))
        p_new = np.where(denom > 0, (wins + eps) / (denom + eps), p)
        # normalisation : moyenne géométrique = 1
        p_new = p_new / np.exp(np.mean(np.log(np.maximum(p_new, eps))))
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    return p


def to_rating(p: np.ndarray) -> np.ndarray:
    return 1500.0 + 400.0 * np.log10(np.maximum(p, 1e-12))


def bootstrap_ratings(n: int, duels: list[tuple[int, int, float]],
                      n_boot: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """IC 95 % par bootstrap sur les duels. Retourne (lo, hi) en points de rating."""
    if not duels:
        return np.full(n, np.nan), np.full(n, np.nan)
    rng = np.random.default_rng(seed)
    duels_arr = np.array(duels, dtype=float)
    samples = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = rng.integers(0, len(duels), len(duels))
        resampled = [tuple(duels_arr[k]) for k in idx]
        resampled = [(int(i), int(j), o) for i, j, o in resampled]
        samples[b] = to_rating(bradley_terry(n, resampled, iters=60))
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def win_matrix(n: int, duels: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """(wins, games) matrices n×n — wins[i,j] = score de i contre j (égalités = 0.5)."""
    w = np.zeros((n, n))
    g = np.zeros((n, n))
    for i, j, o in duels:
        w[i, j] += o
        w[j, i] += 1.0 - o
        g[i, j] += 1
        g[j, i] += 1
    return w, g


def rank_axis(ids: list[str], duels: list[tuple[int, int, float]],
              with_ci: bool = True) -> dict:
    """Résumé complet d'un axe : ratings + IC + matrice de victoires."""
    n = len(ids)
    if not duels:
        return {"ids": ids, "n_duels": 0, "ratings": None}
    p = bradley_terry(n, duels)
    ratings = to_rating(p)
    lo, hi = bootstrap_ratings(n, duels) if with_ci else (np.full(n, np.nan),) * 2
    w, g = win_matrix(n, duels)
    order = list(np.argsort(-ratings))
    return {
        "ids": ids,
        "n_duels": len(duels),
        "ratings": {ids[i]: round(float(ratings[i]), 1) for i in range(n)},
        "ci95": {ids[i]: [round(float(lo[i]), 1), round(float(hi[i]), 1)] for i in range(n)},
        "ranking": [ids[i] for i in order],
        "winrate": {ids[i]: {ids[j]: (round(float(w[i, j] / g[i, j]), 3) if g[i, j] > 0 else None)
                             for j in range(n) if j != i} for i in range(n)},
        "games": {ids[i]: {ids[j]: int(g[i, j]) for j in range(n) if j != i} for i in range(n)},
    }
