"""Sanity check de l'encodeur entraîné.

Produit deux artefacts :
  1) data/features_pca.png : projection PCA 2D des features sur le val set,
     colorée par pokemon_id (hash -> teinte). Si l'encodeur fonctionne, les
     points du même pokemon doivent se grouper.
  2) Stats console : distance moyenne intra-cycle vs inter-cycle, intra-pokemon
     vs inter-pokemon.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from encoder.data import SingleFrames
from encoder.model import SpriteEncoder


@torch.no_grad()
def encode_all(model, loader, device):
    feats, cidxs, fidxs = [], [], []
    for sprite, cidx, f in loader:
        sprite = sprite.to(device, non_blocking=True)
        z = model(sprite)
        feats.append(z.cpu().numpy())
        cidxs.append(cidx.numpy())
        fidxs.append(f.numpy())
    return (np.concatenate(feats), np.concatenate(cidxs), np.concatenate(fidxs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/encoder_v1/best.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", type=Path, default=Path("runs/encoder_v1/features_pca.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SingleFrames(args.data, split=args.split)
    print(f"[viz] {len(ds)} frames")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    model = SpriteEncoder().to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    feats, cidxs, fidxs = encode_all(model, loader, device)
    print(f"[viz] features shape: {feats.shape}")

    # PCA 2D
    p = PCA(n_components=2)
    f2 = p.fit_transform(feats)
    print(f"[viz] PCA explained variance: {p.explained_variance_ratio_}")

    # Charger meta pour avoir pokemon_id par cycle
    meta = json.loads((args.data / "dataset.meta.json").read_text(encoding="utf-8"))
    pokemon_ids = np.array([meta[c]["pokemon_id"] for c in cidxs])
    sources = np.array([meta[c]["source"] for c in cidxs])

    # Rendu matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Panel 1 : couleur = pokemon_id (modulo pour teinte)
    colors = (pokemon_ids % 100) / 100.0
    axes[0].scatter(f2[:, 0], f2[:, 1], c=colors, cmap="hsv", s=6, alpha=0.6)
    axes[0].set_title(f"PCA features ({args.split}) — hue = pokemon_id mod 100")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # Panel 2 : couleur = source
    src_to_int = {s: i for i, s in enumerate(sorted(set(sources)))}
    src_codes = np.array([src_to_int[s] for s in sources])
    sc = axes[1].scatter(f2[:, 0], f2[:, 1], c=src_codes, cmap="tab10", s=6, alpha=0.6)
    axes[1].set_title(f"PCA features ({args.split}) — color = source")
    # Légende
    for s, i in src_to_int.items():
        axes[1].scatter([], [], c=[sc.cmap(sc.norm(i))], label=s)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"[viz] saved {args.out}")

    # Stats : distances intra/inter
    # On échantillonne pour éviter NxN sur tout le split.
    n = min(2000, len(feats))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(feats), n, replace=False)
    F = feats[idx]; C = cidxs[idx]; P = pokemon_ids[idx]
    # Matrice distances cosinus (features déjà L2-norm -> 1 - F@F.T)
    sim = F @ F.T
    dist = 1.0 - sim
    same_cycle = (C[:, None] == C[None, :])
    same_poke = (P[:, None] == P[None, :])
    diag = np.eye(n, dtype=bool)

    intra_cycle = dist[same_cycle & ~diag].mean()
    intra_poke_inter_cycle = dist[same_poke & ~same_cycle].mean() if (same_poke & ~same_cycle).any() else float("nan")
    inter_poke = dist[~same_poke].mean()
    print(f"[stats] mean cosine distance (lower = more similar):")
    print(f"  intra-cycle:                  {intra_cycle:.4f}")
    print(f"  intra-pokemon, inter-cycle:   {intra_poke_inter_cycle:.4f}")
    print(f"  inter-pokemon:                {inter_poke:.4f}")


if __name__ == "__main__":
    main()
