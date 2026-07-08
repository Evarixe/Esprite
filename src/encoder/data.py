"""Dataset PyTorch pour l'apprentissage contrastif.

Chaque item = une paire (view_a, view_b) de 2 frames distinctes du même cycle.
Augmentations indépendantes par vue : miroir horizontal aléatoire + petit shift ±1-2 px.
Pas de permutation de palette (la palette fait partie du signal d'identité).

Output : tensor (16, 32, 32) float32 one-hot par vue.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _one_hot_16(idx_2d: np.ndarray) -> torch.Tensor:
    """idx (32, 32) uint8 0..15 -> (16, 32, 32) float32 one-hot."""
    t = torch.from_numpy(idx_2d.astype(np.int64))  # (32, 32)
    oh = torch.nn.functional.one_hot(t, num_classes=16)  # (32, 32, 16)
    return oh.permute(2, 0, 1).float()                   # (16, 32, 32)


def _augment(idx_2d: np.ndarray, rng: np.random.Generator,
             flip_p: float = 0.5, max_shift: int = 2) -> np.ndarray:
    """Applique flip horizontal + shift entier. Le padding est en index 0 (transparent)."""
    out = idx_2d
    if rng.random() < flip_p:
        out = np.ascontiguousarray(out[:, ::-1])
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    if dx != 0 or dy != 0:
        H, W = out.shape
        shifted = np.zeros_like(out)
        # source slice and dest slice
        y_src = slice(max(0, -dy), H - max(0, dy))
        y_dst = slice(max(0, dy), H - max(0, -dy))
        x_src = slice(max(0, -dx), W - max(0, dx))
        x_dst = slice(max(0, dx), W - max(0, -dx))
        shifted[y_dst, x_dst] = out[y_src, x_src]
        out = shifted
    return out


class ContrastivePairs(Dataset):
    """Yield (view_a, view_b, cycle_idx).

    cycle_idx renvoyé pour debug/diagnostic, non utilisé par la loss.
    """

    def __init__(self, data_dir: Path, split: str = "train",
                 augment: bool = True, seed: int = 0):
        data = np.load(data_dir / "dataset.npz")
        self.cycles = data["cycles"]    # (N, max_frames, 32, 32)
        self.lengths = data["lengths"]  # (N,)
        splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
        self.indices = np.array(splits[split], dtype=np.int64)
        self.augment = augment
        # Un seed par worker via worker_init_fn idéalement ; pour POC simple, seed global
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        cidx = int(self.indices[i])
        L = int(self.lengths[cidx])
        # rng par item, mais qui doit varier à chaque epoch -> on combine avec un nonce
        # On utilise un rng numpy frais sur (seed, cidx, i) — varie entre epochs via i+offset
        # Pour vraie variabilité inter-epoch, le DataLoader doit shuffle.
        rng = np.random.default_rng()
        # Choix de 2 frames distinctes
        if L == 1:
            a_idx = b_idx = 0
        else:
            a_idx, b_idx = rng.choice(L, size=2, replace=False)
        frame_a = self.cycles[cidx, a_idx]
        frame_b = self.cycles[cidx, b_idx]
        if self.augment:
            frame_a = _augment(frame_a, rng)
            frame_b = _augment(frame_b, rng)
        return _one_hot_16(frame_a), _one_hot_16(frame_b), cidx


class SingleFrames(Dataset):
    """Pour évaluation : yield (sprite_one_hot, cycle_idx, frame_idx).
    Sans augmentation, parcourt toutes les frames de tous les cycles d'un split.
    """

    def __init__(self, data_dir: Path, split: str = "val"):
        data = np.load(data_dir / "dataset.npz")
        self.cycles = data["cycles"]
        self.lengths = data["lengths"]
        splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
        indices = splits[split]
        self.items = []
        for cidx in indices:
            for f in range(int(self.lengths[cidx])):
                self.items.append((cidx, f))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        cidx, f = self.items[i]
        return _one_hot_16(self.cycles[cidx, f]), cidx, f
