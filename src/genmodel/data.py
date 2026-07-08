"""Dataset PyTorch pour le transformer génératif.

Stratégie de bucket batching :
  Bucket 'short' = cycles de 2 frames           (4812 cycles)
  Bucket 'long'  = cycles de 8 à 16 frames      (32 cycles, padding au max du batch)

À chaque sample on tire `has_ref` (Bernoulli p=ref_prob) — l'image de référence est
la frame 0 du cycle. Le mask de loss est inchangé : seuls les pixels content et les
séparateurs sont supervisés.

La séparation par bucket maintient des longueurs HOMOGÈNES dans un batch et évite
le padding excessif (le brief y insiste).

Le sampler :
  BucketBatchSampler échantillonne uniformément à l'intérieur d'un bucket puis
  alterne les buckets pour répartir équitablement la vue par le modèle.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenize import encode_cycle, make_loss_mask
from .vocab import MAX_FRAMES


SHORT_BUCKET = "short"   # n_frames == 2
LONG_BUCKET = "long"     # n_frames in [3..16]


class CyclesTokenDataset(Dataset):
    """Yield un dict par item :
        tokens     : (T,) int64
        loss_mask  : (T-1,) int8
        x_pos      : (T,) int8
        y_pos      : (T,) int8
        frame_pos  : (T,) int8
        roles      : (T,) int8
        gen_start  : int scalar (position de <GEN_START>)
        bucket     : str
        cycle_idx  : int
    """

    def __init__(self, data_dir: Path, split: str = "train",
                 ref_prob: float = 0.5, augment_palette: bool = True, seed: int = 0):
        data = np.load(data_dir / "dataset.npz")
        self.cycles = data["cycles"]      # (N, 16, 32, 32)
        self.lengths = data["lengths"]    # (N,) uint8
        splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
        self.indices = np.array(splits[split], dtype=np.int64)
        self.meta = json.loads((data_dir / "dataset.meta.json").read_text(encoding="utf-8"))
        self.ref_prob = ref_prob
        self.augment_palette = augment_palette

        # Pré-calcul des buckets sur le split
        self.buckets: dict[str, list[int]] = {SHORT_BUCKET: [], LONG_BUCKET: []}
        for local_i, ci in enumerate(self.indices):
            L = int(self.lengths[ci])
            self.buckets[SHORT_BUCKET if L == 2 else LONG_BUCKET].append(local_i)

    def __len__(self) -> int:
        return len(self.indices)

    def bucket_of(self, local_i: int) -> str:
        ci = int(self.indices[local_i])
        L = int(self.lengths[ci])
        return SHORT_BUCKET if L == 2 else LONG_BUCKET

    def __getitem__(self, local_i: int) -> dict:
        ci = int(self.indices[local_i])
        L = int(self.lengths[ci])
        meta = self.meta[ci]
        # rng par-item — pour reproductibilité parfaite il faudrait dériver de (epoch, ci)
        rng = np.random.default_rng()
        has_ref = rng.random() < self.ref_prob
        cycle_frames = self.cycles[ci]
        ref = cycle_frames[0] if has_ref else None

        # --- Palette swap (augmentation) ---
        # Permute les indices 1..15. Index 0 (transparence) reste fixe.
        # MÊME permutation appliquée à toutes les frames + à la ref pour préserver
        # la cohérence intra-cycle ET la correspondance ref ↔ frames cibles.
        if self.augment_palette:
            perm = np.zeros(16, dtype=np.uint8)
            perm[0] = 0
            perm[1:] = rng.permutation(np.arange(1, 16, dtype=np.uint8))
            cycle_frames = perm[cycle_frames]                   # (16, 32, 32) lookup
            if ref is not None:
                ref = perm[ref]

        seq = encode_cycle(
            cycle_frames=cycle_frames,
            length=L,
            action_source=meta["action"],
            direction=meta.get("direction"),
            ref_frame_32x32=ref,
        )
        loss_mask = make_loss_mask(seq)

        return {
            "tokens": torch.from_numpy(seq.tokens.astype(np.int64)),
            "loss_mask": torch.from_numpy(loss_mask.astype(np.int8)),
            "x_pos": torch.from_numpy(seq.x_pos.astype(np.int64)),
            "y_pos": torch.from_numpy(seq.y_pos.astype(np.int64)),
            "frame_pos": torch.from_numpy(seq.frame_pos.astype(np.int64)),
            "roles": torch.from_numpy(seq.roles.astype(np.int64)),
            "gen_start": seq.gen_start_idx,
            "bucket": self.bucket_of(local_i),
            "cycle_idx": ci,
        }


def collate_pad(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Pad jusqu'à la longueur max du batch. Le pad est ignoré via loss_mask."""
    T = max(item["tokens"].shape[0] for item in batch)
    B = len(batch)

    out_tokens    = torch.full((B, T), pad_token_id, dtype=torch.long)
    out_x         = torch.zeros((B, T), dtype=torch.long)
    out_y         = torch.zeros((B, T), dtype=torch.long)
    out_f         = torch.zeros((B, T), dtype=torch.long)
    out_roles     = torch.zeros((B, T), dtype=torch.long)
    out_mask      = torch.zeros((B, T - 1), dtype=torch.int8)
    out_attn_mask = torch.zeros((B, T), dtype=torch.bool)   # True = position valide

    for i, item in enumerate(batch):
        L = item["tokens"].shape[0]
        out_tokens[i, :L]    = item["tokens"]
        out_x[i, :L]         = item["x_pos"]
        out_y[i, :L]         = item["y_pos"]
        out_f[i, :L]         = item["frame_pos"]
        out_roles[i, :L]     = item["roles"]
        out_mask[i, :L - 1]  = item["loss_mask"]
        out_attn_mask[i, :L] = True

    return {
        "tokens": out_tokens,
        "x_pos": out_x, "y_pos": out_y, "frame_pos": out_f,
        "roles": out_roles,
        "loss_mask": out_mask,
        "attn_mask": out_attn_mask,
        "buckets": [b["bucket"] for b in batch],
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Échantillonne des batches homogènes par bucket.

    Une 'epoch' = un passage sur toutes les indices, regroupés par bucket.
    Les buckets sont mélangés à chaque epoch ; à l'intérieur d'un bucket, les
    indices sont mélangés puis découpés en batches de taille fixe.

    Si bucket plus petit que batch_size : drop_last=False -> batch incomplet
    autorisé (utile pour le bucket 'long' de 32 cycles).
    """

    def __init__(self, dataset: CyclesTokenDataset, batch_sizes: dict[str, int],
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0
        self._seed = seed

    def set_epoch(self, e: int):
        self._epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        all_batches: list[list[int]] = []
        for bk, local_idxs in self.dataset.buckets.items():
            # Skip les buckets absents de batch_sizes (ex: exclure le bucket long)
            if bk not in self.batch_sizes:
                continue
            idxs = np.array(local_idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            bs = self.batch_sizes[bk]
            n = len(idxs)
            n_full = n // bs
            for k in range(n_full):
                all_batches.append(idxs[k * bs:(k + 1) * bs].tolist())
            rem = n - n_full * bs
            if rem and not self.drop_last:
                all_batches.append(idxs[n_full * bs:].tolist())
        if self.shuffle:
            rng.shuffle(all_batches)
        for b in all_batches:
            yield b

    def __len__(self):
        total = 0
        for bk, local_idxs in self.dataset.buckets.items():
            if bk not in self.batch_sizes:
                continue
            bs = self.batch_sizes[bk]
            n = len(local_idxs)
            if self.drop_last:
                total += n // bs
            else:
                total += (n + bs - 1) // bs
        return total
