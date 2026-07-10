"""Transformer décodeur autoregressif ~50M paramètres.

Spécifications brief Partie 2 :
  dim_model=512, n_layers=16, n_heads=16, dim_head=32, dim_ff=2048
  GELU, RMSNorm pre-norm, BF16 mixed precision, SDPA causale
  Embeddings : token (appris), sinusoïdes seq pos, x/y/frame appris pixel-only

Notes d'implémentation :
  - On utilise torch.nn.functional.scaled_dot_product_attention (SDPA) qui sélectionne
    Flash Attention 2 quand possible sur Blackwell sm_120.
  - Le masque d'attention combine causal + padding via un masque additif 2D.
  - Embeddings x/y/frame conditionnés par un masque is_pixel : on multiplie l'embedding
    par 0 aux positions non-pixel pour éviter de polluer les tokens structurels.
  - Sortie unique 36-way softmax (POC v1) — entmax/sparsemax sur la tête pixel sera
    une ablation v2 (cf brief).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .vocab import VOCAB_SIZE, ROLE, SPRITE_SIZE, MAX_FRAMES, ID, COLOR


# --------------------------------------------------------------------------- norms

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# --------------------------------------------------------------------------- positional

def sinusoidal_table(max_len: int, dim: int) -> torch.Tensor:
    """Table sinusoïdale (max_len, dim), non apprenable."""
    pe = torch.zeros(max_len, dim)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# --------------------------------------------------------------------------- attention

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    # ---------- Static cache forward (1-token step, CUDA Graphs friendly) ----------
    def forward_static(self, x: torch.Tensor,
                       k_buf: torch.Tensor, v_buf: torch.Tensor,
                       pos_idx: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x        : (1, 1, D)
        k_buf,v_buf : (1, H, max_len, head_dim) — buffers preallocated, écrits in-place
        pos_idx  : (1,) long — index où écrire K/V (et où l'on est dans la séquence)
        attn_mask: (1, 1, 1, max_len) du même dtype que x — 0 sur positions valides, -inf sur padding
        Retourne : (1, 1, D)

        Implémentation Q=1 fast path : matmul direct au lieu de SDPA cutlass FMHA.
        Pour Q de longueur 1, le tiling de FMHA est dégénéré (kernel tile 64x64
        pour ne traiter qu'1 ligne utile sur 64). cuBLAS GEMV gère bien le cas
        Q=1 × K=max_len, et l'overhead par kernel reste comparable.
        """
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)  # (1, H, 1, Hd)
        k = k.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # Écriture in-place dans les buffers
        k_buf.index_copy_(2, pos_idx, k)
        v_buf.index_copy_(2, pos_idx, v)

        # === Q=1 fast path ===
        # Q × Kᵀ : (1, H, 1, Hd) @ (1, H, Hd, max_len) -> (1, H, 1, max_len)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k_buf.transpose(-2, -1)) * scale
        # Apply additive mask (broadcasting (1,1,1,max_len) sur (1,H,1,max_len))
        scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        # weights × V : (1, H, 1, max_len) @ (1, H, max_len, Hd) -> (1, H, 1, Hd)
        out = torch.matmul(weights, v_buf)

        out = out.transpose(1, 2).contiguous().view(1, 1, self.n_heads * self.head_dim)
        return self.out(out)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Modes :
          - Sans cache : x = (B, T, D), forward causale standard. Retourne (out, (K, V))
            où K, V sont les tenseurs complets pour cache initial éventuel.
          - Avec cache : x = (B, T_new, D) (souvent T_new=1), kv_cache = (K_past, V_past)
            avec K_past, V_past de shape (B, H, T_past, Hd). On concatène et on calcule
            l'attention de Q (T_new) sur K/V de longueur T_total = T_past + T_new.
            En autoregressif single-token T_new=1, donc pas besoin de masque causal :
            le token attend toutes les positions passées qui sont par construction du
            passé. Si T_new > 1, on construit un masque causal local.
        """
        B, T_new, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T_new, Hd)
        k = k.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            assert key_padding_mask is None, "padding mask non supporté en cached mode"
            K_past, V_past = kv_cache
            k_full = torch.cat([K_past, k], dim=2)  # (B, H, T_past + T_new, Hd)
            v_full = torch.cat([V_past, v], dim=2)
            if T_new == 1:
                # Path dynamique (eager) : on garde SDPA/FMHA (1 kernel) — en eager,
                # le matmul Q=1 en 3 kernels coûte PLUS cher en launch overhead que
                # le gain de tiling. Le matmul direct n'est gagnant QUE dans le graph
                # (cf forward_static), où les launches sont collapsés.
                out = F.scaled_dot_product_attention(q, k_full, v_full,
                                                     is_causal=False, dropout_p=0.0)
            else:
                # Masque causal "local" : Q de longueur T_new sur K de longueur T_total.
                # Q[i] peut voir K[0..T_past + i] inclus.
                T_total = k_full.shape[2]
                T_past = T_total - T_new
                row = torch.arange(T_new, device=q.device).view(-1, 1)
                col = torch.arange(T_total, device=q.device).view(1, -1)
                mask = (col > (T_past + row))  # True où il faut masquer
                attn_mask = torch.zeros(T_new, T_total, dtype=q.dtype, device=q.device)
                attn_mask.masked_fill_(mask, float("-inf"))
                out = F.scaled_dot_product_attention(q, k_full, v_full,
                                                     attn_mask=attn_mask[None, None, :, :],
                                                     is_causal=False, dropout_p=0.0)
            new_kv = (k_full, v_full)
        else:
            # Forward initiale sans cache
            if key_padding_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            else:
                pad = torch.zeros(B, 1, 1, T_new, dtype=q.dtype, device=q.device)
                pad.masked_fill_(~key_padding_mask[:, None, None, :], float("-inf"))
                causal = torch.triu(
                    torch.full((T_new, T_new), float("-inf"), dtype=q.dtype, device=q.device),
                    diagonal=1,
                )
                combined = pad + causal[None, None, :, :]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=combined,
                                                     is_causal=False, dropout_p=0.0)
            new_kv = (k, v)

        out = out.transpose(1, 2).contiguous().view(B, T_new, D)
        return self.out(out), new_kv


# --------------------------------------------------------------------------- block

class FFN(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim, bias=False)
        self.fc2 = nn.Linear(ff_dim, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = FFN(dim, ff_dim)

    def forward(self, x, key_padding_mask, kv_cache=None):
        h, new_kv = self.attn(self.norm1(x), key_padding_mask, kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_kv

    def forward_static(self, x, k_buf, v_buf, pos_idx, attn_mask):
        h = self.attn.forward_static(self.norm1(x), k_buf, v_buf, pos_idx, attn_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- model

class SpriteTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 dim: int = 512, n_layers: int = 16, n_heads: int = 16,
                 ff_dim: int = 2048, max_seq_len: int = 18432,
                 sprite_size: int = SPRITE_SIZE, max_frames: int = MAX_FRAMES,
                 n_identities: int = 1024):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Embeddings appris
        self.tok_emb   = nn.Embedding(vocab_size, dim)
        self.x_emb     = nn.Embedding(sprite_size, dim)
        self.y_emb     = nn.Embedding(sprite_size, dim)
        self.frame_emb = nn.Embedding(max_frames + 1, dim)  # +1 pour l'index 'ref'
        # Conditionnement descriptif hors vocab (spec v2) : identite (table apprise,
        # index 0 = __none__) et couleur (projection RGB continue). Injectes aux
        # positions des tokens-marqueurs ID / COLOR.
        self.id_emb     = nn.Embedding(n_identities, dim)
        self.color_proj = nn.Linear(3, dim, bias=False)

        # Sinusoïdale (buffer non apprenable)
        self.register_buffer("seq_pe", sinusoidal_table(max_seq_len, dim), persistent=False)

        # Pile de blocs
        self.blocks = nn.ModuleList([Block(dim, n_heads, ff_dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Init standard (avant le tying — sinon head.weight serait ré-init après
        # partage et casserait l'égalité des deux usages).
        self.apply(self._init_weights)

        # Weight tying : partage poids embedding token <-> head. L'économie de params
        # est négligeable ici (36×512 = 0.04%), mais l'ablation A/B (cf PROJECT.md) a
        # montré que le tying régularise les tokens RARES en cible (FRAME_SEP/SEQ_END,
        # ~0.1% des positions) et accélère nettement l'apprentissage du structurel —
        # exactement ce qu'on cherche à améliorer. On garde donc le tying.
        self.head.weight = self.tok_emb.weight

        # Gradient checkpointing : recompute des activations au backward pour
        # économiser la VRAM sur longues séquences. Togglé par batch via
        # set_grad_checkpoint() depuis le training loop. Off par défaut.
        self._grad_ckpt = False

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def embed(self, tokens, x_pos, y_pos, frame_pos, roles,
              id_index=None, color_rgb=None, seq_offset: int = 0):
        """Compose les embeddings. x/y/frame uniquement pour rôles pixel ; identite et
        couleur injectees uniquement aux positions des tokens-marqueurs ID / COLOR.

        seq_offset : position absolue du premier token de `tokens` dans la séquence
        complète. Utile en cached inference où on ne passe que les nouveaux tokens.
        """
        h = self.tok_emb(tokens)                           # (B, T, D)
        T = tokens.shape[1]
        h = h + self.seq_pe[seq_offset:seq_offset + T].unsqueeze(0)

        is_pixel = ((roles == ROLE.PREFIX_PIXEL) | (roles == ROLE.CONTENT_PIXEL)).unsqueeze(-1).to(h.dtype)
        h = h + is_pixel * self.x_emb(x_pos)
        h = h + is_pixel * self.y_emb(y_pos)
        h = h + is_pixel * self.frame_emb(frame_pos)

        if id_index is not None:
            is_id = (tokens == ID).unsqueeze(-1).to(h.dtype)
            h = h + is_id * self.id_emb(id_index.clamp(min=0))
        if color_rgb is not None:
            is_color = (tokens == COLOR).unsqueeze(-1).to(h.dtype)
            h = h + is_color * self.color_proj(color_rgb.to(h.dtype) / 255.0)
        return h

    # ---------- Static-cache 1-token forward (graph-capture-friendly) ----------
    def forward_static(self, token, x_pos, y_pos, frame_pos, role,
                       k_bufs: list, v_bufs: list,
                       pos_idx: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs (tous des tenseurs preallocés) :
          token, x_pos, y_pos, frame_pos, role : (1, 1) long
          k_bufs, v_bufs : listes de (1, H, max_len, head_dim) — un par layer
          pos_idx        : (1,) long — position absolue du token courant
          attn_mask      : (1, 1, 1, max_len) — masque additif (0 ou -inf) ; on
                           "active" la position pos_idx en y écrivant 0 ici.
        Retourne : logits (1, 1, vocab).
        """
        # Active la position courante dans le masque (in-place)
        attn_mask.view(-1).index_fill_(0, pos_idx, 0)

        # Embeddings
        h = self.tok_emb(token)                                          # (1, 1, D)
        seq_pe_at = self.seq_pe.index_select(0, pos_idx)                 # (1, D)
        h = h + seq_pe_at.unsqueeze(0)                                   # (1, 1, D)

        is_pixel = ((role == ROLE.PREFIX_PIXEL) | (role == ROLE.CONTENT_PIXEL)).unsqueeze(-1).to(h.dtype)
        h = h + is_pixel * self.x_emb(x_pos)
        h = h + is_pixel * self.y_emb(y_pos)
        h = h + is_pixel * self.frame_emb(frame_pos)

        for i, blk in enumerate(self.blocks):
            h = blk.forward_static(h, k_bufs[i], v_bufs[i], pos_idx, attn_mask)

        h = self.final_norm(h)
        return self.head(h)

    def forward(self, tokens, x_pos, y_pos, frame_pos, roles, attn_mask,
                id_index=None, color_rgb=None,
                kv_cache: list | None = None, seq_offset: int = 0,
                return_cache: bool = False):
        """
        Si kv_cache est None : forward standard sur toute la séquence (training,
        première forward d'inférence). Si return_cache=True, on renvoie aussi le
        nouveau cache (utile pour amorcer le sampling).

        Si kv_cache est une liste de (K, V) par layer : forward sur les `tokens`
        nouveaux uniquement, en réutilisant le cache et en l'étendant. Retourne
        (logits, new_cache).

        id_index (B,T) / color_rgb (B,T,3) : conditionnement descriptif, injecte aux
        positions ID/COLOR (cf embed). None en generation de contenu (pas de ces tokens).
        """
        h = self.embed(tokens, x_pos, y_pos, frame_pos, roles,
                       id_index=id_index, color_rgb=color_rgb, seq_offset=seq_offset)
        new_cache: list | None = [] if (return_cache or kv_cache is not None) else None
        use_ckpt = self._grad_ckpt and self.training and kv_cache is None
        for i, blk in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None else None
            if use_ckpt:
                # recompute au backward : ~+33% compute, mémoire activations effondrée
                h, kv = _ckpt(blk, h, attn_mask, use_reentrant=False)
            else:
                h, kv = blk(h, attn_mask, kv_cache=past)
            if new_cache is not None:
                new_cache.append(kv)
        h = self.final_norm(h)
        logits = self.head(h)
        if new_cache is not None:
            return logits, new_cache
        return logits

    def set_grad_checkpoint(self, enabled: bool):
        """Active/désactive le gradient checkpointing pour les prochains forward
        d'entraînement (sans effet sur l'inférence à cache)."""
        self._grad_ckpt = bool(enabled)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
