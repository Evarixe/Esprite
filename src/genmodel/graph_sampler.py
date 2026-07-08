"""CUDA Graphs sampler — accélère le sampling autoregressif batch=1 launch-bound.

Idée :
  Le forward 1-token via cache dynamique passe 80% de son temps en overhead CPU
  (kernel launches + dispatcher Python). En batch=1 sur 50M params, la compute
  pure GPU est <1.5ms/token mais le wall-clock est 6-8ms. CUDA Graphs capture
  toute la séquence des kernels d'un step en 1 graphe réutilisable, et `replay()`
  saute tout l'overhead.

Architecture :
  - K/V buffers préallocés de taille (1, H, max_len, head_dim) par layer (BF16)
  - Mask additif 1D de taille max_len ; "unmask" la position courante par
    index_fill in-place dans le forward static
  - Input buffers (1, 1) pour token/x/y/f/role, mis à jour entre replays
  - pos_idx scalar tensor (1,), incrémenté entre replays
  - logits écrits dans un output buffer fixe

Cycle :
  1. `prepare_for_prefix(req)` : crée le préfixe en eager, peuple K/V buffers + mask
  2. `capture()` : warmup 2-3 steps puis capture du graph
  3. `step(token, x, y, f, role) -> logits` : replay + retourne logits du dernier token
  4. `reset()` : remet à zéro pos_idx et mask pour une nouvelle génération
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

from .model import SpriteTransformer
from .vocab import ROLE, MAX_FRAMES, PIXELS_PER_FRAME, REF_FRAME_INDEX
from .tokenize import _RASTER_X, _RASTER_Y


class GraphSampler:
    def __init__(self, model: SpriteTransformer, max_len: int = 18432,
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.max_len = max_len
        self.dtype = dtype
        self.device = next(model.parameters()).device

        n_layers = len(model.blocks)
        n_heads = model.blocks[0].attn.n_heads
        head_dim = model.blocks[0].attn.head_dim
        vocab = model.head.out_features

        # Buffers d'input — taille fixe (1, 1)
        self.tok_in  = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.x_in    = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.y_in    = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.f_in    = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.role_in = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # Position absolue courante (1,), mise à jour entre replays
        self.pos_idx = torch.zeros(1, dtype=torch.long, device=self.device)

        # K/V buffers — un par layer
        self.k_bufs = [torch.zeros(1, n_heads, max_len, head_dim, dtype=dtype, device=self.device)
                       for _ in range(n_layers)]
        self.v_bufs = [torch.zeros(1, n_heads, max_len, head_dim, dtype=dtype, device=self.device)
                       for _ in range(n_layers)]

        # Mask additif : -inf partout au départ ; la position pos_idx est unmaskée
        # à chaque step (dans le forward_static)
        self.attn_mask = torch.full((1, 1, 1, max_len), float("-inf"),
                                     dtype=dtype, device=self.device)

        # Logits output buffer
        self.logits_out = torch.zeros(1, 1, vocab, dtype=dtype, device=self.device)

        self.graph: torch.cuda.CUDAGraph | None = None

    # ---------- API ----------

    def reset(self):
        """Repart à zéro pour une nouvelle génération (avant prefix forward)."""
        self.pos_idx.fill_(0)
        self.attn_mask.fill_(float("-inf"))
        for kb in self.k_bufs: kb.zero_()
        for vb in self.v_bufs: vb.zero_()

    @torch.no_grad()
    def prepare_from_prefix(self, prefix_tokens, prefix_roles, prefix_x, prefix_y,
                            prefix_f) -> int:
        """Lance un forward eager sur le préfixe (n'importe quelle longueur), copie
        les K/V dans les buffers statiques, ajuste le mask et pos_idx. Retourne
        la longueur du préfixe (= position où l'on va écrire le 1er token généré).
        """
        device = self.device
        t = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
        x = torch.tensor(prefix_x,      dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(prefix_y,      dtype=torch.long, device=device).unsqueeze(0)
        f = torch.tensor(prefix_f,      dtype=torch.long, device=device).unsqueeze(0)
        r = torch.tensor(prefix_roles,  dtype=torch.long, device=device).unsqueeze(0)
        L = t.shape[1]

        _, kv_cache = self.model(t, x, y, f, r, attn_mask=None, return_cache=True)

        # Recopie dans les buffers statiques aux positions 0..L-1
        for i, (k, v) in enumerate(kv_cache):
            # k, v shape (1, H, L, Hd), dtype = autocast bf16 ou fp32 selon contexte
            self.k_bufs[i][:, :, :L, :].copy_(k.to(self.dtype))
            self.v_bufs[i][:, :, :L, :].copy_(v.to(self.dtype))

        # Unmask les positions 0..L-1
        self.attn_mask[:, :, :, :L] = 0
        # pos_idx pointe vers la position où le PROCHAIN token sera écrit (= L)
        self.pos_idx.fill_(L)
        return L

    def _step_eager(self):
        """Un step en mode eager — utilisé pour warmup et pour la capture du graph."""
        out = self.model.forward_static(
            self.tok_in, self.x_in, self.y_in, self.f_in, self.role_in,
            self.k_bufs, self.v_bufs, self.pos_idx, self.attn_mask,
        )
        self.logits_out.copy_(out)

    @torch.no_grad()
    def capture(self):
        """Capture du graph. À appeler après prepare_from_prefix.

        Important : la capture demande des shapes/pointers constants. Comme
        pos_idx est un tenseur, sa VALEUR peut changer entre replays sans
        recapture. Idem pour les input buffers.
        """
        # Warmup sur un stream séparé pour ne pas polluer la capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_eager()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # On a fait avancer pos_idx de 3 pendant warmup ? Non — _step_eager ne
        # touche pas à pos_idx, c'est nous qui devons l'incrémenter entre steps.
        # Donc la capture est sur "step à la position pos_idx courante" ; au
        # replay, on update pos_idx externe et tout marche.

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_eager()

    @torch.no_grad()
    def migrate_from(self, other: "GraphSampler", valid_len: int):
        """Recopie l'état (K/V 0:valid_len, mask, pos_idx) depuis un sampler plus petit.

        Utilisé lors d'une promotion de bucket : `other` a saturé son buffer, on
        continue la génération dans `self` (max_len plus grand) sans rien perdre.
        """
        for i in range(len(self.k_bufs)):
            self.k_bufs[i][:, :, :valid_len, :].copy_(other.k_bufs[i][:, :, :valid_len, :])
            self.v_bufs[i][:, :, :valid_len, :].copy_(other.v_bufs[i][:, :, :valid_len, :])
        self.attn_mask.fill_(float("-inf"))
        self.attn_mask[:, :, :, :valid_len] = 0
        self.pos_idx.fill_(valid_len)

    @torch.no_grad()
    def step(self, token_id: int, x_val: int, y_val: int,
             f_val: int, role_val: int) -> torch.Tensor:
        """Avance d'un token. Retourne (vocab,) tensor des logits prédits APRÈS
        ce token (= distribution du token suivant)."""
        self.tok_in.fill_(int(token_id))
        self.x_in.fill_(int(x_val))
        self.y_in.fill_(int(y_val))
        self.f_in.fill_(int(f_val))
        self.role_in.fill_(int(role_val))
        if self.graph is None:
            self._step_eager()
        else:
            self.graph.replay()
        # Avance pos_idx pour le prochain step
        self.pos_idx.add_(1)
        # Retourne les logits du dernier token (le seul ; out shape (1, 1, vocab))
        return self.logits_out[0, 0]


# Buckets par défaut : calculés, pas de magic numbers.
#   bucket_N = PREFIX_MAX + N × (PIXELS_PER_FRAME + 1)
# Le (PIXELS_PER_FRAME + 1) = 1024 pixels + 1 token (FRAME_SEP entre frames, ou
# le slot décision SEQ_END/FRAME_SEP après la dernière frame → "voir si fin").
# N ∈ {1, 2, 4, 8, 16}. PREFIX_MAX couvre le pire préfixe (avec ref ~1031) + marge.
PREFIX_MAX = 1056          # 1031 (action+dir+FRAMES_VAL+N+REF_START+1024+REF_END+GEN_START) + 25 marge
_FRAME_BLOCK = PIXELS_PER_FRAME + 1   # 1025
DEFAULT_BUCKETS = [PREFIX_MAX + n * _FRAME_BLOCK for n in (1, 2, 4, 8, 16)]
# -> [2081, 3106, 5156, 9256, 17456]


class GraphSamplerPool:
    """Pool de GraphSampler à différentes tailles `max_len` (buckets).

    Permet de démarrer une génération dans un petit bucket (per-token rapide car
    attention sur peu de colonnes) et de promouvoir vers un bucket plus grand
    quand le buffer sature, en migrant l'état K/V — sans recapture du graph
    courant (chaque bucket a son graph capturé une fois, mis en cache).

    Les samplers sont créés et capturés à la demande (lazy) puis réutilisés entre
    générations.
    """

    def __init__(self, model: SpriteTransformer, buckets: list[int] | None = None,
                 dtype: torch.dtype = torch.bfloat16):
        self.model = model
        self.dtype = dtype
        self.buckets = sorted(buckets if buckets is not None else DEFAULT_BUCKETS)
        self._samplers: dict[int, GraphSampler] = {}

    def smallest_bucket_for(self, length: int) -> int:
        """Plus petit bucket dont max_len >= length. Si aucun, le plus grand."""
        for b in self.buckets:
            if b >= length:
                return b
        return self.buckets[-1]

    def next_bucket(self, current: int) -> int | None:
        """Bucket immédiatement supérieur, ou None si déjà au max."""
        idx = self.buckets.index(current)
        return self.buckets[idx + 1] if idx + 1 < len(self.buckets) else None

    def get(self, bucket: int) -> GraphSampler:
        """Récupère (ou crée + capture) le sampler du bucket. Graph capturé 1 fois."""
        s = self._samplers.get(bucket)
        if s is None:
            s = GraphSampler(self.model, max_len=bucket, dtype=self.dtype)
            self._samplers[bucket] = s
        return s
