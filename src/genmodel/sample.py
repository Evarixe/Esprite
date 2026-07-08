"""Génération autoregressive depuis un checkpoint — sampling LIBRE + parsing post-hoc.

Philosophie :
  - Pendant l'entraînement, on a appris au modèle la grammaire (pixel * 1024 par frame,
    puis FRAME_SEP ou SEQ_END). À l'inférence, on lui laisse la liberté de la respecter
    ou non. Si le modèle est mal entraîné, on doit pouvoir le constater visuellement.

  - On échantillonne dans tout le vocab à chaque position (softmax + température).
  - On stoppe sur SEQ_END ou sur un cap de tokens (sécurité anti-infini).
  - Après la génération, on parse le flot de tokens et on en extrait des frames :
       - 1024 pixel-tokens consécutifs suivis de FRAME_SEP ou SEQ_END → frame VALID
       - frame de moins de 1024 pixels interrompue par un token structurel → INVALID
         (la "trame" courante est paddée à transparent pour l'affichage)
       - frame de plus de 1024 pixels (le modèle n'a pas su s'arrêter) → INVALID
       - cap atteint sans SEQ_END → status global TRUNCATED

Le modèle peut donc générer 1, 3, 7, ou 16 frames sans qu'on le force à un nombre.
La cible `frames` du `GenRequest` n'est utilisée que comme upper bound de sécurité
(`max_frames`), pas comme valeur forcée.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from .vocab import (
    ACTION_TOKEN, DIR_TOKEN, FRAMES_VAL, GEN_START, SEQ_END,
    REF_START, REF_END, FRAME_SEP, PIXELS_PER_FRAME, SPRITE_SIZE,
    REF_FRAME_INDEX, ROLE, MAX_FRAMES, VOCAB_SIZE, is_pixel_token,
)
from .tokenize import _RASTER_X, _RASTER_Y
from .model import SpriteTransformer


@dataclass
class GenRequest:
    action: str = "idle"
    direction: str | None = None
    frames: int = 8                   # cible (cap dur côté sécurité, voir max_frames_safety)
    reference: np.ndarray | None = None
    temperature: float = 0.3
    seed: int | None = None
    max_frames_safety: int = MAX_FRAMES  # cap dur sur le nombre de frames générées
    token_budget_extra: int = 200     # tokens supplémentaires tolérés au-delà du strict
                                       # nécessaire (laisse la place à des trous structurels)


@dataclass
class FrameResult:
    indices: np.ndarray              # (32, 32) uint8 — toujours présent (padding transparent si INVALID)
    status: str                      # 'valid' | 'short' | 'no_separator' | 'unknown_token'
    n_pixels: int                    # nombre de pixel-tokens effectivement émis (avant interruption)
    notes: str = ""


@dataclass
class GenResult:
    frames: list[FrameResult]
    raw_tokens: list[int]            # flot complet généré (après GEN_START, hors préfixe)
    stop_reason: str                 # 'seq_end' | 'token_cap' | 'frame_cap'
    n_valid: int = 0
    n_invalid: int = 0


def _sample_token(logits: torch.Tensor, temperature: float, rng: torch.Generator,
                  forbid_ids: list[int] | None = None) -> int:
    """Sample dans tout le vocab. `forbid_ids` permet d'interdire des tokens
    (typiquement : interdire GEN_START / REF_START / REF_END / FRAMES_VAL dans la
    zone de génération, qui n'ont aucun sens là).
    """
    l = logits.clone()
    if forbid_ids:
        l[forbid_ids] = float("-inf")
    if temperature <= 0:
        return int(l.argmax().item())
    probs = F.softmax(l / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=rng).item())


def build_prefix(req: GenRequest) -> tuple[list[int], list[int], list[int], list[int], list[int], int]:
    tokens, roles, xs, ys, fs = [], [], [], [], []

    def add(t, role, x=0, y=0, f=0):
        tokens.append(t); roles.append(role); xs.append(x); ys.append(y); fs.append(f)

    add(ACTION_TOKEN[req.action], ROLE.PREFIX_NON_PIXEL)
    if req.direction is not None:
        add(DIR_TOKEN[req.direction], ROLE.PREFIX_NON_PIXEL)
    add(FRAMES_VAL, ROLE.PREFIX_NON_PIXEL)
    add(req.frames, ROLE.PREFIX_NON_PIXEL)

    if req.reference is not None:
        add(REF_START, ROLE.PREFIX_NON_PIXEL)
        ref_flat = req.reference.reshape(-1).astype(int).tolist()
        for k, pix in enumerate(ref_flat):
            add(pix, ROLE.PREFIX_PIXEL, x=int(_RASTER_X[k]), y=int(_RASTER_Y[k]), f=REF_FRAME_INDEX)
        add(REF_END, ROLE.PREFIX_NON_PIXEL)

    gen_start_idx = len(tokens)
    add(GEN_START, ROLE.PREFIX_NON_PIXEL)
    return tokens, roles, xs, ys, fs, gen_start_idx


# Tokens interdits dans la zone de génération (ne devraient pas apparaître en sortie).
_FORBIDDEN_IN_GEN = [GEN_START, REF_START, REF_END, FRAMES_VAL,
                     # action et direction tags (22..35) interdits aussi
                     *range(22, VOCAB_SIZE)]


def _parse_token_stream(tokens: list[int]) -> tuple[list[FrameResult], str]:
    """Parse un flot de tokens (après GEN_START, sans inclure SEQ_END terminal) en
    une liste de FrameResult.

    Règle d'extraction :
      - On lit des pixel-tokens (0..15). Quand on en a 1024 consécutifs, c'est une frame.
      - Si on rencontre FRAME_SEP avant 1024 pixels → frame courte (status='short').
      - Si on rencontre SEQ_END → on stoppe (frame courante traitée comme short si <1024).
      - Si on lit 1024 pixels et que le token suivant n'est ni FRAME_SEP ni SEQ_END
        (ex: un autre pixel, ou un token bizarre) → status='no_separator' pour cette frame,
        on consomme le token et on enchaîne immédiatement sur la frame suivante.
      - Si on rencontre un token non-pixel et non-séparateur ni SEQ_END (ex: REF_START,
        un tag d'action, etc.) au milieu d'une frame → status='unknown_token', frame
        courante coupée, on continue avec une nouvelle.
    """
    frames: list[FrameResult] = []
    current_pixels: list[int] = []
    stop_reason = "token_cap"

    def flush(status: str, note: str = ""):
        n = len(current_pixels)
        idx = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=np.uint8)
        if n > 0:
            flat = idx.reshape(-1)
            take = min(n, PIXELS_PER_FRAME)
            flat[:take] = current_pixels[:take]
        frames.append(FrameResult(indices=idx, status=status, n_pixels=n, notes=note))
        current_pixels.clear()

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if is_pixel_token(t):
            current_pixels.append(t)
            i += 1
            if len(current_pixels) == PIXELS_PER_FRAME:
                # Frame pleine ; le séparateur attendu est le prochain token
                if i >= len(tokens):
                    # Cap atteint juste après une frame pleine
                    flush("valid")
                    break
                nxt = tokens[i]
                if nxt == FRAME_SEP:
                    flush("valid")
                    i += 1
                elif nxt == SEQ_END:
                    flush("valid")
                    stop_reason = "seq_end"
                    i += 1
                    break
                else:
                    # Le modèle a oublié le séparateur — on coupe quand même, on note
                    flush("no_separator", note=f"expected FRAME_SEP/SEQ_END, got token {nxt}")
                    # On NE consomme PAS nxt — il sera relu et soit accepté soit invalidé
                    # comme début de la frame suivante.
            continue
        # token non-pixel
        if t == FRAME_SEP:
            # Séparateur prématuré : frame courante est courte
            flush("short", note=f"only {len(current_pixels)} pixels before FRAME_SEP")
            i += 1
        elif t == SEQ_END:
            if current_pixels:
                flush("short", note=f"only {len(current_pixels)} pixels before SEQ_END")
            stop_reason = "seq_end"
            i += 1
            break
        else:
            # Token bizarre au milieu d'une frame (REF_START, action, etc.)
            flush("unknown_token", note=f"unexpected token {t} after {len(current_pixels)} pixels")
            i += 1

    return frames, stop_reason


@torch.no_grad()
def generate(model: SpriteTransformer, req: GenRequest, device: str = "cuda",
             use_graph: bool = True, sampler=None, pool=None,
             timings: dict | None = None) -> GenResult:
    """Sampling autoregressif.

    Par défaut utilise CUDA Graphs (via GraphSampler) qui capture le step en
    1 graphe replayable → ×7 sur batch=1 par rapport au path dynamique. Fallback
    sur le path dynamique (cat-based) si CUDA non dispo ou si capture impossible.

    Le modèle doit être en bf16 statique pour le path graph (cast une fois
    avant l'appel : `model.to(torch.bfloat16)`).

    Args :
      sampler : GraphSampler préalloué (réutilisable entre appels). Si None et
        path graph activé, on en crée un éphémère (coûteux : ~30s d'allocation
        + capture). Pour batcher plusieurs générations, passer le même sampler.
      timings : dict optionnel où on accumule des temps clés ('setup', 'prefix',
        'sample_loop', 'capture') en secondes.
    """
    model.eval()
    gen = torch.Generator(device=device)
    if req.seed is not None:
        gen.manual_seed(int(req.seed))

    tokens, roles, xs, ys, fs, gen_start_idx = build_prefix(req)
    prefix_len = len(tokens)
    max_total_new_tokens = req.max_frames_safety * (PIXELS_PER_FRAME + 1) + req.token_budget_extra

    can_use_graph = (
        use_graph and device == "cuda" and torch.cuda.is_available()
        and next(model.parameters()).dtype == torch.bfloat16
    )

    if pool is not None and can_use_graph:
        return _generate_with_pool(model, req, tokens, roles, xs, ys, fs,
                                    prefix_len, max_total_new_tokens, gen, device,
                                    pool=pool, timings=timings)
    if can_use_graph:
        return _generate_with_graph(model, req, tokens, roles, xs, ys, fs,
                                     prefix_len, max_total_new_tokens, gen, device,
                                     sampler=sampler, timings=timings)
    return _generate_dynamic(model, req, tokens, roles, xs, ys, fs,
                              prefix_len, max_total_new_tokens, gen, device,
                              timings=timings)


def _step_book_keeping(tok: int, current_frame_idx: int, pixel_in_current: int,
                        max_frames_safety: int):
    """Calcule (role, x, y, f) pour le PROCHAIN forward selon le token émis.
    Retourne aussi le nouveau (current_frame_idx, pixel_in_current) et un signal stop."""
    if is_pixel_token(tok):
        role_new = ROLE.CONTENT_PIXEL
        x_new = int(_RASTER_X[pixel_in_current % PIXELS_PER_FRAME])
        y_new = int(_RASTER_Y[pixel_in_current % PIXELS_PER_FRAME])
        f_new = min(current_frame_idx, MAX_FRAMES - 1)
        return role_new, x_new, y_new, f_new, current_frame_idx, pixel_in_current + 1, None
    # token structurel
    role_new = ROLE.CONTENT_SEP
    f_new = min(current_frame_idx, MAX_FRAMES - 1)
    if tok == FRAME_SEP:
        new_frame_idx = current_frame_idx + 1
        if new_frame_idx >= max_frames_safety:
            return role_new, 0, 0, f_new, new_frame_idx, 0, "frame_cap"
        return role_new, 0, 0, f_new, new_frame_idx, 0, None
    if tok == SEQ_END:
        return role_new, 0, 0, f_new, current_frame_idx, pixel_in_current, "seq_end"
    return role_new, 0, 0, f_new, current_frame_idx, pixel_in_current, None


def _generate_with_graph(model, req, tokens, roles, xs, ys, fs,
                          prefix_len, max_total_new_tokens, gen, device,
                          sampler=None, timings: dict | None = None):
    import time
    from .graph_sampler import GraphSampler

    max_len_needed = prefix_len + max_total_new_tokens + 8
    t_setup = time.perf_counter()
    owns_sampler = sampler is None
    if sampler is None:
        sampler = GraphSampler(model, max_len=max_len_needed, dtype=torch.bfloat16)
    else:
        # Vérifie que le sampler a une capacité suffisante
        if sampler.max_len < max_len_needed:
            raise RuntimeError(
                f"Provided sampler max_len={sampler.max_len} < needed {max_len_needed}. "
                "Augmenter max_len à la création."
            )
    sampler.reset()
    sampler.prepare_from_prefix(tokens, roles, xs, ys, fs)
    torch.cuda.synchronize()
    if timings is not None:
        timings["setup_prefix"] = timings.get("setup_prefix", 0.0) + (time.perf_counter() - t_setup)

    # Capture du graph : seulement si le sampler n'en a pas encore
    if sampler.graph is None:
        t_capture = time.perf_counter()
        sampler.capture()
        torch.cuda.synchronize()
        if timings is not None:
            timings["capture"] = timings.get("capture", 0.0) + (time.perf_counter() - t_capture)
        # La capture a écrit dans les buffers à pos_idx=prefix_len. Comme le
        # warmup a 3 fois écrit la même chose puis pos_idx n'a pas bougé, c'est
        # OK pour le premier step.

    # Forward eager pour récupérer les logits qui prédisent le PREMIER token généré
    t_prefix = time.perf_counter()
    t_tens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_tens = torch.tensor(xs,     dtype=torch.long, device=device).unsqueeze(0)
    y_tens = torch.tensor(ys,     dtype=torch.long, device=device).unsqueeze(0)
    f_tens = torch.tensor(fs,     dtype=torch.long, device=device).unsqueeze(0)
    r_tens = torch.tensor(roles,  dtype=torch.long, device=device).unsqueeze(0)
    logits_prefix = model(t_tens, x_tens, y_tens, f_tens, r_tens, attn_mask=None)
    last_logits = logits_prefix[0, -1].float()
    torch.cuda.synchronize()
    if timings is not None:
        timings["prefix_forward"] = timings.get("prefix_forward", 0.0) + (time.perf_counter() - t_prefix)

    raw_emitted: list[int] = []
    current_frame_idx = 0
    pixel_in_current = 0
    stop_reason = "token_cap"

    t_loop = time.perf_counter()
    for n_new in range(max_total_new_tokens):
        tok = _sample_token(last_logits, req.temperature, gen, forbid_ids=_FORBIDDEN_IN_GEN)
        raw_emitted.append(tok)

        role_new, x_new, y_new, f_new, current_frame_idx, pixel_in_current, stop = \
            _step_book_keeping(tok, current_frame_idx, pixel_in_current, req.max_frames_safety)
        if stop in ("seq_end", "frame_cap"):
            stop_reason = stop
            break

        logits = sampler.step(tok, x_new, y_new, f_new, role_new)
        last_logits = logits.float()
    torch.cuda.synchronize()
    if timings is not None:
        timings["sample_loop"] = timings.get("sample_loop", 0.0) + (time.perf_counter() - t_loop)
        timings["n_tokens_emitted"] = timings.get("n_tokens_emitted", 0) + len(raw_emitted)

    return _finalize(raw_emitted, stop_reason)


def _generate_with_pool(model, req, tokens, roles, xs, ys, fs,
                        prefix_len, max_total_new_tokens, gen, device,
                        pool, timings: dict | None = None):
    """Génération avec pool de buckets + promotion dynamique de graph.

    Démarre au plus petit bucket couvrant prefix + 1 frame + décision, et promeut
    (migration K/V + switch de graph) quand le buffer sature. cur_pos est suivi en
    Python (miroir de pos_idx) pour éviter tout sync de lecture du tensor.
    """
    import time

    start_len = prefix_len + 1025  # room pour 1 frame + token-décision
    bucket = pool.smallest_bucket_for(start_len)
    sampler = pool.get(bucket)

    t_setup = time.perf_counter()
    sampler.reset()
    sampler.prepare_from_prefix(tokens, roles, xs, ys, fs)
    if sampler.graph is None:
        sampler.capture()
    torch.cuda.synchronize()
    if timings is not None:
        timings["setup_prefix"] = timings.get("setup_prefix", 0.0) + (time.perf_counter() - t_setup)

    # Logits du 1er token généré (forward eager du préfixe)
    t_prefix = time.perf_counter()
    t_tens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_tens = torch.tensor(xs,     dtype=torch.long, device=device).unsqueeze(0)
    y_tens = torch.tensor(ys,     dtype=torch.long, device=device).unsqueeze(0)
    f_tens = torch.tensor(fs,     dtype=torch.long, device=device).unsqueeze(0)
    r_tens = torch.tensor(roles,  dtype=torch.long, device=device).unsqueeze(0)
    logits_prefix = model(t_tens, x_tens, y_tens, f_tens, r_tens, attn_mask=None)
    last_logits = logits_prefix[0, -1].float()
    torch.cuda.synchronize()
    if timings is not None:
        timings["prefix_forward"] = timings.get("prefix_forward", 0.0) + (time.perf_counter() - t_prefix)

    raw_emitted: list[int] = []
    current_frame_idx = 0
    pixel_in_current = 0
    cur_pos = prefix_len           # miroir Python de pos_idx (pas de sync)
    stop_reason = "token_cap"
    promotions = 0

    t_loop = time.perf_counter()
    for n_new in range(max_total_new_tokens):
        tok = _sample_token(last_logits, req.temperature, gen, forbid_ids=_FORBIDDEN_IN_GEN)
        raw_emitted.append(tok)

        role_new, x_new, y_new, f_new, current_frame_idx, pixel_in_current, stop = \
            _step_book_keeping(tok, current_frame_idx, pixel_in_current, req.max_frames_safety)
        if stop == "seq_end":
            stop_reason = stop
            break

        # Promotion si le buffer courant est plein (on doit écrire à cur_pos)
        if cur_pos >= sampler.max_len:
            nb = pool.next_bucket(sampler.max_len)
            if nb is None:
                stop_reason = "frame_cap"   # déjà au plus gros bucket → on tronque
                break
            new_sampler = pool.get(nb)
            new_sampler.migrate_from(sampler, valid_len=cur_pos)
            if new_sampler.graph is None:
                new_sampler.capture()
            sampler = new_sampler
            promotions += 1

        logits = sampler.step(tok, x_new, y_new, f_new, role_new)
        last_logits = logits.float()
        cur_pos += 1

    torch.cuda.synchronize()
    if timings is not None:
        timings["sample_loop"] = timings.get("sample_loop", 0.0) + (time.perf_counter() - t_loop)
        timings["n_tokens_emitted"] = timings.get("n_tokens_emitted", 0) + len(raw_emitted)
        timings["promotions"] = timings.get("promotions", 0) + promotions
        timings["final_bucket"] = sampler.max_len

    return _finalize(raw_emitted, stop_reason)


def _generate_dynamic(model, req, tokens, roles, xs, ys, fs,
                       prefix_len, max_total_new_tokens, gen, device,
                       timings: dict | None = None):
    """Fallback : path dynamique (concat KV cache) — utilisé si pas de CUDA ou
    modèle pas bf16 statique."""
    t_tens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_tens = torch.tensor(xs,     dtype=torch.long, device=device).unsqueeze(0)
    y_tens = torch.tensor(ys,     dtype=torch.long, device=device).unsqueeze(0)
    f_tens = torch.tensor(fs,     dtype=torch.long, device=device).unsqueeze(0)
    r_tens = torch.tensor(roles,  dtype=torch.long, device=device).unsqueeze(0)
    use_amp = (device == "cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
    with ctx:
        logits, kv_cache = model(t_tens, x_tens, y_tens, f_tens, r_tens,
                                  attn_mask=None, return_cache=True)
    last_logits = logits[0, -1].float()

    raw_emitted: list[int] = []
    current_frame_idx = 0
    pixel_in_current = 0
    cur_seq_pos = prefix_len
    stop_reason = "token_cap"

    for n_new in range(max_total_new_tokens):
        tok = _sample_token(last_logits, req.temperature, gen, forbid_ids=_FORBIDDEN_IN_GEN)
        raw_emitted.append(tok)
        role_new, x_new, y_new, f_new, current_frame_idx, pixel_in_current, stop = \
            _step_book_keeping(tok, current_frame_idx, pixel_in_current, req.max_frames_safety)
        if stop in ("seq_end", "frame_cap"):
            stop_reason = stop
            break

        t_in = torch.tensor([[tok]],      dtype=torch.long, device=device)
        x_in = torch.tensor([[x_new]],    dtype=torch.long, device=device)
        y_in = torch.tensor([[y_new]],    dtype=torch.long, device=device)
        f_in = torch.tensor([[f_new]],    dtype=torch.long, device=device)
        r_in = torch.tensor([[role_new]], dtype=torch.long, device=device)
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()):
            logits, kv_cache = model(t_in, x_in, y_in, f_in, r_in,
                                      attn_mask=None, kv_cache=kv_cache,
                                      seq_offset=cur_seq_pos)
        last_logits = logits[0, -1].float()
        cur_seq_pos += 1

    return _finalize(raw_emitted, stop_reason)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _finalize(raw_emitted: list[int], stop_reason: str) -> GenResult:
    frames, parse_stop = _parse_token_stream(raw_emitted)
    if stop_reason != "seq_end":
        stop_reason = parse_stop if parse_stop == "seq_end" else stop_reason
    n_valid = sum(1 for f in frames if f.status == "valid")
    n_invalid = len(frames) - n_valid
    return GenResult(frames=frames, raw_tokens=raw_emitted, stop_reason=stop_reason,
                     n_valid=n_valid, n_invalid=n_invalid)
