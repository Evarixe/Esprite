"""Encode un cycle (indices palette 32x32) en séquence de tokens pour le modèle.

Layout final d'une séquence :

  Préfixe (sans ref) :
    [<action:X>, <dir:Y>?, <FRAMES_VAL>, <N>]                          (3 ou 4 tokens)

  Préfixe (avec ref) :
    [<action:X>, <dir:Y>?, <FRAMES_VAL>, <N>, <REF_START>, *ref_pixels(1024), <REF_END>]

  Contenu (à générer) :
    [<GEN_START>, *frame0_pixels(1024), <FRAME_SEP>, *frame1_pixels(1024),
     <FRAME_SEP>, ..., <FRAME_SEP>, *frame{L-1}_pixels(1024), <SEQ_END>]

Pour chaque position on retourne aussi :
  - role      : un TokenRole pour le masquage de loss et l'application sélective des
                embeddings spatiaux (x/y/frame appris uniquement pour pixels).
  - x_pos     : 0..31 si rôle pixel, sinon 0 (ignoré).
  - y_pos     : idem.
  - frame_pos : 0..MAX_FRAMES-1 pour pixels content, REF_FRAME_INDEX (=16) pour pixels ref,
                sinon 0 (ignoré).

Le loss_mask vaut 1 sur les positions cibles à apprendre (= tout sauf le préfixe initial
avant <GEN_START>), et 0 ailleurs. Comme on est en next-token prediction, la cible à
position i est le token à position i+1 → on construit le mask sur les cibles.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .vocab import (
    ACTION_TOKEN, DIR_TOKEN, FRAMES_VAL, GEN_START, SEQ_END,
    REF_START, REF_END, FRAME_SEP, SOURCE_ACTION_MAP,
    PIXELS_PER_FRAME, SPRITE_SIZE, REF_FRAME_INDEX, ROLE,
)


@dataclass
class EncodedSequence:
    tokens: np.ndarray      # (T,) int32
    roles: np.ndarray       # (T,) int8
    x_pos: np.ndarray       # (T,) int8
    y_pos: np.ndarray       # (T,) int8
    frame_pos: np.ndarray   # (T,) int8
    gen_start_idx: int      # position du token <GEN_START> dans la séquence

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


# Coordonnées x/y des 1024 pixels d'une frame, ordre raster ligne-par-ligne.
_YS, _XS = np.meshgrid(np.arange(SPRITE_SIZE), np.arange(SPRITE_SIZE), indexing="ij")
_RASTER_Y = _YS.reshape(-1).astype(np.int8)   # 0,0,...,0,1,1,...,1,...
_RASTER_X = _XS.reshape(-1).astype(np.int8)   # 0,1,...,31,0,1,...,31,...


def _frame_to_pixel_tokens(frame_32x32: np.ndarray) -> np.ndarray:
    """frame (32, 32) uint8 -> (1024,) int32 ordre raster."""
    return frame_32x32.reshape(-1).astype(np.int32)


def encode_cycle(cycle_frames: np.ndarray, length: int, action_source: str,
                 direction: str | None, ref_frame_32x32: np.ndarray | None) -> EncodedSequence:
    """
    cycle_frames : (max_frames, 32, 32) uint8 — frames du cycle, padding hors longueur.
    length       : nombre de frames effectives (1..MAX_FRAMES).
    action_source: clé de SOURCE_ACTION_MAP ('idle_overworld'|'idle_combat'|'animated_combat').
    direction    : 'down'|'right'|'left'|'up' ou None.
    ref_frame    : (32, 32) uint8 ou None — image de référence à inclure (peut être
                   une frame du cycle ou n'importe quelle autre).
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    action_name = SOURCE_ACTION_MAP[action_source]

    # --- Préfixe (avant pixels ref) ---
    prefix_pre = [ACTION_TOKEN[action_name]]
    if direction is not None:
        prefix_pre.append(DIR_TOKEN[direction])
    prefix_pre.append(FRAMES_VAL)
    prefix_pre.append(length)   # token 0..15 utilisé comme valeur numérique

    n_pre = len(prefix_pre)

    has_ref = ref_frame_32x32 is not None

    # --- Composition séquence ---
    # On alloue à l'avance pour vitesse.
    n_frames_content = length
    content_tokens_per_frame = PIXELS_PER_FRAME
    # contenu: GEN_START + (frame_pixels + FRAME_SEP) * (L-1) + frame_pixels + SEQ_END
    content_len = 1 + n_frames_content * content_tokens_per_frame + (n_frames_content - 1) + 1
    ref_len = (1 + PIXELS_PER_FRAME + 1) if has_ref else 0
    total_len = n_pre + ref_len + content_len

    tokens = np.zeros(total_len, dtype=np.int32)
    roles = np.zeros(total_len, dtype=np.int8)
    x_pos = np.zeros(total_len, dtype=np.int8)
    y_pos = np.zeros(total_len, dtype=np.int8)
    frame_pos = np.zeros(total_len, dtype=np.int8)

    p = 0
    # Préfixe non-pixel (action, dir, FRAMES_VAL, N)
    for t in prefix_pre:
        tokens[p] = t
        roles[p] = ROLE.PREFIX_NON_PIXEL
        p += 1

    # Référence (optionnelle)
    if has_ref:
        tokens[p] = REF_START; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1
        ref_pixels = _frame_to_pixel_tokens(ref_frame_32x32)
        tokens[p:p + PIXELS_PER_FRAME] = ref_pixels
        roles[p:p + PIXELS_PER_FRAME] = ROLE.PREFIX_PIXEL
        x_pos[p:p + PIXELS_PER_FRAME] = _RASTER_X
        y_pos[p:p + PIXELS_PER_FRAME] = _RASTER_Y
        frame_pos[p:p + PIXELS_PER_FRAME] = REF_FRAME_INDEX
        p += PIXELS_PER_FRAME
        tokens[p] = REF_END; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1

    # GEN_START
    gen_start_idx = p
    tokens[p] = GEN_START; roles[p] = ROLE.PREFIX_NON_PIXEL  # marqueur de transition
    p += 1

    # Frames de contenu
    for fi in range(n_frames_content):
        frame_pixels = _frame_to_pixel_tokens(cycle_frames[fi])
        tokens[p:p + PIXELS_PER_FRAME] = frame_pixels
        roles[p:p + PIXELS_PER_FRAME] = ROLE.CONTENT_PIXEL
        x_pos[p:p + PIXELS_PER_FRAME] = _RASTER_X
        y_pos[p:p + PIXELS_PER_FRAME] = _RASTER_Y
        frame_pos[p:p + PIXELS_PER_FRAME] = fi
        p += PIXELS_PER_FRAME
        if fi < n_frames_content - 1:
            tokens[p] = FRAME_SEP; roles[p] = ROLE.CONTENT_SEP; p += 1

    # SEQ_END
    tokens[p] = SEQ_END; roles[p] = ROLE.CONTENT_SEP; p += 1
    assert p == total_len, f"length mismatch {p} != {total_len}"

    return EncodedSequence(
        tokens=tokens, roles=roles,
        x_pos=x_pos, y_pos=y_pos, frame_pos=frame_pos,
        gen_start_idx=gen_start_idx,
    )


def make_loss_mask(seq: EncodedSequence) -> np.ndarray:
    """Mask des cibles à apprendre.

    Pour next-token prediction, position i prédit token i+1. Le mask de longueur T-1
    vaut 1 si la cible (token i+1) est une position de contenu (CONTENT_PIXEL ou
    CONTENT_SEP), 0 sinon. On apprend donc à prédire :
      - tous les pixels de la zone de génération
      - FRAME_SEP et SEQ_END
    On n'apprend PAS à prédire les tags, FRAMES_VAL, la valeur de N, REF_START/END,
    les pixels de la ref (PREFIX_PIXEL), ni GEN_START.
    """
    target_roles = seq.roles[1:]
    mask = ((target_roles == ROLE.CONTENT_PIXEL) | (target_roles == ROLE.CONTENT_SEP)).astype(np.int8)
    return mask
