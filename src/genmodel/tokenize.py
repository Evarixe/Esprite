"""Encode un cycle en sequence de tokens (modele v2). Cf model_spec_v2.md.

Format :
  [action, dir, N]                                   coeur fixe (dir toujours present : 'none' si None)
  (<TAG_START> desc-tokens... <TAG_END>)?            bloc descriptif optionnel (dropout)
  (<REF_START> *1024 ref <REF_END>)?                 reference optionnelle (dropout)
  <GEN_START> *pixels <FRAME_SEP> *pixels ... <SEQ_END>

desc-tokens = tokens auto-identifiants (game/kind/gender/type(s)/stage/shiny) + marqueurs
`id`/`color` dont la valeur est un EMBEDDING injecte par le modele (side-channels id_index
/ color_rgb). Plus de FRAMES_VAL (N positionnel). loss_mask = contenu seulement (role).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .vocab import (
    ACTION_TOKEN, DIR_TOKEN, GEN_START, SEQ_END, REF_START, REF_END, FRAME_SEP,
    TAG_START, TAG_END, GAME_TOKEN, KIND_TOKEN, GENDER_TOKEN, TYPE_TOKEN, STAGE_TOKEN,
    SHINY, ID, COLOR, SOURCE_ACTION_MAP, PIXELS_PER_FRAME, SPRITE_SIZE, REF_FRAME_INDEX, ROLE,
)


@dataclass
class EncodedSequence:
    tokens: np.ndarray      # (T,) int32
    roles: np.ndarray       # (T,) int8
    x_pos: np.ndarray       # (T,) int8
    y_pos: np.ndarray       # (T,) int8
    frame_pos: np.ndarray   # (T,) int8
    id_index: np.ndarray    # (T,) int32 : index d'identite aux positions ID (0 ailleurs)
    family_index: np.ndarray # (T,) int32 : index de lignee (famille) aux positions ID
    color_rgb: np.ndarray   # (T,3) uint8 : RGB aux positions COLOR (0 ailleurs)
    gen_start_idx: int

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


_YS, _XS = np.meshgrid(np.arange(SPRITE_SIZE), np.arange(SPRITE_SIZE), indexing="ij")
_RASTER_Y = _YS.reshape(-1).astype(np.int8)
_RASTER_X = _XS.reshape(-1).astype(np.int8)


def _desc_tokens(descriptors: dict, colors_rgb, identity_index, family_index):
    """Liste (token, id_index, family_index, rgb) du bloc descriptif, ordre canonique."""
    out = []
    d = descriptors or {}
    if "game" in d:   out.append((GAME_TOKEN[d["game"]], 0, 0, None))
    if "kind" in d:   out.append((KIND_TOKEN[d["kind"]], 0, 0, None))
    if "gender" in d: out.append((GENDER_TOKEN[d["gender"]], 0, 0, None))
    for t in d.get("types", []): out.append((TYPE_TOKEN[t], 0, 0, None))
    if "stage" in d:  out.append((STAGE_TOKEN[d["stage"]], 0, 0, None))
    if d.get("shiny"): out.append((SHINY, 0, 0, None))
    if identity_index or family_index:   # position ID porte id + famille (embeddings sommes)
        out.append((ID, int(identity_index or 0), int(family_index or 0), None))
    for rgb in (colors_rgb or []):
        out.append((COLOR, 0, 0, tuple(int(c) for c in rgb)))
    return out


def encode_cycle(cycle_frames: np.ndarray, length: int, action_source: str,
                 direction: str | None, ref_frame_32x32: np.ndarray | None,
                 descriptors: dict | None = None, colors_rgb=None,
                 identity_index: int | None = None, family_index: int | None = None,
                 drop_tags: bool = False) -> EncodedSequence:
    if length < 1:
        raise ValueError("length must be >= 1")
    action_tok = ACTION_TOKEN[SOURCE_ACTION_MAP[action_source]]
    dir_tok = DIR_TOKEN[direction if direction is not None else "none"]

    desc = [] if drop_tags else _desc_tokens(descriptors, colors_rgb, identity_index, family_index)
    has_tags = len(desc) > 0
    has_ref = ref_frame_32x32 is not None

    # --- longueurs ---
    core_len = 3                                            # action, dir, N
    tag_len = (2 + len(desc)) if has_tags else 0            # TAG_START + desc + TAG_END
    ref_len = (1 + PIXELS_PER_FRAME + 1) if has_ref else 0
    content_len = 1 + length * PIXELS_PER_FRAME + (length - 1) + 1
    total = core_len + tag_len + ref_len + content_len

    tokens = np.zeros(total, np.int32)
    roles = np.zeros(total, np.int8)
    x_pos = np.zeros(total, np.int8)
    y_pos = np.zeros(total, np.int8)
    frame_pos = np.zeros(total, np.int8)
    id_index = np.zeros(total, np.int32)
    family_index = np.zeros(total, np.int32)
    color_rgb = np.zeros((total, 3), np.uint8)

    p = 0
    # coeur : action, dir, N (tous PREFIX_NON_PIXEL)
    for t in (action_tok, dir_tok, length):
        tokens[p] = t; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1

    # bloc descriptif
    if has_tags:
        tokens[p] = TAG_START; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1
        for tok, idx, fam, rgb in desc:
            tokens[p] = tok; roles[p] = ROLE.PREFIX_NON_PIXEL
            if idx: id_index[p] = idx
            if fam: family_index[p] = fam
            if rgb is not None: color_rgb[p] = rgb
            p += 1
        tokens[p] = TAG_END; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1

    # reference
    if has_ref:
        tokens[p] = REF_START; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1
        tokens[p:p + PIXELS_PER_FRAME] = ref_frame_32x32.reshape(-1).astype(np.int32)
        roles[p:p + PIXELS_PER_FRAME] = ROLE.PREFIX_PIXEL
        x_pos[p:p + PIXELS_PER_FRAME] = _RASTER_X
        y_pos[p:p + PIXELS_PER_FRAME] = _RASTER_Y
        frame_pos[p:p + PIXELS_PER_FRAME] = REF_FRAME_INDEX
        p += PIXELS_PER_FRAME
        tokens[p] = REF_END; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1

    # contenu
    gen_start_idx = p
    tokens[p] = GEN_START; roles[p] = ROLE.PREFIX_NON_PIXEL; p += 1
    for fi in range(length):
        tokens[p:p + PIXELS_PER_FRAME] = cycle_frames[fi].reshape(-1).astype(np.int32)
        roles[p:p + PIXELS_PER_FRAME] = ROLE.CONTENT_PIXEL
        x_pos[p:p + PIXELS_PER_FRAME] = _RASTER_X
        y_pos[p:p + PIXELS_PER_FRAME] = _RASTER_Y
        frame_pos[p:p + PIXELS_PER_FRAME] = fi
        p += PIXELS_PER_FRAME
        if fi < length - 1:
            tokens[p] = FRAME_SEP; roles[p] = ROLE.CONTENT_SEP; p += 1
    tokens[p] = SEQ_END; roles[p] = ROLE.CONTENT_SEP; p += 1
    assert p == total, f"length mismatch {p} != {total}"

    return EncodedSequence(tokens=tokens, roles=roles, x_pos=x_pos, y_pos=y_pos,
                           frame_pos=frame_pos, id_index=id_index, family_index=family_index,
                           color_rgb=color_rgb, gen_start_idx=gen_start_idx)


def make_loss_mask(seq: EncodedSequence) -> np.ndarray:
    """Cibles a apprendre (next-token) : contenu seulement (CONTENT_PIXEL/CONTENT_SEP).
    Tout le conditionnement (coeur, bloc TAG, ref, GEN_START) est masque."""
    target_roles = seq.roles[1:]
    return ((target_roles == ROLE.CONTENT_PIXEL) | (target_roles == ROLE.CONTENT_SEP)).astype(np.int8)
