"""Vocabulaire du modèle génératif (36 tokens, voir brief Partie 2).

Layout :
   0..15  : indices palette / valeurs numériques (double usage selon contexte)
   16     : <FRAMES_VAL>           marqueur "le prochain token est un compteur de frames"
   17     : <GEN_START>            début de la zone à générer
   18     : <SEQ_END>              fin de séquence
   19     : <REF_START>            début de l'image de référence
   20     : <REF_END>              fin de l'image de référence
   21     : <FRAME_SEP>             séparateur entre frames
   22..31 : tags d'action (idle, walk, run, attack, cast, hurt, dodge, victory, defeat, combat)
   32..35 : tags de direction (down, right, left, up)

Au POC v1 seuls <action:idle>, <action:combat> et les 4 directions sont effectivement
utilisés. Les autres tags d'action restent dans le vocab pour ne pas avoir à
re-tokeniser quand v2 enrichira le dataset.
"""
from __future__ import annotations
from dataclasses import dataclass

# --- Token IDs ---
N_PALETTE = 16
PIX_FIRST, PIX_LAST = 0, 15

FRAMES_VAL  = 16
GEN_START   = 17
SEQ_END     = 18
REF_START   = 19
REF_END     = 20
FRAME_SEP   = 21

ACTION_FIRST = 22
ACTIONS = ["idle", "walk", "run", "attack", "cast",
           "hurt", "dodge", "victory", "defeat", "combat"]
ACTION_TOKEN = {name: ACTION_FIRST + i for i, name in enumerate(ACTIONS)}

DIR_FIRST = 32
DIRECTIONS = ["down", "right", "left", "up"]
DIR_TOKEN = {name: DIR_FIRST + i for i, name in enumerate(DIRECTIONS)}

VOCAB_SIZE = 36
assert DIR_FIRST + len(DIRECTIONS) == VOCAB_SIZE


def is_pixel_token(t: int) -> bool:
    return 0 <= t < N_PALETTE


def token_name(t: int) -> str:
    if 0 <= t < N_PALETTE: return f"PIX({t})"
    if t == FRAMES_VAL:    return "<FRAMES_VAL>"
    if t == GEN_START:     return "<GEN_START>"
    if t == SEQ_END:       return "<SEQ_END>"
    if t == REF_START:     return "<REF_START>"
    if t == REF_END:       return "<REF_END>"
    if t == FRAME_SEP:     return "<FRAME_SEP>"
    if ACTION_FIRST <= t < ACTION_FIRST + len(ACTIONS):
        return f"<action:{ACTIONS[t - ACTION_FIRST]}>"
    if DIR_FIRST <= t < DIR_FIRST + len(DIRECTIONS):
        return f"<dir:{DIRECTIONS[t - DIR_FIRST]}>"
    return f"<INVALID:{t}>"


# --- Mapping action sources -> tags du vocabulaire ---
# Notre dataset a 3 sources d'action ; le POC en regroupe 2 selon le brief.
SOURCE_ACTION_MAP = {
    "idle_overworld":   "idle",
    "idle_combat":      "combat",
    "animated_combat":  "combat",
    "human_walk":       "walk",      # sheets humains : cycles 3-frames idle->walk->walk
    "human_victory":    "victory",   # sheets humains : 2-frames idle->pose finale
}


# --- Géométrie sprite ---
SPRITE_SIZE = 32
PIXELS_PER_FRAME = SPRITE_SIZE * SPRITE_SIZE   # 1024
MAX_FRAMES = 16
REF_FRAME_INDEX = MAX_FRAMES                    # index 16 réservé pour l'image de référence


@dataclass(frozen=True)
class TokenRole:
    """Rôle d'une position dans la séquence — sert au masquage de loss et aux embeddings."""
    PREFIX_NON_PIXEL = 0   # tags, FRAMES_VAL, valeur de N, REF_START/END, GEN_START (en pré-position)
    PREFIX_PIXEL     = 1   # pixels de l'image de référence
    CONTENT_PIXEL    = 2   # pixels des frames à générer
    CONTENT_SEP      = 3   # FRAME_SEP / SEQ_END dans la zone de génération


ROLE = TokenRole()
