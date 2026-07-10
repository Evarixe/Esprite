"""Vocabulaire du modele generatif v2 (lignee from-scratch). Cf model_spec_v2.md.

Principe : dans le bloc descriptif, chaque valeur d'attribut est un token AUTO-IDENTIFIANT
(pokemon, creature, fire, stage3, male...) globalement unique -> pas de cles, pas de
partage de tokens (les tags sont optionnels et en nombre variable, partager confondrait).
Les 0-15 ne servent donc QUE de pixel ou de compteur N (position+role les distinguent).

`id` et `color` sont des tokens-marqueurs dont la valeur est un EMBEDDING injecte par le
modele (table d'identite / projection RGB continue), pas un token -> vocab petit malgre
des centaines d'identites et des couleurs continues.

Format de sequence (cf spec) :
  [action, dir, N]  (<TAG_START> desc-tokens... <TAG_END>)?  (<REF_START>...<REF_END>)?
  <GEN_START> pixels <FRAME_SEP> pixels ... <SEQ_END>
Plus de FRAMES_VAL (prefixe fixe, N positionnel). Plus de `combat` (Pokemon -> attack).
"""
from __future__ import annotations
from dataclasses import dataclass

# --- Pixels / valeurs (0-15) : pixel OU compteur N (position+role) ---
N_PALETTE = 16
PIX_FIRST, PIX_LAST = 0, 15

# --- Structurels (7) ---
GEN_START = 16
SEQ_END   = 17
REF_START = 18
REF_END   = 19
FRAME_SEP = 20
TAG_START = 21
TAG_END   = 22

# --- Actions (14) ---
ACTION_FIRST = 23
ACTIONS = ["idle", "walk", "run", "jump", "climb", "swim",
           "attack", "shoot", "cast",
           "guard", "dodge", "hurt", "defeat",
           "victory"]
ACTION_TOKEN = {name: ACTION_FIRST + i for i, name in enumerate(ACTIONS)}

# --- Directions (5, avec 'none') ---
DIR_FIRST = ACTION_FIRST + len(ACTIONS)   # 37
DIRECTIONS = ["down", "right", "left", "up", "none"]
DIR_TOKEN = {name: DIR_FIRST + i for i, name in enumerate(DIRECTIONS)}

# --- Descripteurs auto-identifiants (bloc TAG) ---
GAME_FIRST = DIR_FIRST + len(DIRECTIONS)  # 42
GAMES = ["pokemon", "zelda", "mario"]                       # extensible
GAME_TOKEN = {g: GAME_FIRST + i for i, g in enumerate(GAMES)}

KIND_FIRST = GAME_FIRST + len(GAMES)      # 45
KINDS = ["character", "creature"]
KIND_TOKEN = {k: KIND_FIRST + i for i, k in enumerate(KINDS)}

GENDER_FIRST = KIND_FIRST + len(KINDS)    # 47
GENDERS = ["male", "female", "none"]
GENDER_TOKEN = {g: GENDER_FIRST + i for i, g in enumerate(GENDERS)}

TYPE_FIRST = GENDER_FIRST + len(GENDERS)  # 50
TYPES = ["normal", "fire", "water", "grass", "electric", "ice", "fighting",
         "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
         "dragon", "dark", "steel", "fairy"]   # 18 (fairy garde la porte ouverte)
TYPE_TOKEN = {t: TYPE_FIRST + i for i, t in enumerate(TYPES)}

STAGE_FIRST = TYPE_FIRST + len(TYPES)     # 68
STAGES = ["stage1", "stage2", "stage3", "mega"]
STAGE_TOKEN = {s: STAGE_FIRST + i for i, s in enumerate(STAGES)}

SHINY = STAGE_FIRST + len(STAGES)         # 72  (flag : present => shiny)
ID    = SHINY + 1                         # 73  (marqueur, embedding identite)
COLOR = ID + 1                            # 74  (marqueur, embedding couleur RGB)

VOCAB_SIZE = COLOR + 1                     # 75
assert VOCAB_SIZE == 75


def is_pixel_token(t: int) -> bool:
    return 0 <= t < N_PALETTE


def token_name(t: int) -> str:
    if 0 <= t < N_PALETTE: return f"VAL({t})"
    singles = {GEN_START: "GEN_START", SEQ_END: "SEQ_END", REF_START: "REF_START",
               REF_END: "REF_END", FRAME_SEP: "FRAME_SEP", TAG_START: "TAG_START",
               TAG_END: "TAG_END", SHINY: "shiny", ID: "id", COLOR: "color"}
    if t in singles: return f"<{singles[t]}>"
    for first, names, pfx in ((ACTION_FIRST, ACTIONS, "action"), (DIR_FIRST, DIRECTIONS, "dir"),
                              (GAME_FIRST, GAMES, "game"), (KIND_FIRST, KINDS, "kind"),
                              (GENDER_FIRST, GENDERS, "gender"), (TYPE_FIRST, TYPES, "type"),
                              (STAGE_FIRST, STAGES, "stage")):
        if first <= t < first + len(names):
            return f"<{pfx}:{names[t - first]}>"
    return f"<INVALID:{t}>"


# --- Mapping action sources -> tags. `combat` supprime : Pokemon -> attack (geste agressif). ---
SOURCE_ACTION_MAP = {
    "idle_overworld":   "idle",
    "idle_combat":      "attack",
    "animated_combat":  "attack",
    "human_walk":       "walk",
    "human_victory":    "victory",
}
SOURCE_ACTION_MAP.update({a: a for a in ACTIONS})   # sources a action directe (TSR) : identite


# --- Geometrie sprite ---
SPRITE_SIZE = 32
PIXELS_PER_FRAME = SPRITE_SIZE * SPRITE_SIZE   # 1024
MAX_FRAMES = 16
REF_FRAME_INDEX = MAX_FRAMES                    # 16 : image de reference


@dataclass(frozen=True)
class TokenRole:
    """Role d'une position — masquage de loss + embeddings selectifs."""
    PREFIX_NON_PIXEL = 0   # coeur (action/dir/N), bloc TAG, REF_START/END, GEN_START
    PREFIX_PIXEL     = 1   # pixels de l'image de reference
    CONTENT_PIXEL    = 2   # pixels des frames a generer
    CONTENT_SEP      = 3   # FRAME_SEP / SEQ_END


ROLE = TokenRole()
