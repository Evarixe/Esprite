"""Derivation des attributs descriptifs d'un cycle (spec v2) depuis meta + veekun.

Truthful par cycle : on ne renvoie QUE ce qui est connu (on omet le reste). Les couleurs
dominantes ne sont PAS ici (calculees post-augmentation dans data.py). Le type Pokemon
utilise le typage MODERNE de veekun -> fees retrofittees (Melofee->Fee, etc.).

`id` = cle d'identite string (pk<dex> / ch:<perso>). Le registre map ces cles -> index
entier pour la table d'embedding d'identite (build_identity_registry, stable via seed d'ordre).
"""
from __future__ import annotations
import functools
import sqlite3
from pathlib import Path

from .vocab import TYPES, GAMES, KINDS, GENDERS, STAGES

_VEEKUN = Path("assets/veekun-pokedex.sqlite")
_TYPESET = set(TYPES)

# source du dataset -> jeu
SOURCE_GAME = {
    "hgss_overworld":      "pokemon",
    "emerald_combat":      "pokemon",
    "emerald_animated":    "pokemon",
    "human_overworld":     "pokemon",   # sprites de dresseurs Pokemon (Aroma Lady, etc.)
    "tsr_zelda_minishcap": "zelda",
}
# kind pour les personnages (non-Pokemon) : creatures connues -> creature, sinon character.
_TSR_CREATURES = {
    "darknut", "stalfos", "spear_moblin", "octorok", "chuchu", "keaton", "vaati",
    "goomba", "magikoopa", "hammer_bro", "koopa", "bowser", "fawful", "moblin",
    "mc_keaton",
}


@functools.lru_cache(maxsize=1)
def _con():
    if not _VEEKUN.exists():
        return None
    return sqlite3.connect(f"file:{_VEEKUN}?mode=ro", uri=True)


@functools.lru_cache(maxsize=4096)
def _poke_types(species_id: int) -> tuple:
    con = _con()
    if con is None:
        return ()
    q = ("SELECT lower(t.identifier) FROM pokemon p "
         "JOIN pokemon_types pt ON pt.pokemon_id=p.id "
         "JOIN types t ON t.id=pt.type_id "
         "WHERE p.species_id=? AND p.is_default=1 ORDER BY pt.slot")
    return tuple(r[0] for r in con.execute(q, (species_id,)).fetchall() if r[0] in _TYPESET)


@functools.lru_cache(maxsize=4096)
def _poke_stage(species_id: int) -> int:
    """Profondeur dans la chaine d'evolution : base=1, 1re evo=2, 2e evo=3 (cap 3)."""
    con = _con()
    if con is None:
        return 1
    depth, sid = 1, species_id
    for _ in range(5):
        row = con.execute("SELECT evolves_from_species_id FROM pokemon_species WHERE id=?", (sid,)).fetchone()
        if not row or row[0] is None:
            break
        sid = row[0]
        depth += 1
    return min(depth, 3)


def cycle_descriptors(meta: dict) -> dict:
    """Attributs truthful d'un cycle (hors couleurs). Cles possibles : game, kind, gender,
    id, types(list<=2), stage, shiny."""
    d: dict = {}
    game = SOURCE_GAME.get(meta.get("source"))
    if game:
        d["game"] = game
    pid = meta.get("pokemon_id")
    if pid is not None:
        d["kind"] = "creature"
        d["id"] = f"pk{pid}"
        types = list(_poke_types(int(pid)))
        if types:
            d["types"] = types[:2]
        d["stage"] = f"stage{_poke_stage(int(pid))}"
        if meta.get("variant") == "shiny":
            d["shiny"] = True
    else:
        name = str(meta.get("character") or "").lower()
        d["kind"] = "creature" if any(c in name for c in _TSR_CREATURES) else "character"
        if meta.get("character"):
            d["id"] = f"ch:{meta['character']}"
    g = meta.get("gender")
    if g == "female":
        d["gender"] = "female"   # 'regular' = defaut (male/asexue inconnu) -> on n'affirme pas
    return d


def identity_key(meta: dict) -> str | None:
    pid = meta.get("pokemon_id")
    if pid is not None:
        return f"pk{pid}"
    if meta.get("character"):
        return f"ch:{meta['character']}"
    return None


def build_identity_registry(metas: list[dict]) -> dict:
    """Cle d'identite -> index entier (0-based, ordre trie deterministe). Index 0 reserve
    a l'identite INCONNUE/absente (generation generique)."""
    keys = sorted({k for m in metas if (k := identity_key(m)) is not None})
    return {"__none__": 0, **{k: i + 1 for i, k in enumerate(keys)}}
