"""Parsers par source veekun. Chaque parser renvoie un itérable de RawCycle.

Sources retenues pour le POC :
  - HGSS overworld : 32x32 natif, cycles 2-frames (frame1 + frame2/), x {down,right,left,up} x {normal,shiny} x {regular,female}
  - Gen3 Emerald combat : 64x64, cycles 2-frames (main + frame2/), x {normal,shiny}
  - Gen3 Emerald animated : 96x96, GIFs, on filtre à <=16 frames
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator
import numpy as np
from PIL import Image

from .types import RawCycle


# ---------- helpers ----------

def _load_rgba(path: str | Path) -> np.ndarray:
    """Charge un PNG en array (H, W, 4) uint8 RGBA."""
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _gif_frames_rgba(path: str | Path) -> list[np.ndarray]:
    """Extrait toutes les frames d'un GIF en RGBA.

    Important : on convertit chaque frame indépendamment via 'RGBA' pour gérer
    correctement les disposals (apparition progressive de pixels).
    """
    im = Image.open(path)
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append(np.asarray(im.convert("RGBA"), dtype=np.uint8))
    return frames


# ---------- HGSS overworld ----------

def parse_hgss_overworld(root: Path) -> Iterator[RawCycle]:
    """root = .../assets/overworld/pokemon/overworld"""
    DIRS = ("down", "right", "left", "up")

    # Combinaisons (sub_path, variant, gender) à parcourir
    # variant ∈ {normal, shiny}, gender ∈ {regular, female}
    branches = [
        ("", "normal", "regular"),
        ("shiny", "shiny", "regular"),
        ("female", "normal", "female"),
        ("shiny/female", "shiny", "female"),
    ]

    for sub, variant, gender in branches:
        for direction in DIRS:
            base = root / sub / direction if sub else root / direction
            f2 = base / "frame2"
            if not base.is_dir() or not f2.is_dir():
                continue
            for fname in sorted(os.listdir(base)):
                if not fname.endswith(".png") or not fname[:-4].isdigit():
                    continue
                p1 = base / fname
                p2 = f2 / fname
                if not p2.exists():
                    continue
                pokemon_id = int(fname[:-4])
                yield RawCycle(
                    frames=[_load_rgba(p1), _load_rgba(p2)],
                    metadata={
                        "source": "hgss_overworld",
                        "pokemon_id": pokemon_id,
                        "action": "idle_overworld",
                        "direction": direction,
                        "variant": variant,
                        "gender": gender,
                    },
                )


# ---------- Gen3 Emerald combat 2-frames ----------

def parse_emerald_combat(root: Path) -> Iterator[RawCycle]:
    """root = .../assets/generation-3/pokemon/main-sprites/emerald"""
    branches = [
        (root, "normal"),
        (root / "shiny", "shiny"),
    ]
    for base, variant in branches:
        f2 = base / "frame2"
        if not f2.is_dir():
            continue
        for fname in sorted(os.listdir(base)):
            if not fname.endswith(".png") or not fname[:-4].isdigit():
                continue
            p1 = base / fname
            p2 = f2 / fname
            if not p2.exists():
                continue
            pokemon_id = int(fname[:-4])
            yield RawCycle(
                frames=[_load_rgba(p1), _load_rgba(p2)],
                metadata={
                    "source": "emerald_combat",
                    "pokemon_id": pokemon_id,
                    "action": "idle_combat",
                    "direction": None,
                    "variant": variant,
                    "gender": "regular",
                },
            )


# ---------- Gen3 Emerald animated GIFs ----------

def parse_emerald_animated(root: Path, max_frames: int = 16) -> Iterator[RawCycle]:
    """root = .../assets/generation-3/pokemon/main-sprites/emerald/animated

    GIFs > max_frames : sous-échantillonnés uniformément (1 frame sur K, K=ceil(n/max))
    pour capturer l'arc complet de l'animation en <=max_frames, au lieu de jeter.
    Les GIFs <= max_frames sont pris entiers (K=1).

    NB : ces GIFs sont des animations d'ENTRÉE de combat (intro→pose tenue→reboucle),
    pas des cycles idle bouclés. Isolés sous le tag `animated_combat` pour ne pas
    contaminer le signal de bouclage des autres actions.
    """
    if not root.is_dir():
        return
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".gif") or not fname[:-4].isdigit():
            continue
        path = root / fname
        with Image.open(path) as im:
            n = im.n_frames
        if n < 2:
            continue
        frames = _gif_frames_rgba(path)
        if n > max_frames:
            # sous-échantillonnage uniforme sur toute la durée
            import math
            k = math.ceil(n / max_frames)
            frames = frames[::k][:max_frames]
        pokemon_id = int(fname[:-4])
        yield RawCycle(
            frames=frames,
            metadata={
                "source": "emerald_animated",
                "pokemon_id": pokemon_id,
                "action": "animated_combat",
                "direction": None,
                "variant": "normal",
                "gender": "regular",
            },
        )


# ---------- Human overworld sheets ----------

def _detect_cell_width(W: int) -> int | None:
    """Largeur de cellule donnant 9 ou 10 cases. None si indéterminable."""
    for cw in (16, 32):
        if W % cw == 0 and (W // cw) in (9, 10):
            return cw
    return None


def _bg_to_alpha(cell_rgba: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """Met alpha=0 sur les pixels égaux à la couleur de fond (transparence par
    couleur plate → alpha, pour compat avec le reste du pipeline qui lit alpha)."""
    out = cell_rgba.copy()
    is_bg = np.all(out[:, :, :3] == bg_rgb[:3], axis=2)
    out[is_bg, 3] = 0
    return out


def _pad_to_32(cell_rgba: np.ndarray) -> np.ndarray:
    """Pad une cellule (h, w, 4) <=32 vers 32x32 : centré horizontalement,
    aligné en bas verticalement (sprite posé au sol). Padding transparent."""
    h, w = cell_rgba.shape[:2]
    if h > 32 or w > 32:
        # cas inattendu (cellule plus grande que 32) : on recadre au centre-bas
        cell_rgba = cell_rgba[max(0, h - 32):, :32]
        h, w = cell_rgba.shape[:2]
    out = np.zeros((32, 32, 4), dtype=np.uint8)
    x0 = (32 - w) // 2
    y0 = 32 - h            # aligné en bas
    out[y0:y0 + h, x0:x0 + w] = cell_rgba
    return out


def parse_human_overworld(root: Path) -> Iterator[RawCycle]:
    """root = .../assets/human_Overworld_Sprites

    Layout par sheet (9 ou 10 cases de cell_w × H) :
      case 0,1,2 = idle down/up/right ; 3,4 = walk down ; 5,6 = walk up ;
      7,8 = walk right ; 9 = pose finale (victory), absente si 9 cases.

    Cycles produits par perso :
      - walk 4-frames : [idle_dir, walk_a, idle_dir, walk_b] pour down/up/right + left=miroir(right)
        (la frame neutre intercalée entre chaque pas équilibre la démarche — convention GameFreak)
      - walk 2-frames : [walk_a, walk_b] (les deux pas seuls, sans idle intercalé).
        Donne au modèle le contraste walk@N=2 vs walk@N=4 → le nombre de frames devient
        une vraie consigne conditionnée par N, au lieu d'une constante figée au tag walk.
      - victory 2-frames : [idle_down, pose] si la case 9 existe
    Transparence = couleur du pixel (0,0). Chaque frame paddée à 32x32 (centré bas).
    """
    if not root.is_dir():
        return
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".png"):
            continue
        path = root / fname
        sheet = _load_rgba(path)
        H, W = sheet.shape[:2]
        cw = _detect_cell_width(W)
        if cw is None:
            print(f"[skip human] {fname}: largeur {W} non découpable (9/10 cases)")
            continue
        ncell = W // cw
        bg = sheet[0, 0].copy()

        # Découpe les cases, normalise transparence, pad 32x32
        cells = []
        for k in range(ncell):
            c = sheet[:, k * cw:(k + 1) * cw]
            c = _bg_to_alpha(c, bg)
            cells.append(_pad_to_32(c))

        name = fname[:-4]
        # walk 3-frames par direction. (idle_idx, walk_a, walk_b)
        walk_layout = [
            ("down",  0, 3, 4),
            ("up",    1, 5, 6),
            ("right", 2, 7, 8),
        ]
        for direction, i_idle, i_a, i_b in walk_layout:
            if max(i_idle, i_a, i_b) >= ncell:
                continue
            # 4-frames : neutre intercalé entre chaque pas pour équilibrer la démarche
            frames4 = [cells[i_idle], cells[i_a], cells[i_idle], cells[i_b]]
            # 2-frames : les deux pas seuls (contraste de longueur pour la consigne N)
            frames2 = [cells[i_a], cells[i_b]]
            for frames in (frames4, frames2):
                yield RawCycle(frames=frames, metadata={
                    "source": "human_overworld", "character": name,
                    "action": "human_walk", "direction": direction,
                    "variant": "normal", "gender": "regular",
                })
                # left = miroir horizontal de right
                if direction == "right":
                    mirrored = [f[:, ::-1].copy() for f in frames]
                    yield RawCycle(frames=mirrored, metadata={
                        "source": "human_overworld", "character": name,
                        "action": "human_walk", "direction": "left",
                        "variant": "normal", "gender": "regular",
                    })

        # victory 2-frames : idle_down -> pose finale (case 9, si 10 cases)
        if ncell >= 10:
            yield RawCycle(frames=[cells[0], cells[9]], metadata={
                "source": "human_overworld", "character": name,
                "action": "human_victory", "direction": "down",
                "variant": "normal", "gender": "regular",
            })


# ---------- Orchestrateur ----------

def iter_all_sources(assets_root: Path, max_anim_frames: int = 16) -> Iterator[RawCycle]:
    """Itère tous les cycles bruts de toutes les sources retenues."""
    yield from parse_hgss_overworld(
        assets_root / "overworld" / "pokemon" / "overworld"
    )
    yield from parse_emerald_combat(
        assets_root / "generation-3" / "pokemon" / "main-sprites" / "emerald"
    )
    yield from parse_emerald_animated(
        assets_root / "generation-3" / "pokemon" / "main-sprites" / "emerald" / "animated",
        max_frames=max_anim_frames,
    )
    yield from parse_human_overworld(
        assets_root / "human_Overworld_Sprites"
    )
