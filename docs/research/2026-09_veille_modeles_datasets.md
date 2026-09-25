# Veille modèles & datasets — Esprite (sept. 2026)

Journal tenu au fil de l'eau. Périmètre : (1) évolutions de modèle pertinentes pour
Esprite, fondées sur des travaux **publiés depuis janvier 2026** ; (2) datasets
utilisables (âge indifférent). Proposition finale en bas de document.

---

## 0. État des lieux du projet (lecture du code, commit `4f200ac`)

**Tâche** : générer des cycles d'animation pixel-art 32×32, palette indexée 16 couleurs
(0 = transparent), 2 à 16 frames, conditionnés par action (14), direction (5),
nombre de frames N, bloc descriptif optionnel (jeu, kind, genre, types Pokémon, stade,
shiny, embedding d'identité + famille d'évolution, couleurs dominantes RGB) et une
image de référence optionnelle.

**Modèle** (`src/genmodel/model.py`) : décodeur causal ~50 M params, dim 512, 16 couches,
16 têtes, FFN GELU, RMSNorm pre-norm, PE sinusoïdale + embeddings x/y/frame appris,
weight tying. **Un token = un pixel**, ordre raster → 1024 tokens/frame, jusqu'à ~18 k
tokens par séquence (16 frames + ref). Sampling AR token-par-token avec KV-cache statique
et CUDA Graphs.

**Entraînement** : SFT CE pondérée (w_end=40, w_sep=20 pour apprendre à compter les
frames), puis boucle DPO human-in-the-loop (best-of-2, β=1, logπ normalisé en longueur,
lr/10). Augmentation : permutation de palette (indices 1..15).

**Données** (`src/dataset/sources.py`) : veekun (HGSS overworld 2 frames, Emerald combat
2 frames, Emerald animated GIF ≤16 frames), feuilles de dresseurs Pokémon, feuilles TSR
(Zelda Minish Cap) labellisées à la main. D'après `data.py` : **~4 800 cycles de 2 frames
et ~32 cycles longs (8–16 frames)** au moment de la v1 → dataset très petit et très
déséquilibré ; les 12 actions nouvelles de la v2 (run, jump, climb, swim, shoot, cast,
guard, dodge, hurt, defeat…) n'ont quasiment aucun exemple hors TSR.

**Points faibles identifiés à la lecture** (hypothèses à confronter à la veille) :

1. *Coût/longueur de séquence* : 1 token = 1 pixel, O(N·1024) pas de décodage séquentiels,
   attention quadratique jusqu'à 18 k. Les 2 frames successives sont très redondantes
   (souvent < 10 % de pixels changent) mais le modèle re-prédit tout.
2. *Comptage des frames* fragile (1 terminateur pour 1024 pixels) → hacks de pondération.
3. *Cold-start no-ref* (frames vides) → exposure bias.
4. *PE sinusoïdale absolue 1D* alors que la structure est 3D (x, y, t) ; pas de RoPE.
5. *Données* : pénurie d'animations longues et d'actions variées.

---

## 1. Veille modèles (≥ janvier 2026)

_(en cours)_

## 2. Veille techniques transférables (≥ janvier 2026)

_(en cours)_

## 3. Datasets

_(en cours)_

## 4. Proposition

_(à venir)_
