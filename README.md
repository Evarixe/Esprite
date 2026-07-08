# wan_sprites

Génération de **cycles d'animation pixel-art** (sprites 32×32, palette 16 couleurs,
cycles de 2 à 16 frames) par un **transformer autoregressif ~50 M de paramètres**,
puis alignement de la qualité d'animation sur **l'œil humain** via une boucle de
préférences (DPO) — sans classifieur de beauté appris, donc sans risque de Goodhart.

Le dépôt contient le code ; les checkpoints de production sont publiés séparément sur
**Hugging Face** (lien à venir).

---

## Idée

Un sprite animé est une courte séquence de frames partageant une identité visuelle et
une palette. Plutôt que de générer des pixels RGB, on travaille en **indices de
palette** (0–15) sur une grille 32×32, et on modélise l'animation comme une **séquence
de tokens** qu'un décodeur causal produit image par image, conditionné par une action
(`idle`, `walk`, `combat`, `victory`…), une direction, un nombre de frames cible, et
optionnellement une **image de référence**.

Deux étages d'apprentissage :

1. **SFT** (supervised fine-tuning) — le modèle apprend à reproduire les cycles du dataset.
2. **DPO human-in-the-loop** — des duels A/B jugés par un humain réinjectent une
   préférence esthétique directement en gradient, pour améliorer le *mouvement* que les
   métriques de teacher-forcing ne mesurent pas.

---

## Représentation des sprites — choix de design

- **Indices de palette, pas de RGB.** Sprite = grille 32×32 d'`uint8` ∈ [0,15].
  Index `0` = transparence sentinelle ; `1..15` = couleurs d'une **palette RGB unifiée
  par cycle**. Le modèle raisonne sur la structure, pas sur des valeurs continues.
- **Downsampling par ratios entiers** (1:1 / 2:1 / 3:1, mode-par-bloc sur indices,
  **jamais d'interpolation**) — préserve exactement la palette et les bords nets.
- **Cycles ≤ 16 frames.** Les animations plus longues (intros de combat) sont
  **sous-échantillonnées uniformément** (garde l'arc complet) au lieu d'être jetées.
- **Transparence** dérivée du canal alpha, pad 32×32 centré-bas (sprite posé au sol).

---

## Le modèle génératif (`src/genmodel/`)

### Format de séquence — vocabulaire de 36 tokens

```
[<action>, <dir>?, <FRAMES_VAL>, <N>,
 (<REF_START>, *1024 pixels de référence, <REF_END>)?,
 <GEN_START>, *1024 pixels frame0, <FRAME_SEP>, *1024 pixels frame1, …, <SEQ_END>]
```

- **Tokens 0–15 à double usage** (valeur numérique *ou* pixel) — désambiguïsés par le
  **contexte syntaxique** (après `<FRAMES_VAL>` = valeur, sinon = pixel).
- **Image de référence optionnelle**, avec **dropout p=0.5 à l'entraînement** : le modèle
  apprend à générer *avec* (sprite-to-sprite) **et** *sans* ref (text-to-sprite).
- **Rôles par position** (`PREFIX_NON_PIXEL` / `PREFIX_PIXEL` / `CONTENT_PIXEL` /
  `CONTENT_SEP`) pilotent le masquage de loss et les embeddings sélectifs. On n'apprend
  jamais à prédire le conditionnement.

### Architecture

- **Decoder-only causal**, pre-norm RMSNorm, FFN GELU. `dim=512`, `16 couches`,
  `16 têtes`, `dim_ff=2048` — **50,41 M paramètres**.
- **Embeddings** : token appris (36×512), position de séquence sinusoïdale, et
  **positions x / y / frame apprises appliquées uniquement aux pixels**.
- **Weight tying** `head.weight = tok_emb.weight` : l'économie de params est négligeable,
  mais le tying **régularise les tokens rares en cible** (`FRAME_SEP` / `SEQ_END`) et
  accélère l'apprentissage du structurel — validé par ablation A/B.

### Le problème central : faire respecter le nombre de frames

Une séquence de N frames = N×1024 pixels mais **1 seul terminateur par frame**, soit
`f ≈ 0,098 %` des positions de contenu — **indépendant de N**. Les décisions de longueur
sont noyées sous les pixels, et le modèle « compte » mal les frames. Deux réponses :

- **Reweighting de loss** (`LossWeights`, dans `loss.py`) — un poids par position dérivé
  de `(rôle, classe de la cible)`, qui unifie plusieurs leviers :

  | coef | cible | rôle |
  |---|---|---|
  | `w_color` | pixel couleur 1–15 | pousse le dessin |
  | `w_transp` | pixel transparent 0 | casse le « pari-sûr » transparent |
  | `w_sep` | `FRAME_SEP` | grammaire « continue » |
  | `w_end` | `SEQ_END` | **le compte** (levier maximal) |
  | `w_ref` | pixels de référence dé-masqués | reconstruction = amorçage from-scratch |
  | `w_ref_end` | `REF_END` | fin de bloc ref |

- **Métrique d'écart `SEQ_END`** (`eval_metrics.py`, non différentiable) : mesure
  l'écart signé entre la position réelle et attendue du terminateur (négatif = trop
  court, positif = déborde, `no_seq_end` = runaway). Sur la lignée SFT, le biais signé
  s'est effondré de `+1188` à `≈ 0` avec les steps, à `w_end` constant.

### Dé-masquer la référence — le cold-start no-ref

Diagnostic : en génération *sans* ref, le modèle produisait des frames **entièrement
vides**. Ce n'est pas un manque de données mais un **exposure bias** — en teacher
forcing, il n'a jamais eu à *amorcer* un sprite de zéro (la ref ou le teacher lui
révèlent toujours le contenu). Solution : **dé-masquer les pixels de la référence**
(`w_ref>0`) pour entraîner explicitement le geste « dessiner un sprite from-scratch pour
ces tags ». Premier amorçage no-ref non-nul obtenu ainsi, température écartée comme cause.

---

## Boucle human-in-the-loop : SFT + DPO unifiés

**Modèle mental (important) :** il n'y a *pas* « un DPO à côté ». Il y a **un seul
workflow d'entraînement** où les steps viennent en **types** : SFT-court, SFT-long,
**DPO**. Tous partagent **la même horloge (step de lignée) et la même cosine LR globale**
— un step DPO lit `lr_at(step)` exactement comme un step SFT. Le DPO est un *type de
step*, pas un programme séparé.

Loss DPO : `−log σ( β · [ (logπ(c) − logπ_ref(c)) − (logπ(r) − logπ_ref(r)) ] )`,
référence figée = checkpoint courant. **Pas de reward model** (l'humain arbitre
directement) → **pas de Goodhart**.

Recette validée par sweep β/lr :

- **`logπ` length-normalized** (moyenne par token de contenu, pas somme) — sinon la somme
  récompense mécaniquement les séquences longues (**biais de longueur → runaway**) et
  amplifie le gradient ∝ L.
- **β = 1.0** — β contrôle la *saturation* (self-limiting du drift), pas seulement la
  vitesse : β trop bas ne sature jamais → dérive illimitée / perte de fidélité à la ref.
- **LR_DPO = LR_SFT / 10** — le DPO est un raffinement, il tape 10× plus doux que la
  construction SFT (lr trop haut = morphing d'identité, trop bas = inerte).
- **Masque structurel** — le DPO ne porte que sur les pixels ; `SEQ_END`/`FRAME_SEP` sont
  exclus (la grammaire de comptage, fragile, est laissée au SFT).
- **Paires same-model best-of-2** — chaque item généré avec **2 seeds** du checkpoint
  courant, l'humain choisit le meilleur tirage → longueurs similaires, biais de longueur
  natif nul, self-refinement propre.
- **Pool = dataset complet** (le best-of-2 ne compare rien à la vérité terrain → aucune
  fuite), **variété forcée** (strates source/action/direction, sujets distincts,
  fraction with/no-ref, mémoire anti-répétition) et **fenêtre glissante** sur les N
  dernières campagnes pour diluer l'overfit sur ~50 votes.

Un cycle typique : `~3k SFT → génère 50 items × 2 seeds → 50 duels simples → DPO 1 epoch`.
Le DPO représente ~5–8 % des steps ; sa fréquence est pilotée par la **tolérance de vote
humaine**, pas par un idéal théorique.

L'**arène** (`src/arena/`, `arena.html`) sert les duels A/B en aveugle (anims en loop
synchronisé, ref au centre, identités révélées après le vote) et classe les checkpoints
en **Bradley-Terry**. L'orchestrateur (`loop.py`) est une **machine à états
ré-invocable** (pause humaine au vote) qui écrit toute la lignée — historique continu,
tête `last.pt`, snapshots, campagnes — dans **un seul dossier de run**.

---

## Encodeur contrastif auxiliaire (`src/encoder/`)

Un petit encodeur (1,07 M params, NT-Xent / SimCLR) apprend un vecteur 128-D d'identité
visuelle invariant à la pose (paires = 2 frames d'un même cycle). Outil d'analyse et de
validation ; **pas** utilisé comme récompense (cf. anti-Goodhart).

---

## Monitoring (`src/monitor/`, `dashboard.html`)

Serveur stdlib sans dépendance : heartbeat live, historique, télémétrie GPU
(util / VRAM / température / puissance), et un canal de contrôle
(pause / stop / checkpoint) piloté depuis le dashboard. Le training écrit
`heartbeat.json` / `history.json` ; le dashboard poll toutes les ~2 s.

---

## Installation

Environnement géré par [uv](https://docs.astral.sh/uv/). PyTorch **nightly cu128**
(support Blackwell / sm_120, testé RTX 5090 32 GiB).

```bash
uv sync
```

## Utilisation

```bash
set PYTHONPATH=src            # Windows ; export sous Unix

# Dataset
uv run python build_dataset.py

# Entraînement du générateur (SFT)
uv run python -m genmodel.train --out runs/gen --total-steps 20000

# Boucle human-in-the-loop (SFT + DPO)
uv run python loop.py init --ckpt runs/gen/last.pt --global-step 20000 --horizon 40000
uv run python loop.py run    # SFT + génère la campagne, puis s'arrête au vote
# ... voter dans l'arène (commande imprimée) ...
uv run python loop.py run    # rejoue : lance le DPO, cycle suivant

# Monitoring
uv run python -m monitor.server --run runs/dpo_loop --port 8765
```

## Structure

```
src/
  dataset/     construction du dataset (sources, palette, splits, downsampling)
  genmodel/    modèle génératif : vocab, tokenize, model, loss, sample, dpo
  arena/       arène d'éval humaine (serveur, campagnes, ranking Bradley-Terry)
  encoder/     encodeur contrastif auxiliaire
  monitor/     dashboard live (heartbeat, gpu, control, serveur)
tools/       benchmarks, profiling, éval, visualisations (hors workflow principal)
loop.py        orchestrateur de la boucle unifiée SFT + DPO
dpo_train.py   étape DPO (fenêtre glissante sur les campagnes)
dpo_campaign.py génération de campagne DPO same-model 2-seeds
build_dataset.py   construction du dataset
render_samples.py  rendu de grilles d'échantillons
dashboard.html / arena.html  fronts
```

## Statut

POC v1 — recherche en cours. Les checkpoints de production seront publiés sur
Hugging Face.
