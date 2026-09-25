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

> Méthode : arxiv.org / alphaxiv.org bloqués par le proxy de la session → dates vérifiées
> via Hugging Face Papers (`published_at`), préfixe d'ID arXiv (AAMM) et dates de création
> des dépôts HF/GitHub. Les éléments lus seulement via extraits de recherche sont signalés.

**Constat principal** : quasiment **aucune publication académique 2026** ne traite
directement la génération d'*animations* de sprites pixel-art. Les travaux 2026 centrés
sprites sont des **LoRA communautaires** sur gros modèles image/vidéo ; l'apport
académique 2026 vient de techniques voisines (boucles, AR pixel-space, diffusion
discrète, DPO robuste, benchmarks d'animation).

| # | Travail | Date (vérif.) | Nature | Idée à reprendre pour Esprite |
|---|---|---|---|---|
| 1 | **svntax-dev** `pixel_spritesheet_4walk_small_lora_v1` / `…_4walk_combat_32x48_v1` — [HF](https://hf.co/svntax-dev/pixel_spritesheet_4walk_small_lora_v1), [HF](https://hf.co/svntax-dev/pixel_spritesheet_4walk_combat_32x48_v1) | 2026-02-01 / 2026-03-18 (dépôts) | LoRA FLUX.2-klein-4B puis Qwen-Image-Edit-2511 ; grilles 4×4 (walk×3 ×4 dirs, 32×32) et 6×4 (walk/attack/hurt, 32×48) ; downscale k-centroid ×4 ; mode *edit* : réf → sheet | L'analogue le plus proche (32 px, action×direction, réf). **Enseignant hors-ligne** pour synthétiser des cycles longs → k-centroid → quantif palette 16 → tri humain |
| 2 | **Loopy** : seamless looping video — [arXiv 2608.23090](https://arxiv.org/abs/2608.23090), [projet](https://donghaotian123.github.io/Loopy) | 2026-08-24 (ACM TOG 45(6)) | DiT vidéo ; décalage de PE *par couche* (couche « ancre » dominante) → temps perçu circulaire ; RGBA ; note que forcer frame₀ = frame_N ⇒ vidéos quasi-statiques | **Encodage temporel circulaire** (angle 2π·t/N) pour les cycles ; méfiance vis-à-vis de l'effondrement statique (aussi en DPO) |
| 3 | **PRA** — Parallel Rollout Approximation for pixel-space AR — [arXiv 2606.27978](https://arxiv.org/abs/2606.27978) | 2026-06-26 | AR direct sur patchs de pixels, sans tokenizer ; attaque (i) l'erreur par pas et (ii) l'exposure bias du teacher forcing en construisant en parallèle des entrées « type inférence » ; FID 1.94 (511 M) | Esprite = AR pixel-space en teacher forcing sur 18 k tokens → **corruption contrôlée du contexte** à l'entraînement (tokens remplacés par prédictions du modèle / couleurs voisines) contre la dérive des frames tardives et le cold-start |
| 4 | **Nemotron-Labs-Diffusion-Image** — [arXiv 2606.29814](https://arxiv.org/abs/2606.29814) | 2026-06-29 | Diffusion discrète masquée T2I ; *édition* de tokens déjà démasqués ; **Grouped Cross-Entropy** (crédit aux tokens voisins dans l'espace d'embedding) | (a) alternative non-raster : décodage parallèle + inpainting de frames ; (b) **CE à crédit partiel sur couleurs voisines** (distance palette) — utile en low-data |
| 5 | **LottieGPT** — [arXiv 2604.11792](https://arxiv.org/abs/2604.11792) | 2026-04-13 | Tokenizer natif d'animation vectorielle (keyframes, calques) ; dataset 660 K animations ; fine-tune Qwen-VL | Même thèse (animation = séquence AR) avec **compression** : encoder frame₀ puis **deltas** inter-frames |
| 6 | **Mystic07** `flux-lora-spritesheet` — [HF](https://hf.co/Mystic07/flux-lora-spritesheet) | 2026-04-27 (dépôt) | LoRA r32 FLUX.2-klein-9B sur ~18 706 sheets style **LPC** | Confirme LPC comme gisement massif de cycles longs multi-directions → à ingérer **directement** (§3) |
| 7 | **fal** `flux-2-klein-4b-spritesheet-lora` — [HF](https://hf.co/fal/flux-2-klein-4b-spritesheet-lora) | 2026-01-19 (dépôt) | LoRA edit : objet → grille 2×2 multi-vues ; **48 paires** curées | Peu de paires bien choisies suffisent à spécialiser un gros modèle → génération de **directions manquantes** |
| 8 | **ViPO / Poly-DPO** — [arXiv 2604.24953](https://arxiv.org/abs/2604.24953) | 2026-04-29 | Terme polynomial ajoutant une confiance adaptative au DPO selon le bruit des préférences ; se réduit au DPO sur données propres | **Remplacement drop-in** de `dpo_loss` (votes humains peu nombreux et bruités) ; annotation par critère |
| 9 | **AnimationBench** — [arXiv 2604.15299](https://arxiv.org/abs/2604.15299), [GitHub](https://github.com/VideoVerses/AnimationBench) | 2026-04 (jour non vérifié ; extraits) | Benchmark I2V d'animation : 12 principes de l'animation + préservation d'identité | Grille de critères pour l'arène/DPO et l'éval |
| 10 | **DreamActor-M2** (ByteDance) — [arXiv 2601.21716](https://arxiv.org/abs/2601.21716) ; **Wan-Animate-2** (Alibaba Tongyi) — [arXiv 2608.06009](https://arxiv.org/abs/2608.06009) | 2026-01-29 ; 2026-08-07 | Animation de personnage depuis réf en in-context ; paires pseudo cross-identity auto-amorcées ; contrôle de point de vue par texte ; Lite par Self-Forcing ; poids Base annoncés | Idée données **cross-identity** (même mouvement, perso différent) ; enseignant potentiel |
| 11 | **pixel-forge** — [GitHub](https://github.com/cochranblock/pixel-forge) | 2026-03-19 (dépôt) | 3 petits modèles de diffusion EDM 1–17 M, sprites 32×32, palettes ; Rust (archi/données peu documentées) | Baseline de taille comparable (statique seulement) |
| 12 | **Texel Splatting** — [arXiv 2603.14587](https://arxiv.org/abs/2603.14587), [code](https://github.com/dylanebert/texel-splatting) | 2026-03 | Rendu pixel-art 3D stable en perspective (cubemap → quads monde) | **Source synthétique** : rendre des modèles 3D riggés animés en 32×32 sans scintillement, toutes directions |
| 13 | **SpriteToMesh** — [arXiv 2602.21153](https://arxiv.org/abs/2602.21153) | 2026-02 | Segmenteur sprite/masque entraîné sur >100 K paires issues de 172 jeux → maillage Spine2D | Détourage/nettoyage des sheets ; augmentation par déformation squelettique |
| 14 | **texel-studio** — [GitHub](https://github.com/EYamanS/texel-studio) | 2026-04-01 (dépôt) | Agent VLM qui peint pixel par pixel sous palette stricte (statique) | VLM comme **pré-filtre** avant le vote humain |

**Écartés (antérieurs à 2026 ou hors sujet)** : *Sprite Sheet Diffusion* (arXiv 2412.03685,
déc. 2024 — pourtant la référence académique la plus proche), Wan2.2 pixel-animate (déc. 2025),
LoRA Teluv (août 2025), Voxify3D (déc. 2025), PixelDiT (prépub. nov. 2025), *Deep
Sprite-based Image Models* (2604.19480 — « sprite » au sens décomposition d'image, faux ami),
Retro Diffusion (dates non vérifiables).

**Lecture pour Esprite** : l'état de l'art 2026 ne propose pas de modèle « clé en main »
qui ferait mieux sur la même tâche à 50 M params ; il fournit (a) des **enseignants**
lourds pour fabriquer des données, et (b) des **briques** transférables : PE temporelle
circulaire, anti-exposure-bias, CE à crédit partiel, compression inter-frames, DPO robuste.

## 2. Veille techniques transférables (≥ janvier 2026)

_(en cours)_

## 3. Datasets

_(en cours)_

## 4. Proposition

_(à venir)_
