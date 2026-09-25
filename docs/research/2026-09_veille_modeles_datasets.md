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
4. *Positions* : déjà 3D (embeddings x / y / frame appris, pixels seulement) + PE de séquence
   sinusoïdale ; encodage **absolu additif** — seule piste restante : passer en relatif (RoPE 3D).
   _(corrigé après retour de l'auteur : ce n'est pas un point faible majeur.)_
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

> Dates vérifiées via HF Papers (`published_at`) + préfixe AAMM + ligne « Published » du
> HTML arXiv quand disponible. Seuls résumés/débuts d'articles lus (arXiv bloqué). Les
> « applications Esprite » sont des **extrapolations**, pas des résultats des papiers.
> Aucun travail 2026 spécifiquement pixel-art/sprite trouvé côté HF Papers.

### 2.1 Anti-exposure-bias, cold-start, dérive inter-frames (AR)

| Travail | Date | Idée | Application Esprite | Coût |
|---|---|---|---|---|
| **In-Context Forcing** — [2608.05237](https://huggingface.co/papers/2608.05237) (ShanghaiTech, Tencent Youtu, ZJU) | 2026-08-05 | Contexte trop propre ⇒ raccourci de copie ; bruitage/masquage **progressif** du contexte (plus fort sur les frames proches) | `tokenize.py`/`loss.py` : remplacer une fraction des pixels de t−1 par couleur aléatoire (taux décroissant avec la distance) → casse la copie triviale, robustesse à ses propres erreurs | ≈ 0 |
| **VideoAR** — [2601.05966](https://huggingface.co/papers/2601.05966) (Baidu ERNIE) | 2026-01-09 | Next-scale intra-frame + next-frame causal ; *Random Frame Mask*, *Cross-Frame Error Correction*, *Multi-scale Temporal RoPE* ; curriculum durée/résolution | Random Frame Mask + **curriculum 2→4→16 frames** ; (option lourde) next-scale 8→16→32 intra-frame | faible / élevé |
| **Self Gradient Forcing** — [2607.20368](https://huggingface.co/papers/2607.20368) (JD) | 2026-07-22 | Rollout sans gradient fidèle à l'inférence, puis reconstruction parallèle où le contexte auto-généré reçoit du gradient sur ses K/V | Générer t−1 sans grad, entraîner t sur ce contexte ; séquences courtes ⇒ rollout bon marché ; cible dérive + cold-start | ~2× step |
| Context Forcing — [2602.06028](https://huggingface.co/papers/2602.06028) | 2026-02-05 | Teacher long-contexte → student | Faible (séquences courtes) | — |

(+ **PRA**, §1 #3, même famille.)

### 2.2 Accélération du décodage (frames redondantes)

| Travail | Date | Idée | Application Esprite |
|---|---|---|---|
| **SSD** — Spatially Speculative Decoding — [2606.20543](https://huggingface.co/papers/2606.20543) (Rutgers) | 2026-06-18 | Têtes prédisant aussi le token *sous* le courant ; proposition de lignes entières vérifiées ; jusqu'à 13,3× sur Janus-Pro | `model.py` : têtes auxiliaires x+1 / y+1 **et « même (x,y) à t+1 »** ; `sample.py` : vérification. Taux d'acceptation attendu très élevé sur frames redondantes |
| **MuLo-SD** — [2601.05149](https://huggingface.co/papers/2601.05149) (Qualcomm) | 2026-01-08 | Brouillon basse-rés + ré-échantillonnage *local* au rejet | `sample.py` seul : **frame t−1 comme brouillon gratuit** de t, vérifiée en une passe (rééchantillonnage local = approximation à valider) |
| VC-Attention — 2609.15810 | 2026-09-14 | Attention FP8 training-free, kernels ciblant RTX 5090 | Accélération inférence (gain annoncé sur le noyau seul) |
| ReHyAt — [2601.04342](https://huggingface.co/papers/2601.04342) (Qualcomm) | 2026-01-07 | Softmax local + linéaire global, récurrent | Basse priorité à 18 k tokens / 50 M |

### 2.3 Représentation : positions et redondance inter-frames

| Travail | Date | Idée | Application Esprite |
|---|---|---|---|
| **LeRoPE** — [2607.10134](https://huggingface.co/papers/2607.10134) (UCSD) | 2026-07-11 | Fréquences RoPE apprenables ; testé from scratch **dès 52 M** | `model.py` : les positions sont **déjà 3D** (x/y/frame appris) ; option mineure = passage en **relatif** (RoPE 3D à fréquences apprenables, t circulaire à la Loopy). Priorité basse |
| Partial RoPE — [2603.11611](https://huggingface.co/papers/2603.11611) | 2026-03-12 | RoPE sur une fraction des dims suffit | Réserver une partie de chaque tête à la position |
| **DeltaTok** — [2604.04913](https://huggingface.co/papers/2604.04913) (Amazon, TU/e, JHU) | 2026-04-06 | Une frame = un token « delta » (continu) ; entraînement multi-hypothèses | Transposition discrète (extrapolation) : pour t ≥ 1, vocab **KEEP + 16 couleurs** → la CE ne réapprend plus les pixels statiques |
| Echo-Infinity 2606.04527 ; TIE 2605.10543 | 2026-06 / 2026-05 | Ancrage RoPE de frames « sink » ; RoPE d'intervalles | Réf à position temporelle fixe ; encodage de la durée [0, N] |

### 2.4 Contrôle de longueur / comptage des frames

| Travail | Date | Idée | Application Esprite |
|---|---|---|---|
| **SmartCrop** — [2603.06123](https://huggingface.co/papers/2603.06123) | 2026-03-06 | La longueur de sortie est lisible dans la représentation du prompt (probe) | Conséquence : **ne plus faire « compter » le modèle** — N est déjà dans le préfixe ; rendre la structure déterministe (voir §4) |
| **ρ-EOS** — [2601.22527](https://huggingface.co/papers/2601.22527) (ICML) | 2026-01-30 | Densité implicite d'EOS pour étendre/contracter, sans entraînement | `sample.py` : monitorer P(SEQ_END)/P(FRAME_SEP) aux frontières de frame |
| DreamOn — [2602.01326](https://huggingface.co/papers/2602.01326) | 2026-02-01 (papier ; blog probablement 2025) | États [expand]/[delete] dans la diffusion | Voie MDM : ajuster le nombre de frames pendant le débruitage |

### 2.5 Alignement de préférence

| Travail | Date | Idée | Application Esprite |
|---|---|---|---|
| **RealAlign** — [2605.19839](https://huggingface.co/papers/2605.19839) | 2026-05-19 | Données réelles = gagnantes vs échantillons générés/perturbés, sans annotation | `dpo.py` : **paires synthétiques** (gagnant = cycle réel ; perdant = frame dupliquée/supprimée, boucle cassée, bruit palette, pixels orphelins, décalage 1 px, ou sortie modèle) mélangées aux votes |
| **AR-CoPO** — [2603.17461](https://huggingface.co/papers/2603.17461) | 2026-03-18 | Rollouts à préfixe partagé divergent sur un chunk → paires localisées | `dpo_campaign.py` : best-of-2 **partageant les frames 0..k** ; logπ sur les seuls tokens divergents → variance ↓ |
| **VAR RL Done Right** — [2601.02256](https://huggingface.co/papers/2601.02256) (Tsinghua, ByteDance) | 2026-01-05 | GRPO pour VAR, propagation de masque spatio-temporelle | DPO restreint/pondéré sur la **zone de différence** gagnant/perdant |
| UDM-GRPO — [2604.18518](https://huggingface.co/papers/2604.18518) ; V-GRPO — [2604.23380](https://huggingface.co/papers/2604.23380) | 2026-04-20 ; 2026-04-25 | GRPO stable pour diffusion discrète uniforme ; GRPO via ELBO | Uniquement si bascule MDM ; récompenses vérifiables (compte, boucle, palette) |

### 2.6 Voie « diffusion discrète masquée » (MDM) et régime de données

| Travail | Date | Idée | Application Esprite |
|---|---|---|---|
| **MDM-Prime-v2** — [2603.16077](https://huggingface.co/papers/2603.16077) | 2026-03-17 | Masquage partiel au niveau de sous-tokens binaires ; 21,8× plus efficace en calcul qu'AR (texte) | 16 couleurs = **4 bits** → sous-tokens masquables (opacité connue avant teinte) |
| CWDFM — [2607.21427](https://huggingface.co/papers/2607.21427) (Meta FAIR) | 2026-07-23 | CE pondérée par la densité de contexte révélé | Pondération par voisins révélés (4-voisinage + t±1) |
| ProSeCo — [2602.11590](https://huggingface.co/papers/2602.11590) ; Info-Gain Sampler — [2602.18176](https://huggingface.co/papers/2602.18176) ; LoMDM — [2602.02112](https://huggingface.co/papers/2602.02112) ; Tri-Modal MDM design space — [2602.21472](https://huggingface.co/papers/2602.21472) (Apple) | 2026-02/03 | Auto-correction ; ordre de révélation par gain d'info ; ordre appris ; réglages par défaut MDM | Briques si bascule MDM |
| **Abra** — [2608.17286](https://huggingface.co/papers/2608.17286) | 2026-08-18 | Lois d'échelle T2I : optimum ≈ 200 tokens image/param | Esprite ≈ **< 1 token/param** (≈ 5 k cycles, majoritairement 2 frames, pour 50 M) → régime très sous-alimenté en données : plus de données > plus de params ; tester 15–25 M |

### 2.7 Discussion : plus de pixels par passe *vs* décodage spéculatif

Question de l'auteur : plutôt que spéculer, générer plusieurs pixels par passe ?

| Option | Principe | Exact ? | Effet entraînement | Gain inférence | Risque |
|---|---|---|---|---|---|
| A. Têtes parallèles naïves (k pixels indépendants par passe) | k têtes, échantillonnage indépendant | **Non** — suppose les k pixels indépendants sachant le passé | nul | ×k | Incohérences locales (pixels orphelins, contours cassés) : fatal en pixel-art. Acceptable **seulement comme drafter** (type Medusa/SSD) |
| B. Spéculatif (brouillon = frame t−1 / ref / têtes SSD) | le modèle vérifie un bloc en une passe | **Oui** (sans perte) | nul (ou têtes légères) | ∝ longueur moyenne des séries acceptées ; ~1 passe par pixel *changé* | Gain faible sur frames très animées ; ne réduit pas le coût d'entraînement |
| C. **Hiérarchique global/local** (patch 4×4 : gros transformer par patch, petit décodeur AR intra-patch — lignée MegaByte / RQ-Transformer) | 1 passe du gros modèle = 16 pixels, décodés exactement par un petit modèle | **Oui** | séquence du gros modèle ÷16 (18 k → ~1,1 k) ⇒ entraînement bien moins cher, batchs plus gros | ~×10+ sur le gros modèle | Refonte `model.py`/`sample.py`/`graph_sampler.py` ; DPO inchangé (vraisemblance exacte) |
| D. + **flag KEEP par patch** pour t ≥ 1 | un patch inchangé vs t−1 = 1 token, pas de décodage local | Oui | apprend explicitement « ce qui bouge » ; rééquilibre la CE | énorme sur idle/2-frames (majorité des patchs inchangés) | Choix de la taille de patch |
| E. Diffusion discrète masquée (MaskGIT/MDM) | ~8–16 passes par frame, tous pixels en parallèle | Approx. (ELBO) | réputé meilleur en régime data-constrained | fort | Changement de paradigme ; DPO → variante ELBO |

**Avis** : A est à éviter comme sampler final. B est gratuit et immédiat (inférence seule). Le
vrai levier « plusieurs pixels par passe » est **C + D** : exact, divise aussi le coût
d'**entraînement** (précieux pour multiplier les epochs sur un petit dataset), et le flag KEEP
encode directement la redondance inter-frames. B reste compatible par-dessus C.

**Écartés** (antérieurs à 2026 ou dates incohérentes) : Diffusion beats AR in data-constrained
settings (2507.15857), Diffusion LMs are Super Data Learners (2511.03276), MaskGRPO, URSA,
Self Forcing (2506.08009), RandAR/ARPG, Pref-GRPO, LLaDA 1.5, 2512.14549, 2606.29066.

## 3. Datasets

> Vérifié par clones `--filter=blob:none` (arbres, licences, `tracker.json`, échantillons PNG/XML
> analysés PIL) et README HF. Bloqués par le proxy (non vérifiés en direct) : itch.io,
> opengameart, kaggle, spriters-resource, wiki PMDO, Showdown. Recomptes indépendants
> faits sur PMD (3339 `AnimData.xml`, 3318 `Walk-Anim.png`, 1026 entrées tracker, licence
> CC BY-NC 4.0) et sur la licence LPC Revised (CC-BY 3.0 / OGA-BY 3.0).

### 3.1 PMD SpriteCollab — ★ 5/5 — [github.com/PMDCollab/SpriteCollab](https://github.com/PMDCollab/SpriteCollab)

- **3339 jeux de sprites** (981 espèces de base, 748 formes, 1436 shiny, 96 femelles, 78 femelles
  shiny) ; dépôt actif (dernier commit 2026-09-25).
- Par jeu : `<Anim>-Anim.png` (+ Offsets/Shadow), `AnimData.xml` (FrameWidth/Height, `Durations`
  par frame, Rush/Hit/ReturnFrame, alias `CopyOf` à dédupliquer), `credits.txt`.
- **8 directions** par feuille (0 bas, 2 droite, 4 haut, 6 gauche → les 4 d'Esprite).
- Animations les plus couvertes (sur 3339) : Rotate, Idle, Walk, Swing, Double, Sleep, Attack,
  Hop, Charge, Hurt (~3280–3340), Shoot (2801) ; puis ~700–870 : Cringe, Eat, Pose, Nod, Faint,
  Float, Tumble, Sit, LeapForth… ; rares : Strike, RearUp, SpAttack, Hover, Withdraw…
- Longueurs médianes : Walk 4, Idle 4, Attack 11, Hurt 2, Charge 10, Shoot 12, Swing 9,
  Double 16, Hop 10, Rotate 9, Faint 4, Tumble 8 — **exactement les cycles longs qui manquent**.
- Taille : bbox max par frame médiane 26 px, **85 % tiennent en 32×32 à 1:1**, 100 % à 2:1
  (attention : l'union sur un cycle Attack atteint ~48 px à cause du déplacement).
- Couleurs : **100 % ≤ 15 couleurs + transparent** → compatible palette Esprite sans quantif.
- **Cohérent avec le conditionnement existant** : n° de dex dans le chemin → types, stade,
  famille (veekun), shiny, forme, genre.
- Volume estimé : ~3339 × ~11 anims × 4 dirs ≈ **~150 k cycles de 2–16 frames** (vs ~5 k aujourd'hui).
- Licence : **CC BY-NC 4.0** ; **666 jeux crédités « CHUNSOFT »** = extraits du jeu officiel
  (non couvrables par la licence) ; IP Pokémon (même risque que veekun). Recherche non
  commerciale défendable ; publication de checkpoint = mention NC + crédits ; isoler Chunsoft.

### 3.2 LPC — ★ 4,5/5

- **Universal LPC Spritesheet Character Generator** — [GitHub](https://github.com/liberatedpixelcup/Universal-LPC-Spritesheet-Character-Generator) :
  88 130 calques, 6 types de corps, combinaisons ~illimitées ; 64×64, 4 directions ;
  spellcast 7, thrust 8, walk 8(+1), slash 6, shoot 13, hurt 6 (1 dir), climb 6 (1 dir),
  idle 2, jump 5, sit 3, emote 3, run 8, combat_idle 2, backslash 13, halfslash 6.
  Perso ~30×51 → **2:1 ≈ 15×26 dans 32×32**. Composite ~29 couleurs → **quantif ≤ 15 requise**.
  Licences par calque (OGA-BY, GPL, CC-BY-SA, CC-BY, CC0) → **filtrer sur CC0/CC-BY/OGA-BY**
  pour publier. Effort moyen (compositeur à écrire via `sheet_definitions/*.json`), mais
  générateur infini avec **identité cohérente entre actions** et référence gratuite.
- **ElizaWy « LPC Revised »** — [GitHub](https://github.com/ElizaWy/LPC) : **CC-BY 3.0 / OGA-BY 3.0
  uniquement** (licence la plus sûre) ; Walk, Run, Idle, Jump, Climb, Sit, Emotes, Combat 1h
  (idle/slash/backslash/halfslash) ; pas de hurt/shoot/spellcast.
- Dérivés prêts : `carlosuperb/lpc-action-pixel-art-diffusion` (HF, CC-BY-SA-3.0, 2,7 Go,
  walk/thrust/slash × 4 dirs, labels implicites) ; YingzhenLi/Sprites (DSVAE 2018 : 1296 persos
  × 9 action-directions × 8 frames ; HF `TalBarami/msd_sprites`, licence déclarée douteuse).

### 3.3 Autres

| Source | Constat | Licence | Score |
|---|---|---|---|
| **Ninja Adventure** (Pixel-boy) — [superpowers-asset-packs](https://github.com/sparklinlabs/superpowers-asset-packs) | ≥ 25 persos + 22 monstres, 16×16, 4 dirs, walk 4 / attaque / saut ; 8–15 couleurs ; 1:1 dans 32 | **CC0** | 3,5 |
| **PokeAPI/sprites** — [GitHub](https://github.com/PokeAPI/sprites) | GIF BW animés 1173 (+shiny/dos/femelles), 40–120 px, 30–90 frames (idle combat → dédup + sous-échantillonnage) ; Showdown mélange pixel et rendus 3D (filtrer sur nb couleurs) | © TPC (le CC0 ne couvre pas les images) | 3 |
| 0x72 DungeonTileset II | idle/run 4, vue droite (non vérifié, itch bloqué) | CC0 | 2,5 |
| spraix_1024 (HF) | 560 anims avec labels texte riches, mais redimensionnées 1024 (grille perdue), sources hétérogènes | GPL déclarée, douteuse | 2 |
| Spriters Resource (autres jeux) | énorme, étiquetage manuel | tous droits réservés | 2 (recherche) |
| À exclure | Limbicnation/pixel-art-character (synthétique IA, NSFW partiel), pseudo-pixel statiques (diffusiondb-pixelart, nerijs) | — | — |

### 3.4 Mapping vers les 14 actions Esprite

| Action | PMD SpriteCollab | LPC |
|---|---|---|
| idle | Idle (Sleep en variante) | idle, combat_idle |
| walk | Walk | walk |
| run | — | run |
| jump | Hop, Jump, LeapForth | jump |
| climb | — | climb (up seulement) |
| swim | — (Float/Hover/Sink : faible) | — |
| attack | Attack, Strike, QuickStrike, Swing, Double, MultiStrike, Slam, Bite, Punch, Kick… | slash, thrust, backslash, halfslash |
| shoot | Shoot (hors alias de Charge), SpAttack, Emit | shoot |
| cast | Charge, SpAttack, RearUp, Rotate | spellcast |
| guard | Withdraw (158) | — |
| dodge | Tumble/TumbleBack (approx.) | — |
| hurt | Hurt, Pain, Cringe, Injured | hurt (1 dir) |
| defeat | Faint, Laying | hurt (chute) |
| victory | Pose, Appeal, Dance, Twirl, Nod | emote |

**Complémentarité** : PMD couvre les créatures (attack/shoot/cast/hurt/defeat/victory, cycles
longs, 4 dirs), LPC couvre les humanoïdes (run/climb/cast/shoot). **Restent orphelins : swim,
guard, dodge** (sources faibles) → candidats à la génération synthétique (enseignants §1) ou à
une fusion d'actions dans le vocab.

## 4. Proposition

Principe directeur : Esprite est **limité par les données** (≈ 5 k cycles, < 1 token/param ;
cf. Abra §2.6), pas par l'architecture. L'ordre des chantiers suit ce constat : **données
d'abord**, puis corrections d'entraînement bon marché, puis refonte d'architecture une fois
le volume de données connu, et DPO en dernier.

### Phase 1 — Données (priorité absolue)

1. **Ingestion PMD SpriteCollab** (`src/dataset/sources.py` → `parse_pmd_spritecollab`) :
   - lire `AnimData.xml`, résoudre/dédupliquer les `CopyOf`, garder les lignes 0/2/4/6
     (down/right/up/left) ; ignorer Offsets/Shadow ;
   - découpe par FrameWidth/Height, **ancrage par cycle** (bbox d'union, calage bas, comme
     `tsr_ingest.anchor_cycle`) ; 1:1 si l'union tient en 32, sinon 2:1 mode-par-bloc, sinon rejet ;
   - > 16 frames → sous-échantillonnage uniforme (règle existante) ; palettes déjà ≤ 15 couleurs ;
   - mapping d'actions §3.4 ; nom PMD d'origine conservé en méta (futur descripteur fin) ;
   - descripteurs gratuits via le n° de dex (types, stade, famille, shiny, forme, genre) ;
   - **exclure (ou isoler dans une source à part) les 666 jeux « CHUNSOFT »** ; garder
     `credits.txt` pour l'attribution.
   - Attendu : **~150 k cycles**, dont une majorité > 2 frames → le bucket « long » cesse
     d'être anecdotique.
2. **Splits par identité** (`src/dataset/splits.py`) : aujourd'hui le split est par cycle,
   stratifié (action, source) ; avec PMD, shiny/directions/actions d'une même espèce
   fuiraient entre train et val. Split par `identity_key` (ou par famille) pour la val.
3. **LPC** pour les humanoïdes : d'abord **ElizaWy LPC Revised** (CC-BY/OGA-BY, licence la
   plus sûre), puis compositeur ULPC filtré sur calques CC0/CC-BY/OGA-BY → run, climb, cast,
   shoot, jump + identité cohérente entre actions ; 2:1 + quantification ≤ 15 couleurs.
4. Appoint CC0 : **Ninja Adventure** (16×16, 1:1).
5. Trous restants (**swim, guard, dodge**) : décider entre (a) fusion/abandon dans le vocab,
   (b) synthèse via enseignants (LoRA svntax / Wan-Animate-2 → k-centroid → quantif → tri
   humain dans l'arène existante).

Après la phase 1, le rapport tokens/param passe d'≈ 1 à un ordre de grandeur de ~20
(estimation grossière) : **le 50 M redevient raisonnable** ; ne pas réduire le modèle avant
d'avoir les données.

### Phase 2 — Correctifs d'entraînement bon marché (modèle actuel)

6. **Bruitage du contexte** (In-Context Forcing / VideoAR Random Frame Mask) dans
   `data.py`/`tokenize.py` : remplacer une fraction des pixels des frames précédentes (et
   parfois de la ref) par une couleur aléatoire, taux plus fort sur t−1 ; la loss reste sur
   les cibles propres. Vise la copie triviale, la dérive et le cold-start.
7. **Curriculum de longueur** 2 → 4 → 16 frames (le sampler de buckets s'y prête).
8. **Décodage spéculatif** dans `sample.py`/`graph_sampler.py` : brouillon = frame t−1 (ou
   la ref pour la frame 0), vérification par blocs — sans perte, sans ré-entraînement.
9. **Structure imposée à l'inférence** : N est dans le préfixe, donc forcer `FRAME_SEP`
   après chaque 1024 pixels et `SEQ_END` après N frames (masque de logits) — supprime le
   runaway sans toucher l'entraînement ; `w_sep`/`w_end` peuvent ensuite redescendre.

### Phase 3 — Architecture v3 (une fois les données en place)

10. **Hiérarchique global/local** (§2.7 option C) : gros transformer par patch 4×4
    (séquence ÷16 : ~18 k → ~1,1 k), petit décodeur AR intra-patch (2–4 couches) —
    vraisemblance exacte conservée (DPO inchangé), entraînement bien moins cher.
11. **Flag KEEP par patch** pour t ≥ 1 (option D) : un patch inchangé = 1 token → la
    redondance inter-frames devient explicite, gros gain sur idle/2-frames.
12. Le spéculatif (8) reste compatible par-dessus. RoPE 3D relative : optionnel, basse priorité.
13. Alternative à garder en réserve : diffusion discrète masquée (MDM, 4 bits/pixel
    MDM-Prime-v2) si l'AR plafonne — changement de paradigme, DPO à adapter (ELBO).

### Phase 4 — DPO

14. **Paires synthétiques** type RealAlign dans `dpo.py` : gagnant = cycle réel, perdant =
    même cycle dégradé (frame dupliquée/supprimée, boucle cassée, pixels orphelins, décalage
    1 px, bruit de palette) — mélangées aux votes, jamais à leur place.
15. **Paires à préfixe partagé** (AR-CoPO) dans `dpo_campaign.py` : les 2 seeds partagent les
    frames 0..k ; logπ sur les tokens divergents seulement → moins de variance par vote.
16. **Poly-DPO** (ViPO) comme remplacement de `dpo_loss` à tester en A/B (votes bruités).

### Points de vigilance

- **Licences** : PMD = CC BY-NC 4.0 + IP Pokémon + 666 jeux Chunsoft ; LPC = mix par calque
  (SA/GPL à filtrer pour publier). Pour publier des checkpoints sur HF, prévoir une lignée
  « publiable » (LPC Revised + CC0 + ULPC filtré) distincte de la lignée recherche.
- Les dates des travaux 2026 ont été vérifiées via HF Papers (arXiv bloqué) : seuls les
  résumés/débuts d'articles ont été lus ; les transpositions à Esprite sont des hypothèses
  à valider par ablation.
