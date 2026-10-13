# Brief pour Claude Code : cascade FM multi-échelle CROSCIM (branche `croscim`)

À lire avec `formulation/CROSCIM_formulation.pdf` (référence des notations) et les figures 1 à 4.
Le dépôt cible est `maxbeauchamp/4dvarnet-starter`, branche `croscim` (dernier commit vu : `902e8da "solve FM fine scale"`).
**Respecter `CLAUDE.md` et `AGENTS.md` du dépôt.** En particulier :
- échanger en français ;
- ne jamais changer en silence les hypothèses scientifiques, l'ordre des canaux, la normalisation, la géométrie des patchs ou les définitions de loss ;
- ne jamais committer de données, de checkpoints ou de figures ;
- faire des changements chirurgicaux.

## 0. Ce qu'on veut

Faire évoluer le solveur **FM Swin** existant (seul solveur gardé) vers :
- une **cascade de 3 générateurs** de flow matching conditionnels, de grossier à fin : C = 32 km, M = 4 km, H = 1 km, sur une grille polaire stéréographique imbriquée (×8, puis ×4) ;
- un **état décomposé exactement**, X = (X_C, δX_M, δX_H) avec R δ = 0. Pas de décodeur appris ;
- des **paramètres sous-maille explicites** S_C, supervisés par des statistiques calculées à la volée sur les observations, plus un petit résidu latent ψ_C ;
- un **contexte observationnel** h_ℓ construit à la volée : encodeurs locaux, puis une pyramide montante Π↑ en cross-attention ;
- **deux réseaux** : 𝒩₀ (initialisation, Stage 1), puis 𝒩₁ (transition, Stage 2, rollout en mode A).

Contrainte forte : **tout est à la volée, sans aucun précalcul hors ligne** (contextes, vérités terrain, statistiques, masques, partition entrée/cible).

## 1. Briques existantes à réutiliser (ne pas réécrire)

| Fichier | Ce qu'on garde |
|---|---|
| `contrib/CROSCIM/solvers/flow_swin_backbone.py` | `SwinUNetBackbone` : un backbone par générateur G_ℓ |
| `contrib/CROSCIM/solvers/flowmatching_swin_solver.py` | `FMSwinWrapper`, `FMSolver` : interpolant, cible u = x₁ − ε, conditionnement |
| `contrib/CROSCIM/solvers/utils_flowmatching.py` | `neglogpdf`, `neglogcdf`, `masked_average`, `LogScaleModel`, `FlowMatchingSampler` (Heun) |
| `contrib/CROSCIM/models/models_flowmatching.py` | `Lit4dVarNet_CROSCIM_FlowMatching`, EMA (`_make_ema`), `StochasticEnsembleTestMixin` |
| `contrib/CROSCIM/dataloaders/data_multires_supervised.py` | lecture des sources et grilles polaires stéréographiques ; à adapter vers des grilles imbriquées alignées |
| `config/xp/CROSCIM/FM_Swin_solvers/*.yaml` | point de départ des configs Hydra |

## 2. Plan par phases

Chaque phase est testable et validée avant la suivante.

**Phase 1 : grille et opérateurs.**
- Grille imbriquée 32 / 4 / 1 km : EPSG:3413-like, lon0 = −45, échelle vraie à 70N, domaine [−3200, 3200] km, arêtes alignées.
- Opérateurs `R` (moyenne pondérée par la fraction d'océan), `P` (injection) et `Π₀ = I − PR`.
- *Tests* : RP = I ; R(δ) ≈ 1e−15 ; recomposition exacte. Les tests unitaires d'abord.

**Phase 2 : dataloader à la volée.**
- Pour chaque batch :
  - partition Y^in / Y^tgt (fauchées, traces ou blocs retirés au hasard) ;
  - grillage 𝒢_ℓ (moyenne pondérée par σ⁻²) ;
  - masques de couverture ;
  - détails cibles δX̂ = X̂ − P R X̂, seulement sur les parents couverts au-delà de γ ;
  - statistiques Ŝ_C = 𝒮_C(X̂_H), avec masque et n_eff.
- Le format des observations peut s'inspirer de WeatherGenerator (tables plates + index temporel), mais le dataloader de WeatherGenerator n'est pas réutilisé.
- *Tests* : R δ̂ = 0 ; masques cohérents ; jamais de cible issue de Y^in seul.

**Phase 3 : G_C seul (premier résultat).**
- FM sur Z_C = (X_C, S_C) ; contexte h_C = encodeurs de C + forçage ERA5, sans pyramide au départ.
- Loss NLL masquée par variable × niveau × pixel ; plancher σ_S pour S.
- *Critères* : score sur Y^tgt ; histogrammes de rang ; contrôlabilité de S.

**Phase 4 : cascade G_M, G_H.**
- Conditionnement par un échantillon EMA du niveau supérieur, sans gradient, avec un petit bruit η.
- Ajouter Π↓[f, Φ_C] (injection + embedding de position de l'enfant).
- Bruit projeté par Π₀ pour les δ.

**Phase 5 : pyramide montante.**
- Encodeurs locaux :
  - tokens + cross-attention par maille pour les observations irrégulières (softmax segmentée) ;
  - CNN masqué à pas pour ASIP.
- Π↑ : cross-attention de K requêtes sur les enfants.
- Pour ASIP à 500 m : encodeur fusionné par tuile C, appliqué uniquement aux tuiles non vides, avec gradient checkpointing.
- Ajouter Ψ_ℓ → ψ_ℓ, ainsi que les losses L_cyc et L_ψ.
- Dropout de groupe sur le contexte (p_∅), pour apprendre à la fois v(·|h) et v(·|∅).

**Phase 6 : 𝒩₁ (transition).**
- Initialisé depuis 𝒩₀. L'état d'entrée est un échantillon de 𝒩₀.
- Disponibilité des observations μ ~ Bernoulli ; pushforward sur 2–4 pas.
- La cible de ψ est donnée par l'encodeur EMA appliqué aux observations de t+Δ.
- Rollout en mode A : G_C¹ seul ; les niveaux fins sont générés à la demande par G_M⁰ et G_H⁰ avec h = ∅.

**Phase 7 : diagnostics.**
- Écart de loss quand on retire S (I↑) ;
- fonctions de réponse avec bruit apparié ;
- comparaison mode A contre mode B sur la zone couverte par le SAR (voir la section 7 du document de formulation).

## 3. Décisions (confirmées le 2026-10-13)

1. **Variables** : 𝒱_C = {SIC, SIT, neige}, 𝒱_M = {δSIC, δSIT}, 𝒱_H = {δSIC}. Pas d'IST. u, v reportés en v2 (pas de source de dérive en v1).
2. **S_C v1** : 9 canaux (histogramme SIC à 4 classes, sic_std, edge_density, anisotropie × 2, sit_std), γ_SIC = 0,5, γ_SIT = 0,1, résolution effective commune 1 km. Définitions complètes : section 8 de la formulation.
3. **Fenêtres** : W = 15 jours pour 𝒩₀ (ASIP : à H sur les 3 derniers jours, à M sur les 15 jours, agrégé à la volée dans les workers) ; Δ = 1 jour pour 𝒩₁, rollout appris jusqu'à 7 jours avec un curriculum de 1 à 7 pas. Tuiles de 64×64 mailles C.
4. **Sources v1** (`/Odyssey/public/CROSCIM_dataset`) : ASIP (SIC), CIMR (SIC, SIT mince), CRISTAL (SIT, neige ; regrillé à 5 km mais traité comme du L2), ERA5 (`atm5km`). Grille : projection native d'ASIP (EPSG:3411). Masque terre/mer : Natural Earth 10 m.
5. **Branche** : `croscim-fm-cascade`, créée depuis `croscim`. Code dans `croscim/`.

**Dataloader** : conversion unique des NetCDF en Zarr (`scripts/convert_cascade_sources.py` : recadrage et changement de format seulement). Plus aucune interpolation pyresample, ni coarsening, ni gridref. Les workers découpent des tranches Zarr, et tout le reste est calculé sur GPU dans `on_after_batch_transfer`.

## 4. État d'implémentation (branche `croscim-fm-cascade`)

Toutes les phases sont codées et testées sur données synthétiques (`pytest tests`). Rien n'a encore tourné sur les vraies données.

| Fichier (`croscim/`) | Contenu |
|---|---|
| `grid.py` | grille 32/4/1 km (EPSG:3411), `R`, `P`, `Π₀`, décomposition / recomposition |
| `obs.py` | 𝒢_ℓ sans interpolation, partition, δ̂, 𝒮_C (dur et doux) |
| `ocean.py` | fraction d'océan ω_H depuis Natural Earth |
| `data.py` | dataset Zarr + `build_batch` GPU ; `horizon` pour les séquences de 𝒩₁ |
| `encoders.py` | tokens par maille, CNN masqués, encodeur ASIP fusionné, Π↑, Π↓, U_ℓ |
| `generators.py` | un générateur FM (Swin existant, LogScaleModel, censure, Heun, Π₀, guidance) |
| `cascade.py` | 𝒩₀ (`LitCascadeN0`) : phases 3 à 5 ; la phase 3 = `model.levels=[C] model.pyramid=false` |
| `transition.py` | 𝒩₁ (`LitCascadeN1`) : phase 6 |
| `rollout.py` | modes A et B, interventions, I↑ / I↓, réponse, contrôlabilité, comparaison A/B |

Configs : `config/xp/N0.yaml`, `N1.yaml`. Conversion des données : `scripts/convert_sources.py`. Formulation versionnée : `docs/formulation.tex`. L'ancien code CROSCIM (solveurs 4DVarNet, UNet, GNN, CM, ancien FM, dataloaders legacy) a été retiré de la branche ; les briques FM réutilisées sont dans `croscim/fm.py` et `croscim/backbone.py`.

Écarts au § 1 : `Lit4dVarNet_CROSCIM_FlowMatching` n'est pas réutilisé tel quel (il dépend de l'ancien format de batch) ; ses briques FM le sont. `_make_ema` est recopié dans `cascade.py`.

## 5. Contenu du zip

- `formulation/` : `CROSCIM_formulation.pdf` (+ `.tex`), la référence des notations.
- `figures/` : figures 1 à 4 en PDF et PNG, ainsi que `old_notation/` (vue d'ensemble et poster, encore en anciennes notations z_ℓ, O).
- `figures_source/` : scripts matplotlib qui génèrent les figures, à partir de champs synthétiques ; il n'est pas nécessaire pour le code du modèle.
