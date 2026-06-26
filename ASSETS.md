# Fichiers lourds non versionnés (gitignore) — 4dvarnet-starter

Inventaire des fichiers volumineux exclus de git mais nécessaires pour faire
tourner les expériences. Établi le 2026-06-25 à partir de ce qui est réellement
présent sur disque (le `.gitignore` liste aussi beaucoup de fichiers obsolètes /
sorties de tests qui n'existent plus).

## Masques

| Fichier | Taille | Usage |
|---|---|---|
| `contrib/CROSCIM/mask_PanArctic.nc` | 5.1 G | Masque grille pan-arctique — CROSCIM (SIT) |
| `contrib/DMI/ASIP_OSISAF/mask_PanArctic.nc` | — | Référencé par les notebooks ASIP_OSISAF mais **ABSENT** du disque (à régénérer/récupérer) |

> `contrib/CROSCIM/scripts/weights_10.nc` / `weights_50.nc` sont gitignorés mais absents.

## Checkpoints — `ckpt/`
### CROSCIM (épaisseur de glace, SIT)
| Fichier | Taille |
|---|---|
| `ckpt/CROSCIM/base_croscim_UNet_unrolling_sit_supervised_forecast.ckpt` | 59 M |
| `ckpt/CROSCIM/base_croscim_UNet_unrolling_sit_supervised_forecast_res10.ckpt` | 36 M |
| `ckpt/CROSCIM/base_croscim_UNet_sit_UOAI_supervised_forecast.ckpt` | 36 M |
| `ckpt/CROSCIM/base_croscim_UNet_sit_UOAI_supervised_forecast_res10.ckpt` | 23 M |

> De nombreux autres `ckpt/CROSCIM/base_croscim_*.ckpt` (variantes 4DVarNet/UNet
> sic, res50…) et `Notebooks/**/*.ckpt` sont listés dans `.gitignore` mais ne
> sont plus présents sur disque.

## Datasets externes utilisés par les notebooks

- `/dmidata/users/maxb/ASIP_OSISAF_dataset/` (L3 + `PREPROC/asip_database*.nc`) — notebooks ASIP_OSISAF
- `/dmidata/users/maxb/CROSCIM_dataset/` — notebooks CROSCIM

> Les 3 jeux de données des notebooks **consistency** sont documentés dans le
> repo `4dvarnet-starter-devs` (voir son `ASSETS.md`), où vivent ces notebooks.
