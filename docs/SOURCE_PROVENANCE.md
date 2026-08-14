# Source And Dataset Provenance

This document records source-code transfer provenance and current dataset roles
for the publication repository. It is a current-status document; historical
protocol and preregistration files remain unchanged.

## Repository Lineage

Current repository:

`zeeynepkaratas/cervical-morphometry-aware-segmentation`

Source repository:

`zeeynepkaratas/cervical-morphometry-conformal`

Source commit SHA:

`094723a576fb8c44f60cd5c1efd0a4ef1e4b7e7d`

Source status before transfer:

Clean working tree on `codex/strict-conformal-protocol`.

## Copied Or Adapted Files

- `src/data_prep/load_herlev.py`: copied. Herlev loader and official mask handling retained.
- `src/data_prep/group_split.py`: copied. General cell-level leakage helpers retained.
- `src/segmentation/unet_model.py`: copied. U-Net architecture retained.
- `src/segmentation/train_unet.py`: copied. RGB preprocessing, `128 x 128` padding, target mapping, and baseline segmentation loss retained.
- `src/measurements/morphometry.py`: copied. Official `compute_nc_ratio` and `compute_circularity` retained.
- `src/degradation/apply_degradations.py`: copied. Gaussian blur/noise functions reused for locked internal analyses.
- `src/utils/config.py`: adapted. Retains project constants for the pilot repository.
- `experiments/exp1_dice_correlation.py`: selectively adapted into `src/pilot/metrics.py` for Dice helpers, mask-to-target conversion, and prediction restoration.
- `results/tables/herlev_group_split.json`: copied to `data/splits/original_herlev_group_split.json`.

SIPaKMeD data-only loading was added in this repository after outcome-blind
preflight:

- `src/data_prep/load_sipakmed.py`
- `experiments/run_sipakmed_preflight.py`
- `experiments/run_sipakmed_external_replication.py`

## Dataset Roles

| Dataset | Role | Status |
| --- | --- | --- |
| Herlev | INTERNAL / PRIMARY DEVELOPMENT | Main development, fair-repeat, morphometry, closing, decomposition, and matched-null analyses |
| SIPaKMeD | CONFIRMATORY / LOCKED EXTERNAL REPLICATION | Official isolated-cell data preflighted outcome-blind, protocol locked before outcomes, external replication complete |
| Cx22 | HISTORICALLY EXPLORED / SUPPORTIVE CONTEXT | Not treated as untouched confirmatory external validation |

## Herlev

Herlev raw images and masks are not committed. The final repository includes
split metadata, derived result artefacts, and frozen checkpoint hashes. The
primary frozen checkpoint family is the Herlev baseline U-Net family with seeds
`20260803`, `20260804`, and `20260805`.

## SIPaKMeD

SIPaKMeD raw archives and extracted raw images are local-only and gitignored.
The committed provenance is:

- Eligibility manifest: `results/sipakmed_preflight/sipakmed_eligibility_manifest.csv`
- Manifest SHA256: `e86e99441866fb03d457e3867032ecee4e6792285c6647d93e35089b923de1cd`
- Preflight report: `docs/SIPAKMED_PREFLIGHT_REPORT.md`
- Final protocol lock: `docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md`
- External replication report: `docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md`
- External replication artefacts: `results/sipakmed_external_replication/`

SIPaKMeD contour semantics were established by direct inspection of official
files: the cytoplasm contour is the whole-cell outer boundary including the
nucleus, and the cytoplasm-only mask is constructed as whole-cell mask AND NOT
nucleus mask.

## Cx22

Cx22 was explored historically as a possible external source, but the final
repository does not treat Cx22 as an untouched confirmatory external validation
dataset. No Cx22 correction result is included in the final study.

## Major Protocol And Result Milestones

| Milestone | Commit / tag |
| --- | --- |
| SIPaKMeD technical preflight GO | `ab66b33314d23fda686e236150c354a915f21a7f` |
| SIPaKMeD final protocol lock | `b0cb7f6d727503861a36b0d754f18c5d999cffe8` |
| SIPaKMeD annotated lock tag | `sipakmed-external-lock-20260813` |
| SIPaKMeD external replication result | `c3fc4e0fc72a92ce50649d286c0814b224b2ca11` |
| Primary closing preregistration | `44b3a67943b0abeb0b94402d2f58dea39b7bd46d` |
| Primary closing result | `7b74f70c67c7a51a257d1eac9c02bd10fc43129e` |
| A decomposition preregistration / result | `ba7f6a5014bfaaab62374dbfb3e294a6e90e0136`, `3bc99a609e1fc9cbd08e929c9487c353fda706c6` |
| B dual-matched null preregistration / result | `f934e4dbdae682cbc22c2697ca268aac20fc5ca9`, `65bce25c4c04beb5f63b911c2c587870a89983f6` |

## Frozen Checkpoint Family

| Seed | Checkpoint | SHA256 |
| ---: | --- | --- |
| 20260803 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260803_best_val_dice.pt` | `8c52642452c674bc4b929a7b643c9f69f0fc600d2f7b5eafdf733287e98eed51` |
| 20260804 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260804_best_val_dice.pt` | `d8bf421f71b0ba899a9d70700f6bdb1155110fd91af8617daff97325aed2b6af` |
| 20260805 | `results/pilot/fair_repeat/checkpoints/baseline_seed_20260805_best_val_dice.pt` | `a8e1563e5dc86f67ed5b393a8f1fc78a96165b727cedd766aa8c1131bc5fdc26` |

## Not Committed

The repository intentionally does not commit:

- raw Herlev images/masks,
- raw SIPaKMeD archives or extracted images,
- large raw-data archives,
- local runtime logs,
- resume partial JSONL files,
- local virtual environments.

## License

No source `LICENSE` file was present in the local source checkout. This
repository includes a placeholder `LICENSE` noting that status.
