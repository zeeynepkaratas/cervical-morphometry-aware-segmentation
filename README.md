# Cervical Morphometry-Aware Segmentation Analysis

This repository contains the publication-track code, locked protocols, and derived statistical artefacts for a controlled cervical-cell segmentation reliability study. The final study focuses on morphometry-aware reliability, clean/perturbed morphometric behavior, and a frozen non-retraining closing correction for nucleus circularity.

The repository is not a clinical validation package. It contains no patient-identifying data and does not include the raw Herlev dataset.

## Scope

Final scientific components:

- Clean Herlev segmentation and morphometry reliability analysis.
- N/C-aware training pilot and fair-repeat evidence.
- Circularity mechanism analysis and counterfactual checks.
- Frozen closing correction confirmatory analysis.
- Area/perimeter decomposition of the closing correction.
- Prospective dual-matched local random null validation.

Not part of the final active study:

- Cx22 external correction replication. A feasibility audit found the current workspace not suitable for clean Cx22 external replication because generated image/label pairs were unavailable; no Cx22 correction outcome is included.
- New training, new inference, new lambda sweeps, or new corruption sweeps beyond the committed artefacts.

## Data

The Herlev dataset is not committed. Provide it locally with `HERLEV_DATA_DIR` or place it in the source-layout fallback described in [data/README.md](data/README.md).

```powershell
$env:HERLEV_DATA_DIR = "C:\path\to\herlev\dataset"
```

The committed split files are:

- [data/splits/original_herlev_group_split.json](data/splits/original_herlev_group_split.json)
- [data/splits/pilot_train_validation_split.json](data/splits/pilot_train_validation_split.json)

## Environment

Install the project dependencies:

```bash
pip install -r requirements.txt
```

`requirements-lock-windows-py313.txt` records the Windows/Python 3.13 environment used for repository validation. Dependency versions are intentionally conservative; do not modernize them as part of result reproduction.

## Repository Structure

- `configs/`: locked pilot configuration.
- `data/`: committed split metadata only; no raw data.
- `docs/`: protocols, preregistrations, provenance notes, and experiment status.
- `experiments/`: executable orchestration scripts for the committed analyses.
- `src/`: reusable loaders, morphometry functions, segmentation model, correction logic, and analysis helpers.
- `tests/`: deterministic unit and integrity tests.
- `results/`: committed derived tables, summaries, and figures required for scientific provenance.

## Final Study Pipeline

| Stage | Script / module | Main input | Main output | Training involved | Scientific role |
| --- | --- | --- | --- | --- | --- |
| Data and split validation | `src/data_prep/load_herlev.py`, `experiments/make_pilot_split.py` | Local Herlev data, committed split metadata | `data/splits/*.json` | No | Reproducible cell-level split handling |
| Frozen baseline and N/C-aware pilot | `experiments/run_fair_repeat.py` | Herlev train/validation split | `results/pilot/fair_repeat/` | Yes, historical/committed only | Model and N/C-aware pilot evidence |
| Core final analysis | `experiments/run_core_final_analysis.py` | Existing fair-repeat predictions | `results/final_analysis/` | No | Clean/degraded morphometry and conformal summaries |
| Circularity mechanism | `experiments/run_circularity_mechanism.py` | Existing predictions and masks | `results/circularity_mechanism/` | No | Boundary/perimeter mechanism evidence |
| Counterfactual checks | `experiments/run_counterfactual_analysis.py` | Mechanism outputs | `results/circularity_mechanism/*counterfactual*` | No | Area-preserving counterfactual evidence |
| Closing correction | `src/closing_correction_confirmatory.py` | Frozen baseline checkpoints and test split | `results/closing_correction/` | No | Primary frozen correction result |
| Area/perimeter decomposition | `src/closing_area_perimeter_decomposition.py` | Frozen closing outputs | `results/closing_correction/mechanism_decomposition/` | No | Mechanistic decomposition of closing effect |
| Dual-matched null | `src/closing_dual_matched_null.py` | Frozen closing outputs | `results/closing_correction/dual_matched_null/` | No | Prospective specificity control |

## Key Frozen Artefacts

- Primary closing preregistration: `44b3a67943b0abeb0b94402d2f58dea39b7bd46d`
- Primary closing implementation: `9728957826c58028c83654446dafa826887d7be8`
- Primary frozen result: `7b74f70c67c7a51a257d1eac9c02bd10fc43129e`
- A decomposition preregistration/result: `ba7f6a5014bfaaab62374dbfb3e294a6e90e0136`, `3bc99a609e1fc9cbd08e929c9487c353fda706c6`
- B dual-matched null preregistration/result: `f934e4dbdae682cbc22c2697ca268aac20fc5ca9`, `65bce25c4c04beb5f63b911c2c587870a89983f6`
- Frozen baseline checkpoint SHA256: `8c52642452c674bc4b929a7b643c9f69f0fc600d2f7b5eafdf733287e98eed51`

## Documentation

- [docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md): canonical status table for final, supporting, negative-control, provenance-only, and not-run components.
- [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md): source-repository transfer notes.
- [docs/closing_correction_preregistration.md](docs/closing_correction_preregistration.md): locked closing correction protocol.
- [docs/closing_area_perimeter_decomposition_preregistration.md](docs/closing_area_perimeter_decomposition_preregistration.md): A decomposition protocol.
- [docs/closing_dual_matched_null_preregistration.md](docs/closing_dual_matched_null_preregistration.md): B dual-matched null protocol.

## Reproduction Guardrails

Do not rerun training, inference, bootstrap, or correction analyses unless intentionally reproducing the committed artefacts under a separate, documented protocol. Do not change frozen splits, checkpoints, kernels, model architecture, or result files during manuscript cleanup.

## Disclaimer

This repository supports a computational robustness and morphometry reliability study. It does not provide clinical validation, clinical software, or approval for clinical use.
