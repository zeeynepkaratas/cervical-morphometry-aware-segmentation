# Cervical Morphometry-Aware Segmentation

This repository contains the publication-track code, locked protocols, and
derived artefacts for a cervical-cell segmentation reliability study.

The project question is whether high segmentation overlap reliably implies
cell-level morphometric fidelity, and whether apparent post-segmentation
correction gains are specific to the correction operation itself.

This is not a clinical validation package. It contains no patient-identifying
data and does not commit raw image datasets.

## Current Status

Publication repository state: `MANUSCRIPT_READY_AFTER_FINAL_TEXT_INTEGRATION`.

The completed study supports a restrained interpretation:

Overlap and morphometric fidelity are related but not interchangeable. A
geometry-matched, GT-blind null reproduced most of the apparent circularity
improvement obtained by fixed morphological closing in both the primary case
study and the locked external SIPaKMeD replication.

## Datasets

| Dataset | Role | Status |
| --- | --- | --- |
| Herlev | Internal / primary development study | Main segmentation, morphometry, closing, decomposition, and null-control analyses |
| SIPaKMeD | Confirmatory / locked external replication | Outcome-blind preflight, final protocol lock, and frozen external replication complete |
| Cx22 | Historically explored / supportive context only | Not treated as untouched confirmatory external validation |

SIPaKMeD external replication used the locked eligibility manifest:

- Total isolated cells inspected: 4049
- Technically eligible isolated cells: 4000
- Parent-source images recovered for clustered inference: 966 in the preflight inventory; 963 represented among the final eligible evaluation set
- No SIPaKMeD training, fine-tuning, preprocessing tuning, or checkpoint selection

Raw Herlev and SIPaKMeD files are intentionally not tracked by Git. Local data
paths are documented by the committed manifests and preflight reports.

## Model

The frozen model family is an RGB U-Net with three semantic classes:

- `0`: background / other
- `1`: cytoplasm-only
- `2`: nucleus

Input preprocessing is RGB, `[0, 1]` normalization, aspect-ratio-preserving
resize, center padding, and `128 x 128` model input. SIPaKMeD predictions are
restored to native image geometry before morphometric measurement.

The locked SIPaKMeD external replication evaluated all three frozen Herlev
baseline seeds and did not choose a best external seed.

## Main Study Components

1. Segmentation-vs-morphometry reliability analysis
2. Fixed `3 x 3` elliptical closing case intervention
3. Area/perimeter decomposition of the closing effect
4. GT-blind dual geometry-matched null control
5. Outcome-blind, prospectively locked SIPaKMeD external replication

## SIPaKMeD External Replication Headline

Final aggregate uses cell-level median across the three locked baseline seeds
and parent-source cluster bootstrap confidence intervals.

| Outcome | Estimate |
| --- | ---: |
| Nucleus Dice | 0.664344 |
| Circularity AE | 0.185073 |
| Dice-circularity AE Spearman rho | -0.505682 |
| Closing benefit | 0.054938 |
| Matched-null benefit | 0.052296 |
| Closing-specific residual | 0.002306 |

Full results are in
[`docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md`](docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md)
and `results/sipakmed_external_replication/`.

## Key Protocol And Result Commits

| Milestone | Commit / tag |
| --- | --- |
| SIPaKMeD technical preflight GO | `ab66b33314d23fda686e236150c354a915f21a7f` |
| SIPaKMeD final external protocol lock | `b0cb7f6d727503861a36b0d754f18c5d999cffe8` |
| SIPaKMeD lock tag | `sipakmed-external-lock-20260813` |
| SIPaKMeD external replication result | `c3fc4e0fc72a92ce50649d286c0814b224b2ca11` |
| Primary closing preregistration | `44b3a67943b0abeb0b94402d2f58dea39b7bd46d` |
| Primary closing result | `7b74f70c67c7a51a257d1eac9c02bd10fc43129e` |
| A decomposition preregistration / result | `ba7f6a5014bfaaab62374dbfb3e294a6e90e0136`, `3bc99a609e1fc9cbd08e929c9487c353fda706c6` |
| B dual-matched null preregistration / result | `f934e4dbdae682cbc22c2697ca268aac20fc5ca9`, `65bce25c4c04beb5f63b911c2c587870a89983f6` |

## Repository Structure

- `configs/`: locked pilot configuration.
- `data/`: committed split metadata and data documentation only.
- `docs/`: protocols, preregistrations, provenance, final reports, and status.
- `experiments/`: orchestration scripts for committed analyses.
- `src/`: loaders, morphometry functions, U-Net model, correction logic, and analysis helpers.
- `tests/`: deterministic unit and integrity tests.
- `results/`: committed derived tables, summaries, and figures required for scientific provenance.

## Reproduction Entry Points

Do not rerun expensive analyses during manuscript cleanup. For intentional
reproduction under the locked protocols, the main entry points are:

| Purpose | Entry point |
| --- | --- |
| Herlev split handling | `experiments/make_pilot_split.py` |
| Fair-repeat model evidence | `experiments/run_fair_repeat.py` |
| Core final analysis | `experiments/run_core_final_analysis.py` |
| Circularity mechanism analysis | `experiments/run_circularity_mechanism.py` |
| Counterfactual checks | `experiments/run_counterfactual_analysis.py` |
| Closing correction | `src/closing_correction_confirmatory.py` |
| Area/perimeter decomposition | `src/closing_area_perimeter_decomposition.py` |
| Dual-matched null | `src/closing_dual_matched_null.py` |
| SIPaKMeD technical preflight | `experiments/run_sipakmed_preflight.py` |
| SIPaKMeD frozen external replication | `experiments/run_sipakmed_external_replication.py` |

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements-lock-windows-py313.txt` records the Windows/Python 3.13
environment used for repository validation. Do not upgrade packages as part of
publication cleanup unless a real reproducibility problem is identified.

## Documentation

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md): canonical current status.
- [`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md): source and dataset role provenance.
- [`docs/SIPAKMED_PREFLIGHT_REPORT.md`](docs/SIPAKMED_PREFLIGHT_REPORT.md): outcome-blind SIPaKMeD technical preflight.
- [`docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md`](docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md): final locked external protocol.
- [`docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md`](docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md): frozen external replication results.
- [`docs/FINAL_PUBLICATION_REPOSITORY_AUDIT.md`](docs/FINAL_PUBLICATION_REPOSITORY_AUDIT.md): final repository audit.

## Guardrails

Do not change frozen splits, eligibility, checkpoints, preprocessing, kernels,
model architecture, bootstrap rules, result files, or protocol locks during
publication cleanup. New analyses should use a separate documented protocol.

## Disclaimer

This repository supports a computational robustness and morphometry reliability
study. It does not provide clinical software, clinical validation, or approval
for clinical use.
