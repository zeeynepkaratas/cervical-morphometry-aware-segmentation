# Experiment Status

This document is the current navigation aid for the publication repository. It
records component status without reinterpreting numerical results or rewriting
historical preregistrations.

Overall state: `MANUSCRIPT_READY_AFTER_FINAL_TEXT_INTEGRATION`

New scientific experiments planned for the current manuscript: `NONE`

## Final Study Components

| Component | Status | Role | Included in final study | Canonical evidence |
| --- | --- | --- | --- | --- |
| Herlev split and data validation | COMPLETE | Internal / primary development data split and leakage control | YES | `data/splits/original_herlev_group_split.json`, `data/splits/pilot_train_validation_split.json`, `results/final_analysis/data_leakage_audit.csv` |
| Baseline segmentation model family | COMPLETE | Frozen primary model family | YES | `results/pilot/fair_repeat/`, checkpoint hashes in `results/sipakmed_external_replication/checkpoint_hashes.json` |
| N/C-aware model pilot | COMPLETE | Tests whether N/C-aware loss changes clean N/C error without harming segmentation | YES, as supporting evidence | `results/pilot/fair_repeat/`, `docs/FAIR_REPEAT_PROTOCOL.md` |
| Core clean/degraded morphometry analysis | COMPLETE | Clean and deterministic degradation evidence | YES | `results/final_analysis/` |
| Frozen-clean conformal summaries | COMPLETE | Coverage and target-specific fragility context | YES, as supporting evidence | `results/final_analysis/conformal_results.csv`, `results/final_analysis/target_specific_coverage_report.md` |
| Circularity mechanism analysis | COMPLETE | Boundary/perimeter mechanism evidence | YES | `docs/circularity_mechanism_report.md`, `results/circularity_mechanism/` |
| Area-preserving counterfactual mechanism check | COMPLETE | Tests strict area-clamped mechanism interpretation | YES, as supporting/limiting evidence | `docs/counterfactual_decomposition_report.md`, `results/circularity_mechanism/*counterfactual*` |
| Fixed closing case study | COMPLETE | Frozen non-retraining `3 x 3` elliptical closing intervention | YES | `docs/closing_correction_preregistration.md`, `results/closing_correction/summary.json`, `results/closing_correction/verdict.json` |
| A: closing area/perimeter decomposition | COMPLETE | Decomposes the closing effect | YES | `docs/closing_area_perimeter_decomposition_preregistration.md`, `results/closing_correction/mechanism_decomposition/` |
| B: prospective dual-matched null | COMPLETE | GT-blind geometry-matched specificity control for the Herlev closing case study | YES | `docs/closing_dual_matched_null_preregistration.md`, `results/closing_correction/dual_matched_null/` |
| SIPaKMeD technical preflight | GO / COMPLETE | Outcome-blind external dataset feasibility, eligibility, contour, and grouping audit | YES | `docs/SIPAKMED_PREFLIGHT_REPORT.md`, `results/sipakmed_preflight/`, commit `ab66b33314d23fda686e236150c354a915f21a7f` |
| SIPaKMeD final protocol lock | COMPLETE | Prospective external replication lock before outcomes | YES | `docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md`, tag `sipakmed-external-lock-20260813` |
| SIPaKMeD external replication | COMPLETE | Confirmatory / locked external replication | YES | `docs/SIPAKMED_EXTERNAL_REPLICATION_REPORT.md`, `results/sipakmed_external_replication/`, commit `c3fc4e0fc72a92ce50649d286c0814b224b2ca11` |

## Provenance And Context Components

| Component | Status | Role | Included in final study | Canonical evidence |
| --- | --- | --- | --- | --- |
| Original H5 post-processing selection | PROVENANCE ONLY | Explains the earlier joint-safety `NO_SAFE_POSTPROCESSING` decision | NO, preserved for chronology | `docs/circularity_mechanism_hypotheses.md`, `results/circularity_mechanism/postprocessing_selection.json` |
| N/C equality audit | PROVENANCE ONLY | Confirms B-analysis N/C equality is expected by design, not a copy bug | NO, preserved in audit trail | `src/closing_dual_matched_null.py`, `results/closing_correction/dual_matched_null/` |
| Safety-hierarchy provenance audit | PROVENANCE ONLY | Confirms clean-primary hierarchy was frozen before the reported closing result while disclosed as post-hoc relative to H5 | NO, informs manuscript wording | `docs/closing_correction_preregistration.md`, Git commits `44b3a679` and `7b74f70c` |
| Boundary-aware fine-tune and plain fine-tune control | NEGATIVE CONTROL | Separates boundary-aware fine-tune effects from continued training | NO, preserved as control context | `results/boundary_pilot/` |
| Noise-morphometry consistency pilot | ARCHIVED | Earlier exploratory branch/output | NO | Preserved in Git history and ignored local artefacts if present |
| Cx22 feasibility audit | HISTORICALLY EXPLORED / SUPPORTIVE | Outcome-blind technical feasibility context | NO | Audit conclusion: `NOT_SUITABLE_FOR_CLEAN_EXTERNAL_REPLICATION` |
| Cx22 external correction replication | NOT PART OF FINAL STUDY | Not run as an untouched confirmatory external validation | NO | No Cx22 correction result is included |

## Final Pipeline Summary

The active final manuscript pipeline uses Herlev as the internal development
study and SIPaKMeD as the prospectively locked external replication. Cx22 is
retained only as historically explored context and must not be described as an
untouched confirmatory external validation.

The fixed closing result is reported together with the GT-blind geometry-matched
null. The null reproduced most of the apparent circularity improvement, so the
publication-facing interpretation is about specificity and morphometric
reliability, not about closing as a uniquely validated correction method.
