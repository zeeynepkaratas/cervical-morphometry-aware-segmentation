# Experiment Status

This document is a navigation aid for the publication repository. It records whether each component is part of the final scientific study, supporting evidence, negative-control context, provenance only, or not part of the final study. It does not reinterpret numerical results.

| Component | Status | Role | Included in final study | Canonical evidence |
| --- | --- | --- | --- | --- |
| Herlev split and data validation | FINAL | Locked data split and leakage control | YES | `data/splits/original_herlev_group_split.json`, `data/splits/pilot_train_validation_split.json`, `results/final_analysis/data_leakage_audit.csv` |
| Baseline segmentation model family | FINAL | Frozen primary model family | YES | `results/pilot/fair_repeat/`, checkpoint SHA256 in README |
| N/C-aware model pilot | SUPPORTING | Tests whether N/C-aware loss changes clean N/C error without harming segmentation | YES, as supporting evidence | `results/pilot/fair_repeat/`, `docs/FAIR_REPEAT_PROTOCOL.md` |
| Core clean/degraded morphometry analysis | FINAL | Clean and deterministic degradation evidence | YES | `results/final_analysis/` |
| Frozen-clean conformal summaries | SUPPORTING | Coverage and target-specific fragility context | YES, as supporting evidence | `results/final_analysis/conformal_results.csv`, `results/final_analysis/target_specific_coverage_report.md` |
| Original H5 post-processing selection | PROVENANCE ONLY | Explains the earlier joint-safety `NO_SAFE_POSTPROCESSING` decision | NO, but preserved for chronology | `docs/circularity_mechanism_hypotheses.md`, `results/circularity_mechanism/postprocessing_selection.json` |
| Circularity mechanism analysis | FINAL | Boundary/perimeter mechanism evidence | YES | `docs/circularity_mechanism_report.md`, `results/circularity_mechanism/` |
| Area-preserving counterfactual mechanism check | SUPPORTING | Tests strict area-clamped mechanism interpretation | YES, as supporting/limiting evidence | `docs/counterfactual_decomposition_report.md`, `results/circularity_mechanism/*counterfactual*` |
| Primary closing correction | FINAL | Frozen non-retraining correction result | YES | `docs/closing_correction_preregistration.md`, `results/closing_correction/summary.json`, `results/closing_correction/verdict.json` |
| A: closing area/perimeter decomposition | FINAL | Decomposes the closing correction mechanism | YES | `docs/closing_area_perimeter_decomposition_preregistration.md`, `results/closing_correction/mechanism_decomposition/` |
| B: prospective dual-matched null | FINAL | Specificity control for closing correction | YES | `docs/closing_dual_matched_null_preregistration.md`, `results/closing_correction/dual_matched_null/` |
| N/C equality audit | PROVENANCE ONLY | Confirms B-analysis N/C equality is expected by design, not a copy bug | NO, but preserved in audit trail | Code path: `src/closing_dual_matched_null.py`, artefacts in `results/closing_correction/dual_matched_null/` |
| Safety-hierarchy provenance audit | PROVENANCE ONLY | Confirms clean-primary hierarchy was frozen before the reported closing result while disclosed as post-hoc relative to H5 | NO, but informs manuscript wording | `docs/closing_correction_preregistration.md`, Git commits `44b3a679` and `7b74f70c` |
| Boundary-aware fine-tune and plain fine-tune control | NEGATIVE CONTROL | Separates boundary-aware fine-tune effects from continued training | NO, but preserved as control context | `results/boundary_pilot/` |
| Noise-morphometry consistency pilot | ARCHIVED | Earlier exploratory branch/output; not part of final active pipeline | NO | Preserved in Git history and ignored local artefacts if present |
| Cx22 feasibility audit | NOT PART OF FINAL STUDY | Outcome-blind technical feasibility only | NO | Audit conclusion: `NOT_SUITABLE_FOR_CLEAN_EXTERNAL_REPLICATION` |
| Cx22 external correction replication | NOT PART OF FINAL STUDY | Not run | NO | No Cx22 correction result is included |

## Final Pipeline Summary

The active final pipeline is Herlev-only and uses the committed frozen splits, canonical morphometry implementation, frozen baseline checkpoint family, primary closing correction, A decomposition, and B dual-matched null validation. Cx22 is not an active correction result in this repository.
