# Circularity Mechanism Analysis — Final Report

## Executive Summary
This report presents the findings of the pre-registered circularity mechanism analysis. The goal was to explain why conformal prediction coverage for the circularity metric degrades significantly under Gaussian noise while the N/C ratio remains robust, and to test if a simple post-processing step could mitigate this.

**Decision Conclusion:** `STRONG_MECHANISM_SUPPORTED`
The analysis provides strong evidence that circularity degradation is driven by high-frequency boundary irregularities (perimeter expansion) which disproportionately affect circularity but leave area (and thus N/C ratio) relatively intact. However, a universal morphological post-processing fix could not be safely applied without violating Dice and N/C safety constraints.

## H1 & H2: Area–Perimeter Divergence (Phase A)
By calculating perimeter directly from the prediction masks (using OpenCV contours via deterministic re-inference) rather than algebraically deriving it, we decoupled the mathematical artifacts and decomposed the error under Gaussian noise:

- **Perimeter vs. Area Correlation:** The increase in circularity absolute error under Gaussian noise correlates more strongly with direct perimeter error changes ($\rho = 0.586$) than with area error changes ($\rho = 0.408$).
- This supports the hypothesis that noise heavily corrupts the perimeter (boundary), disrupting circularity predictions, while area integrals remain relatively stable.
- (Prior derived-perimeter correlations were artificially inflated due to mathematical coupling; the direct-mask evidence provides independent confirmation).

## H3 & H4: Controlled Mask Perturbations (Phase B)
We subjected ground-truth masks to controlled deterministic perturbations, grouping them by the resulting Dice loss to ensure fair comparisons. Cytoplasm masks were strictly constrained to conserve the original predicted cell boundary.

At a matched **mild Dice loss band (0.95–0.98)**:
- **Boundary perturbations** (jitter, indentations, protrusions) caused massive drops in circularity (mean change $\approx -0.36$).
- **Shape perturbations** (erosion, dilation) caused negligible changes in circularity (mean absolute change $\approx 0.018$).
- Conversely, boundary perturbations changed the N/C ratio only modestly (mean change $\approx -0.003$), while shape perturbations had a slightly larger effect on N/C (mean change $\approx 0.025$).

This double dissociation supports the hypothesis regarding the discrepancy in conformal coverage: Gaussian noise induces boundary-type errors that disrupt circularity but largely preserve the N/C ratio.

## H5: Post-Processing Mitigation (Phase C)
We tested four candidate morphological operations (opening, closing, opening $\rightarrow$ closing, closing $\rightarrow$ opening) using a fixed $3\times3$ elliptical kernel on the validation set. Predictions were strictly constrained such that post-processed masks never expanded beyond the initial predicted cell boundary (avoiding ground-truth leakage).

**Result: NO_SAFE_POSTPROCESSING**
None of the candidate operations passed all pre-registered safety constraints. While morphological smoothing did reduce circularity error in some cases, it consistently caused unacceptable degradation in clean Dice scores (loss $> 0.005$) or worsened the N/C ratio error beyond the allowed 5% threshold. As a result, test set evaluation for post-processing was skipped.

## Data Leakage Audit (Phase 0)
An automated audit verified that:
- There is zero overlap between the training, validation, calibration, and test sets.
- All cell IDs are unique within their respective splits.
- The post-processing selection protocol strictly used only the validation set, keeping the test set unseen.
- (All 13 checks passed).

## Conclusion
The fragility of the circularity metric under Gaussian noise is strongly associated with the mathematical property of contour-based perimeter measurements in the presence of high-frequency pixel noise. Simple morphological post-processing is too blunt an instrument to smooth the boundaries without harming other critical metrics like Dice scores. Future work may require model-level interventions or uncertainty-aware metric definitions.
