# Circularity Mechanism Analysis — Final Report

## Executive Summary
This report presents the findings of the pre-registered circularity mechanism analysis. The goal was to explain why conformal prediction coverage for the circularity metric degrades significantly under Gaussian noise while the N/C ratio remains robust, and to test if a simple post-processing step could mitigate this.

**Decision Conclusion:** `STRONG_MECHANISM_SUPPORTED`
The analysis provides strong evidence that circularity degradation is driven by high-frequency boundary irregularities (perimeter expansion) which disproportionately affect circularity but leave area (and thus N/C ratio) relatively intact. However, a universal morphological post-processing fix could not be safely applied without violating Dice and N/C safety constraints.

## H1 & H2: Area–Perimeter Divergence (Phase A)
By algebraically deriving perimeter from the existing area and circularity predictions (using the locked formula $C = 4\pi A / P^2$), we decomposed the error under Gaussian noise:

- **Perimeter vs. Area Correlation:** The increase in circularity absolute error under Gaussian noise correlates significantly more strongly with perimeter error changes ($\rho = 0.57$) than with area error changes ($\rho = 0.38$). 
- **N/C robustness:** Conversely, N/C absolute error changes correlate strongly with area error changes ($\rho = 0.60$) but weakly with perimeter error changes ($\rho = 0.18$).
- This confirms that noise heavily corrupts the perimeter (boundary), destroying circularity predictions, while area integrals (and therefore N/C) remain stable.

## H3 & H4: Controlled Mask Perturbations (Phase B)
We subjected ground-truth masks to controlled deterministic perturbations, grouping them by the resulting Dice loss to ensure fair comparisons. 

At a matched **mild Dice loss band (0.95–0.98)**:
- **Boundary perturbations** (jitter, indentations, protrusions) caused massive drops in circularity (mean change $\approx -0.32$).
- **Shape perturbations** (erosion, dilation) caused negligible changes in circularity (mean absolute change $\approx 0.02$).
- Conversely, boundary perturbations changed the N/C ratio only modestly (mean absolute change $\approx 0.07$), while shape perturbations had a larger effect on N/C (mean absolute change $\approx 0.12$).

This double dissociation perfectly explains the discrepancy in conformal coverage: Gaussian noise induces boundary-type errors that destroy circularity but preserve the N/C ratio.

## H5: Post-Processing Mitigation (Phase C)
We tested four candidate morphological operations (opening, closing, opening $\rightarrow$ closing, closing $\rightarrow$ opening) using a fixed $3\times3$ elliptical kernel on the validation set.

**Result: NO_SAFE_POSTPROCESSING**
None of the candidate operations passed all pre-registered safety constraints. While morphological smoothing did reduce circularity error, it consistently caused unacceptable degradation in clean Dice scores ($>0.005$) or worsened the N/C ratio error beyond the allowed 5% threshold. As a result, test set evaluation for post-processing was skipped.

## Data Leakage Audit (Phase 0)
An automated audit verified that:
- There is zero overlap between the training, validation, calibration, and test sets.
- All cell IDs are unique within their respective splits.
- The post-processing selection protocol strictly used only the validation set, keeping the test set unseen.
- (All 13 checks passed).

## Conclusion
The fragility of the circularity metric under Gaussian noise is an inherent mathematical property of contour-based perimeter measurements in the presence of high-frequency pixel noise. While the mechanism is now fully understood, simple morphological post-processing is too blunt an instrument to fix the boundaries without harming other critical metrics. Future work may require model-level interventions or uncertainty-aware metric definitions.
