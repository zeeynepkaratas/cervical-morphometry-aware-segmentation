# Circularity Mechanism Analysis — Final Report

## Executive Summary
This report presents the findings of the pre-registered circularity mechanism analysis. The goal was to explain why conformal prediction coverage for the circularity metric degrades significantly under Gaussian noise while the N/C ratio remains comparatively more robust than circularity, and to test if a simple post-processing step could mitigate this.

**Decision Conclusion:** `STRONG_MECHANISM_SUPPORTED`
The analysis provides strong evidence that circularity degradation was more strongly associated with direct perimeter error than with area error, while controlled boundary perturbations affected circularity substantially more than N/C ratio. Although several morphological operations improved mean circularity error, Dice, and/or N/C error on the validation set, none qualified as safe because each increased the invalid prediction rate. Therefore, no candidate was selected and the test set remained unseen.

## H1 & H2: Area–Perimeter Divergence (Phase A)
By calculating perimeter directly from the prediction masks (using OpenCV contours via deterministic re-inference) rather than algebraically deriving it, we decoupled the mathematical artifacts and decomposed the error under Gaussian noise.

- **Correlation:** The increase in circularity absolute error under Gaussian noise correlates more strongly with direct perimeter error changes ($\rho \approx 0.586$) than with area error changes ($\rho \approx 0.408$).
- **Regression:** A cell-aggregated OLS regression ($R^2 = 0.503$) indicates that while perimeter error is a strong positive predictor of circularity degradation (coefficient: `0.382`), the concurrent drop in foreground Dice score is also a major explanatory factor (coefficient: `-0.502`). Area error showed a weak negative association (coefficient: `-0.104`). Perimeter is a substantial mechanism, but not the *sole* cause of metric fragility.

## H3 & H4: Controlled Mask Perturbations (Phase B)
We subjected ground-truth masks to controlled deterministic perturbations, grouping them by the resulting Dice loss to ensure fair comparisons. Cytoplasm masks were strictly constrained to conserve the original predicted cell boundary.

At a matched **mild Dice loss band (0.95–0.98)**:
- **Boundary perturbations** (jitter, indentations, protrusions) caused substantial reductions in circularity (mean signed change $\approx -0.32484$, mean absolute change $\approx 0.32486$).
- **Shape perturbations** (erosion, dilation) caused negligible changes in circularity (mean signed change $\approx 0.01428$, mean absolute change $\approx 0.01824$).
- Conversely, boundary perturbations changed the N/C ratio only modestly (mean signed change $\approx 0.00727$, mean absolute change $\approx 0.08137$), while shape perturbations had a noticeably larger effect on N/C (mean signed change $\approx 0.03726$, mean absolute change $\approx 0.15305$).

This double dissociation supports the hypothesis regarding the discrepancy in conformal coverage: Gaussian noise induces boundary-type errors that disrupt circularity but largely preserve the N/C ratio.

## H5: Post-Processing Mitigation (Phase C)
We tested four candidate morphological operations (opening, closing, opening $\rightarrow$ closing, closing $\rightarrow$ opening) using a fixed $3\times3$ elliptical kernel on the validation set. Predictions were strictly constrained such that post-processed masks never expanded beyond the initial predicted cell boundary (avoiding ground-truth leakage).

**Result: NO_SAFE_POSTPROCESSING**
Although several morphological operations improved mean circularity error, Dice, and/or N/C error on the validation set, none qualified as safe because each increased the invalid prediction rate. Specifically, the closing operation improved average metrics but was rejected entirely due to the increase in invalid predictions. Therefore, no candidate was selected and the test set remained unseen.

## Data Leakage Audit (Phase 0)
An automated audit verified that:
- There is zero overlap between the training, validation, calibration, and test sets.
- All cell IDs are unique within their respective splits.
- The post-processing selection protocol strictly used only the validation set, keeping the test set unseen.
- (All 13 checks passed).

## Conclusion
The fragility of the circularity metric under Gaussian noise is strongly associated with the mathematical property of contour-based perimeter measurements in the presence of high-frequency pixel noise. Simple morphological post-processing is too blunt an instrument to smooth the boundaries without harming other critical metrics like invalid prediction rates. Future work may require model-level interventions or uncertainty-aware metric definitions.
