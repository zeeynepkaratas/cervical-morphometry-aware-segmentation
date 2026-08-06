# Validated Counterfactual Analysis Report

This document reports the final, validated findings of the pre-registered Exact Log-Ratio Area-Perimeter Decomposition and Area-Preserving Boundary Counterfactual Perturbation analyses. It strictly adheres to cell-level clustered statistics and area conservation rules documented in `docs/counterfactual_protocol_deviations.md`.

## 1. Exact Log-Ratio Decomposition (Analysis A)
**Purpose:** To algebraically decompose the predicted change in unclipped direct log-circularity into additive area and perimeter components by strictly comparing paired predictions (Clean vs. Corrupted) from the exact same cells and seeds.

### 1.1 Methodology
- Excluded predictions with $Area \le 0$ or $Perimeter \le 0$.
- Formula: $\Delta \log(C_{raw\_direct}) = \log(A_{corr} / A_{clean}) - 2 \log(P_{corr} / P_{clean})$
- Residual error validation: $\max|\text{Residual}| < 1 \times 10^{-10}$ required across all paired instances.

### 1.2 Results
- **Valid Clean-Corrupted Pairs Analyzed:** $3,270$ (from $4,372$ input rows)
- **Mean Absolute Perimeter Contribution:** $0.2491$
- **Mean Absolute Area Contribution:** $0.1860$
- **Perimeter Dominant Fraction:** $65.23\%$ (95% CI: $[62.20\%, 68.26\%]$)
- **Maximum Identity Residual:** $8.88 \times 10^{-16}$

**Interpretation:** Perimeter contribution exceeded area contribution in $65.2\%$ of valid clean–corrupted prediction pairs. Both perimeter and area errors strictly quantify the circularity change, but perimeter fluctuations generally carry a larger absolute contribution.

## 2. Area-Preserving Counterfactual (Analysis D)
**Purpose:** To strictly isolate the effect of boundary roughness on circularity by holding nucleus area constant ($\le 2\%$ contour tolerance and $0$ pixel count change) and strictly pairing counterfactual instances to shape perturbation instances within the exact same cell in the Mild Dice band.

### 2.1 Methodology
- **Eligibility:** Strict $0$ pixel-count change combined with absolute OpenCV contour area relative change $\le 2\%$.
- **Strict Matching:** Counterfactual instances were deterministically paired with Shape instances (erosion/dilation) from the exact same cell via minimum absolute Dice difference sorting within the $0.95–0.98$ target Dice band, without replacement. 
- **Statistical Model:** 10,000 iterations of Cell-ID clustered bootstrap over the *cell-averaged* paired differences.

### 2.2 Results
- **Total Valid Matched Pairs:** 203 instances across 182 unique cells
- **Unmatched Counterfactual Instances:** 11
- **Mean Paired Circularity Difference (Counterfactual - Shape):** $-0.0023$
- **Positive Difference Cell Fraction:** $45.6\%$
- **Mean Counterfactual Absolute N/C Change:** $0.0000$ (Pixel count was exactly preserved, and OpenCV contour area remained within $\pm 2\%$)
- **Cell-level Bootstrap 95% CI (Difference):** $[-0.00668, 0.00192]$

### 2.3 Validated Decision: `ADDITIONAL_SUPPORT_NOT_OBSERVED`
**Reasoning:**
The 95% Confidence Interval for the cell-level paired difference in absolute circularity change between counterfactual boundary perturbations and matched shape perturbations is $[-0.00668, 0.00192]$. Because the CI contains zero and the mean difference is negatively signed, boundary perturbations (when area is completely clamped) did not reliably produce greater circularity change than regular shape perturbations in this controlled cohort.

**Conclusion context:** Under strict pixel-count preservation, contour-area tolerance, within-cell Dice matching, and cell-level bootstrap, area-preserving boundary perturbations did not produce greater circularity change than regular shape perturbations. This does not invalidate the observed perimeter association in real model predictions, but it does not support a strictly area-independent boundary mechanism. The mechanism may involve combined boundary and area changes.

*(Note: No new models were trained. Frozen model behavior and GT-mask counterfactual experiments represent independent lines of evidence. The decomposition is a strict algebraic identity, not a statistical regression model explaining variance across populations).*
