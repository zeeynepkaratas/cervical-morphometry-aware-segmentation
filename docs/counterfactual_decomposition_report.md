# Validated Counterfactual Analysis Report

This document reports the final, validated findings of the pre-registered Exact Log-Ratio Area-Perimeter Decomposition and Area-Preserving Boundary Counterfactual Perturbation analyses. It reflects corrections documented in `docs/counterfactual_protocol_deviations.md`.

## 1. Exact Log-Ratio Decomposition (Analysis A)
**Purpose:** To algebraically decompose the predicted change in unclipped direct log-circularity into additive area and perimeter components by strictly comparing paired predictions (Clean vs. Corrupted) from the exact same cells and seeds.

### 1.1 Methodology
- Excluded predictions with $Area \le 0$ or $Perimeter \le 0$.
- Formula: $\Delta \log(C_{raw\_direct}) = \log(A_{corr} / A_{clean}) - 2 \log(P_{corr} / P_{clean})$
- Residual error validation: $\max|\text{Residual}| < 1 \times 10^{-15}$ across all paired instances.

### 1.2 Results
- **Valid Clean-Corrupted Pairs Analyzed:** $3,270$
- **Mean Absolute Perimeter Contribution:** $0.2491$
- **Mean Absolute Area Contribution:** $0.1860$
- **Perimeter Dominant Fraction:** $65.23\%$
- **Maximum Identity Residual:** $8.88 \times 10^{-16}$

**Interpretation:** Perimeter contribution exceeded area contribution in $65.2\%$ of valid clean–corrupted prediction pairs. Both perimeter and area errors strictly quantify the circularity change, but perimeter fluctuations generally carry a larger absolute contribution.

## 2. Area-Preserving Counterfactual (Analysis D)
**Purpose:** To strictly isolate the effect of boundary roughness on circularity by holding nucleus area constant ($\le 2\%$ contour tolerance and $0$ pixel count change) and strictly pairing counterfactual instances to shape perturbation instances within the exact same cell in the Mild Dice band.

### 2.1 Methodology
- **Eligibility:** Strict 0 pixel-count change combined with absolute OpenCV contour area change $\le 2\%$.
- **Strict Matching:** Counterfactual instances were paired with Shape instances (erosion/dilation) from the exact same cell via deterministic minimum Dice difference sorting within the $0.95–0.98$ target Dice band.
- **Statistical Model:** 10,000 iterations of Cell-ID clustered bootstrap over the paired cell differences.

### 2.2 Results
- **Total Valid Matched Pairs:** 49 instances across 48 unique cells
- **Mean Paired Circularity Difference (Counterfactual - Shape):** $-0.0088$
- **Positive Difference Cell Fraction:** $25.0\%$
- **Mean Counterfactual N/C Change:** $0.0000$ (Area strictly preserved)
- **Bootstrap 95% CI (Difference):** $[-0.0124, -0.0054]$

### 2.3 Validated Decision: `ADDITIONAL_SUPPORT_NOT_OBSERVED`
**Reasoning:**
The 95% Confidence Interval for the difference between the absolute circularity change of counterfactual boundary perturbations and matched shape perturbations is $[-0.0124, -0.0054]$. The difference is strictly negative.

Therefore, boundary perturbations (when area is completely clamped) did not influence circularity significantly more than shape perturbations in this controlled cohort. Area-independent boundary effects did not receive strong additional support.

**Conclusion context:** The existing direct-perimeter, regression, and original Dice-matched perturbation findings remain valid, but the mechanism may involve combined boundary and area changes. No existing mechanisms are invalidated, but the claim of strict area-independence is not supported by counterfactual intervention.

*(Note: No new models were trained. Frozen model behavior and GT-mask counterfactual experiments represent independent lines of evidence. The decomposition is a strict algebraic identity, not a statistical regression model explaining variance across populations).*
