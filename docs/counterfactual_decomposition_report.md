# Counterfactual & Exact Decomposition Report

This document reports the findings of the pre-registered Exact Log-Ratio Area-Perimeter Decomposition and Area-Preserving Boundary Counterfactual Perturbation analyses. 

## 1. Exact Log-Ratio Decomposition (Analysis A)
**Purpose:** To algebraically decompose the predicted change in log-circularity into additive area and perimeter components.

### 1.1 Methodology
- Valid direct-mask predictions from real model outputs were extracted.
- Formula: $\Delta \log(C) = \log(A_{corr} / A_{clean}) - 2 \log(P_{corr} / P_{clean})$
- Predictions with $Area \le 0$ or $Perimeter \le 0$ were excluded.

### 1.2 Results
- **Mean Absolute Perimeter Contribution:** $0.3706$
- **Mean Absolute Area Contribution:** $0.3209$
- **Perimeter Dominant Fraction:** $59.95\%$
- **Mean Identity Residual:** $-0.0605$

**Interpretation:** In approximately 60% of cases, the perimeter contribution to the circularity error outweighs the area contribution in absolute terms. The magnitudes confirm that while area errors contribute significantly to the circularity deviation, the direct perimeter measurement is the dominant factor causing circularity collapse.

## 2. Area-Preserving Counterfactual (Analysis D)
**Purpose:** To isolate the effect of boundary roughness on circularity by holding nucleus area constant ($\pm 2\%$ tolerance) and comparing against standard shape perturbations.

### 2.1 Methodology
- A strict exact pixel-exchange strategy was used. Deterministic pairs of protrusions and indentations of exactly $k$ pixels were added/removed from the boundary.
- **Eligibility:** 182 distinct cells successfully matched the topological constraints (no holes, single component, strict containment) and achieved a Mild Dice band score ($0.95 \le \text{Dice} \le 0.98$).

### 2.2 Results
Within the matched `0.95–0.98` Dice band:
- **Matched Cells:** 182
- **Shape Perturbation (Erosion/Dilation) Abs Circularity Change:** $0.0158$
- **Counterfactual Boundary Perturbation Abs Circularity Change:** $0.0213$
- **Difference:** $+0.0054$
- **Shape N/C Change:** $0.1403$
- **Counterfactual N/C Change:** $0.0000$ (Area strictly preserved)

### 2.3 Pre-registered Decision: `PARTIAL_ADDITIONAL_SUPPORT`
**Reasoning:**
While the area-preserving boundary perturbations *did* cause a larger absolute circularity change ($0.0213$) than the matched shape perturbations ($0.0158$), the difference magnitude ($+0.0054$) is relatively modest. 

Area-independent boundary effects did not receive strong additional support. The existing direct-perimeter, regression, and original Dice-matched perturbation findings remain valid, but the mechanism may involve combined boundary and area changes. The pre-registered decision is therefore categorized as `PARTIAL_ADDITIONAL_SUPPORT`.

*(Note: No new models were trained. Frozen model behavior and GT-mask counterfactual experiments represent independent lines of evidence. The decomposition is an algebraic identity, not a statistical regression model.)*
