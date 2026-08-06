# Pre-registration Plan: Counterfactual and Exact Decomposition Analyses

This document defines the pre-registered protocol for two linked analyses:
1. Exact log-ratio area–perimeter decomposition
2. Area-preserving boundary counterfactual perturbation

These analyses supplement the prior circularity mechanism evidence without altering the fundamental deep learning pipeline, external datasets, or initiating new training loops.

## 1. Data Sources
- The primary data source is the existing direct-mask measurements previously computed in `results/circularity_mechanism/real_prediction_area_perimeter.csv` tracking predictions across different noise severities and seeds.
- For counterfactual analysis, deterministic ground truth masks (Herlev dataset) are used. 
- Frozen checkpoints are only used to infer any missing required metrics (if applicable, though current cached direct mask predictions are preferred).

## 2. Exact Log-Ratio Area-Perimeter Decomposition
This is an exact algebraic identity accounting, not a statistical regression model explaining error variability across populations.

**Formula:**
$$ \Delta \log(C) = \log(C_{corrupted} / C_{clean}) $$
$$ \text{Area Contribution} = \log(A_{corrupted} / A_{clean}) $$
$$ \text{Perimeter Contribution} = -2 \times \log(P_{corrupted} / P_{clean}) $$
**Identity Validation:** $\Delta \log(C) = \text{Area Contribution} + \text{Perimeter Contribution}$

### Invalid/Degenerate Filtering
Predictions are explicitly excluded from the log-decomposition if they fall into the following conditions:
- Predicted nucleus area is $\le 0$.
- Predicted direct perimeter is $\le 0$.
- Non-finite values in prediction metrics.
- Missing or entirely empty predicted nucleus mask.
- Pre-defined project invalid criteria.
Exclusion events are strictly logged per condition, severity, and seed. Invalid instances are never silently dropped.

### Outcomes
- Primary metrics per cell: `delta_log_circularity`, `area_contribution`, `perimeter_contribution`, absolute contributions, dominant component, and identity residual.
- Summary outcomes: Difference between absolute perimeter and area contributions, fraction of cells where perimeter dominates, consistency across severity/seeds, signed relationships, and confirmation of identical bounds via residual checks ($\approx 0$).

## 3. Area-Preserving Boundary Counterfactual Perturbation
**Goal:** Determine how much circularity changes when boundary complexity (direct perimeter) is altered while nucleus area is intentionally held constant.

### 3.1. Exact Pixel-Exchange Strategy (No Iterative Tuning)
The algorithm relies on an exact pixel-exchange heuristic using pre-defined perturbation budgets, ensuring equal addition and subtraction of pixels without iterative searching or scaling.

- **Perturbation Budgets:** Let $k$ be the target pixel exchange budget. Three pre-defined budget levels will be tested: `Low`, `Medium`, and `High` (defined statically based on average nucleus sizes prior to execution).
- **Exchange:** $k$ pixels are removed from the boundary (indentation) and exactly $k$ valid pixels are added to another section of the boundary (protrusion).
- **Area Tolerance:** The resulting relative area difference $\frac{|A_{pert} - A_{orig}|}{A_{orig}} \le 0.02$ serves *only* as a strict ceiling for eligibility due to rasterization artifacts. It is *not* an optimization target. 

### 3.2. Candidate Generation and Selection
**Selection bias is strictly prohibited.** Selection is entirely blind to resulting circularity, perimeter, or N/C metrics.

- **Candidate Order:** A deterministic seeded sequence of phase pairs (protrusion angle and indentation angle) is generated based on the `cell_id` hash. A minimum positional separation is enforced between the two points.
- **Maximum Attempts:** A strict maximum of $N_{max} = 20$ attempts will be made per cell per budget. 
- **First Valid Rule:** The *first* candidate in the deterministic sequence that satisfies all topological constraints is selected.
- **Exclusion:** If 20 attempts fail, the cell is logged as `no_valid_counterfactual` and excluded. No further parameters will be searched.

### 3.3. Topological Constraints (Eligibility)
A candidate is only valid if it satisfies all of the following:
- Area tolerance $\le 2\%$.
- Perturbed nucleus is strictly within `original_cell_mask` (nucleus + cytoplasm).
- Nucleus is not empty.
- Nucleus remains a single connected component.
- No new topological holes are created.
- Cytoplasm is exactly `original_cell_mask & ~perturbed_nucleus` (strict disjointness).

### 3.4. Evaluation and Matching
- **Comparison Group:** The counterfactuals will be compared against the existing shape perturbations (erosion/dilation).
- **Primary Dice Band:** Only perturbations resulting in a Mild Dice score ($0.95 \le \text{Dice} \le 0.98$) against the unperturbed GT nucleus will be analyzed.
- **Matching:** Counterfactuals will be deterministically matched to shape perturbations based on Dice, initial nucleus area, and initial circularity where statistical power allows. Matching drop-out rates will be explicitly reported.
- **Statistical Unit:** Cell-ID clustered bootstrapping will be used to ensure valid standard errors for multiple perturbations derived from the same cell.

### 3.5. Reporting Requirements
The final report must detail:
- Total cells, eligible cells, failed cells (with failure reasons).
- Relationships between eligibility and initial nucleus area/circularity.
- Actual distribution of area tolerance and exact exchanged pixel counts.
- Distribution of candidate attempt counts (1 to 20).

## 4. Interpretation Guidelines
Based on the Counterfactual results, the conclusions are pre-locked to:

- **`STRONG_ADDITIONAL_SUPPORT`:** Area tolerance is achieved for the vast majority of eligible cells. The area-preserving boundary perturbation produces a significantly larger absolute circularity change than the matched shape perturbation, with a 95% CI strictly > 0 for the difference. The direct perimeter shifts in the expected direction, whereas N/C ratio impact remains minimal. Consistency across seeds/replicates is maintained.
- **`PARTIAL_ADDITIONAL_SUPPORT`:** The effect trends in the anticipated direction but the 95% CI crosses zero, or the magnitude of circularity change is considerably smaller than predicted. Perimeter changes but N/C separation is blurry, or support exists only at subset severity bands.
- **`ADDITIONAL_SUPPORT_NOT_OBSERVED`:** Boundary perturbations (when area is clamped) do not influence circularity significantly more than shape perturbations, or the effect is reversed. Perimeter fails to change even if area preservation holds.
*(Note: An unobserved outcome does not invalidate the existing regression or direct-mask evidence, but requires reframing the mechanism to acknowledge combined boundary/area dependencies).*

## 5. Technical Stopping Criteria
The counterfactual phase will be aborted and reported as technically unviable if:
- $\pm 2\%$ area tolerance cannot be achieved for a significant majority of eligible cells within the maximum attempts.
- Topological constraints (connected components, holes) routinely fail across most cells.
- Too few cells remain in the 0.95–0.98 Mild Dice band to perform meaningful statistical comparison.
- Algorithm eligibility demonstrates severe pathological biases toward only specific cell classes.
