# Pre-registration Plan: Counterfactual and Exact Decomposition Analyses

This document defines the pre-registered protocol for two linked analyses:
1. Exact log-ratio area–perimeter decomposition
2. Area-preserving boundary counterfactual perturbation

These analyses supplement the prior circularity mechanism evidence without altering the fundamental deep learning pipeline, external datasets, or initiating new training loops.

## 1. Data Sources
- The primary data source is the existing direct-mask measurements previously computed in the `results/circularity_mechanism/real_prediction_area_perimeter.csv` tracking predictions across different noise severities and seeds.
- For counterfactual analysis, deterministic ground truth masks (Herlev dataset) are used. 
- Frozen checkpoints are only used to infer any missing required metrics (if applicable, though current cached direct mask predictions are preferred).

## 2. Inclusion/Exclusion Criteria and Invalid Predictions
Predictions are explicitly excluded from the log-decomposition if they fall into the following **invalid/degenerate** conditions:
- Predicted nucleus area is $\le 0$.
- Predicted direct perimeter is $\le 0$.
- Non-finite values in prediction metrics.
- Missing or entirely empty predicted nucleus mask.
- Pre-defined project invalid criteria (e.g., severe mask overlap rendering N/C impossible).
Exclusion events are strictly logged per condition, severity, and seed. Invalid instances are never silently dropped.

## 3. Exact Log-Ratio Area-Perimeter Decomposition
This is an exact algebraic identity accounting, not a statistical regression model explaining error variability across populations.

**Formula:**
$$ \Delta \log(C) = \log(C_{corrupted} / C_{clean}) $$
$$ \text{Area Contribution} = \log(A_{corrupted} / A_{clean}) $$
$$ \text{Perimeter Contribution} = -2 \times \log(P_{corrupted} / P_{clean}) $$
**Identity Validation:** $\Delta \log(C) = \text{Area Contribution} + \text{Perimeter Contribution}$

**Outputs:**
- Primary metrics per cell: `delta_log_circularity`, `area_contribution`, `perimeter_contribution`, absolute contributions, dominant component, and identity residual.
- Summary outcomes: Difference between absolute perimeter and area contributions, fraction of cells where perimeter dominates, consistency across severity/seeds, signed relationships, and confirmation of identical bounds via residual checks ($\approx 0$).

## 4. Area-Preserving Boundary Counterfactual Perturbation
**Goal:** Determine how much circularity changes when boundary complexity (direct perimeter) is altered while nucleus area is intentionally held constant.

**Algorithm & Topology Constraints:**
- Deterministic morphological operation pairs (e.g., equal local additions and subtractions on the boundary) altering boundary roughness without a uniform scaling (not uniform erosion/dilation).
- **Tolerance:** Area preservation must be strictly bounded within $\pm 2\%$ relative to the original initial nucleus area: 
  $$ \frac{|A_{perturbed} - A_{original}|}{A_{original}} \le 0.02 $$
- **Mask Semantics:** 
  - $ \text{original\_cell\_mask} = \text{nucleus\_gt} \cup \text{cytoplasm\_gt} $
  - $ \text{perturbed\_cytoplasm} = \text{original\_cell\_mask} \cap \sim\text{perturbed\_nucleus} $
- Topology must restrict the perturbed nucleus to strictly lie within the original cell mask. The perturbed nucleus cannot be empty, should preferably maintain a single connected component, and avoid new disjoint islands or topological holes.
- All pseudo-randomness for boundary localization is driven by identical seeded functions bound to the unique `cell_id`.

**Comparison Constraints:**
- Evaluated specifically on cells achieving a **Mild Dice band (0.95–0.98)** against original shape perturbation statistics to ensure parity in segmentation loss.
- Additional matching on nucleus dice, initial area, and initial circularity may be applied as long as statistical power persists.

**Outputs:**
- Primary: Mean absolute circularity change compared between Area-Preserving Boundary Perturbation vs. Matched Shape Perturbation.
- Secondary: Signed circularity change, direct perimeter change, absolute/signed N/C change, area preservation error metrics, valid condition rates, Dice distributions, topological violations.

## 5. Statistical Framework
- All confidence intervals (CI) employ **Cell-ID clustered bootstrapping** to properly account for repeated measurements across models, severities, and random seeds.

## 6. Interpretation Guidelines
Based on the Area-Preserving Counterfactual results, the conclusions are pre-locked to:
- **Strong Support:** Area tolerance is achieved for the vast majority of eligible cells. The area-preserving boundary perturbation produces a significantly larger absolute circularity change than the matched shape perturbation, with a 95% CI strictly > 0 for the difference. The direct perimeter shifts in the expected direction, whereas N/C ratio impact remains minimal. Consistency across seeds/replicates is maintained.
- **Partial Support:** The effect trends in the anticipated direction but the 95% CI crosses zero, or the magnitude of circularity change is considerably smaller than predicted. Perimeter changes but N/C separation is blurry, or support exists only at subset severity bands.
- **Not Supported:** Boundary perturbations (when area is clamped) do not influence circularity significantly more than shape perturbations. Perimeter fails to change even if area preservation holds, or the result is entirely driven by a few artifactual cell outliers. *(Note: If not supported, the mechanism should be reframed to involve simultaneous area and boundary shifts, without invalidating prior direct-perimeter correlations).*

## 7. Technical Stopping Criteria
The counterfactual phase will be aborted and reported as technically unviable if:
- $\pm 2\%$ area tolerance cannot be achieved for a significant majority of eligible cells.
- The perturbation strategy consistently violates topological rules (e.g., massive disconnecting components).
- Too few cells remain in the target 0.95–0.98 Mild Dice band to perform meaningful statistical comparison.
- The algorithm depends heavily on tuning specific sizes/classes, or necessitates new hyperparameter searches post-plan lockdown.
