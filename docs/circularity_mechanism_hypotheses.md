# Circularity Mechanism Analysis — Pre-registered Hypotheses

**Timestamp:** 2026-08-05T11:38:00+03:00
**Branch:** `feature/circularity-mechanism-analysis`
**Starting commit:** `2c4c3b1e2076c1b45d3eabdc81c4c44cd4fd6b9a`
**Tag:** `pre_consistency_pivot_2c4c3b1`

These hypotheses are recorded **before** any new analysis is performed.
No results from this analysis have been seen at the time of writing.

---

## H1 — Area–perimeter divergence under noise

**Statement:** Under Gaussian noise, predicted nucleus area changes will be
more limited than predicted nucleus perimeter changes.

**Rationale:** Gaussian noise introduces high-frequency boundary irregularities
that inflate contour length disproportionately relative to region area.
Area is an integral quantity that averages over interior pixels and is
therefore more robust to boundary perturbations.

**Operationalisation:** For each cell, compute
`|Δperimeter_error| / |Δarea_error|` where Δ = noise − clean. The median
ratio across cells should exceed 1.0 for the hypothesis to be supported.

---

## H2 — Circularity error is perimeter-driven

**Statement:** The increase in circularity absolute error under Gaussian noise
will show stronger correlation with the increase in perimeter error than with
the increase in area error.

**Rationale:** Circularity = 4πA/P². Since P appears squared in the
denominator, small perimeter perturbations have an amplified effect on
circularity. If noise primarily corrupts boundaries, the circularity
degradation should track perimeter degradation more closely than area
degradation.

**Operationalisation:** Spearman correlation between Δcircularity_error and
Δperimeter_error should be stronger (higher |ρ|) than between
Δcircularity_error and Δarea_error. The difference should be consistent
across at least 2 of 3 seeds.

---

## H3 — Boundary-type perturbations disproportionately affect circularity

**Statement:** Among controlled mask perturbations producing similar Dice loss,
boundary jitter, small protrusions, and small indentations will cause larger
circularity changes than erosion or dilation.

**Rationale:** Erosion and dilation uniformly shrink or expand a mask, changing
area and perimeter in a correlated way that approximately preserves shape
(and thus circularity). Boundary jitter, protrusions, and indentations alter
local curvature without large area changes, disproportionately increasing
perimeter and thus reducing circularity.

**Operationalisation:** At matched Dice bands, mean |circularity change| for
jitter/protrusion/indentation perturbations should exceed mean |circularity
change| for erosion/dilation perturbations.

---

## H4 — N/C ratio is boundary-robust

**Statement:** Small boundary irregularities will change circularity
substantially while changing N/C ratio only modestly.

**Rationale:** N/C = nucleus_area / cytoplasm_area. Boundary perturbations that
increase perimeter without substantially changing area will have minimal
effect on N/C but large effect on circularity. This asymmetry would explain
why conformal coverage for circularity degrades more than for N/C under
noise.

**Operationalisation:** For boundary-type perturbations (jitter, protrusions,
indentations), the ratio |circularity_change| / |N/C_change| should be
substantially greater than 1.0.

---

## H5 — A single fixed post-processing step can safely reduce circularity fragility

**Statement:** A morphological operation (opening, closing, or their
composition) selected and frozen on the validation set will reduce nucleus
circularity error on the test set under Gaussian noise, without causing
unacceptable Dice or N/C degradation.

**Rationale:** If circularity fragility is driven by boundary irregularities
introduced by noise, a simple morphological smoothing should remove small
spurious boundary features, thereby improving circularity accuracy. The
intervention must be safe: Dice loss ≤ 0.005, N/C error increase ≤ 5%,
and consistent across seeds.

**Operationalisation:** The selected post-processing must reduce mean
circularity absolute error in at least 2 of 3 seeds on the test set, with
combined improvement of at least ~10%, while meeting all safety constraints.
If no candidate meets the safety criteria, the result is
`NO_SAFE_POSTPROCESSING`.

---

## Decision criteria

The decision criteria (STRONG_MECHANISM_SUPPORTED,
STRONG_MECHANISM_PLUS_SAFE_FIX, PARTIAL_MECHANISM, MECHANISM_NOT_SUPPORTED)
are defined in the task specification and will not be modified after seeing
results.
