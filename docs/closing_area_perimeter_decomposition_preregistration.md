# Closing Area/Perimeter Counterfactual Decomposition Preregistration

## Status

This is a prospective secondary mechanistic analysis of the existing
`CLOSING_CORRECTION_CONFIRMATORY` experiment. It does not change the primary
correction result or the preregistered `CORRECTION_SUPPORTED` verdict.

No training, inference, new correction, new threshold, subgroup search, endpoint
selection, kernel change, or iteration change is allowed.

## Research Question

Observed closing-induced circularity absolute-error improvement is attributable
to which geometric channel: area change, perimeter change, or their joint
nonlinear effect?

The analysis uses the frozen test correction outputs already written under
`results/closing_correction/`.

## Inputs

The primary input is:

`results/closing_correction/test_results_per_seed.csv`

This file contains the frozen raw and closing per-seed outputs for each test
cell and model. No new model forward pass is performed.

Because the project canonical circularity is OpenCV contour based, the
decomposition uses the exact canonical area implied by the saved circularity
and perimeter:

`A_canonical = C * P^2 / (4*pi)`

This preserves exact consistency with the saved project circularity values.
The pixel-count area remains available descriptively in the existing correction
outputs but is not used for the exact circularity identity.

## Counterfactual Definitions

For each cell, model, and seed, read or derive:

- `A_raw`
- `P_raw`
- `A_close`
- `P_close`
- `C_gt`

Four circularity values are then computed:

1. Raw:
   `C_raw = 4*pi*A_raw/(P_raw^2)`

2. Area-only counterfactual:
   `C_area_only = 4*pi*A_close/(P_raw^2)`

3. Perimeter-only counterfactual:
   `C_perimeter_only = 4*pi*A_raw/(P_close^2)`

4. Full closing:
   `C_full = 4*pi*A_close/(P_close^2)`

`C_raw` and `C_full` must match the saved raw and closing circularity values
within floating-point tolerance. A mismatch outside tolerance is a bug.

## Absolute Error Definitions

For each row:

- `AE_raw = abs(C_raw - C_gt)`
- `AE_area_only = abs(C_area_only - C_gt)`
- `AE_perimeter_only = abs(C_perimeter_only - C_gt)`
- `AE_full = abs(C_full - C_gt)`

Benefit convention:

- `benefit_area = AE_raw - AE_area_only`
- `benefit_perimeter = AE_raw - AE_perimeter_only`
- `benefit_full = AE_raw - AE_full`

Positive benefit means improvement.

## Signed Biases

The following signed circularity biases are reported:

- `bias_raw = C_raw - C_gt`
- `bias_area_only = C_area_only - C_gt`
- `bias_perimeter_only = C_perimeter_only - C_gt`
- `bias_full = C_full - C_gt`

## Exact Interaction

Circularity-value effects:

- `delta_C_area = C_area_only - C_raw`
- `delta_C_perimeter = C_perimeter_only - C_raw`
- `delta_C_full = C_full - C_raw`
- `interaction_C = delta_C_full - delta_C_area - delta_C_perimeter`

The identity

`delta_C_full = delta_C_area + delta_C_perimeter + interaction_C`

must hold within floating-point tolerance.

No arbitrary `>50%` attribution rule is used. Absolute-error effects are
reported as counterfactual benefits; no additive identity is expected for AE
because absolute value is nonlinear.

## Main Mechanistic Summaries

For `benefit_area`, `benefit_perimeter`, and `benefit_full`, report:

- mean
- median
- IQR
- 95% cell-level bootstrap CI
- percent of cells with benefit greater than zero

Also report direct paired comparison:

`benefit_perimeter - benefit_area`

Positive values mean the perimeter-only counterfactual provides more
circularity AE improvement than the area-only counterfactual. Negative values
mean the area-only counterfactual provides more improvement.

This comparison is reported numerically and is not forced into a binary winner.

## Under/Over Circular Stratification

Subgroups are fixed before analysis:

- under-circular: `C_raw < C_gt`
- over-circular: `C_raw > C_gt`
- equal: `C_raw == C_gt`

For each subgroup, report:

- `n`
- `AE_raw`
- `AE_area_only`
- `AE_perimeter_only`
- `AE_full`
- `benefit_area`
- `benefit_perimeter`
- `benefit_full`
- `interaction_C`

## Supportive Directional Expectations

H1: Under-circular cells are expected to show larger positive full-closing
benefit than over-circular cells.

H2: If the original perimeter-driven fragility mechanism contributes to closing
benefit, perimeter-only benefit should be larger than area-only benefit,
especially in cells with high `abs_perimeter_contribution`.

H3: If area change materially explains the aggregate closing effect, the
area-only counterfactual will show non-negligible positive AE benefit.

H4: Full-closing effect may not equal the arithmetic sum of AE benefits because
circularity and absolute error are nonlinear.

These are supportive mechanistic hypotheses and are not promoted to primary
correction endpoints.

## Existing Perimeter Contribution Link

Use the existing `abs_perimeter_contribution` from:

`results/circularity_mechanism/log_decomposition_per_cell.csv`

For baseline primary analysis, compute Spearman rank correlations:

- `abs_perimeter_contribution` versus `benefit_perimeter`
- `abs_perimeter_contribution` versus `benefit_full`

N/C-aware sensitivity may be reported only if it is available from existing
artifacts without new inference.

## Analysis Unit

The statistical unit is `cell_id`. Seeds are not independent samples.

Counterfactual values are computed per seed first, then cell-level median
effects are computed. Summaries and bootstrap intervals use the `n=184`
cell-level median values.

## Bootstrap

Use 10,000 percentile bootstrap iterations at the cell level.

The bootstrap seed is deterministic and SHA256-derived. Python built-in
`hash()` is not used.

## Self Checks

Mandatory checks:

1. `C_raw` matches saved raw circularity.
2. `C_full` matches saved closing circularity.
3. The interaction identity residual is floating-point small.
4. No cells are dropped silently.

Report:

- maximum absolute identity residual
- mean absolute identity residual
- input row counts
- output cell counts

## Hierarchy

The primary paper result is unchanged. The existing closing correction primary
PASS and `CORRECTION_SUPPORTED` verdict remain unchanged.

This analysis is a prospective secondary mechanistic decomposition.
