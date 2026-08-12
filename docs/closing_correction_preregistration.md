# Closing Correction Preregistration

## Project role

This is not a new standalone project. It is Contribution 3 of the existing
cervical morphometry reliability project: mechanism-guided non-retraining
correction.

Contributions 1 and 2 remain independent of the outcome of this correction:

1. Clean real-cell overlap versus morphometry dissociation.
2. Perimeter-driven circularity fragility through exact decomposition,
   area/perimeter contribution analysis, area-preserving counterfactuals, and
   the existing mechanism analysis.

Synthetic Gaussian blur/noise evidence is secondary stress-test evidence, not
primary evidence for this correction.

## Test exposure disclosure

The frozen test set has `n=184` cells. The test set has previously been used
for mechanism characterization and decomposition.

The specific correction tested here was not selected or tuned on the test set:

- kernel
- operation
- fallback
- null control
- success rule

This analysis is classified as `SEMI-CONFIRMATORY`.

## Protocol deviation

The original H5 hierarchy jointly considered clean and Gaussian noise safety,
and the prior postprocessing selection returned `NO_SAFE_POSTPROCESSING`.

The locked protocol here uses clean performance as the primary correction
evidence and synthetic stress tests only as secondary/supplementary evidence.
This hierarchy redesign contains a post-hoc element and is disclosed.

Selection-bias classification:

- candidate/kernel: `LOW`
- safety-rule redesign: `MODERATE`

## Locked intervention

The correction is fixed before running the test evaluation:

- kernel: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))`
- operation: `cv2.MORPH_CLOSE`
- iterations: `1`

No other kernel, operation, iteration count, threshold, model, or checkpoint
selection will be tried in this protocol.

## Per-cell GT-independent fallback

For each cell, the corrected nucleus mask falls back to the raw prediction if
any of the following GT-independent conditions occur after closing:

- corrected nucleus mask is empty
- no valid corrected nucleus contour exists
- resulting cytoplasm-only prediction is invalid or zero area

The fallback decision must not use ground truth.

## Primary model family

The primary analysis is the baseline model family across the three frozen
seeds: `20260803`, `20260804`, `20260805`.

The N/C-aware model family is analyzed only as a secondary/sensitivity
analysis. The two model families are not mixed in the primary pool.

## Statistical unit

The real statistical unit is `cell_id`. The three seeds are not treated as
independent samples.

For each cell, seed-median paired effects are computed first. The primary
uncertainty analysis then uses `n=184` cell-level bootstrap.

## Primary endpoint

The primary endpoint is circularity absolute error.

For each cell:

`delta_closing = AE_closing - AE_raw`

The success direction is negative.

Primary correction success requires both:

1. The 95% cell-level bootstrap CI for `delta_closing` is entirely below zero.
2. The percent of cells with `AE_closing < AE_raw` is greater than 50%.

## Mechanism endpoint

The mechanism endpoint is perimeter absolute error.

For each cell:

`delta_perimeter = perimeter_AE_closing - perimeter_AE_raw`

## Safety endpoints

Safety endpoints are:

- nucleus Dice
- foreground Dice
- N/C absolute error
- nucleus area absolute error
- invalid rate

For error endpoints, positive paired deltas are worse. For Dice endpoints,
negative paired deltas are worse.

If the CI for a safety endpoint is entirely in the worse direction, this is a
safety failure. Otherwise the conclusion is phrased as no statistically
detectable deterioration. Invalid rate must be non-increasing.

Do not use language such as safety proven.

## Roundness audit

Two subgroups are mandatory:

- raw `pred_circularity < GT`
- raw `pred_circularity > GT`

A continuous supportive analysis is also mandatory:

`raw circularity signed bias = pred_circularity - gt_circularity`

versus:

`closing benefit = AE_raw - AE_closing`

Pre-specified hypothesis direction: more under-circular raw predictions are
expected to show higher closing benefit. This is supportive/mechanistic and
not primary.

## Existing perimeter mechanism relation

For each cell, relate `abs_perimeter_contribution` to closing benefit.

Pre-specified hypothesis direction: higher perimeter contribution is expected
to be associated with higher closing benefit.

This is a supportive mechanistic analysis, not a primary endpoint. No p-value
hunting is permitted.

## Decomposition uncertainty

For the existing decomposition summary, produce cell-level bootstrap confidence
intervals around:

- R2 where available from the existing decomposition/regression output
- perimeter-dominant fraction

This is not a new primary test. It strengthens existing mechanism reporting.

## Cx22

If existing external Cx22 results are present in the repository, they are not
used for tuning this correction. They may only be audited as existing
real-external-domain evidence and motivation/generalization context.

No threshold, operation, or kernel is selected on Cx22.

## Synthetic degradation

Gaussian noise and Gaussian blur are not primary evidence. They are
secondary/supplementary stress-test evidence only. No new severity sweep is
included in this protocol.

## Null control

The null control is local random-location, perimeter-reduction-matched boundary
perturbation.

The old random-direction null is not used.

For each cell:

1. Use the raw predicted nucleus mask.
2. Compute the locked closing mask.
3. Set `target_delta_P = P_closing - P_raw`.
4. Set `target_reduction = abs(target_delta_P)`.
5. Extract valid local boundary points from the raw prediction.
6. Initialize deterministic RNG from:
   `SHA256(f"{cell_id}|{model}|{seed}|closing_null_v1")`
   and convert the first eight hex characters to a uint32 seed.
7. Select random local boundary locations.
8. Apply only small perimeter-reducing local boundary edits.
9. Do not inspect GT, prediction/GT error direction, or the pixel locations
   changed by closing.
10. Match `abs(P_null - P_raw)` to `target_reduction` with a locked tolerance
    of `+/-10%`.
11. If `+/-10%` cannot be achieved, use the closest achievable mask and do not
    drop the cell.
12. Record `null_target_reduction`, `null_achieved_reduction`,
    `null_matching_error_abs`, and `null_matching_error_pct`.
13. If the null mask is invalid, fall back to the raw prediction using the same
    GT-independent validity rules.

The local edit is deterministic for a given cell/model/seed and consists of
removing one randomly selected raw boundary nucleus pixel at a time when that
removal decreases the OpenCV external-contour perimeter and leaves a valid
non-empty nucleus and non-zero cytoplasm-only region. Candidate pixels are
ordered by the deterministic RNG seed above. The algorithm never uses
ground-truth geometry or the closing-modified pixel set.

Closing and null both report:

- `delta_area_closing`
- `delta_area_null`
- `abs(delta_area_closing - delta_area_null)`

The null is perimeter-matched only. It is not area-matched.

## Null specificity test

Use direct cell-level paired contrast, not independent CI overlap.

For each cell:

`delta_closing = AE_closing - AE_raw`

`delta_null = AE_null - AE_raw`

`delta_specific = delta_closing - delta_null`

Mechanism-specific advantage requires the 95% bootstrap CI for
`delta_specific` to be entirely below zero.

## Verdict categories

The verdict is produced programmatically:

- `CORRECTION_SUPPORTED`: primary passes, no statistically detectable safety
  deterioration, invalid rate non-increasing, and null specificity passes.
- `CORRECTION_UNSPECIFIC`: primary passes, safety acceptable, and null
  specificity fails.
- `CORRECTION_UNSAFE`: primary passes and one or more safety endpoints
  significantly deteriorate.
- `CORRECTION_NOT_SUPPORTED`: primary fails.

No manual relabeling is allowed.

## Analysis hierarchy

Primary confirmatory-like:

- clean baseline model family
- circularity absolute error

Mechanism specificity:

- closing versus perimeter-reduction-matched null

Safety:

- Dice
- N/C absolute error
- area absolute error
- invalid rate

Supportive mechanistic:

- under-circular versus over-circular subgroup
- continuous signed circularity bias versus closing benefit
- absolute perimeter contribution versus closing benefit
- decomposition bootstrap CI

Sensitivity:

- N/C-aware model family

Secondary/supplementary:

- synthetic Gaussian noise/blur

## Test-run restrictions

After this preregistration is committed and pushed, the test run may use:

- frozen test `n=184`
- primary baseline model family, three seeds
- N/C-aware model family as sensitivity only

No training, lambda selection, kernel sweep, operation sweep, new corruption,
checkpoint reselection, or hyperparameter sweep is allowed.
