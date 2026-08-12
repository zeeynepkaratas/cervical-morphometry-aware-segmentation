# Closing Dual-Matched Local Random Null Preregistration

## Epistemic Status

This is a `PROSPECTIVE SECONDARY MECHANISM-VALIDATION ANALYSIS` for the existing
closing correction experiment. It is not a primary analysis, not a replacement
confirmatory analysis, and not retrospective optimization.

The primary correction result remains unchanged:

- frozen test cells: `184`
- primary correction verdict: `CORRECTION_SUPPORTED`

This B analysis exists because the preregistered primary null control showed
substantial mismatch in perimeter magnitude and area behavior. This secondary
analysis was specified after observing that limitation, but before generating
any dual-matched-null result.

## Scientific Question

If a local random boundary perturbation is constrained to reproduce the
closing-induced geometric changes in both area and perimeter, does it produce
the same circularity-error improvement as morphological closing?

If closing remains better than a genuinely dual-matched random local
perturbation, this strengthens evidence that its benefit is not explained only
by gross changes in area and perimeter. If closing and matched null perform
similarly, or if the null performs better, specificity is not supported. This
does not alter the already established primary finding that closing improved
circularity absolute error.

## Inputs And Inference Status

Existing closing outputs are consumed first:

- `results/closing_correction/test_results_per_seed.csv`
- `results/closing_correction/area_delta_comparison.csv`
- `results/closing_correction/verdict.json`

These scalar outputs do not persist raw prediction masks. A local random null
requires mask topology, so frozen-checkpoint inference is required only to
reconstruct the existing raw predictions for the frozen test cells. No model is
trained, selected, tuned, or changed. Preprocessing and checkpoints remain
frozen.

Primary B analysis uses the baseline model family and seeds:

- `20260803`
- `20260804`
- `20260805`

The final inferential unit is `cell_id`; seed-level effects are aggregated by
cell median before bootstrap.

## Forbidden Leakage Variables

The null generator must not use:

- GT mask geometry to choose perturbation locations
- GT circularity
- circularity absolute error
- signed circularity error
- under/over-circular status
- whether closing helped a cell
- test outcome
- target AE improvement
- any objective involving closeness to GT
- any objective involving circularity

The matching objective uses only geometric changes induced by closing relative
to the raw prediction:

- `target_dA`
- `target_dP`
- candidate `dA`
- candidate `dP`

GT is used only after the null mask is frozen, for evaluation.

## Target Definitions

For each raw predicted mask:

- `A_raw`: predicted nucleus pixel count
- `P_raw`: OpenCV external-contour perimeter

For its existing frozen 3x3 ellipse, one-iteration closing:

- `A_close`
- `P_close`

Signed targets:

- `target_dA = A_close - A_raw`
- `target_dP = P_close - P_raw`

Direction matters. The null attempts to reproduce both signed changes.

## Local Random Null Algorithm

The algorithm is local, randomized, boundary-based, GT-blind, closing-location
blind, and deterministic under fixed seed.

For each cell and seed:

1. Reconstruct the raw predicted nucleus and cytoplasm masks from the frozen
   baseline checkpoint.
2. Compute the frozen closing mask using a 3x3 elliptical kernel, one
   `cv2.MORPH_CLOSE` iteration, constrained to the predicted cell mask.
3. Compute `target_dA` and `target_dP`.
4. Initialize the candidate null at the raw predicted nucleus mask.
5. Candidate add pixels are cytoplasm-only pixels inside the predicted cell
   that are 8-neighbor adjacent to the current nucleus.
6. Candidate remove pixels are current nucleus boundary pixels.
7. A proposal randomly chooses add/remove, with 75% probability assigned to the
   operation that moves current signed `dA` toward `target_dA`; the remaining
   25% allows perimeter matching to recover from area-only direction.
8. A proposal changes exactly one local pixel. No global erosion, dilation, or
   morphology operator is applied as the null.
9. The proposal is rejected if it makes the nucleus empty, makes cytoplasm-only
   zero, creates no valid OpenCV external contour, or creates class overlap.
10. The proposal is accepted only if it improves the pre-specified dual
    matching score below.
11. The best mask encountered during the fixed proposal budget is used.
12. If no valid improvement is found, the null falls back to the raw prediction.

The algorithm never identifies notch locations, gap locations, high-error
regions, locations changed by closing, or any GT-related locations.

## Deterministic Seed

For each cell and seed:

`SHA256("dual_matched_null_v1|" + cell_id + "|" + seed_id)`

The first eight hex characters are converted to uint32. Python built-in
`hash()` is not used.

## Candidate Budget

The fixed budget is:

- random local proposals per cell/seed: `2048`
- maximum accepted local edits per cell/seed: `256`

The run is performed once. The budget, seed, score, floors, and fallback are
not changed after seeing B results.

## Matching Objective

For each candidate:

`candidate_dA = A_candidate - A_raw`

`candidate_dP = P_candidate - P_raw`

Area and perimeter errors:

`eA = abs(candidate_dA - target_dA) / max(abs(target_dA), sqrt(max(A_raw, 1)))`

`eP = abs(candidate_dP - target_dP) / max(abs(target_dP), sqrt(max(P_raw, 1)))`

The joint score is:

`joint_error = max(eA, eP)`

Ties are broken by lower `eA + eP`, then by earlier proposal order.

These floors are deterministic functions of raw prediction geometry and are
used to avoid unstable division by near-zero targets. Circularity, GT, and AE
are not used.

## Matching Diagnostics

For every cell/seed report:

- `target_dA`
- `null_dA`
- area matching error
- `target_dP`
- `null_dP`
- perimeter matching error
- joint matching error
- invalid proposal count
- accepted edit count
- fallback flag

Aggregate diagnostics report mean, median, IQR, P90, and P95 for area,
perimeter, and joint matching errors. Signed target and achieved distributions
are also reported.

## Matching Status

No universal biological percentage threshold is asserted. Matching status is
evaluated in two stages:

1. Quantify matching quality.
2. Label the B specificity result as `ADEQUATE` only if the dual null improves
   both mean area matching error and mean perimeter matching error relative to
   the original preregistered null diagnostics and the achieved mean signed
   `dA` and `dP` have the same directions as the target means. Otherwise label
   it `LIMITED`.

The original null reference values are:

- mean area mismatch: `103.72826086956522`
- mean perimeter mismatch: `4.728883951470472`

This matching status is independent of whether closing beats the null.

## B Comparison

After the null masks are frozen without GT-based optimization, compute GT
circularity absolute error.

Per cell:

- `AE_raw`
- `AE_closing`
- `AE_dual_null`
- `benefit_closing = AE_raw - AE_closing`
- `benefit_null = AE_raw - AE_dual_null`
- `specific_advantage = benefit_closing - benefit_null`

Equivalent:

`specific_advantage = AE_dual_null - AE_closing`

Positive means closing is better.

Report mean, median, 95% cell-level bootstrap CI, and percent of cells where
closing is better than the dual null.

Bootstrap:

- 10,000 iterations
- cell-level
- deterministic SHA256-derived seed

## B Verdict Logic

B does not have authority to change `CORRECTION_SUPPORTED`.

Possible secondary conclusions:

- `DUAL_MATCHED_SPECIFICITY_SUPPORTED`: matching status is `ADEQUATE`, mean
  specific advantage is positive, and 95% CI excludes zero on the positive
  side.
- `SPECIFICITY_NOT_SUPPORTED`: matching status is `ADEQUATE`, but closing does
  not outperform null with CI excluding zero.
- `LIMITED_BY_MATCHING`: matching status is `LIMITED`, regardless of apparent
  AE difference.
- `NULL_OUTPERFORMS_CLOSING`: matching status is `ADEQUATE` and null
  significantly outperforms closing.

No fifth category will be introduced after seeing results.

## Safety And Heterogeneity

For dual null and closing, report descriptive/supportive:

- nucleus Dice
- foreground Dice
- N/C AE
- area AE
- invalid rate

Under/over-circular heterogeneity may be reported descriptively using the
previously frozen definition:

- under: `C_raw < C_gt`
- over: `C_raw > C_gt`

Subgroup status is never used to generate or match the null.

## Quality Checks

Before the final B verdict verify:

- `n=184` cells represented
- no silent exclusions
- no GT leakage in candidate selection
- no circularity-based candidate selection
- no result-dependent rerun
- same raw/closing values reproduce existing frozen outputs within tolerance
- original primary files unchanged
- A decomposition files unchanged
- original null files unchanged

This is the final new experiment for this manuscript.
