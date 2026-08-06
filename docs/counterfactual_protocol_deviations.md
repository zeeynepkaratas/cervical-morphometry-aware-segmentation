# Counterfactual Protocol Deviations

This document serves as a scientific transparency record outlining deviations from the originally registered protocol in `docs/counterfactual_decomposition_plan.md`. These deviations occurred during the initial implementation and execution phases and are documented here before producing the final validated results. The original plan is kept intact for historical accuracy.

## Documented Deviations

1. **Undisclosed Numerical Budgets:** The pre-registered plan mentioned fixed budgets (Low, Medium, High) but did not explicitly specify the numerical values (10, 30, and 60 pixels respectively) prior to execution.
2. **Undisclosed Budget Scaling:** The implementation dynamically halved the pixel budget for cells with initial area `< 500`. This scaling logic was not pre-specified.
3. **Improper Pairing and Matching:** The first iteration of the runner script extracted overlapping cells but compared group averages rather than strictly matching and pairing individual shape and counterfactual instances *within the same cell* based on nucleus Dice. It also failed to use cell-level clustered bootstrapping for confidence intervals.
4. **Flawed Decomposition Identity:** The exact log-ratio decomposition incorrectly computed ratios between *predictions* and *ground truth* (`log(predicted / true)`). The correct algebraic identity for the mechanism analysis must evaluate the ratio between the *corrupted prediction* and the *clean prediction* for the exact same cell and seed.
5. **Clipped Circularity Artifact:** The initial decomposition identity failed to evaluate cleanly (`residual = -0.0605`) because it utilized standard circularity metrics which are often clipped at 1.0. Exact decomposition requires `unclipped direct circularity` ($4\pi A / P^2$) for the identity to close accurately ($< 1e-10$).
6. **Contour Area Ambiguity:** The initial area-preserving implementation guaranteed `pixel_count_change == 0`, but did not independently verify that the resulting OpenCV `contourArea` relative change remained $\le 2\%$. Circularity relies on contour area, so both conditions must hold for true eligibility.
7. **Premature Conclusion:** The initial report concluded `PARTIAL_ADDITIONAL_SUPPORT` based on unvalidated, unpaired, and non-bootstrapped group differences. This conclusion is retracted pending the corrected paired analysis.

## Corrective Actions

These deviations will be addressed in the updated scripts:
- The fixed budgets (10, 30, 60) and area threshold scaling (<500) are now locked.
- The exact decomposition will correctly pair clean and corrupted predictions for the same cell and seed, utilizing unclipped raw circularity.
- The area-preserving perturbation will strictly enforce `abs(contour_area_change_rel) <= 0.02` in addition to pixel-count equality.
- The counterfactual comparison will use strict within-cell pairing by matching the nearest shape instance (by Dice) to each counterfactual instance, resolving ties deterministically.
- A 10,000 iteration cell-level paired bootstrap will compute the 95% CI.
- Final decisions will strictly map the bootstrapped confidence intervals and paired differences to the predefined categories.
