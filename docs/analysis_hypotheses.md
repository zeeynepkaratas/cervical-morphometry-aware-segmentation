# Pre-registered Degradation and Conformal Hypotheses

Timestamp: `2026-08-04T10:12:27+03:00`

Branch: `feature/nc-aware-pilot`

Commit before final-analysis inference: `b6c7aaff118bb598b5baf73e620871c582a80210`

Working tree note: only prior quick-prescan untracked artifacts were present; no final blur/noise or conformal inference had been run before this file was created.

## H1 - Degradation Severity and Morphometric Error

As blur and noise severity increase, N/C absolute error and circularity absolute error will generally increase.

This hypothesis will not be interpreted as requiring several-fold error increases. Evaluation will use:

- monotonic or mostly ordered change with severity,
- whether most cells move in the same direction,
- paired effect size,
- clustered bootstrap confidence intervals.

## H2 - Circularity Fragility

Because nuclear circularity is more sensitive to boundary geometry, it is expected to degrade faster or more consistently than N/C ratio under blur and noise.

## H3 - Dice-Morphometry Decoupling

Dice change will not fully explain N/C and circularity error changes. Cells or conditions with similar Dice may show clearly different morphometric errors.

## H4 - N/C-aware Loss Instability

The `lambda_nc=0.10` N/C-aware model will not show consistent superiority over baseline across all three seeds under blur and noise.

## H5 - Frozen Conformal Coverage

Conformal intervals calibrated and frozen only on clean calibration cells are expected to show a tendency toward below-nominal coverage as degradation severity increases. This drop is expected to be more pronounced for circularity.
