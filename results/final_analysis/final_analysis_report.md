# Selective Final Analysis Report

Decision: **ORTA AMA SAVUNULABİLİR**

The broad 17-figure expansion was intentionally not run. The selective
completion focuses on conformal coverage loss, severity trends, Dice-
morphometry decoupling, and the negative/unstable N/C-aware intervention.

## Coverage

Frozen clean-calibration intervals are below nominal in most degraded
conditions. Mean coverage delta versus clean across degraded rows is
`-0.0360`; rows with at least 5 percentage-point drop:
`13/72`.

## Severity Trends

Trend labels are written to `severity_trends.csv`. Fully monotonic behavior is
not required for the story; the stronger observation is that coverage loss is
common while morphometric error trends are mixed.

## N/C-aware Model

N/C-aware has lower mean N/C error than baseline in `5/21`
test seed-condition rows. This remains a negative or unstable intervention,
not a successful method claim.

## Recommended Scope

Keep the study centered on reliability: Dice overlap alone does not guarantee
morphometric measurement reliability, and frozen clean conformal intervals lose
coverage under distribution shift. Treat blur/noise morphometric error and the
N/C-aware loss as supporting, not central, evidence.
