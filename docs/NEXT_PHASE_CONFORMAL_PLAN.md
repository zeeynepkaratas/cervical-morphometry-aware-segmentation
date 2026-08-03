# Next Phase Conformal Plan

This plan is only for a future phase if the pilot returns `GO` or `CONDITIONAL_GO`.

The old conformal project remains untouched. The selected N/C-aware checkpoint would be exported with the same metric names and per-cell output schema used by the source strict conformal pipeline.

Planned handoff:

- Marginal split conformal: use the source calibration protocol with new model predictions and unchanged official N/C/circularity measurements.
- Strict one-variant-per-cell analysis: preserve cell-level independence by sampling one degradation variant per original cell per repeat.
- Joint N/C-circularity coverage: reuse source joint coverage code against paired absolute errors.
- Failure-aware coverage: stratify by invalid N/C/circularity flags and segmentation degeneracy diagnostics.
- Interval width comparison: compare baseline versus N/C-aware checkpoint intervals under the same calibration/test splits.

This phase is not implemented or run in the pilot task.
