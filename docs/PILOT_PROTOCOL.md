# Pilot Protocol

The pilot uses the source Herlev split metadata but only the original train pool. Calibration and test cells are forbidden for model development.

Locked comparison:

- Baseline: `lambda_nc=0`
- Candidates: `lambda_nc=0.05`, `0.10`, `0.20`
- Same U-Net, RGB preprocessing, 128x128 aspect-ratio-preserving resize, optimizer, seed, batch size, and clean training images.
- Primary checkpoint rule before seeing results: `best_composite_validation`.
- Secondary checkpoint rule for comparison: `best_val_dice`.

Validation conditions are clean, Gaussian blur severities `[1, 2, 3]`, and Gaussian noise severities `[10, 20, 30]`. Noise is deterministically seeded by base seed, cell ID, and severity label.

Final morphometry is computed only after argmax predictions are restored to original image size. The differentiable soft N/C loss is not used as the final metric.
