# Fair Repeat Protocol

This protocol checks whether the small clean-validation N/C improvement from
the quick prescan repeats under paired initial conditions.

Locked scope:

- Seeds: `20260803`, `20260804`, `20260805`
- Models: baseline `lambda_nc=0` and N/C-aware `lambda_nc=0.10`
- Max epochs: `25`
- Early stopping patience: `6`
- Primary checkpoint rule: `best_val_dice`
- Validation: clean Herlev pilot validation cells only

For each seed, one initial U-Net state is created and hashed. Both models load
that same state. Each model then receives its own fresh optimizer. The train
order hash is also matched within each seed.

Final evaluation loads the `best_val_dice` checkpoint for every model and seed.
Per-cell metrics are computed after argmax predictions are restored to original
image size. The official N/C definition remains:

```text
nucleus_area / cytoplasm_only_area
```

Negative paired deltas mean the N/C-aware model improved the corresponding
error. Checkpoints and initial `.pt` states are local artifacts and are not
committed.
