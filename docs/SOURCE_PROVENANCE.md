# Source Provenance

Source repository: `zeeynepkaratas/cervical-morphometry-conformal`

Source commit SHA: `094723a576fb8c44f60cd5c1efd0a4ef1e4b7e7d`

Source status before transfer: clean working tree on `codex/strict-conformal-protocol`.

## Copied Or Adapted Files

- `src/data_prep/load_herlev.py`: copied. Herlev loader and official mask handling retained.
- `src/data_prep/group_split.py`: copied. General cell-level leakage helpers retained.
- `src/segmentation/unet_model.py`: copied. U-Net architecture retained.
- `src/segmentation/train_unet.py`: copied. RGB preprocessing, 128x128 padding, target mapping, and baseline segmentation loss retained.
- `src/measurements/morphometry.py`: copied. Official `compute_nc_ratio` and `compute_circularity` retained.
- `src/degradation/apply_degradations.py`: copied. Gaussian blur/noise functions reused; pilot config exposes only these two families.
- `src/utils/config.py`: adapted. Removed Cx22, CQR, Mondrian, and conformal runtime configuration; retained Herlev pilot constants.
- `experiments/exp1_dice_correlation.py`: selectively adapted into `src/pilot/metrics.py` for Dice helpers, mask-to-target conversion, and prediction restoration.
- `results/tables/herlev_group_split.json`: copied to `data/splits/original_herlev_group_split.json`.

## Not Copied

Cx22 loaders, Cx22 experiments, conformal modules, Mondrian/CQR experiments, broad robustness experiments, and large raw data/checkpoint artifacts were intentionally not copied.

## License

No source `LICENSE` file was present in the local source checkout. This repo includes a placeholder `LICENSE` noting that status.
