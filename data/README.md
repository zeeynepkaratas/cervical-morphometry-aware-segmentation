# Data

Do not commit raw Herlev images, downloaded archives, checkpoints, or large result dumps.

Pilot commands resolve Herlev data in this order:

1. `--data-dir`
2. `HERLEV_DATA_DIR`
3. `data.herlev_data_dir` in `configs/pilot.yaml`
4. `../cervical-morphometry-conformal/data/raw/herlev`

Committed split files:

- `splits/original_herlev_group_split.json`: copied from the source repository.
- `splits/pilot_train_validation_split.json`: generated from the source train pool only.
