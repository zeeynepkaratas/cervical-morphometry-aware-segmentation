# Closing Correction Reproducibility

The preregistered closing-correction analysis lives in:

- protocol: `docs/closing_correction_preregistration.md`
- implementation: `src/closing_correction_confirmatory.py`
- outputs: `results/closing_correction/`

Re-run command, after setting the Herlev raw-data location:

```powershell
$env:HERLEV_DATA_DIR = "C:\path\to\herlev"
python -B src\closing_correction_confirmatory.py
```

The run performs no training, no new corruption, no lambda search, no kernel
search, no operation search, and no checkpoint reselection. It uses the frozen
test split from `data/splits/original_herlev_group_split.json` and the frozen
fair-repeat checkpoints under `results/pilot/fair_repeat/checkpoints/`.
