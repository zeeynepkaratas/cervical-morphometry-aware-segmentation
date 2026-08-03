# Cervical Morphometry-Aware Segmentation Pilot

This repo is a separate Herlev-only pilot derived from `zeeynepkaratas/cervical-morphometry-conformal`. The old project is preserved as the strict conformal analysis source; this repo tests one narrow pivot without changing that source checkout.

Research question: can adding a differentiable N/C-ratio target reduce N/C measurement error without materially lowering segmentation Dice, and does the effect survive Gaussian blur and Gaussian noise?

The official N/C definition is locked as:

```text
N/C ratio = nucleus_area / cytoplasm_only_area
```

It is not `nucleus_area / whole_cell_area`. Class indices remain `0=background`, `1=cytoplasm-only`, `2=nucleus`.

## Scope

The pilot uses only Herlev. Cx22 is out of scope because this is a go/no-go pilot, not external validation. Conformal prediction is also out of scope during this phase; outputs and metric names are kept compatible with a later strict conformal handoff.

The only model change is the loss:

```text
total_loss = CrossEntropy + foreground Dice + lambda_nc * SmoothL1(log(pred_nc), log(true_nc))
```

Baseline uses `lambda_nc=0`, which numerically matches the source segmentation loss.

## Data

Raw Herlev images are not committed. Provide data with:

```bash
set HERLEV_DATA_DIR=C:\path\to\herlev
```

or pass `--data-dir`. If neither is provided, commands try the local source checkout at `../cervical-morphometry-conformal/data/raw/herlev`.

## Commands

```bash
python -m pytest -q
python experiments/make_pilot_split.py --config configs/pilot.yaml
python experiments/train_pilot_models.py --config configs/pilot.yaml
python experiments/evaluate_pilot_models.py --config configs/pilot.yaml
python experiments/summarize_pilot.py --config configs/pilot.yaml
python experiments/run_pilot.py --config configs/pilot.yaml
```

Results are written under `results/pilot/`, including per-cell predictions, metric summaries, loss-scale diagnostics, training histories, `pilot_summary.json`, and `pilot_decision.md`.

## Decision Rule

GO requires most pre-declared checks to pass: at least roughly 10% mean N/C absolute-error reduction, foreground Dice drop no worse than 0.01, no meaningful invalid N/C increase, improvement in at least two of clean/blur/noise families, and no serious circularity harm. Otherwise the decision is `CONDITIONAL_GO` or `NO_GO`.

No success is claimed until the pilot has actually run.
