# Cervical Morphometry-Aware Segmentation Analysis

## Project Purpose
This repository provides a controlled evaluation of morphometric reliability, conformal prediction coverage, and a deep circularity mechanism analysis for cervical cell segmentation. The project investigates whether adding a differentiable N/C-ratio target reduces N/C measurement error without harming segmentation performance, and provides evidence explaining why circularity predictions degrade under Gaussian noise while N/C ratio predictions remain robust.

## Data and Structure
This repository does **not** include the raw Herlev image dataset due to licensing and size constraints. 

To run the analysis, provide the Herlev dataset locally via the `HERLEV_DATA_DIR` environment variable:
```bash
set HERLEV_DATA_DIR=C:\path\to\herlev\dataset
```

Expected data structure:
```text
herlev/
  ├── smear2005/ (or similar raw folders containing .bmp)
  └── ...
```

## Environment Setup
The project uses standard PyTorch and data science libraries.
```bash
pip install -r requirements.txt
pip install torch torchvision
```

## Branch Structure
The project maintains a safe, strict separation of concerns via branching:
- `main`: The secure baseline containing the finalized target-specific conformal coverage and pilot results.
- `feature/circularity-mechanism-analysis`: Contains the deep-dive mechanism analysis evaluating boundary vs. shape perturbation under noise.

## Core Commands

### Running the Mechanism Analysis
To reproduce the circularity mechanism analysis (Phase A-D):
```bash
set HERLEV_DATA_DIR=C:\path\to\herlev\dataset
python experiments/run_circularity_mechanism.py --phase all
```
This generates all outputs, data leakage audits, and statistical figures under `results/circularity_mechanism/`.

### Running Tests
To run the automated test suite, which enforces leakage prevention, perturbation determinism, and safety constraints:
```bash
python -m pytest tests/ -v
```

## Scientific Findings Summary
The circularity mechanism analysis **provides evidence** that under Gaussian noise, high-frequency boundary irregularities cause severe degradation of the perimeter measurements. This disproportionately affects circularity error ($\rho = 0.57$) while leaving the overall area integrals (and thus the N/C ratio) relatively intact. 

Controlled shape vs. boundary perturbation tests **support** the hypothesis that boundary-level noise destroys circularity while preserving area-based morphometrics. Our evaluation explored morphological post-processing to mitigate this, but found no single static operator that could safely restore circularity without causing unacceptable Dice loss or N/C degradation.

## Reproducibility
All perturbations use deterministic, hashed seeds anchored to unique cell IDs to ensure exact reproducibility across multiple runs. Clustered bootstrapping is used for all confidence intervals to account for correlated cell observations across seeds and severities.
