# Cervical Morphometry-Aware Segmentation Analysis

## Project Purpose
This repository provides a controlled evaluation of morphometric reliability, conformal prediction coverage, and a deep circularity mechanism analysis for cervical cell segmentation. The project investigates whether adding a differentiable N/C-ratio target reduces N/C measurement error without harming segmentation performance, and provides evidence explaining why circularity predictions degrade under Gaussian noise while N/C ratio predictions remain comparatively more robust than circularity under Gaussian noise.

## Data, Privacy, and Clinical Disclaimer
* **Herlev dataset is NOT included in this repository.**
* The user must obtain the data from an authorized and official source.
* **This repository contains NO patient or personal data.**
* The repository only contains source code, configuration files, and derived anonymous statistical results.
* **The results of this study DO NOT constitute clinical validation or approval for clinical use.** It is strictly a controlled evaluation and mechanism analysis.

To run the analysis, provide the dataset locally via the `HERLEV_DATA_DIR` environment variable:
* **Windows (Command Prompt):** `set HERLEV_DATA_DIR=C:\path\to\herlev\dataset`
* **Windows (PowerShell):** `$env:HERLEV_DATA_DIR="C:\path\to\herlev\dataset"`
* **Linux/macOS:** `export HERLEV_DATA_DIR=/path/to/herlev/dataset`

## Environment Setup
The project uses standard PyTorch and data science libraries.
```bash
pip install -r requirements.txt
```
To reproduce the exact verified environment, refer to `requirements-lock.txt`.

## Branch Structure
The project maintains a safe, strict separation of concerns via branching:
- `main`: The secure baseline containing the finalized target-specific conformal coverage and pilot results.
- `feature/circularity-mechanism-analysis`: Contains the deep-dive mechanism analysis evaluating boundary vs. shape perturbation under noise.
- `manuscript/v1`: The canonical manuscript branch containing the fully validated statistical mechanisms, deterministic bootstrapping, and evidence manifests. This branch represents all repository documentation moving forward.

## Scientific Findings Summary
The circularity mechanism analysis provides evidence that under Gaussian noise, high-frequency boundary irregularities cause severe degradation of the perimeter measurements. Circularity degradation was more strongly associated with direct perimeter error ($\rho = 0.586$) than with area error, while controlled boundary perturbations affected circularity substantially more than the N/C ratio. 

**Exact Log-Ratio Decomposition:** 
Algebraic decomposition showed that perimeter contributions exceeded area contributions in most valid clean–corrupted pairs ($65.23\%$ perimeter dominant, CI: $[62.20\%, 68.26\%]$). 

**Area-Clamped Counterfactual:** 
An area-preserving counterfactual boundary perturbation was matched deterministically against shape perturbations (erosion/dilation) strictly within the same cell and Dice band. The cell-level clustered 95% Confidence Interval for the difference was $[-0.00668, 0.00192]$. Because the boundary perturbations did not consistently exceed the shape perturbations when area was fully restricted, the decision `ADDITIONAL_SUPPORT_NOT_OBSERVED` was reached. 

**Limitation:** 
The area-clamped counterfactual analysis did not provide additional evidence that boundary perturbations produce greater circularity change than matched shape perturbations. This limits a strictly area-independent interpretation without invalidating the observed perimeter association. The mechanism may involve combined boundary and area changes. N/C ratio predictions are not entirely unaffected by noise, but they remain comparatively more robust than circularity. Note that test-set predictions were used during the sensitivity analysis phase, meaning the test set is no longer "untouched" for future unrelated benchmarking.

## End-to-End Reproducibility
All perturbations use deterministic, hashed seeds anchored to unique cell IDs to ensure exact reproducibility. 

**Command Sequence:**
1. Preparation of existing frozen prediction and direct-mask evaluation tables.
2. `python src/mask_perturbation.py` (Creates the controlled perturbation CSV).
3. `python experiments/run_counterfactual_analysis.py` (Runs the exact decomposition and area-preserving matched counterfactual runner, outputting to JSON/CSV manifests).
4. `python -m pytest tests/ -v` (Validates deterministic logic, leakages, and topological constraints).
