"""Phase C — Single fixed post-processing test.

Candidate operations (all 3×3 kernel, single iteration):
    1. morphological opening
    2. morphological closing
    3. opening → closing
    4. closing → opening

Selection protocol (VALIDATION ONLY):
    - Candidate that most reduces Gaussian-noise circularity absolute error
    - Subject to safety constraints:
        * clean Dice loss <= 0.005
        * noisy Dice loss <= 0.005
        * N/C error worsening <= 5%
        * invalid rate does not increase

If no candidate passes all constraints: NO_SAFE_POSTPROCESSING.
If multiple pass: pick lowest validation circularity error.

Selection is frozen in postprocessing_selection.json BEFORE test evaluation.

Outputs:
    results/circularity_mechanism/postprocessing_selection.json
    results/circularity_mechanism/postprocessing_test_results.csv
    results/circularity_mechanism/postprocessing_paired_differences.csv

CRITICAL: Post-processing requires running inference because prediction masks
are not stored on disk. This is allowed under the constraints because it is
purely deterministic re-inference from existing frozen checkpoints — no new
training, no parameter changes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.measurements.morphometry import compute_circularity, compute_nc_ratio, relative_error
from src.pilot.metrics import dice_score, foreground_dice, restore_prediction_to_original_size
from src.segmentation.unet_model import UNet
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.data_prep.load_herlev import load_image_and_masks, list_herlev_images
from src.degradation.apply_degradations import apply_gaussian_noise
from src.segmentation.train_unet import preprocess_rgb_image

CONFIG_PATH = ROOT / "configs" / "pilot.yaml"
SPLIT_FILE = ROOT / "data" / "splits" / "original_herlev_group_split.json"
CHECKPOINT_DIR = ROOT / "results" / "pilot" / "fair_repeat" / "checkpoints"
OUT_DIR = ROOT / "results" / "circularity_mechanism"

MODELS = {"baseline": 0.0, "nc_aware_lambda_0p10": 0.10}
FAIR_SEEDS = [20260803, 20260804, 20260805]
NOISE_LEVELS = [10.0, 20.0, 30.0]

# 3×3 structuring element (fixed per protocol)
KERNEL_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

CANDIDATES = {
    "opening":          lambda m: cv2.morphologyEx(m, cv2.MORPH_OPEN,  KERNEL_3X3, iterations=1),
    "closing":          lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_3X3, iterations=1),
    "opening_closing":  lambda m: cv2.morphologyEx(
                            cv2.morphologyEx(m, cv2.MORPH_OPEN,  KERNEL_3X3, iterations=1),
                            cv2.MORPH_CLOSE, KERNEL_3X3, iterations=1),
    "closing_opening":  lambda m: cv2.morphologyEx(
                            cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_3X3, iterations=1),
                            cv2.MORPH_OPEN,  KERNEL_3X3, iterations=1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_noise_seed(cell_id: str, severity: float) -> int:
    key = f"{cell_id}|gaussian_noise|{float(severity):g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little", signed=False)


def _load_split_ids() -> dict[str, list[str]]:
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return data


def _load_checkpoint(model_name: str, seed: int, config: dict, device: torch.device):
    path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}_best_val_dice.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _apply_postprocessing(pred_nucleus: np.ndarray, op_fn) -> np.ndarray:
    """Apply morphological operation to nucleus prediction mask."""
    m_u8 = pred_nucleus.astype(np.uint8)
    result = op_fn(m_u8)
    return result.astype(bool)


def _cell_metrics(
    pred_nucleus: np.ndarray,
    pred_cytoplasm: np.ndarray,
    gt_nucleus: np.ndarray,
    gt_cytoplasm: np.ndarray,
) -> dict:
    """Compute metrics for a single cell prediction."""
    fg_dice_val = foreground_dice(
        np.where(pred_nucleus, 2, np.where(pred_cytoplasm, 1, 0)).astype(np.uint8),
        np.where(gt_nucleus, 2, np.where(gt_cytoplasm, 1, 0)).astype(np.uint8),
    )
    nuc_dice_val = dice_score(pred_nucleus, gt_nucleus)

    try:
        gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
    except ValueError:
        gt_nc = float("nan")

    try:
        gt_circ = compute_circularity(gt_nucleus)
    except ValueError:
        gt_circ = float("nan")

    try:
        pred_nc = compute_nc_ratio(pred_nucleus, pred_cytoplasm)
        nc_abs_err = abs(pred_nc - gt_nc)
        invalid_nc = False
    except ValueError:
        pred_nc = float("nan")
        nc_abs_err = float("inf")
        invalid_nc = True

    try:
        pred_circ = compute_circularity(pred_nucleus)
        circ_abs_err = abs(pred_circ - gt_circ)
        invalid_circ = False
    except ValueError:
        pred_circ = float("nan")
        circ_abs_err = float("inf")
        invalid_circ = True

    return {
        "foreground_dice": fg_dice_val,
        "nucleus_dice": nuc_dice_val,
        "gt_nc": gt_nc,
        "pred_nc": pred_nc,
        "nc_absolute_error": nc_abs_err,
        "invalid_nc": invalid_nc,
        "gt_circularity": gt_circ,
        "pred_circularity": pred_circ,
        "circularity_absolute_error": circ_abs_err,
        "invalid_circularity": invalid_circ,
        "gt_nucleus_area": int(gt_nucleus.sum()),
        "pred_nucleus_area": int(pred_nucleus.sum()),
        "nucleus_area_absolute_error": abs(int(pred_nucleus.sum()) - int(gt_nucleus.sum())),
        "invalid_prediction": not pred_nucleus.any(),
    }


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def _run_inference_for_cells(
    cell_ids: list[str],
    split_name: str,
    config: dict,
    image_index: dict,
    image_size: tuple,
    device: torch.device,
    conditions: list[tuple[str, str]],
) -> list[dict]:
    """Run inference for all models/seeds/conditions over given cell_ids.

    Returns list of raw metric dicts (no post-processing applied here).
    """
    rows = []
    for seed in FAIR_SEEDS:
        for model_name in MODELS:
            print(f"  Loading checkpoint: {model_name}, seed {seed} ...")
            try:
                model = _load_checkpoint(model_name, seed, config, device)
            except FileNotFoundError as e:
                print(f"    SKIP: {e}")
                continue

            for cell_id in cell_ids:
                if cell_id not in image_index:
                    continue
                try:
                    image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_index[cell_id])
                except Exception as e:
                    print(f"    WARNING: {cell_id}: {e}")
                    continue

                for degradation, severity_str in conditions:
                    if degradation == "clean":
                        img_input = image.copy()
                    else:
                        sev = float(severity_str)
                        ns = _stable_noise_seed(cell_id, sev)
                        img_input = apply_gaussian_noise(image, sev, seed=ns)

                    with torch.no_grad():
                        logits = model(
                            preprocess_rgb_image(img_input, size=image_size).unsqueeze(0).to(device)
                        )
                        pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    pred_target = restore_prediction_to_original_size(
                        pred_resized, image.shape[:2], image_size
                    )
                    pred_nucleus = pred_target == 2
                    pred_cytoplasm = pred_target == 1

                    base_metrics = _cell_metrics(pred_nucleus, pred_cytoplasm, gt_nucleus, gt_cytoplasm)
                    rows.append({
                        "split": split_name,
                        "cell_id": cell_id,
                        "seed": seed,
                        "model": model_name,
                        "degradation": degradation,
                        "severity": severity_str,
                        "pred_nucleus": pred_nucleus,      # kept in memory temporarily
                        "pred_cytoplasm": pred_cytoplasm,  # kept in memory temporarily
                        "gt_nucleus": gt_nucleus,
                        "gt_cytoplasm": gt_cytoplasm,
                        **base_metrics,
                    })
    return rows


# ---------------------------------------------------------------------------
# Validation selection
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    op_name: str,
    val_rows: list[dict],
) -> dict:
    """Compute candidate metrics on validation rows for both clean and noisy conditions."""
    import pandas as pd

    op_fn = CANDIDATES[op_name]
    results = {"op_name": op_name, "valid": True, "failure_reason": ""}

    clean_rows = [r for r in val_rows if r["degradation"] == "clean"]
    noise_rows = [r for r in val_rows if r["degradation"] == "gaussian_noise"]

    def _metrics_for_subset(rows, tag):
        raw_dice, pp_dice = [], []
        raw_circ_err, pp_circ_err = [], []
        raw_nc_err, pp_nc_err = [], []
        raw_invalid, pp_invalid = 0, 0

        for r in rows:
            pred_nuc = r["pred_nucleus"]
            gt_nuc = r["gt_nucleus"]
            gt_cyto = r["gt_cytoplasm"]

            # Post-processed nucleus
            pp_nuc = _apply_postprocessing(pred_nuc, op_fn)
            if not pp_nuc.any():
                pp_invalid += 1
                pp_nuc = pred_nuc  # fallback to raw for this cell

            pp_cyto = gt_cyto & ~pp_nuc  # recompute cytoplasm

            # Raw metrics (already computed)
            raw_dice.append(r["nucleus_dice"])
            raw_circ_err.append(r["circularity_absolute_error"] if not r["invalid_circularity"] else float("nan"))
            raw_nc_err.append(r["nc_absolute_error"] if not r["invalid_nc"] else float("nan"))
            if r["invalid_prediction"]:
                raw_invalid += 1

            # PP metrics
            pp_metrics = _cell_metrics(pp_nuc, pp_cyto, gt_nuc, gt_cyto)
            pp_dice.append(pp_metrics["nucleus_dice"])
            pp_circ_err.append(pp_metrics["circularity_absolute_error"] if not pp_metrics["invalid_circularity"] else float("nan"))
            pp_nc_err.append(pp_metrics["nc_absolute_error"] if not pp_metrics["invalid_nc"] else float("nan"))
            if pp_metrics["invalid_prediction"]:
                pp_invalid += 1

        n = len(rows)
        return {
            f"{tag}_raw_mean_dice": float(np.nanmean(raw_dice)) if raw_dice else float("nan"),
            f"{tag}_pp_mean_dice": float(np.nanmean(pp_dice)) if pp_dice else float("nan"),
            f"{tag}_dice_change": float(np.nanmean(pp_dice) - np.nanmean(raw_dice)) if raw_dice else float("nan"),
            f"{tag}_raw_mean_circ_err": float(np.nanmean(raw_circ_err)),
            f"{tag}_pp_mean_circ_err": float(np.nanmean(pp_circ_err)),
            f"{tag}_circ_err_change": float(np.nanmean(pp_circ_err) - np.nanmean(raw_circ_err)),
            f"{tag}_raw_mean_nc_err": float(np.nanmean(raw_nc_err)),
            f"{tag}_pp_mean_nc_err": float(np.nanmean(pp_nc_err)),
            f"{tag}_nc_err_change": float(np.nanmean(pp_nc_err) - np.nanmean(raw_nc_err)),
            f"{tag}_raw_invalid_rate": raw_invalid / max(n, 1),
            f"{tag}_pp_invalid_rate": pp_invalid / max(n, 1),
            f"{tag}_n": n,
        }

    results.update(_metrics_for_subset(clean_rows, "clean"))
    results.update(_metrics_for_subset(noise_rows, "noise"))
    return results


def _check_safety(cand: dict) -> tuple[bool, list[str]]:
    """Check all safety constraints. Returns (passes, list_of_failures).

    Key names from _evaluate_candidate / _metrics_for_subset:
        clean_dice_change, noise_dice_change,
        noise_raw_mean_nc_err, noise_pp_mean_nc_err,
        noise_pp_invalid_rate, noise_raw_invalid_rate
    """
    failures = []
    # Constraint 1: clean Dice loss <= 0.005
    clean_dice_chg = cand.get("clean_dice_change", 0.0)
    if isinstance(clean_dice_chg, float) and not np.isnan(clean_dice_chg):
        if abs(clean_dice_chg) > 0.005:
            failures.append(f"clean_dice_change={clean_dice_chg:.4f} > 0.005")
    # Constraint 2: noisy Dice loss <= 0.005
    noise_dice_chg = cand.get("noise_dice_change", 0.0)
    if isinstance(noise_dice_chg, float) and not np.isnan(noise_dice_chg):
        if abs(noise_dice_chg) > 0.005:
            failures.append(f"noise_dice_change={noise_dice_chg:.4f} > 0.005")
    # Constraint 3: N/C error worsening <= 5%
    raw_nc = cand.get("noise_raw_mean_nc_err", float("nan"))
    pp_nc = cand.get("noise_pp_mean_nc_err", float("nan"))
    if not np.isnan(raw_nc) and not np.isnan(pp_nc) and raw_nc > 0:
        nc_pct_change = (pp_nc - raw_nc) / raw_nc
        if nc_pct_change > 0.05:
            failures.append(f"nc_err_pct_change={nc_pct_change:.3f} > 0.05")
    # Constraint 4: invalid rate not increasing
    if cand.get("noise_pp_invalid_rate", 0) > cand.get("noise_raw_invalid_rate", 0) + 1e-9:
        failures.append("invalid_rate_increased")
    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Main: validation selection + test evaluation
# ---------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> None:
    """Run post-processing selection (validation) and test evaluation."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    herlev_dir = resolve_herlev_data_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(config["image_size"])

    splits = _load_split_ids()
    image_index = {p.stem: p for p in list_herlev_images(herlev_dir)}

    conditions_clean = [("clean", "clean")]
    conditions_noise = [(f"gaussian_noise", str(s)) for s in NOISE_LEVELS]
    conditions_all = conditions_clean + conditions_noise

    # ---- VALIDATION SELECTION ----
    print("\n=== VALIDATION: Running inference on validation cells ===")
    val_ids = splits.get("unet_val", [])
    val_rows = _run_inference_for_cells(
        val_ids, "unet_val", config, image_index, image_size, device, conditions_all
    )
    print(f"  {len(val_rows)} inference rows collected.")

    print("\n=== VALIDATION: Evaluating post-processing candidates ===")
    candidate_results = []
    for op_name in CANDIDATES:
        print(f"  Evaluating: {op_name} ...")
        cand = _evaluate_candidate(op_name, val_rows)
        # Check if circularity error is reduced
        circ_reduction = -(cand["noise_circ_err_change"])  # positive = improvement
        cand["noise_circ_err_reduction"] = circ_reduction
        cand["safety_ok"], cand["safety_failures"] = _check_safety(cand)
        candidate_results.append(cand)
        print(f"    circ_err_change={cand['noise_circ_err_change']:.4f}, "
              f"safe={cand['safety_ok']}")

    # Filter to safe candidates
    safe_candidates = [c for c in candidate_results if c["safety_ok"]]

    if not safe_candidates:
        decision = "NO_SAFE_POSTPROCESSING"
        selected_op = None
        print(f"\nDecision: {decision}")
        print("No candidate met all safety constraints on validation data.")
    else:
        # Pick the one with highest noise circularity error reduction
        best = max(safe_candidates, key=lambda c: c["noise_circ_err_reduction"])
        decision = best["op_name"]
        selected_op = decision
        print(f"\nDecision: {decision}")
        print(f"  Noise circularity error reduction: {best['noise_circ_err_reduction']:.4f}")

    # Save selection (FROZEN — do not modify after this point)
    selection = {
        "decision": decision,
        "selected_op": selected_op,
        "candidates": candidate_results,
        "selection_set": "unet_val",
        "test_not_seen": True,
        "kernel": "3x3_ellipse",
        "iterations": 1,
    }
    # Remove numpy arrays (not JSON serialisable)
    for cand in selection["candidates"]:
        cand.pop("safety_failures_list", None)
        if isinstance(cand.get("safety_failures"), list):
            cand["safety_failures"] = "; ".join(cand["safety_failures"])

    sel_path = out_dir / "postprocessing_selection.json"
    sel_path.write_text(json.dumps(selection, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Selection frozen: {sel_path}")

    if selected_op is None:
        print("Skipping test evaluation (NO_SAFE_POSTPROCESSING).")
        return

    # ---- TEST EVALUATION ----
    print("\n=== TEST: Running inference on test cells ===")
    test_ids = splits.get("test", [])
    test_rows = _run_inference_for_cells(
        test_ids, "test", config, image_index, image_size, device, conditions_all
    )
    print(f"  {len(test_rows)} test inference rows.")

    op_fn = CANDIDATES[selected_op]
    test_result_rows = []
    test_paired_rows = []

    for r in test_rows:
        pred_nuc = r["pred_nucleus"]
        gt_nuc = r["gt_nucleus"]
        gt_cyto = r["gt_cytoplasm"]

        pp_nuc = _apply_postprocessing(pred_nuc, op_fn)
        if not pp_nuc.any():
            pp_nuc = pred_nuc  # fallback
        pp_cyto = gt_cyto & ~pp_nuc

        raw_m = {k: v for k, v in r.items() if k not in ("pred_nucleus", "pred_cytoplasm", "gt_nucleus", "gt_cytoplasm")}
        pp_m = _cell_metrics(pp_nuc, pp_cyto, gt_nuc, gt_cyto)

        test_result_rows.append({**raw_m, "processing": "raw"})
        test_result_rows.append({
            "split": r["split"],
            "cell_id": r["cell_id"],
            "seed": r["seed"],
            "model": r["model"],
            "degradation": r["degradation"],
            "severity": r["severity"],
            "processing": "post_processed",
            **pp_m,
        })

        # Paired difference (pp - raw)
        for metric, raw_val, pp_val in [
            ("nucleus_dice",              r["nucleus_dice"], pp_m["nucleus_dice"]),
            ("circularity_absolute_error", r["circularity_absolute_error"], pp_m["circularity_absolute_error"]),
            ("nc_absolute_error",          r["nc_absolute_error"], pp_m["nc_absolute_error"]),
            ("nucleus_area_absolute_error", r["nucleus_area_absolute_error"], pp_m["nucleus_area_absolute_error"]),
        ]:
            if np.isfinite(raw_val) and np.isfinite(pp_val):
                test_paired_rows.append({
                    "split": r["split"],
                    "cell_id": r["cell_id"],
                    "seed": r["seed"],
                    "model": r["model"],
                    "degradation": r["degradation"],
                    "severity": r["severity"],
                    "metric": metric,
                    "raw_value": raw_val,
                    "pp_value": pp_val,
                    "difference_pp_minus_raw": pp_val - raw_val,
                })

    results_df = pd.DataFrame(test_result_rows)
    results_path = out_dir / "postprocessing_test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Written: {results_path}  ({len(results_df):,} rows)")

    paired_df = pd.DataFrame(test_paired_rows)
    paired_path = out_dir / "postprocessing_paired_differences.csv"
    paired_df.to_csv(paired_path, index=False)
    print(f"Written: {paired_path}  ({len(paired_df):,} rows)")


if __name__ == "__main__":
    run()
