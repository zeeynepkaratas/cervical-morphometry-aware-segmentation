"""Run the three-seed fair paired N/C-aware pilot repeat."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.losses.morphometry_loss import combined_segmentation_nc_loss
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.fair_repeat import (
    FAIR_LAMBDAS,
    FAIR_SEEDS,
    MAX_EPOCHS,
    MIN_DELTA,
    PATIENCE,
    RunSpec,
    decide,
    evaluate_model,
    paired_differences,
    paired_summary,
    prepare_initial_state,
    seed_everything,
    summarize_rows,
    train_order_hash,
    write_csv,
)
from src.segmentation.train_unet import HerlevSegmentationDataset
from src.segmentation.train_unet import preprocess_rgb_image
from src.segmentation.unet_model import UNet
from src.pilot.metrics import masks_to_target, restore_prediction_to_original_size


def _device(config: dict) -> torch.device:
    return torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_split(config: dict) -> dict:
    return json.loads((ROOT_DIR / config["splits"]["pilot_split"]).read_text(encoding="utf-8"))


def _load_resume(path: Path, model: UNet, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, float, int, list[dict]]:
    if not path.exists():
        return 1, -1.0, 0, []
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_val_dice"]),
        int(checkpoint["epochs_without_improvement"]),
        checkpoint.get("history", []),
    )


def train_run(config: dict, data_dir: Path, spec: RunSpec) -> dict:
    """Train one fair-repeat run with epoch-level resume."""
    out_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat"
    ckpt_dir = out_dir / "checkpoints"
    init_dir = out_dir / "initial_states"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split = _load_split(config)
    train_ids = split["train_ids"]
    val_ids = split["validation_ids"]
    image_size = tuple(config["image_size"])
    batch_size = int(config["training"]["batch_size"])
    base_channels = int(config["training"]["base_channels"])
    device = _device(config)

    init_path, init_hash = prepare_initial_state(spec.seed, base_channels, init_dir)
    seed_everything(spec.seed)
    model = UNet(base_channels=base_channels).to(device)
    init_payload = torch.load(init_path, map_location=device)
    model.load_state_dict(init_payload["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    train_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=train_ids, image_size=image_size)
    order_hash = train_order_hash(train_ds, batch_size, spec.seed)
    val_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=val_ids, image_size=image_size)

    last_path = ckpt_dir / f"{spec.run_id}_last.pt"
    start_epoch, best_val_dice, epochs_without_improvement, history = _load_resume(last_path, model, optimizer, device)
    if start_epoch > MAX_EPOCHS:
        return _run_result(spec, history, init_hash, order_hash, resumed=True)

    run_start = time.time()
    resumed = start_epoch > 1
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        totals = {"total": 0.0, "seg": 0.0, "nc": 0.0}
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(spec.seed + epoch),
        )
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            parts = combined_segmentation_nc_loss(model(images), targets, lambda_nc=spec.lambda_nc)
            parts.total_loss.backward()
            optimizer.step()
            totals["total"] += float(parts.total_loss.item())
            totals["seg"] += float(parts.segmentation_loss.item())
            totals["nc"] += float(parts.nc_loss.item())
        n_batches = max(1, len(train_loader))
        val_rows = evaluate_model(model, val_ids, data_dir, image_size, device, seed=spec.seed, model_name=spec.model)
        val_summary = summarize_rows(val_rows)
        epoch_seconds = time.time() - epoch_start
        row = {
            "seed": spec.seed,
            "model": spec.model,
            "lambda_nc": spec.lambda_nc,
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "estimated_remaining_seconds": (MAX_EPOCHS - epoch) * epoch_seconds,
            "train_total_loss": totals["total"] / n_batches,
            "train_segmentation_loss": totals["seg"] / n_batches,
            "train_raw_nc_loss": totals["nc"] / n_batches,
            "validation_foreground_dice": val_summary["foreground_dice"],
            "validation_nucleus_dice": val_summary["nucleus_dice"],
            "validation_cytoplasm_dice": val_summary["cytoplasm_dice"],
            "validation_mean_nc_absolute_error": val_summary["mean_nc_absolute_error"],
            "validation_median_nc_absolute_error": val_summary["median_nc_absolute_error"],
            "validation_invalid_nc_rate": val_summary["invalid_nc_rate"],
            "initial_state_sha256": init_hash,
            "train_order_sha256": order_hash,
        }
        history.append(row)
        if epoch == 1:
            _append_loss_diag(out_dir, row)
        print(
            f"{spec.run_id} epoch {epoch:03d}/{MAX_EPOCHS:03d} "
            f"epoch_seconds={epoch_seconds:.1f} eta_seconds={row['estimated_remaining_seconds']:.1f} "
            f"train_total={row['train_total_loss']:.4f} seg={row['train_segmentation_loss']:.4f} "
            f"nc={row['train_raw_nc_loss']:.4f} val_fg_dice={row['validation_foreground_dice']:.4f} "
            f"val_nc_mean={row['validation_mean_nc_absolute_error']:.4f} "
            f"val_nc_median={row['validation_median_nc_absolute_error']:.4f}",
            flush=True,
        )
        improved = row["validation_foreground_dice"] > best_val_dice + MIN_DELTA
        if improved:
            best_val_dice = row["validation_foreground_dice"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": spec.seed,
                    "lambda_nc": spec.lambda_nc,
                    "model": spec.model,
                    "best_val_dice": best_val_dice,
                    "initial_state_sha256": init_hash,
                    "initial_state_path": str(init_path),
                    "train_order_sha256": order_hash,
                    "config": _checkpoint_config(config),
                },
                ckpt_dir / f"{spec.run_id}_best_val_dice.pt",
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "seed": spec.seed,
                "lambda_nc": spec.lambda_nc,
                "model": spec.model,
                "best_val_dice": best_val_dice,
                "epochs_without_improvement": epochs_without_improvement,
                "initial_state_sha256": init_hash,
                "initial_state_path": str(init_path),
                "train_order_sha256": order_hash,
                "history": history,
                "config": _checkpoint_config(config),
            },
            last_path,
        )
        _write_history(out_dir, spec, history, time.time() - run_start, init_hash, order_hash)
        if epochs_without_improvement >= PATIENCE:
            break
    return _run_result(spec, history, init_hash, order_hash, resumed=resumed)


def _checkpoint_config(config: dict) -> dict:
    return {
        "image_size": config["image_size"],
        "base_channels": int(config["training"]["base_channels"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_size": int(config["training"]["batch_size"]),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "primary_checkpoint_rule": "best_val_dice",
    }


def _append_loss_diag(out_dir: Path, row: dict) -> None:
    path = out_dir / "loss_scale_diagnostic.csv"
    weighted = float(row["lambda_nc"]) * float(row["train_raw_nc_loss"])
    total = float(row["train_segmentation_loss"]) + weighted
    diag = {
        "seed": row["seed"],
        "model": row["model"],
        "epoch": row["epoch"],
        "mean_segmentation_loss": row["train_segmentation_loss"],
        "mean_raw_nc_loss": row["train_raw_nc_loss"],
        "weighted_nc_loss": weighted,
        "mean_total_loss": total,
        "weighted_nc_fraction": weighted / total if total else 0.0,
    }
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diag.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(diag)


def _write_history(out_dir: Path, spec: RunSpec, history: list[dict], elapsed: float, init_hash: str, order_hash: str) -> None:
    path = out_dir / f"training_history_{spec.run_id}.json"
    path.write_text(
        json.dumps(
            {
                "run_id": spec.run_id,
                "seed": spec.seed,
                "lambda_nc": spec.lambda_nc,
                "initial_state_sha256": init_hash,
                "train_order_sha256": order_hash,
                "primary_checkpoint_rule": "best_val_dice",
                "elapsed_seconds": elapsed,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_result(spec: RunSpec, history: list[dict], init_hash: str, order_hash: str, resumed: bool) -> dict:
    return {
        "seed": spec.seed,
        "model": spec.model,
        "run_id": spec.run_id,
        "completed_epochs": max([row["epoch"] for row in history], default=0),
        "resumed": resumed,
        "initial_state_sha256": init_hash,
        "train_order_sha256": order_hash,
        "training_time": sum(float(row.get("epoch_seconds", 0)) for row in history),
    }


def evaluate_best_checkpoints(config: dict, data_dir: Path) -> list[dict]:
    """Evaluate all best-val-Dice checkpoints on clean validation cells."""
    out_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat"
    ckpt_dir = out_dir / "checkpoints"
    split = _load_split(config)
    rows = []
    device = _device(config)
    for seed in FAIR_SEEDS:
        for lambda_nc in FAIR_LAMBDAS:
            spec = RunSpec(seed=seed, lambda_nc=lambda_nc)
            path = ckpt_dir / f"{spec.run_id}_best_val_dice.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing best-val-Dice checkpoint: {path}")
            checkpoint = torch.load(path, map_location=device)
            model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model_rows = evaluate_model(
                model,
                split["validation_ids"],
                data_dir,
                tuple(config["image_size"]),
                device,
                seed=seed,
                model_name=spec.model,
            )
            for row in model_rows:
                row["best_epoch"] = int(checkpoint["epoch"])
            rows.extend(model_rows)
    write_csv(out_dir / "per_cell_predictions.csv", rows)
    return rows


def summarize_all(per_cell_rows: list[dict], run_results: list[dict], config: dict, data_dir: Path) -> dict:
    """Create all fair-repeat summaries and plots."""
    out_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat"
    pair_rows = paired_differences(per_cell_rows)
    write_csv(out_dir / "paired_cell_differences.csv", pair_rows)
    by_seed_model = []
    grouped = defaultdict(list)
    for row in per_cell_rows:
        grouped[(row["seed"], row["model"])].append(row)
    history_by_run = _history_index(out_dir)
    for (seed, model), rows in sorted(grouped.items()):
        summary = summarize_rows(rows)
        run_id = f"{model}_seed_{seed}"
        summary.update(
            seed=seed,
            model=model,
            best_epoch=int(rows[0].get("best_epoch", 0)),
            training_time=history_by_run.get(run_id, {}).get("training_time", float("nan")),
        )
        by_seed_model.append(summary)
    deltas = []
    for seed in FAIR_SEEDS:
        b = next(row for row in by_seed_model if row["seed"] == seed and row["model"] == "baseline")
        n = next(row for row in by_seed_model if row["seed"] == seed and row["model"] == "nc_aware_lambda_0p10")
        deltas.append(
            {
                "seed": seed,
                "model": "paired_delta",
                "foreground_dice_drop": b["foreground_dice"] - n["foreground_dice"],
                "mean_nc_absolute_error_delta": n["mean_nc_absolute_error"] - b["mean_nc_absolute_error"],
                "median_nc_absolute_error_delta": n["median_nc_absolute_error"] - b["median_nc_absolute_error"],
                "invalid_nc_rate_delta": n["invalid_nc_rate"] - b["invalid_nc_rate"],
                "mean_circularity_absolute_error_delta": n["mean_circularity_absolute_error"] - b["mean_circularity_absolute_error"],
            }
        )
    write_csv(out_dir / "metrics_by_seed.csv", by_seed_model + deltas)
    class_rows = _class_summary(per_cell_rows, pair_rows)
    write_csv(out_dir / "metrics_by_class.csv", class_rows)
    pair_summaries = [paired_summary([r for r in pair_rows if r["seed"] == seed], f"seed_{seed}") for seed in FAIR_SEEDS]
    combined_pair = paired_summary(pair_rows, "combined")
    decision, reasons = decide(deltas, combined_pair)
    summary = {
        "run_results": run_results,
        "metrics_by_seed": by_seed_model,
        "paired_seed_summaries": pair_summaries,
        "paired_combined_summary": combined_pair,
        "decision": decision,
        "decision_reasons": reasons,
    }
    write_csv(out_dir / "fair_repeat_summary.csv", pair_summaries + [combined_pair])
    (out_dir / "fair_repeat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_decision_md(out_dir / "fair_repeat_decision.md", summary)
    _make_plots(out_dir, per_cell_rows, pair_rows, by_seed_model, data_dir, config)
    return summary


def _history_index(out_dir: Path) -> dict[str, dict]:
    results = {}
    for path in out_dir.glob("training_history_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        results[payload["run_id"]] = {
            "training_time": sum(float(row.get("epoch_seconds", 0)) for row in payload.get("history", [])),
        }
    return results


def _read_per_cell_csv(path: Path) -> list[dict]:
    """Read per-cell CSV with numeric and boolean fields restored."""
    numeric_fields = {
        "seed",
        "nucleus_dice",
        "cytoplasm_dice",
        "foreground_dice",
        "gt_nucleus_area",
        "pred_nucleus_area",
        "nucleus_area_absolute_error",
        "gt_cytoplasm_area",
        "pred_cytoplasm_area",
        "cytoplasm_area_absolute_error",
        "gt_nc",
        "pred_nc",
        "nc_absolute_error",
        "nc_relative_error",
        "gt_circularity",
        "pred_circularity",
        "circularity_absolute_error",
        "best_epoch",
    }
    bool_fields = {"invalid_nc", "invalid_circularity"}
    rows = []
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        for key in numeric_fields & set(row):
            row[key] = float(row[key]) if row[key] not in {"", "nan"} else float("nan")
            if key in {"seed", "gt_nucleus_area", "pred_nucleus_area", "gt_cytoplasm_area", "pred_cytoplasm_area", "best_epoch"}:
                row[key] = int(row[key])
        for key in bool_fields & set(row):
            row[key] = row[key] == "True"
        rows.append(row)
    return rows


def _class_summary(per_cell_rows: list[dict], pair_rows: list[dict]) -> list[dict]:
    out = []
    classes = sorted({row["class"] for row in per_cell_rows})
    for cls in classes:
        base = [row for row in per_cell_rows if row["class"] == cls and row["model"] == "baseline"]
        nc = [row for row in per_cell_rows if row["class"] == cls and row["model"] == "nc_aware_lambda_0p10"]
        pairs = [row for row in pair_rows if row["class"] == cls]
        base_s = summarize_rows(base)
        nc_s = summarize_rows(nc)
        rel = (
            (nc_s["mean_nc_absolute_error"] - base_s["mean_nc_absolute_error"]) / base_s["mean_nc_absolute_error"]
            if base_s["mean_nc_absolute_error"]
            else float("nan")
        )
        out.append(
            {
                "class": cls,
                "n": len(base),
                "baseline_mean_nc_error": base_s["mean_nc_absolute_error"],
                "baseline_median_nc_error": base_s["median_nc_absolute_error"],
                "nc_aware_mean_nc_error": nc_s["mean_nc_absolute_error"],
                "nc_aware_median_nc_error": nc_s["median_nc_absolute_error"],
                "relative_change": rel,
                "win_rate": float(np.mean([row["delta_nc_error"] < 0 for row in pairs])) if pairs else float("nan"),
                "dice_change": nc_s["foreground_dice"] - base_s["foreground_dice"],
                "circularity_error_change": nc_s["mean_circularity_absolute_error"] - base_s["mean_circularity_absolute_error"],
            }
        )
    return out


def _write_decision_md(path: Path, summary: dict) -> None:
    combined = summary["paired_combined_summary"]
    text = [
        "# Fair Repeat Decision",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"Combined N/C-aware win rate: `{combined.get('win_rate_nc_aware')}`",
        f"Bootstrap 95% CI for mean delta: `{combined.get('bootstrap_ci_low')}`, `{combined.get('bootstrap_ci_high')}`",
        f"Permutation p-value: `{combined.get('permutation_pvalue')}`",
        "",
        "Reasons:",
        *[f"- {reason}" for reason in summary["decision_reasons"]],
        "",
        "Negative delta means N/C-aware improved N/C absolute error.",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def _make_plots(out_dir: Path, per_cell_rows: list[dict], pair_rows: list[dict], by_seed_model: list[dict], data_dir: Path, config: dict) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds = FAIR_SEEDS
    width = 0.35
    for metric, filename, ylabel in [
        ("mean_nc_absolute_error", "seed_mean_nc_error.png", "Mean N/C absolute error"),
        ("median_nc_absolute_error", "seed_median_nc_error.png", "Median N/C absolute error"),
    ]:
        plt.figure(figsize=(7, 4))
        x = np.arange(len(seeds))
        base = [next(row[metric] for row in by_seed_model if row["seed"] == s and row["model"] == "baseline") for s in seeds]
        nc = [next(row[metric] for row in by_seed_model if row["seed"] == s and row["model"] == "nc_aware_lambda_0p10") for s in seeds]
        plt.bar(x - width / 2, base, width, label="baseline")
        plt.bar(x + width / 2, nc, width, label="N/C-aware")
        plt.xticks(x, [str(s) for s in seeds])
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=160)
        plt.close()
    deltas = np.array([row["delta_nc_error"] for row in pair_rows], dtype=float)
    plt.figure(figsize=(7, 4))
    plt.hist(deltas[np.isfinite(deltas)], bins=40)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Delta N/C error (N/C-aware - baseline)")
    plt.ylabel("Cells")
    plt.tight_layout()
    plt.savefig(fig_dir / "paired_delta_distribution.png", dpi=160)
    plt.close()
    plt.figure(figsize=(5, 5))
    b = [row["baseline_nc_error"] for row in pair_rows]
    n = [row["nc_aware_nc_error"] for row in pair_rows]
    plt.scatter(b, n, s=10, alpha=0.6)
    limit = max([v for v in b + n if np.isfinite(v)] or [1])
    plt.plot([0, limit], [0, limit], color="black", linewidth=1)
    plt.xlabel("Baseline N/C error")
    plt.ylabel("N/C-aware N/C error")
    plt.tight_layout()
    plt.savefig(fig_dir / "baseline_vs_nc_aware_nc_error.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.scatter([r["delta_foreground_dice"] for r in pair_rows], [r["delta_nc_error"] for r in pair_rows], s=10, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Delta foreground Dice")
    plt.ylabel("Delta N/C error")
    plt.tight_layout()
    plt.savefig(fig_dir / "dice_delta_vs_nc_delta.png", dpi=160)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.scatter([r["delta_nc_error"] for r in pair_rows], [r["delta_circularity_error"] for r in pair_rows], s=10, alpha=0.6)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Delta N/C error")
    plt.ylabel("Delta circularity error")
    plt.tight_layout()
    plt.savefig(fig_dir / "nc_delta_vs_circularity_delta.png", dpi=160)
    plt.close()
    _make_example_panels(fig_dir, per_cell_rows, pair_rows, data_dir, config, improved=True)
    _make_example_panels(fig_dir, per_cell_rows, pair_rows, data_dir, config, improved=False)


def _make_example_panels(fig_dir: Path, per_cell_rows: list[dict], pair_rows: list[dict], data_dir: Path, config: dict, improved: bool) -> None:
    selected = sorted(pair_rows, key=lambda row: row["delta_nc_error"], reverse=not improved)[:5]
    if not selected:
        return
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    ckpt_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat" / "checkpoints"
    device = _device(config)
    image_size = tuple(config["image_size"])
    fig, axes = plt.subplots(len(selected), 4, figsize=(10, 2.5 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, 0)
    model_cache = {}
    for row_index, pair in enumerate(selected):
        image_path = image_by_id[pair["cell_id"]]
        image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
        panels = [image, masks_to_target(gt_nucleus, gt_cytoplasm)]
        for model_name in ("baseline", "nc_aware_lambda_0p10"):
            key = (pair["seed"], model_name)
            if key not in model_cache:
                spec_name = f"{model_name}_seed_{pair['seed']}"
                checkpoint = torch.load(ckpt_dir / f"{spec_name}_best_val_dice.pt", map_location=device)
                model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                model_cache[key] = model
            with torch.no_grad():
                logits = model_cache[key](torch.unsqueeze(preprocess_rgb_image(image, size=image_size), 0).to(device))
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                panels.append(restore_prediction_to_original_size(pred, image.shape[:2], image_size))
        titles = ["RGB", "GT", "Baseline", "N/C-aware"]
        for col, panel in enumerate(panels):
            ax = axes[row_index, col]
            ax.imshow(panel, cmap=None if col == 0 else "viridis")
            ax.set_title(titles[col])
            ax.axis("off")
        axes[row_index, 0].set_ylabel(f"{pair['cell_id']}\nΔ={pair['delta_nc_error']:.3f}")
    plt.tight_layout()
    plt.savefig(fig_dir / ("top5_improved_panels.png" if improved else "top5_worsened_panels.png"), dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if not (args.train or args.evaluate or args.summarize):
        args.train = args.evaluate = args.summarize = True
    config = load_config(ROOT_DIR / args.config)
    data_dir = resolve_herlev_data_dir(config, args.data_dir)
    run_results = []
    out_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.train:
        for seed in FAIR_SEEDS:
            for lambda_nc in FAIR_LAMBDAS:
                run_results.append(train_run(config, data_dir, RunSpec(seed=seed, lambda_nc=lambda_nc)))
        (out_dir / "run_results.json").write_text(json.dumps(run_results, indent=2), encoding="utf-8")
    elif (out_dir / "run_results.json").exists():
        run_results = json.loads((out_dir / "run_results.json").read_text(encoding="utf-8"))
    per_cell_rows = []
    if args.evaluate:
        per_cell_rows = evaluate_best_checkpoints(config, data_dir)
    elif args.summarize:
        per_cell_rows = _read_per_cell_csv(out_dir / "per_cell_predictions.csv")
    if args.summarize:
        summary = summarize_all(per_cell_rows, run_results, config, data_dir)
        print(json.dumps({"decision": summary["decision"], "paired_combined_summary": summary["paired_combined_summary"]}, indent=2))


if __name__ == "__main__":
    main()
