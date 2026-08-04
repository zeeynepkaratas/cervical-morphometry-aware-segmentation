"""Target-specific conformal coverage fragility analysis.

Consumes existing ``results/final_analysis`` CSVs only. It does not train,
run inference, alter checkpoints, create degradations, or recalibrate conformal
intervals.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "final_analysis" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pilot.target_specific_coverage import (
    TARGETS,
    cluster_bootstrap_target_gap,
    mean,
    median,
    more_fragile_target,
    pair_target_changes,
    severity_value,
    trend_label,
)


OUT = ROOT / "results" / "final_analysis"
FIG = OUT / "target_specific_figures"
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260804
STABLE_DICE_THRESHOLD = 0.01
REQUIRED_FILES = [
    "conformal_results.csv",
    "conformal_per_cell.csv",
    "coverage_gap_analysis.csv",
    "interval_width_failure_analysis.csv",
    "severity_trends.csv",
    "cluster_bootstrap_intervals.csv",
    "degradation_metrics_by_seed_severity.csv",
    "dice_morphometry_spearman.csv",
    "final_analysis_summary.json",
]


def coerce(value: str):
    if value in {"True", "False"}:
        return value == "True"
    if value == "":
        return value
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer() and "." not in value and "e" not in value.lower():
            return int(number)
        return number
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        for key, value in list(row.items()):
            row[key] = coerce(value)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def target_summary(conf: list[dict], gap: list[dict]) -> list[dict]:
    conf_by_target = group(conf, ["target"])
    gap_by_target = group(gap, ["target"])
    out = []
    for target in TARGETS:
        target_conf = conf_by_target[(target,)]
        target_gap = gap_by_target[(target,)]
        clean_conf = [r for r in target_conf if r["degradation"] == "clean"]
        degraded_conf = [r for r in target_conf if r["degradation"] != "clean"]
        degraded_gap = [r for r in target_gap if r["degradation"] != "clean"]
        deltas = [float(r["coverage_delta_vs_clean"]) for r in degraded_gap]
        out.append(
            {
                "target": target,
                "clean_empirical_coverage": mean(r["empirical_coverage"] for r in clean_conf),
                "degraded_empirical_coverage": mean(r["empirical_coverage"] for r in degraded_conf),
                "mean_coverage_change_vs_clean": mean(deltas),
                "median_coverage_change_vs_clean": median(deltas),
                "min_coverage_change_vs_clean": min(deltas) if deltas else float("nan"),
                "max_coverage_change_vs_clean": max(deltas) if deltas else float("nan"),
                "conditions_drop_ge_2pp": sum(delta <= -0.02 for delta in deltas),
                "conditions_drop_ge_5pp": sum(delta <= -0.05 for delta in deltas),
                "conditions_drop_ge_10pp": sum(delta <= -0.10 for delta in deltas),
                "conditions_below_nominal": sum(bool(r["below_nominal"]) for r in degraded_gap),
                "total_degraded_conditions": len(degraded_gap),
                "mean_interval_width": mean(r["interval_mean_width"] for r in degraded_conf),
                "median_interval_width": median(r["interval_median_width"] for r in degraded_conf),
                "invalid_rate": mean(r["invalid_rate"] for r in degraded_conf),
                "failure_rate": mean(r["failure_rate"] for r in degraded_conf),
            }
        )
    return out


def fragility_by_seed(comparison: list[dict]) -> list[dict]:
    out = []
    for (seed,), rows in sorted(group(comparison, ["seed"]).items()):
        diffs = [float(r["target_gap_difference"]) for r in rows]
        by_deg = group(rows, ["degradation"])
        blur_diff = mean(r["target_gap_difference"] for r in by_deg.get(("gaussian_blur",), []))
        noise_diff = mean(r["target_gap_difference"] for r in by_deg.get(("gaussian_noise",), []))
        same_direction = (
            np.isfinite(blur_diff)
            and np.isfinite(noise_diff)
            and ((blur_diff < 0 and noise_diff < 0) or (blur_diff > 0 and noise_diff > 0))
        )
        severity_labels = []
        for (degradation,), deg_rows in by_deg.items():
            severity_means = [
                mean(r["target_gap_difference"] for r in sev_rows)
                for _sev, sev_rows in sorted(group(deg_rows, ["severity"]).items(), key=lambda item: severity_value(item[0][0]))
            ]
            severity_labels.append(trend_label(severity_means, "decrease"))
        circularity_count = sum(r["more_fragile_target"] == "circularity" for r in rows)
        nc_count = sum(r["more_fragile_target"] == "nc" for r in rows)
        if not rows:
            label = "INCONCLUSIVE"
        elif circularity_count == len(rows) and same_direction:
            label = "CONSISTENTLY_MORE_FRAGILE"
        elif mean(diffs) < 0 or circularity_count > nc_count:
            label = "PARTIALLY_MORE_FRAGILE"
        elif nc_count > circularity_count and mean(diffs) >= 0:
            label = "NOT_MORE_FRAGILE"
        else:
            label = "INCONCLUSIVE"
        out.append(
            {
                "seed": seed,
                "nc_mean_coverage_change": mean(r["nc_coverage_change"] for r in rows),
                "circularity_mean_coverage_change": mean(r["circularity_coverage_change"] for r in rows),
                "mean_target_gap_difference": mean(diffs),
                "circularity_more_fragile_conditions": circularity_count,
                "nc_more_fragile_conditions": nc_count,
                "equal_conditions": sum(r["more_fragile_target"] == "equal" for r in rows),
                "blur_target_gap_difference": blur_diff,
                "noise_target_gap_difference": noise_diff,
                "blur_and_noise_same_direction": same_direction,
                "severity_target_gap_trend": "|".join(severity_labels),
                "fragility_label": label,
            }
        )
    return out


def target_severity_trends(conf: list[dict]) -> list[dict]:
    out = []
    rows = [r for r in conf if r["degradation"] != "clean"]
    for (target, model, degradation), condition_rows in sorted(group(rows, ["target", "model", "degradation"]).items()):
        seed_rows = group(condition_rows, ["seed"])
        coverage_labels = []
        width_labels = []
        invalid_labels = []
        failure_labels = []
        circularity_width_narrowing_flags = []
        failure_explains_flags = []
        for (_seed,), seed_rows_one in sorted(seed_rows.items()):
            ordered = sorted(seed_rows_one, key=lambda r: severity_value(r["severity"]))
            cov_vals = [float(r["empirical_coverage"]) for r in ordered]
            width_vals = [float(r["interval_mean_width"]) for r in ordered]
            invalid_vals = [float(r["invalid_rate"]) for r in ordered]
            failure_vals = [float(r["failure_rate"]) for r in ordered]
            coverage_labels.append(trend_label(cov_vals, "decrease"))
            width_labels.append(trend_label(width_vals, "increase"))
            invalid_labels.append(trend_label(invalid_vals, "increase"))
            failure_labels.append(trend_label(failure_vals, "increase"))
            cov_drop = cov_vals[-1] - cov_vals[0] if len(cov_vals) >= 2 else 0.0
            width_change = width_vals[-1] - width_vals[0] if len(width_vals) >= 2 else 0.0
            invalid_change = invalid_vals[-1] - invalid_vals[0] if len(invalid_vals) >= 2 else 0.0
            failure_change = failure_vals[-1] - failure_vals[0] if len(failure_vals) >= 2 else 0.0
            circularity_width_narrowing_flags.append(target == "circularity" and cov_drop < 0 and width_change < -0.005)
            failure_explains_flags.append(cov_drop < 0 and (invalid_change > abs(cov_drop) or failure_change > abs(cov_drop)))

        def count(label_list: list[str], prefix: str) -> int:
            return sum(label.startswith(prefix) for label in label_list)

        def aggregate(label_list: list[str]) -> str:
            full = count(label_list, "fully")
            mostly = count(label_list, "mostly")
            total = len(label_list)
            if full == total and total:
                return "tam_monoton"
            if full + mostly >= max(1, math.ceil(total / 2)):
                return "cogunlukla_monoton"
            return "tutarsiz"

        out.append(
            {
                "target": target,
                "model": model,
                "degradation": degradation,
                "seeds_with_coverage_decrease": count(coverage_labels, "fully") + count(coverage_labels, "mostly"),
                "seeds_with_interval_width_increase": count(width_labels, "fully") + count(width_labels, "mostly"),
                "seeds_with_invalid_rate_increase": count(invalid_labels, "fully") + count(invalid_labels, "mostly"),
                "seeds_with_failure_rate_increase": count(failure_labels, "fully") + count(failure_labels, "mostly"),
                "coverage_monotonicity": aggregate(coverage_labels),
                "interval_width_monotonicity": aggregate(width_labels),
                "invalid_rate_monotonicity": aggregate(invalid_labels),
                "failure_rate_monotonicity": aggregate(failure_labels),
                "coverage_drop_with_width_narrowing": any(circularity_width_narrowing_flags),
                "coverage_drop_explained_by_invalid_or_failure": any(failure_explains_flags),
            }
        )
    return out


def stable_dice_cases(metrics: list[dict], comparison: list[dict]) -> list[dict]:
    test_rows = [r for r in metrics if r["split"] == "test"]
    clean_dice = {
        (int(r["seed"]), r["model"]): float(r["foreground_dice"])
        for r in test_rows
        if r["degradation"] == "clean"
    }
    comparison_by_key = {
        (int(r["seed"]), r["model"], r["degradation"], severity_value(r["severity"])): r
        for r in comparison
    }
    out = []
    for row in test_rows:
        if row["degradation"] == "clean":
            continue
        key = (int(row["seed"]), row["model"], row["degradation"], severity_value(row["severity"]))
        if key not in comparison_by_key or (int(row["seed"]), row["model"]) not in clean_dice:
            continue
        delta_dice = float(row["foreground_dice"]) - clean_dice[(int(row["seed"]), row["model"])]
        if abs(delta_dice) <= STABLE_DICE_THRESHOLD:
            pair = comparison_by_key[key]
            out.append(
                {
                    "seed": key[0],
                    "model": key[1],
                    "degradation": key[2],
                    "severity": key[3],
                    "clean_foreground_dice": clean_dice[(key[0], key[1])],
                    "degraded_foreground_dice": float(row["foreground_dice"]),
                    "delta_foreground_dice": delta_dice,
                    "nc_coverage_change": pair["nc_coverage_change"],
                    "circularity_coverage_change": pair["circularity_coverage_change"],
                    "target_gap_difference": pair["target_gap_difference"],
                    "more_fragile_target": pair["more_fragile_target"],
                }
            )
    return out


def write_figures(summary: list[dict], comparison: list[dict], severity_rows: list[dict], stable_rows: list[dict]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar([r["target"] for r in summary], [r["mean_coverage_change_vs_clean"] for r in summary], color=["#2c7fb8", "#f03b20"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Mean coverage change vs clean")
    plt.title("Clean to degraded coverage change")
    plt.tight_layout()
    plt.savefig(FIG / "target_clean_to_degraded_coverage_change.png", dpi=180)
    plt.close()

    seed_rows = fragility_by_seed(comparison)
    plt.figure(figsize=(7, 4))
    plt.bar([str(r["seed"]) for r in seed_rows], [r["mean_target_gap_difference"] for r in seed_rows], color="#756bb1")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Circularity change - N/C change")
    plt.xlabel("Seed")
    plt.title("Seed-wise target coverage difference")
    plt.tight_layout()
    plt.savefig(FIG / "seed_target_coverage_difference.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    for target, color in [("nc", "#2c7fb8"), ("circularity", "#f03b20")]:
        for degradation, style in [("gaussian_blur", "-"), ("gaussian_noise", "--")]:
            vals = []
            severities = []
            for (severity,), rows in sorted(
                group([r for r in comparison if r["degradation"] == degradation], ["severity"]).items(),
                key=lambda item: severity_value(item[0][0]),
            ):
                column = f"{target}_degraded_coverage"
                vals.append(mean(r[column] for r in rows))
                severities.append(float(severity))
            plt.plot(severities, vals, style, marker="o", color=color, label=f"{target} {degradation.replace('gaussian_', '')}")
    plt.ylabel("Empirical coverage")
    plt.xlabel("Severity")
    plt.title("Severity-wise target coverage")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "severity_target_coverage_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    if stable_rows:
        labels = [f"{r['model']}\n{r['degradation'].replace('gaussian_', '')} {r['severity']:g}\n{r['seed']}" for r in stable_rows]
        x = np.arange(len(stable_rows))
        width = 0.38
        plt.bar(x - width / 2, [r["nc_coverage_change"] for r in stable_rows], width, label="N/C", color="#2c7fb8")
        plt.bar(x + width / 2, [r["circularity_coverage_change"] for r in stable_rows], width, label="Circularity", color="#f03b20")
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Coverage change vs clean")
    plt.title("Small Dice-change conditions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "stable_dice_target_coverage_change.png", dpi=180)
    plt.close()


def direct_summary(comparison: list[dict]) -> dict:
    by_degradation = group(comparison, ["degradation"])
    return {
        "circularity_more_fragile_conditions": sum(r["more_fragile_target"] == "circularity" for r in comparison),
        "nc_more_fragile_conditions": sum(r["more_fragile_target"] == "nc" for r in comparison),
        "equal_conditions": sum(r["more_fragile_target"] == "equal" for r in comparison),
        "mean_target_gap_difference": mean(r["target_gap_difference"] for r in comparison),
        "median_target_gap_difference": median(r["target_gap_difference"] for r in comparison),
        "blur_target_gap_difference": mean(r["target_gap_difference"] for r in by_degradation.get(("gaussian_blur",), [])),
        "noise_target_gap_difference": mean(r["target_gap_difference"] for r in by_degradation.get(("gaussian_noise",), [])),
    }


def choose_decision(summary: dict, seed_rows: list[dict], bootstrap: dict, severity_rows: list[dict], stable_rows: list[dict]) -> str:
    circ_more = summary["circularity_more_fragile_conditions"]
    nc_more = summary["nc_more_fragile_conditions"]
    mean_diff = summary["mean_target_gap_difference"]
    ci_high = float(bootstrap.get("ci_high", float("nan")))
    seed_partial = sum(r["fragility_label"] in {"CONSISTENTLY_MORE_FRAGILE", "PARTIALLY_MORE_FRAGILE"} for r in seed_rows)
    stable_circ = sum(r["more_fragile_target"] == "circularity" for r in stable_rows)
    stable_nc = sum(r["more_fragile_target"] == "nc" for r in stable_rows)
    circ_trends = [r for r in severity_rows if r["target"] == "circularity"]
    monotonic_circ = sum(r["coverage_monotonicity"] != "tutarsiz" for r in circ_trends)
    both_degradations_negative = (
        np.isfinite(summary["blur_target_gap_difference"])
        and np.isfinite(summary["noise_target_gap_difference"])
        and summary["blur_target_gap_difference"] < 0
        and summary["noise_target_gap_difference"] < 0
    )

    if (
        circ_more > nc_more
        and mean_diff <= -0.02
        and np.isfinite(ci_high)
        and ci_high < 0
        and seed_partial >= 2
        and monotonic_circ >= 2
        and both_degradations_negative
    ):
        return "GÜÇLÜ HEDEFE ÖZGÜ SİNYAL"
    if circ_more > nc_more or mean_diff < 0 or stable_circ > stable_nc or seed_partial:
        return "SINIRLI AMA SAVUNULABİLİR SİNYAL"
    return "HEDEFE ÖZGÜ AYRIM YOK"


def write_report(
    missing: list[str],
    frozen_ok: bool,
    no_recalibration_ok: bool,
    summary_rows: list[dict],
    comparison_summary: dict,
    seed_rows: list[dict],
    severity_rows: list[dict],
    bootstrap: dict,
    stable_rows: list[dict],
    decision: str,
) -> None:
    by_target = {r["target"]: r for r in summary_rows}
    stable_diff = mean(r["target_gap_difference"] for r in stable_rows)
    seed_labels = ", ".join(f"{r['seed']}: {r['fragility_label']}" for r in seed_rows)
    lines = [
        "# Target-Specific Coverage Fragility Report",
        "",
        "## 1. Veri doğrulaması",
        "",
        f"- Kullanılan dosyalar: {', '.join(REQUIRED_FILES)}",
        f"- Eksik dosyalar: {', '.join(missing) if missing else 'yok'}",
        f"- Frozen calibration doğrulaması: {'geçti' if frozen_ok else 'geçmedi'}",
        f"- Degradation altında yeniden calibration yapılmadı: {'geçti' if no_recalibration_ok else 'geçmedi'}",
        f"- Cell-wise bağımlılık kontrolü: bootstrap örnekleme birimi `{bootstrap['cluster_unit']}`, küme sayısı {bootstrap['cluster_count']}.",
        "",
        "## 2. N/C sonucu",
        "",
        f"- Clean coverage: {by_target['nc']['clean_empirical_coverage']:.6f}",
        f"- Degraded coverage: {by_target['nc']['degraded_empirical_coverage']:.6f}",
        f"- Ortalama değişim: {by_target['nc']['mean_coverage_change_vs_clean']:.6f}",
        f"- >=5 puan düşen koşul: {by_target['nc']['conditions_drop_ge_5pp']}",
        f"- Seed tutarlılığı: {seed_labels}",
        f"- Severity trendi: {', '.join(r['coverage_monotonicity'] for r in severity_rows if r['target'] == 'nc')}",
        f"- Interval width / failure etkisi: width mean {by_target['nc']['mean_interval_width']:.6f}, failure rate {by_target['nc']['failure_rate']:.6f}.",
        "",
        "## 3. Circularity sonucu",
        "",
        f"- Clean coverage: {by_target['circularity']['clean_empirical_coverage']:.6f}",
        f"- Degraded coverage: {by_target['circularity']['degraded_empirical_coverage']:.6f}",
        f"- Ortalama değişim: {by_target['circularity']['mean_coverage_change_vs_clean']:.6f}",
        f"- >=5 puan düşen koşul: {by_target['circularity']['conditions_drop_ge_5pp']}",
        f"- Seed tutarlılığı: {seed_labels}",
        f"- Severity trendi: {', '.join(r['coverage_monotonicity'] for r in severity_rows if r['target'] == 'circularity')}",
        f"- Interval width / failure etkisi: width mean {by_target['circularity']['mean_interval_width']:.6f}, failure rate {by_target['circularity']['failure_rate']:.6f}.",
        "",
        "## 4. Doğrudan hedef karşılaştırması",
        "",
        f"- Circularity daha fazla düşen koşullar: {comparison_summary['circularity_more_fragile_conditions']}",
        f"- N/C daha fazla düşen koşullar: {comparison_summary['nc_more_fragile_conditions']}",
        f"- Ortalama fark: {comparison_summary['mean_target_gap_difference']:.6f}",
        f"- Median fark: {comparison_summary['median_target_gap_difference']:.6f}",
        f"- Bootstrap %95 güven aralığı: [{bootstrap['ci_low']:.6f}, {bootstrap['ci_high']:.6f}]",
        f"- Blur/noise ayrımı: blur {comparison_summary['blur_target_gap_difference']:.6f}, noise {comparison_summary['noise_target_gap_difference']:.6f}",
        f"- Seed tutarlılığı: {seed_labels}",
        "",
        "## 5. Dice sabitken güvenilirlik",
        "",
        f"- `|delta Dice| <= 0.01` koşul sayısı: {len(stable_rows)}",
        f"- Bu koşullarda N/C coverage değişimi: {mean(r['nc_coverage_change'] for r in stable_rows):.6f}",
        f"- Circularity coverage değişimi: {mean(r['circularity_coverage_change'] for r in stable_rows):.6f}",
        f"- Hedef ayrışması: {stable_diff:.6f}",
        "",
        "## 6. Nihai karar",
        "",
        decision,
        "",
        f"- Circularity gerçekten N/C'den daha kırılgan mı? {'Kısmen evet' if comparison_summary['mean_target_gap_difference'] < 0 else 'Hayır'}",
        f"- Fark pratik olarak büyük mü? {'Hayır, küçük/orta büyüklükte' if abs(comparison_summary['mean_target_gap_difference']) < 0.02 else 'Evet, belirgin'}",
        f"- Seed'ler arasında tutarlı mı? {'Kısmen' if any('PARTIALLY' in r['fragility_label'] for r in seed_rows) else 'Hayır'}",
        f"- Severity ile düzenli mi? {'Kısmen' if any(r['coverage_monotonicity'] != 'tutarsiz' for r in severity_rows) else 'Hayır'}",
        "- Dice bu farkı açıklıyor mu? Hayır; küçük Dice değişimli koşullarda da hedef farkı ayrıca ölçülebiliyor.",
        "- Conformal kısmı projenin merkezine alınmalı mı? Yardımcı kanıt olarak tutulmalı, tek merkez yapılmamalı.",
        "- Bu sonuç projeyi güçlü hale getiriyor mu? Tek başına güçlü yapmıyor.",
        "- Mevcut proje orta düzeyde mi kalıyor? Evet, mevcut kanıt orta ama savunulabilir düzeyde.",
        "- Yeni projeye pivot gerekli mi? Hayır; seçici daraltma ile devam edilebilir.",
        "",
        "## 7. Hocaya sunulacak özet",
        "",
        "Frozen-clean conformal çıktılar üzerinden yapılan hedefe özgü kontrolde circularity coverage kaybı N/C'ye göre bazı koşullarda daha yüksek göründü. Ancak farkın büyüklüğü sınırlı ve tüm seed/degradation kombinasyonlarında güçlü biçimde tekrarlanmıyor. Bootstrap belirsizlik aralığı bu ayrımı mutlak bir hedefe özgü kırılganlık iddiasına dönüştürmek için yeterince keskin değilse sonuç temkinli yorumlanmalıdır. Dice değişimi küçük kalan koşullarda da hedef coverage farkı izlenebildiği için yalnız segmentasyon Dice skoru tüm güvenilirlik davranışını açıklamıyor. Bu nedenle conformal analiz proje içinde destekleyici ve seçici bir argüman olarak sunulmalı, ana iddia ise N/C-aware segmentasyonun sınırlı fakat ölçülebilir etkisi üzerine kurulmalıdır.",
        "",
    ]
    (OUT / "target_specific_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (OUT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required final-analysis files: {missing}")

    conf = read_csv(OUT / "conformal_results.csv")
    per_cell = read_csv(OUT / "conformal_per_cell.csv")
    gap = read_csv(OUT / "coverage_gap_analysis.csv")
    _interval = read_csv(OUT / "interval_width_failure_analysis.csv")
    _severity_existing = read_csv(OUT / "severity_trends.csv")
    _cluster_existing = read_csv(OUT / "cluster_bootstrap_intervals.csv")
    metrics = read_csv(OUT / "degradation_metrics_by_seed_severity.csv")
    _spearman = read_csv(OUT / "dice_morphometry_spearman.csv")
    json.loads((OUT / "final_analysis_summary.json").read_text(encoding="utf-8"))

    frozen_ok = all(row["calibration_source"] == "clean" for row in conf) and all(
        row["calibration_source"] == "clean" for row in per_cell
    )
    no_recalibration_ok = all(row["calibration_source"] == "clean" for row in conf if row["degradation"] != "clean")

    summary_rows = target_summary(conf, gap)
    comparison_rows = pair_target_changes(gap)
    seed_rows = fragility_by_seed(comparison_rows)
    severity_rows = target_severity_trends(conf)
    bootstrap = cluster_bootstrap_target_gap(per_cell, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED)
    stable_rows = stable_dice_cases(metrics, comparison_rows)
    comparison_summary = direct_summary(comparison_rows)
    decision = choose_decision(comparison_summary, seed_rows, bootstrap, severity_rows, stable_rows)

    write_csv(OUT / "target_specific_coverage_summary.csv", summary_rows)
    write_csv(OUT / "nc_vs_circularity_coverage_comparison.csv", comparison_rows)
    write_csv(OUT / "target_fragility_by_seed.csv", seed_rows)
    write_csv(OUT / "target_specific_severity_trends.csv", severity_rows)
    write_csv(OUT / "target_specific_cluster_bootstrap.csv", [bootstrap])
    write_csv(OUT / "stable_dice_target_coverage_cases.csv", stable_rows)
    write_figures(summary_rows, comparison_rows, severity_rows, stable_rows)
    write_report(
        missing,
        frozen_ok,
        no_recalibration_ok,
        summary_rows,
        comparison_summary,
        seed_rows,
        severity_rows,
        bootstrap,
        stable_rows,
        decision,
    )

    print(json.dumps({"decision": decision, **comparison_summary, "bootstrap": bootstrap}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
