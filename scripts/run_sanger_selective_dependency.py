"""Externally evaluate standardized selective dependency on frozen Sanger controls.

Project Score BAGEL and DepMap Chronos values are not directly subtractable.
For every held-out external lineage, paired non-held Broad/Sanger screens define
per-gene means and standard deviations separately on each platform.  Sanger
truth and Broad predictions are then compared as training-only standardized
deviations.  This is an exploratory analysis because earlier external outcomes
were inspected before this endpoint was specified.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
from prepare_sanger_external_controls import read_label_subset
from run_selective_dependency import score_models


TOP_K = 10
Z_THRESHOLD = -1.0
MIN_CALIBRATION_N = 20
GAIN_METRICS = ("ndcg_at_10", "selective_precision_at_10", "top10_overlap", "spearman", "regret")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_header(path: Path):
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))[1:]


def paired_standardization(broad, sanger, quantile=0.10):
    """Pairwise platform moments and a Broad lower-tail prior, all on GPU."""
    b = torch.as_tensor(broad, dtype=torch.float64, device="cuda")
    s = torch.as_tensor(sanger, dtype=torch.float64, device="cuda")
    valid = torch.isfinite(b) & torch.isfinite(s)
    count = valid.sum(dim=0)
    b0, s0 = torch.where(valid, b, 0), torch.where(valid, s, 0)
    denominator = count.clamp_min(1)
    bmean, smean = b0.sum(dim=0) / denominator, s0.sum(dim=0) / denominator
    bdev = torch.where(valid, b - bmean, torch.nan)
    sdev = torch.where(valid, s - smean, torch.nan)
    bsd = torch.sqrt(torch.nanmean(bdev * bdev, dim=0))
    ssd = torch.sqrt(torch.nanmean(sdev * sdev, dim=0))
    usable = (count >= MIN_CALIBRATION_N) & (bsd > 1e-8) & (ssd > 1e-8)
    bz = bdev / bsd
    prior = torch.nanquantile(bz, quantile, dim=0)
    for values in (bmean, smean, bsd, ssd, prior):
        values[~usable] = torch.nan
    return tuple(values.cpu().numpy() for values in (bmean, bsd, smean, ssd, prior, count))


def zscore(values, mean, sd):
    return (values - mean) / sd


def summarize_methods(metrics):
    return metrics.groupby("method", as_index=False)[list(GAIN_METRICS) + ["gene_n"]].mean()


def paired_deltas(metrics, method, reference="training_standardized_prior"):
    subset = metrics[metrics.method.isin([method, reference])]
    wide = subset.pivot(index=["ModelID", "heldout_lineage"], columns="method", values=list(GAIN_METRICS))
    wide = wide.dropna(subset=[(metric, method) for metric in GAIN_METRICS] +
                              [(metric, reference) for metric in GAIN_METRICS])
    result = wide.index.to_frame(index=False)
    result["comparison"] = f"{method}_vs_{reference}"
    for metric in GAIN_METRICS:
        left, right = wide[(metric, method)], wide[(metric, reference)]
        result["regret_reduction" if metric == "regret" else f"{metric}_gain"] = (
            right.to_numpy() - left.to_numpy() if metric == "regret" else left.to_numpy() - right.to_numpy())
    return result


def bootstrap_mean(values, draws, generator):
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    index = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    means = tensor[index].mean(dim=1)
    q = torch.quantile(means, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return [float(value) for value in q.cpu()]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--frozen-dir", type=Path, default=root / "outputs/sanger_external_controls_frozen_v1")
    parser.add_argument("--prediction-dir", type=Path, default=root / "outputs/sanger_external_controls_v1")
    parser.add_argument("--external-dir", type=Path, default=root / "data/raw/external_sanger_20260914")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_selective_dependency_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.draws < 1000 or args.seed < 0:
        parser.error("draws至少1000且seed必须非负")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device("cuda")
    print("【阶段 1/5】校验冻结外部队列、既有预测与跨平台身份", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    controls = pd.read_csv(args.frozen_dir / "frozen_models.csv")
    with np.load(args.frozen_dir / "matrices.npz", allow_pickle=False) as archive:
        frozen = {name: archive[name] for name in archive.files}
    with np.load(args.prediction_dir / "predictions.npz", allow_pickle=False) as archive:
        predictions = {name: archive[name] for name in archive.files}
    if (frozen["model_ids"].tolist() != controls.matched_broad_id.tolist() or
            predictions["model_ids"].tolist() != frozen["model_ids"].tolist() or
            predictions["genes"].tolist() != genes.tolist()):
        raise ValueError("冻结模型、预测或基因顺序不一致")
    primary = frozen["primary_gene_mask"].astype(bool)
    if primary.sum() != 13120 or np.any(primary & common_essential):
        raise ValueError("主评价非 common-essential 基因集合变化")

    scaled_path = args.external_dir / "project_score_release1_scaled_bf.tsv.gz"
    header = set(label_header(scaled_path))
    annotation = pd.read_csv(args.external_dir / "model_list_20260814.csv")
    sid_names = annotation.groupby("model_id").model_name.agg(
        lambda values: list(dict.fromkeys(values.dropna().astype(str))))
    calibration_rows, calibration_names = [], []
    for index, row in enumerate(models.itertuples()):
        if pd.isna(row.SangerModelID) or row.SangerModelID not in sid_names:
            continue
        matches = [name for name in sid_names[row.SangerModelID] if name in header]
        if len(matches) == 1:
            calibration_rows.append(index)
            calibration_names.append(matches[0])
    calibration_rows = np.asarray(calibration_rows, dtype=int)
    if len(calibration_rows) < 100 or len(set(calibration_names)) != len(calibration_names):
        raise ValueError("唯一配对 Broad-Sanger 校准模型不足或标签列重复")
    if models.iloc[calibration_rows].OncotreeLineage.eq("Kidney").any():
        raise ValueError("跨平台校准队列意外包含 Kidney 模型")
    control_lineages = sorted(controls.BroadLineage.unique())
    calibration_counts = {}
    for lineage in control_lineages:
        target_patients = set(controls.loc[controls.BroadLineage.eq(lineage), "BroadPatientID"])
        keep = ((models.iloc[calibration_rows].OncotreeLineage.to_numpy() != lineage) &
                ~models.iloc[calibration_rows].PatientID.isin(target_patients).to_numpy())
        calibration_counts[lineage] = int(keep.sum())
    print(f"【校准队列】配对模型 {len(calibration_rows)}｜癌系 {models.iloc[calibration_rows].OncotreeLineage.nunique()}｜每个外部折 {min(calibration_counts.values())}-{max(calibration_counts.values())}", flush=True)
    print(f"【固定规则】逐基因双平台 z 残差｜每基因至少 {MIN_CALIBRATION_N} 对｜Top-{TOP_K}｜z≤{Z_THRESHOLD:g}｜不调参", flush=True)
    if args.dry_run:
        for lineage in control_lineages:
            print(f"  {lineage}｜外部模型 {controls.BroadLineage.eq(lineage).sum()}｜非同癌系校准 {calibration_counts[lineage]}", flush=True)
        print("【检查通过】未使用校准或目标标签计算指标，未写入结果。", flush=True)
        return

    print("【阶段 2/5】读取配对 Sanger 标签并做整癌系校准审计", flush=True)
    sanger_bf, label_audit = read_label_subset(scaled_path, calibration_names, genes, binary=False)
    sanger_loss = -sanger_bf
    calibration_metric_rows = []
    calibration_labels = models.iloc[calibration_rows].OncotreeLineage.to_numpy()
    calibration_patients = models.iloc[calibration_rows].PatientID.to_numpy()
    for number, lineage in enumerate(sorted(set(calibration_labels)), 1):
        held_local = np.flatnonzero(calibration_labels == lineage)
        train_local = np.flatnonzero((calibration_labels != lineage) &
                                     ~np.isin(calibration_patients, calibration_patients[held_local]))
        train_global, held_global = calibration_rows[train_local], calibration_rows[held_local]
        bmean, bsd, smean, ssd, _, count = paired_standardization(
            matrices["dependency"][train_global], sanger_loss[train_local])
        truth = zscore(matrices["dependency"][held_global], bmean, bsd)
        calibrated = zscore(sanger_loss[held_local], smean, ssd)
        centered = sanger_loss[held_local] - smean
        rows, _ = score_models(truth, {"sanger_z_residual": calibrated,
                                      "sanger_centered_unscaled": centered},
                               models.index.to_numpy()[held_global], calibration_labels[held_local],
                               genes, primary, TOP_K, Z_THRESHOLD)
        calibration_metric_rows.extend(rows)
        print(f"  校准折 {number:02d}/{len(set(calibration_labels))} {lineage}｜训练配对 {len(train_local)}｜留出 {len(held_local)}｜可用基因中位数 {int(np.nanmedian(count[primary]))}", flush=True)
    calibration_metrics = pd.DataFrame(calibration_metric_rows)
    calibration_summary = summarize_methods(calibration_metrics)

    print("【阶段 3/5】对冻结外部模型构造训练内标准化残差", flush=True)
    external_rows, ranking_rows, fold_rows = [], [], []
    for number, lineage in enumerate(control_lineages, 1):
        held = np.flatnonzero(controls.BroadLineage.to_numpy() == lineage)
        target_patients = set(controls.iloc[held].BroadPatientID)
        train = np.flatnonzero((models.OncotreeLineage.to_numpy() != lineage) &
                               ~models.PatientID.isin(target_patients).to_numpy())
        keep = np.isin(calibration_rows, train)
        paired_global = calibration_rows[keep]
        paired_sanger = sanger_loss[keep]
        bmean, bsd, smean, ssd, prior, count = paired_standardization(
            matrices["dependency"][paired_global], paired_sanger)
        full_mean = baseline.observed_mean(matrices["dependency"][train], axis=0)
        if not np.allclose(predictions["training_gene_mean"][held], full_mean[None, :],
                           equal_nan=True, atol=1e-12, rtol=0):
            raise ValueError(f"{lineage} 既有预测的训练均值不一致")
        truth = zscore(-frozen["sanger_scaled_bf"][held], smean, ssd)
        direct = zscore(predictions["direct_expression_only"][held], bmean, bsd)
        prior_matrix = np.broadcast_to(prior, truth.shape)
        rows, ranks = score_models(truth, {"training_standardized_prior": prior_matrix,
                                          "direct_expression_z_residual": direct},
                                   controls.matched_broad_id.to_numpy()[held],
                                   controls.BroadLineage.to_numpy()[held], genes, primary,
                                   TOP_K, Z_THRESHOLD)
        external_rows.extend(rows); ranking_rows.extend(ranks)
        mapped_local = np.flatnonzero(controls.iloc[held].sanger_expression_mapped.to_numpy(dtype=bool))
        if len(mapped_local):
            mapped_rows = held[mapped_local]
            mapped_truth = truth[mapped_local]
            mapped = zscore(predictions["mapped_expression_only"][mapped_rows], bmean, bsd)
            rows, ranks = score_models(mapped_truth, {"mapped_expression_z_residual": mapped},
                                       controls.matched_broad_id.to_numpy()[mapped_rows],
                                       controls.BroadLineage.to_numpy()[mapped_rows], genes, primary,
                                       TOP_K, Z_THRESHOLD)
            external_rows.extend(rows)
            ranking_rows.extend([row for row in ranks if row["ranking"] != "oracle"])
        fold_rows.append({"heldout_lineage": lineage, "external_n": len(held),
                          "paired_calibration_n": len(paired_global),
                          "usable_primary_gene_n": int(np.sum(np.isfinite(bsd) & primary)),
                          "patient_overlap_n": 0})
        print(f"  外部折 {number:02d}/{len(control_lineages)} {lineage}｜模型 {len(held)}｜校准 {len(paired_global)}｜可用基因 {fold_rows[-1]['usable_primary_gene_n']}", flush=True)

    print("【阶段 4/5】GPU计算外部配对增益与bootstrap区间", flush=True)
    external_metrics = pd.DataFrame(external_rows)
    direct_delta = paired_deltas(external_metrics, "direct_expression_z_residual")
    mapped_delta = paired_deltas(external_metrics, "mapped_expression_z_residual")
    all_deltas = pd.concat([direct_delta, mapped_delta], ignore_index=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    bootstrap_rows = []
    gain_columns = [column for column in direct_delta if column.endswith("_gain") or column == "regret_reduction"]
    for comparison, group in all_deltas.groupby("comparison"):
        for metric in gain_columns:
            for estimand, values in (("model_weighted", group[metric].to_numpy()),
                                     ("lineage_equal", group.groupby("heldout_lineage")[metric].mean().to_numpy())):
                interval = bootstrap_mean(values, args.draws, generator)
                bootstrap_rows.append({"comparison": comparison, "metric": metric,
                                       "estimand": estimand, "unit_n": len(values),
                                       "mean": float(values.mean()), "ci_low": interval[0],
                                       "bootstrap_median": interval[1], "ci_high": interval[2]})

    print("【阶段 5/5】保存校准审计、外部指标与运行记录", flush=True)
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    calibration_metrics.to_csv(temporary / "calibration_metrics.csv.gz", index=False)
    calibration_summary.to_csv(temporary / "calibration_summary.csv", index=False)
    external_metrics.to_csv(temporary / "external_metrics.csv.gz", index=False)
    all_deltas.to_csv(temporary / "paired_deltas.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(temporary / "bootstrap.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv.gz", index=False)
    pd.DataFrame(fold_rows).to_csv(temporary / "folds.csv", index=False)
    run = {
        "status": "exploratory_cross_platform_standardized_selective_dependency",
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(0), "dtype": "float64",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "design": {"external_model_n": len(controls), "external_lineage_n": len(control_lineages),
                   "paired_calibration_model_n": len(calibration_rows),
                   "paired_calibration_lineage_n": int(models.iloc[calibration_rows].OncotreeLineage.nunique()),
                   "primary_gene_n": int(primary.sum()), "minimum_gene_calibration_n": MIN_CALIBRATION_N,
                   "top_k": TOP_K, "z_threshold": Z_THRESHOLD},
        "rules": {
            "calibration": "Per-gene mean and population SD estimated separately on paired Broad Chronos and Sanger BAGEL screens after excluding the target lineage and patients",
            "truth": "Sanger loss standardized with training Sanger moments",
            "prediction": "Frozen Broad-scale prediction standardized with paired-training Broad moments",
            "prior": "10th percentile of paired-training Broad standardized dependency",
            "primary_comparison": "Direct Broad expression prediction versus training standardized selectivity prior",
            "secondary_comparison": "Mapped Sanger expression prediction versus the same prior in the 17-model availability subset",
            "bootstrap": "Model weighted and heldout-lineage equal; external controls have unique patients",
        },
        "label_audit": label_audit,
        "source_sha256": {"baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "frozen_matrices": sha256(args.frozen_dir / "matrices.npz"),
                          "predictions": sha256(args.prediction_dir / "predictions.npz"),
                          "sanger_scaled_bf": sha256(scaled_path), "script": sha256(Path(__file__))},
        "limitations": [
            "The standardized residual endpoint was specified after absolute external outcomes were inspected.",
            "Gene-wise z scores align location and scale but do not prove BAGEL and Chronos measure identical biology.",
            "Paired calibration models are availability-selected and contain no Kidney model.",
            "Common-essential exclusion uses a release-wide annotation.",
            "The mapped-expression comparison contains only 17 models.",
            "The frozen external cohort contains only one strict ccRCC model, so it cannot test ccRCC specificity.",
        ],
    }
    run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(output)

    calibration_view = calibration_summary.set_index("method")
    print("【跨平台校准审计】", flush=True)
    for method in ("sanger_centered_unscaled", "sanger_z_residual"):
        row = calibration_view.loc[method]
        print(f"  {method}｜NDCG {row.ndcg_at_10:.4f}｜Top-10重叠 {row.top10_overlap:.4f}｜Spearman {row.spearman:.4f}", flush=True)
    boot = pd.DataFrame(bootstrap_rows)
    for comparison in all_deltas.comparison.unique():
        row = boot[(boot.comparison == comparison) & (boot.metric == "ndcg_at_10_gain") &
                   (boot.estimand == "model_weighted")].iloc[0]
        positive = int((all_deltas[all_deltas.comparison == comparison].ndcg_at_10_gain > 0).sum())
        print(f"【外部主结果】{comparison}｜模型 {int(row.unit_n)}｜ΔNDCG {row['mean']:+.4f}｜正向 {positive}｜95%区间 [{row.ci_low:+.4f}, {row.ci_high:+.4f}]", flush=True)
    renal = direct_delta[direct_delta.ModelID.eq("ACH-000411")]
    if len(renal):
        print(f"【769-P探索性】标准化残差 ΔNDCG {renal.ndcg_at_10_gain.iloc[0]:+.4f}｜单模型", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】先看校准审计是否有效，再解释外部增益；本结果不能证明 ccRCC 或患者功能依赖。", flush=True)


if __name__ == "__main__":
    main()
