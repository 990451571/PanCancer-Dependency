#!/usr/bin/env python3
"""Apply DepMap-selected baseline hyperparameters to frozen Sanger controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline
import run_selective_dependency as selective
from prepare_sanger_external_controls import read_label_subset
from run_sanger_selective_dependency import paired_standardization, zscore
from run_selective_dependency_benchmark import (
    METHOD_LABELS, bootstrap_deltas, knn_predictions, paired_deltas,
    pcr_ridge_predictions, summarize_metrics,
)


TOP_K = 10
Z_THRESHOLD = -1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_header(path: Path):
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))[1:]


def external_kernels(models, genes, matrices, train, held_lineages, held_expression,
                     held_drivers, chunk=2048):
    torch = baseline.TORCH
    annotation = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    annotation_held = torch.zeros((len(held_expression), len(train)), dtype=torch.float64, device="cuda")
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    train_lineage = (labels[train, None] == categories[None, :]).astype(float)
    external_lineage = (held_lineages[:, None] == categories[None, :]).astype(float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_indices = [gene_index[gene] for gene in baseline.DRIVERS]
    blocks = [
        (train_lineage, external_lineage),
        (matrices["mutation"][train][:, driver_indices], held_drivers),
        (baseline.observed_mean(matrices["expression"][train], axis=1)[:, None],
         baseline.observed_mean(held_expression, axis=1)[:, None]),
    ]
    for train_values, held_values in blocks:
        baseline.add_kernel(annotation, annotation_held, train_values, held_values)

    expression = torch.zeros_like(annotation)
    expression_held = torch.zeros_like(annotation_held)
    train_norm = torch.zeros(len(train), dtype=torch.float64, device="cuda")
    held_norm = torch.zeros(len(held_expression), dtype=torch.float64, device="cuda")
    raw_train = matrices["expression"][train]
    for start in range(0, len(genes), chunk):
        x = torch.as_tensor(raw_train[:, start:start + chunk], dtype=torch.float64, device="cuda")
        hx = torch.as_tensor(held_expression[:, start:start + chunk], dtype=torch.float64, device="cuda")
        mean = torch.nan_to_num(torch.nanmean(x, dim=0), nan=0.0)
        x = torch.where(torch.isfinite(x), x, mean) - mean
        hx = torch.where(torch.isfinite(hx), hx, mean) - mean
        sd = torch.sqrt((x * x).mean(dim=0))
        keep = sd > 1e-8
        x, hx = x[:, keep] / sd[keep], hx[:, keep] / sd[keep]
        expression.add_(x @ x.T)
        expression_held.add_(hx @ x.T)
        train_norm.add_((x * x).sum(dim=1))
        held_norm.add_((hx * hx).sum(dim=1))
    return {
        "annotation": (annotation, annotation_held),
        "expression": (expression, expression_held),
        "full": (annotation + expression, annotation_held + expression_held),
        "train_norm": train_norm,
        "held_norm": held_norm,
    }


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--frozen-dir", type=Path, default=root / "outputs/sanger_external_controls_frozen_v1")
    parser.add_argument("--historical-prediction-dir", type=Path, default=root / "outputs/sanger_external_controls_v1")
    parser.add_argument("--external-dir", type=Path, default=root / "data/raw/external_sanger_20260914")
    parser.add_argument("--benchmark-dir", type=Path, default=root / "outputs/selective_dependency_benchmark_v1")
    parser.add_argument("--protocol", type=Path, default=root / "configs/dependency_benchmark_protocol_20260916.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_baseline_benchmark_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device("cuda")
    print("【阶段 1/5】校验冻结Sanger队列和DepMap所选参数", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    controls = pd.read_csv(args.frozen_dir / "frozen_models.csv")
    with np.load(args.frozen_dir / "matrices.npz", allow_pickle=False) as archive:
        frozen = {name: archive[name] for name in archive.files}
    with np.load(args.historical_prediction_dir / "predictions.npz", allow_pickle=False) as archive:
        historical = {name: archive[name] for name in archive.files}
    benchmark_run = json.loads((args.benchmark_dir / "run.json").read_text())
    selected = benchmark_run["selected_parameters"]
    primary = frozen["primary_gene_mask"].astype(bool)
    if primary.sum() != 13120 or np.any(primary & common_essential):
        raise ValueError("Sanger主评价基因集合变化")
    all_lineages = sorted(controls.BroadLineage.unique())
    lineages = [lineage for lineage in all_lineages if lineage in selected]
    excluded = controls.loc[~controls.BroadLineage.isin(lineages),
                            ["matched_broad_id", "BroadLineage", "BroadPatientID"]].copy()
    if len(lineages) < 10:
        raise ValueError("具有对应DepMap所选参数的Sanger癌系不足")

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
    if len(calibration_rows) < 100:
        raise ValueError("Broad-Sanger校准模型不足")
    benchmark_model_n = int(controls.BroadLineage.isin(lineages).sum())
    print(f"【冻结队列】原始Sanger模型 {len(controls)}｜公平比较 {benchmark_model_n}｜癌系 {len(lineages)}｜主评价基因 {primary.sum()}", flush=True)
    if len(excluded):
        print("【预设排除】" + "｜".join(
            f"{row.BroadLineage}:{row.matched_broad_id}" for row in excluded.itertuples())
              + "｜原因：DepMap外层样本不足20，无合法内层参数", flush=True)
    if args.dry_run:
        print("【检查通过】未读取Sanger功能标签、未拟合模型、未写入结果。", flush=True)
        return

    print("【阶段 2/5】读取固定校准标签并按癌系构造外部真值", flush=True)
    sanger_bf, label_audit = read_label_subset(scaled_path, calibration_names, genes, binary=False)
    sanger_loss = -sanger_bf
    metric_rows, ranking_rows, audit_rows = [], [], []
    labels, patients = models.OncotreeLineage.to_numpy(), models.PatientID.to_numpy()
    print("【阶段 3/5】使用DepMap所选参数生成Sanger预测", flush=True)
    for number, lineage in enumerate(lineages, 1):
        tick = time.monotonic()
        held = np.flatnonzero(controls.BroadLineage.to_numpy() == lineage)
        target_patients = set(controls.iloc[held].BroadPatientID)
        train = np.flatnonzero((labels != lineage) & ~models.PatientID.isin(target_patients).to_numpy())
        calibration_keep = np.isin(calibration_rows, train)
        paired_global = calibration_rows[calibration_keep]
        bmean, bsd, smean, ssd, prior, _ = paired_standardization(
            matrices["dependency"][paired_global], sanger_loss[calibration_keep])
        truth = zscore(-frozen["sanger_scaled_bf"][held], smean, ssd)
        mean = baseline.observed_mean(matrices["dependency"][train], axis=0)
        residual = matrices["dependency"][train] - mean
        kernels = external_kernels(
            models, genes, matrices, train, controls.BroadLineage.to_numpy()[held],
            frozen["direct_expression"][held], frozen["driver_mutation"][held])
        predictions = {"training_selectivity_prior": np.broadcast_to(prior, truth.shape)}

        annotation_alpha = float(selected[lineage]["annotation_ridge"][0])
        raw = baseline.masked_ridge(*kernels["annotation"], matrices["dependency"][train],
                                    [annotation_alpha])[annotation_alpha]
        predictions["annotation_ridge"] = zscore(raw, bmean, bsd)

        k = int(selected[lineage]["expression_knn"][0])
        knn_residual = knn_predictions(*kernels["expression"], kernels["train_norm"],
                                       kernels["held_norm"], residual, [k])[k]
        predictions["expression_knn"] = zscore(knn_residual + mean, bmean, bsd)

        rank, pcr_alpha = selected[lineage]["expression_pcr_ridge"]
        pcr_residual = pcr_ridge_predictions(*kernels["expression"], residual,
                                             [int(rank)], [float(pcr_alpha)])[(int(rank), float(pcr_alpha))]
        predictions["expression_pcr_ridge"] = zscore(pcr_residual + mean, bmean, bsd)

        tuned_alpha = float(selected[lineage]["expression_kernel_ridge_tuned"][0])
        alphas = sorted(set([100000.0, tuned_alpha]))
        full = baseline.masked_ridge(*kernels["full"], matrices["dependency"][train], alphas)
        predictions["expression_kernel_ridge_frozen"] = zscore(full[100000.0], bmean, bsd)
        predictions["expression_kernel_ridge_tuned"] = zscore(full[tuned_alpha], bmean, bsd)
        historical_error = float(np.nanmax(np.abs(full[100000.0] - historical["direct_expression_only"][held])))
        if historical_error > 1e-10:
            raise ValueError(f"{lineage} 冻结表达预测复算不一致：{historical_error}")

        rows, ranks = selective.score_models(
            truth, predictions, controls.matched_broad_id.to_numpy()[held],
            controls.BroadLineage.to_numpy()[held], genes, primary, TOP_K, Z_THRESHOLD)
        metric_rows.extend(rows); ranking_rows.extend(ranks)
        audit_rows.append({"heldout_lineage": lineage, "external_n": len(held),
                           "depmap_train_n": len(train), "paired_calibration_n": len(paired_global),
                           "historical_prediction_max_abs_error": historical_error,
                           "patient_overlap_n": 0})
        print(f"  {number:02d}/{len(lineages)} {lineage}｜模型 {len(held)}｜训练 {len(train)}｜{time.monotonic()-tick:.1f}秒", flush=True)

    print("【阶段 4/5】计算外部配对指标和GPU bootstrap", flush=True)
    metrics = pd.DataFrame(metric_rows)
    overall, lineage_summary = summarize_metrics(metrics)
    deltas = paired_deltas(metrics)
    external_units = controls.set_index("matched_broad_id")[["BroadPatientID"]].rename(
        columns={"BroadPatientID": "PatientID"})
    bootstrap = bootstrap_deltas(deltas, external_units, args.draws, args.seed)

    print("【阶段 5/5】保存外部公平比较与来源审计", flush=True)
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        metrics.to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv.gz", index=False)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        lineage_summary.to_csv(temporary / "lineage_summary.csv", index=False)
        deltas.to_csv(temporary / "paired_deltas.csv.gz", index=False)
        bootstrap.to_csv(temporary / "bootstrap.csv", index=False)
        pd.DataFrame(audit_rows).to_csv(temporary / "folds.csv", index=False)
        excluded.to_csv(temporary / "excluded_models.csv", index=False)
        run = {
            "status": "sanger_selective_dependency_baseline_benchmark_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "device": baseline.TORCH.cuda.get_device_name(0),
            "dtype": "float64",
            "protocol_sha256": sha256(args.protocol),
            "depmap_benchmark_run_sha256": sha256(args.benchmark_dir / "run.json"),
            "script_sha256": sha256(Path(__file__)),
            "design": {"frozen_external_model_n": len(controls),
                       "benchmark_external_model_n": benchmark_model_n,
                       "excluded_external_model_n": len(excluded),
                       "external_lineage_n": len(lineages),
                       "primary_gene_n": int(primary.sum()), "sanger_label_used_for_tuning": False},
            "label_audit": label_audit,
            "rules": {
                "hyperparameters": "Copied from the corresponding DepMap whole-lineage outer fold",
                "truth": "Sanger loss standardized with non-target paired Sanger moments",
                "prediction": "Direct Broad expression prediction standardized with paired Broad moments",
                "test_isolation": "Sanger labels were used only after all hyperparameters were fixed",
            },
            "limitations": [
                "Cross-platform z calibration is descriptive and does not make BAGEL and Chronos biologically identical.",
                "The 66-model cohort is availability-selected and contains too few ccRCC models for subtype claims.",
                "This benchmark cannot validate patient functional dependency.",
                "Pleura and Prostate each lacked a >=20-model DepMap outer lineage and were excluded rather than borrowing another lineage's hyperparameters.",
            ],
        }
        run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise

    primary_view = overall[overall.estimand.eq("lineage_equal")].set_index("method")
    print("【Sanger外部结果｜癌系等权】", flush=True)
    for method in METHOD_LABELS:
        row = primary_view.loc[method]
        print(f"  {METHOD_LABELS[method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}｜Spearman {row.spearman:.4f}", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】Sanger只检验跨平台细胞系排序；不能证明ccRCC患者依赖。", flush=True)


if __name__ == "__main__":
    main()
