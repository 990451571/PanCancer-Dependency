"""GPU evaluation of frozen pan-cancer controls against pure Sanger outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline
from run_sanger_phase_a import ALPHA, DRIVERS, TOP_K, score


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expression_kernel(models, genes, matrices, train, expression, driver_mutation):
    torch = baseline.TORCH
    kernel = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    held_kernel = torch.zeros((len(expression), len(train)), dtype=torch.float64, device="cuda")
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    train_lineage = (labels[train, None] == categories[None, :]).astype(float)
    held_lineage = np.zeros((len(expression), len(categories)), dtype=float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in DRIVERS]
    held_means = np.asarray([baseline.observed_mean(row, axis=0) for row in expression])[:, None]
    blocks = [
        (train_lineage, held_lineage),
        (matrices["mutation"][np.ix_(train, driver_columns)], driver_mutation),
        (baseline.observed_mean(matrices["expression"][train], axis=1)[:, None], held_means),
        (matrices["expression"][train], expression),
    ]
    used = 0
    for train_values, held_values in blocks:
        used += baseline.add_kernel(kernel, held_kernel, train_values, held_values)
    return kernel, held_kernel, used


def summarize(metrics):
    rows = []
    keys = ("ndcg_at_10", "binary_dependency_precision_at_10", "top10_overlap", "spearman")
    reference = metrics[metrics.method == "training_gene_mean"].set_index("model_id")
    for method in ("direct_expression_only", "mapped_expression_only"):
        current = metrics[metrics.method == method].copy()
        if current.empty:
            continue
        for metric in keys:
            delta = current.set_index("model_id")[metric] - reference.loc[current.model_id, metric].to_numpy()
            rows.append({"comparison": f"{method}_vs_training_gene_mean", "metric": metric,
                         "model_n": len(delta), "mean_delta": float(delta.mean()),
                         "median_delta": float(delta.median()), "positive_n": int((delta > 0).sum()),
                         "zero_n": int((delta == 0).sum()), "negative_n": int((delta < 0).sum())})
    paired = metrics[metrics.method.isin(["direct_expression_only", "mapped_expression_only"])]
    pivot = paired.pivot(index="model_id", columns="method", values=list(keys)).dropna()
    for metric in keys:
        delta = pivot[(metric, "mapped_expression_only")] - pivot[(metric, "direct_expression_only")]
        rows.append({"comparison": "mapped_vs_direct_expression", "metric": metric,
                     "model_n": len(delta), "mean_delta": float(delta.mean()),
                     "median_delta": float(delta.median()), "positive_n": int((delta > 0).sum()),
                     "zero_n": int((delta == 0).sum()), "negative_n": int((delta < 0).sum())})
    return pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--frozen-dir", type=Path,
                        default=root / "outputs/sanger_external_controls_frozen_v1")
    parser.add_argument("--phase-a-results", type=Path, default=root / "outputs/sanger_phase_a_769p_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_external_controls_v1")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="只检查冻结输入、显卡、癌系和患者隔离")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device(args.device)
    print("【阶段 1/4】校验冻结对照、输入哈希和分组隔离", flush=True)
    frozen_run = json.loads((args.frozen_dir / "run.json").read_text())
    if frozen_run["status"] != "external_controls_frozen_before_prediction_evaluation":
        raise ValueError("泛癌外部对照未在预测前冻结")
    controls = pd.read_csv(args.frozen_dir / "frozen_models.csv")
    with np.load(args.frozen_dir / "matrices.npz", allow_pickle=False) as archive:
        control_data = {name: archive[name] for name in archive.files}
    models, genes, matrices, _ = baseline.load_data(args.baseline_dir)
    if (control_data["model_ids"].tolist() != controls.matched_broad_id.tolist() or
            control_data["genes"].tolist() != genes.tolist() or len(controls) != 66):
        raise ValueError("冻结对照的模型或基因顺序不一致")
    primary = control_data["primary_gene_mask"].astype(bool)
    if primary.sum() != 13120:
        raise ValueError("主评价基因数量变化")
    lineages = sorted(controls.BroadLineage.unique())
    current_patients = set(models.PatientID)
    if set(controls.BroadPatientID) & current_patients:
        raise ValueError("冻结对照患者出现在当前队列")
    print(f"【冻结队列】模型 {len(controls)}｜癌系 {len(lineages)}｜双输入 {controls.sanger_expression_mapped.sum()}｜主评价基因 {primary.sum()}", flush=True)
    print(f"【固定规则】整癌系留出｜α={ALPHA:g}｜不调参", flush=True)
    if args.dry_run:
        for lineage in lineages:
            held_n = int(controls.BroadLineage.eq(lineage).sum())
            train_n = int(models.OncotreeLineage.ne(lineage).sum())
            print(f"  {lineage}：训练 {train_n}｜外部验证 {held_n}", flush=True)
        print("【检查通过】CUDA、冻结状态、模型/基因顺序及患者隔离正常；未训练、未评价功能结果。", flush=True)
        return

    print("【阶段 2/4】按癌系构建 GPU 核并生成固定模型预测", flush=True)
    metrics_rows, ranking_rows, split_rows = [], [], []
    direct_predictions = np.full((len(controls), len(genes)), np.nan, dtype=float)
    mapped_predictions = np.full_like(direct_predictions, np.nan)
    reference_predictions = np.full_like(direct_predictions, np.nan)
    feature_counts = set()
    primary_genes = genes[primary]
    for number, lineage in enumerate(lineages, 1):
        fold_started = time.monotonic()
        held = np.flatnonzero(controls.BroadLineage.to_numpy() == lineage)
        patients = set(controls.iloc[held].BroadPatientID)
        train = np.flatnonzero((models.OncotreeLineage.to_numpy() != lineage) &
                               ~models.PatientID.isin(patients).to_numpy())
        if patients & set(models.iloc[train].PatientID):
            raise ValueError(f"{lineage} 患者隔离失败")
        mapped_held = held[controls.iloc[held].sanger_expression_mapped.to_numpy(dtype=bool)]
        expression = np.vstack([control_data["direct_expression"][held],
                                control_data["mapped_expression"][mapped_held]])
        mutation = np.vstack([control_data["driver_mutation"][held],
                              control_data["driver_mutation"][mapped_held]])
        kernel, held_kernel, feature_n = expression_kernel(models, genes, matrices, train, expression, mutation)
        feature_counts.add(feature_n)
        prediction = baseline.masked_ridge(
            kernel, held_kernel, matrices["dependency"][train], [ALPHA])[ALPHA]
        direct_predictions[held] = prediction[:len(held)]
        if len(mapped_held):
            mapped_predictions[mapped_held] = prediction[len(held):]
        reference_predictions[held] = baseline.observed_mean(matrices["dependency"][train], axis=0)
        split_rows.append({"heldout_lineage": lineage, "train_n": len(train),
                           "external_n": len(held), "mapped_external_n": len(mapped_held),
                           "patient_overlap_n": 0})
        print(f"  癌系 {number}/{len(lineages)} {lineage}｜训练 {len(train)}｜验证 {len(held)}｜映射输入 {len(mapped_held)}｜{time.monotonic()-fold_started:.1f}秒", flush=True)

    print("【阶段 3/4】计算冻结口径的逐模型指标", flush=True)
    truth_matrix = -control_data["sanger_scaled_bf"][:, primary]
    binary_matrix = control_data["sanger_binary_dependency"][:, primary]
    prediction_sets = {"training_gene_mean": reference_predictions[:, primary],
                       "direct_expression_only": direct_predictions[:, primary],
                       "mapped_expression_only": mapped_predictions[:, primary]}
    for row, control in controls.iterrows():
        oracle_saved = False
        for method, values in prediction_sets.items():
            prediction = values[row]
            if not np.isfinite(prediction).any():
                continue
            metrics, selected, oracle = score(
                truth_matrix[row], binary_matrix[row], prediction, primary_genes)
            metrics_rows.append({"model_id": control.matched_broad_id, "model_name": control.model_name,
                                 "lineage": control.BroadLineage, "strict_ccrcc": bool(control.strict_ccrcc),
                                 "method": method, "alpha": ALPHA if method != "training_gene_mean" else None,
                                 **metrics})
            for rank, index in enumerate(selected, 1):
                ranking_rows.append({"model_id": control.matched_broad_id, "model_name": control.model_name,
                                     "method": method, "list": "predicted_top10", "rank": rank,
                                     "Gene": primary_genes[index], "prediction": prediction[index],
                                     "sanger_loss": truth_matrix[row, index],
                                     "sanger_binary_dependency": int(binary_matrix[row, index])})
            if not oracle_saved:
                for rank, index in enumerate(oracle, 1):
                    ranking_rows.append({"model_id": control.matched_broad_id, "model_name": control.model_name,
                                         "method": "oracle", "list": "sanger_top10", "rank": rank,
                                         "Gene": primary_genes[index], "prediction": None,
                                         "sanger_loss": truth_matrix[row, index],
                                         "sanger_binary_dependency": int(binary_matrix[row, index])})
                oracle_saved = True
    metrics = pd.DataFrame(metrics_rows)
    summary = summarize(metrics)
    phase_a_check = "not_available"
    if (args.phase_a_results / "metrics.csv").exists():
        phase_a = pd.read_csv(args.phase_a_results / "metrics.csv")
        current = metrics[metrics.model_name.eq("769-P")]
        mapping = {"training_gene_mean": "training_gene_mean", "expression_only": "direct_expression_only"}
        keys = ["ndcg_at_10", "binary_dependency_precision_at_10", "top10_overlap", "spearman"]
        for _, old in phase_a.iterrows():
            new = current[current.method.eq(mapping[old.method])].iloc[0]
            if any(abs(float(old[key]) - float(new[key])) > 1e-12 for key in keys):
                raise ValueError("769-P 泛癌复算与阶段 A 结果不一致")
        phase_a_check = "exact_match"

    print("【阶段 4/4】保存泛癌分布、排名和预测矩阵", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    metrics.to_csv(temporary / "metrics.csv", index=False)
    summary.to_csv(temporary / "summary.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv", index=False)
    pd.DataFrame(split_rows).to_csv(temporary / "lineage_splits.csv", index=False)
    np.savez_compressed(temporary / "predictions.npz", model_ids=control_data["model_ids"], genes=genes,
                        training_gene_mean=reference_predictions,
                        direct_expression_only=direct_predictions,
                        mapped_expression_only=mapped_predictions)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "external_controls_completed_fixed_pan_cancer_context",
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64",
        "design": {"model_n": len(controls), "lineage_n": len(lineages),
                   "mapped_input_model_n": int(controls.sanger_expression_mapped.sum()),
                   "alpha": ALPHA, "top_k": TOP_K,
                   "training": "current cohort excluding each external model's complete lineage and patient",
                   "aggregation": "model-level descriptive distribution; no ccRCC inferential test"},
        "feature_n": sorted(feature_counts), "phase_a_769p_replay": phase_a_check,
        "summary": summary.to_dict("records"),
        "source_sha256": {"frozen_run": sha256(args.frozen_dir / "run.json"),
                          "frozen_models": sha256(args.frozen_dir / "frozen_models.csv"),
                          "frozen_matrices": sha256(args.frozen_dir / "matrices.npz"),
                          "baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "script": sha256(Path(__file__))},
        "limitations": frozen_run["limitations"] + [
            "Mapped-versus-direct comparison has only 17 models.",
            "The renal direct-input subset contains only 769-P.",
            "Positive pan-cancer performance would weaken, not support, a claim of ccRCC specificity."]}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    print("【泛癌结果】", flush=True)
    for comparison in ("direct_expression_only_vs_training_gene_mean",
                       "mapped_expression_only_vs_training_gene_mean",
                       "mapped_vs_direct_expression"):
        rows = summary[summary.comparison.eq(comparison)].set_index("metric")
        print(f"  {comparison}｜模型 {int(rows.model_n.iloc[0])}｜ΔNDCG {rows.loc['ndcg_at_10','mean_delta']:+.4f}（正 {int(rows.loc['ndcg_at_10','positive_n'])}）｜Δ命中率 {rows.loc['binary_dependency_precision_at_10','mean_delta']:+.4f}｜ΔSpearman {rows.loc['spearman','mean_delta']:+.4f}", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f} 秒｜结果 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
