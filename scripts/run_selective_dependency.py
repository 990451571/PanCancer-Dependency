"""Evaluate expression-based selective dependency under whole-lineage holdout.

The target is dependency minus the training-fold gene mean.  Because the
existing per-target ridge includes an intercept, fitting this centered target
is algebraically equivalent to subtracting the same mean from its raw-score
prediction.  The comparator is a training-only gene selectivity prior: the
10th percentile of centered training outcomes for each gene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline


METHODS = ("training_selectivity_prior", "expression_residual")
METHOD_LABELS = {"training_selectivity_prior": "训练选择性先验", "expression_residual": "表达残差模型"}
METRICS = ("ndcg_at_10", "selective_precision_at_10", "top10_overlap", "spearman", "regret")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zeros(rows: int, columns: int):
    return baseline.TORCH.zeros((rows, columns), dtype=baseline.TORCH.float64, device="cuda")


def expression_kernel(models, genes, matrices, train, held):
    """Build the frozen expression-only kernel without unused modality blocks."""
    kernel, held_kernel = zeros(len(train), len(train)), zeros(len(held), len(train))
    counts = {}
    annotations = baseline.annotation_features(models, genes, matrices, train, held)
    for name in ("lineage", "drivers", "expression_mean"):
        if name not in annotations:
            continue
        counts[name] = baseline.add_kernel(kernel, held_kernel, *annotations[name])
    counts["expression"] = baseline.add_kernel(
        kernel, held_kernel, matrices["expression"][train], matrices["expression"][held])
    return kernel, held_kernel, counts


def training_targets(y_train: np.ndarray, quantile: float):
    """Compute gene means and lower-tail residual prior on GPU."""
    torch = baseline.TORCH
    values = torch.as_tensor(y_train, dtype=torch.float64, device="cuda")
    observed_n = torch.isfinite(values).sum(dim=0)
    mean = torch.nanmean(values, dim=0)
    residual = values - mean
    prior = torch.nanquantile(residual, quantile, dim=0)
    invalid = observed_n < 2
    mean[invalid] = torch.nan
    prior[invalid] = torch.nan
    return mean.cpu().numpy(), prior.cpu().numpy(), observed_n.cpu().numpy()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, including deterministic handling of ties."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = average_ranks(x), average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / denominator) if denominator > 1e-12 else np.nan


def score_models(truth, predictions, model_ids, lineages, genes, stratum, top_k, threshold):
    """Score both methods on one paired observed universe per model."""
    discounts = 1 / np.log2(np.arange(2, top_k + 2))
    metrics, rankings = [], []
    for row, model_id in enumerate(model_ids):
        valid = stratum & np.isfinite(truth[row])
        for prediction in predictions.values():
            valid &= np.isfinite(prediction[row])
        available = np.flatnonzero(valid)
        if len(available) < top_k:
            continue
        oracle = available[np.argsort(truth[row, available], kind="stable")[:top_k]]
        ideal = float(np.dot(np.maximum(0, -truth[row, oracle]), discounts))
        for method, prediction in predictions.items():
            selected = available[np.argsort(prediction[row, available], kind="stable")[:top_k]]
            metrics.append({
                "ModelID": model_id,
                "heldout_lineage": lineages[row],
                "stratum": "non_common_essential" if stratum.dtype == bool and not stratum.all() else "all",
                "method": method,
                "gene_n": len(available),
                "selective_truth_n": int(np.sum(truth[row, available] <= threshold)),
                "ndcg_at_10": float(np.dot(np.maximum(0, -truth[row, selected]), discounts) / ideal)
                if ideal > 0 else np.nan,
                "selective_precision_at_10": float(np.mean(truth[row, selected] <= threshold)),
                "top10_overlap": len(set(selected) & set(oracle)) / top_k,
                "spearman": spearman(prediction[row, available], truth[row, available]),
                "regret": float(truth[row, selected].mean() - truth[row, oracle].mean()),
            })
            if not stratum.all():
                for rank, index in enumerate(selected, 1):
                    rankings.append({"ModelID": model_id, "heldout_lineage": lineages[row],
                                     "ranking": method, "rank": rank, "Gene": genes[index],
                                     "predicted_residual": prediction[row, index],
                                     "true_residual": truth[row, index]})
        if not stratum.all():
            for rank, index in enumerate(oracle, 1):
                rankings.append({"ModelID": model_id, "heldout_lineage": lineages[row],
                                 "ranking": "oracle", "rank": rank, "Gene": genes[index],
                                 "predicted_residual": np.nan, "true_residual": truth[row, index]})
    return metrics, rankings


def summarize(metrics: pd.DataFrame):
    lineage = metrics.groupby(["heldout_lineage", "stratum", "method"], as_index=False)[list(METRICS)].mean()
    keys = ["ModelID", "heldout_lineage", "stratum"]
    wide = metrics.pivot(index=keys, columns="method", values=list(METRICS))
    deltas = wide.index.to_frame(index=False)
    for metric in METRICS:
        left = wide[(metric, "expression_residual")].to_numpy()
        right = wide[(metric, "training_selectivity_prior")].to_numpy()
        name = "regret_reduction" if metric == "regret" else f"{metric}_gain"
        deltas[name] = right - left if metric == "regret" else left - right
    gain_columns = [column for column in deltas if column.endswith("_gain") or column == "regret_reduction"]
    rows = []
    for stratum, current in metrics.groupby("stratum"):
        for method, group in current.groupby("method"):
            rows.append({"estimand": "model_weighted", "stratum": stratum, "method": method,
                         "model_n": group.ModelID.nunique(), "lineage_n": group.heldout_lineage.nunique(),
                         **group[list(METRICS)].mean().to_dict()})
            by_lineage = group.groupby("heldout_lineage")[list(METRICS)].mean()
            rows.append({"estimand": "lineage_equal", "stratum": stratum, "method": method,
                         "model_n": group.ModelID.nunique(), "lineage_n": len(by_lineage),
                         **by_lineage.mean().to_dict()})
        current_delta = deltas[deltas.stratum == stratum]
        for estimand, values in (("model_weighted", current_delta[gain_columns].mean()),
                                 ("lineage_equal", current_delta.groupby("heldout_lineage")[gain_columns].mean().mean())):
            rows.append({"estimand": estimand, "stratum": stratum, "method": "expression_minus_prior",
                         "model_n": len(current_delta), "lineage_n": current_delta.heldout_lineage.nunique(),
                         **values.to_dict()})
    return lineage, deltas, pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/selective_dependency_lolo_v1")
    parser.add_argument("--minimum-lineage-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=100000)
    parser.add_argument("--prior-quantile", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--residual-threshold", type=float, default=-0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.minimum_lineage_size < 2 or args.alpha <= 0 or not 0 < args.prior_quantile < 0.5:
        parser.error("minimum-lineage-size、alpha或prior-quantile不合法")
    if args.top_k < 1 or not np.isfinite(args.residual_threshold):
        parser.error("top-k或residual-threshold不合法")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    baseline.configure_device("cuda")
    print("【阶段 1/4】校验输入并冻结整癌系留出", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(source)
    labels = models.OncotreeLineage.to_numpy()
    patients = models.PatientID.to_numpy()
    counts = models.OncotreeLineage.value_counts()
    lineages = sorted(counts[counts >= args.minimum_lineage_size].index)
    splits, split_rows = {}, []
    for lineage in lineages:
        held = np.flatnonzero(labels == lineage)
        train = np.flatnonzero((labels != lineage) & ~np.isin(patients, patients[held]))
        if set(patients[train]) & set(patients[held]):
            raise ValueError(f"患者泄漏：{lineage}")
        splits[lineage] = (train, held)
        split_rows.append({"heldout_lineage": lineage, "train_n": len(train), "heldout_n": len(held),
                           "heldout_patient_n": len(set(patients[held])),
                           "patient_overlap_n": len(set(patients[train]) & set(patients[held]))})
    print(f"【冻结规则】癌系 {len(lineages)}｜模型 {len(models)}｜基因 {len(genes)}｜α={args.alpha:g}｜先验分位数 {args.prior_quantile:g}", flush=True)
    print(f"【主评价】排除 common-essential｜Top-{args.top_k}｜选择性阈值 残差≤{args.residual_threshold:g}｜不调参", flush=True)
    if args.dry_run:
        for lineage, (train, held) in splits.items():
            print(f"  {lineage}｜训练 {len(train)}｜留出 {len(held)}｜患者重叠 0", flush=True)
        print("【检查通过】未拟合模型、未使用留出标签计算指标、未写入输出。", flush=True)
        return

    y = matrices["dependency"].astype(float)
    metric_rows, ranking_rows, feature_counts = [], [], {}
    print("【阶段 2/4】逐癌系构建表达 GPU 核并拟合固定模型", flush=True)
    for number, (lineage, (train, held)) in enumerate(splits.items(), 1):
        tick = time.monotonic()
        mean, prior, observed_n = training_targets(y[train], args.prior_quantile)
        kernel, held_kernel, feature_counts[lineage] = expression_kernel(models, genes, matrices, train, held)
        raw_prediction = baseline.masked_ridge(kernel, held_kernel, y[train], [args.alpha])[args.alpha]
        truth_residual = y[held] - mean
        predictions = {
            "training_selectivity_prior": np.broadcast_to(prior, truth_residual.shape),
            "expression_residual": raw_prediction - mean,
        }
        for stratum in (np.ones(len(genes), dtype=bool), ~common_essential):
            rows, ranks = score_models(truth_residual, predictions, models.index.to_numpy()[held],
                                       labels[held], genes, stratum, args.top_k, args.residual_threshold)
            metric_rows.extend(rows)
            ranking_rows.extend(ranks)
        primary = pd.DataFrame(metric_rows)
        current = primary[(primary.heldout_lineage == lineage) &
                          (primary.stratum == "non_common_essential")]
        means = current.groupby("method").ndcg_at_10.mean()
        delta = means["expression_residual"] - means["training_selectivity_prior"]
        print(f"  {number:02d}/{len(lineages)} {lineage}｜训练 {len(train)}｜留出 {len(held)}｜"
              f"先验 {means['training_selectivity_prior']:.4f}｜表达 {means['expression_residual']:.4f}｜ΔNDCG {delta:+.4f}｜"
              f"{time.monotonic() - tick:.1f}秒", flush=True)

    print("【阶段 3/4】汇总配对指标与 ccRCC 探索性子集", flush=True)
    metrics = pd.DataFrame(metric_rows)
    lineage_summary, deltas, overall = summarize(metrics)
    ccrcc_ids = set(models.index[models.clear_cell_renal_cell_carcinoma])
    ccrcc_metrics = metrics[(metrics.ModelID.isin(ccrcc_ids)) &
                            (metrics.stratum == "non_common_essential")].copy()
    ccrcc_summary = ccrcc_metrics.groupby("method", as_index=False)[list(METRICS)].mean()
    ccrcc_summary.insert(0, "model_n", ccrcc_metrics.ModelID.nunique())

    print("【阶段 4/4】保存必要结果与审计记录", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        metrics.to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv.gz", index=False)
        lineage_summary.to_csv(temporary / "lineage_summary.csv", index=False)
        deltas.to_csv(temporary / "paired_deltas.csv", index=False)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        ccrcc_summary.to_csv(temporary / "ccrcc_summary.csv", index=False)
        pd.DataFrame(split_rows).to_csv(temporary / "splits.csv", index=False)
        run = {
            "status": "exploratory_selective_dependency_lolo",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "input_sha256": {name: sha256(source / name) for name in ("audit.json", "matrices.npz", "models.csv", "gene_coverage.csv")},
            "script_sha256": sha256(Path(__file__)),
            "compute": {"device": "cuda", "dtype": "float64", "torch": baseline.TORCH.__version__,
                        "gpu": baseline.TORCH.cuda.get_device_name(0)},
            "model_n": len(models), "gene_n": len(genes), "lineage_n": len(lineages),
            "feature_counts": feature_counts, "elapsed_seconds": time.monotonic() - started,
            "rules": {
                "split": "Whole-lineage holdout with patient-overlap purging",
                "target": "Held dependency minus each training-fold gene mean",
                "equivalence": "With a per-target intercept, residual-target fitting equals raw fitting followed by subtracting the training gene mean",
                "prior": f"Training-only {args.prior_quantile:g} quantile of gene-centered dependency",
                "model": "Expression-only linear-kernel ridge with lineage, five driver indicators, expression mean and expression matrix",
                "alpha": f"Fixed at {args.alpha:g}; no selection on residual metrics",
                "primary_stratum": "Non-common-essential genes; release-wide annotation",
                "ranking": "Paired observed universe; lower residual is more selective",
                "ndcg_relevance": "max(0, -true residual)",
                "selective_hit": f"true residual <= {args.residual_threshold:g}; heuristic threshold",
                "regret": "Mean selected true residual minus oracle mean; lower is better",
            },
            "limitations": [
                "This target and comparator were defined after inspecting absolute-dependency results, so this run is exploratory.",
                "The selectivity prior can favor genes with broad lower-tail variability and is not a context-specific model.",
                "The residual threshold is a heuristic effect-size cutoff, not a calibrated essentiality label.",
                "Common-essential annotation is release-wide rather than estimated inside each training fold.",
                "Cell-line cross-lineage validation does not establish patient transfer, normal-tissue safety or ccRCC specificity.",
                "The ccRCC subset has 12 models and is exploratory.",
            ],
        }
        run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise

    primary = overall[(overall.estimand == "lineage_equal") &
                      (overall.stratum == "non_common_essential")].set_index("method")
    print("【核心结果｜癌系等权、非 common-essential】", flush=True)
    for method in METHODS:
        row = primary.loc[method]
        print(f"  {METHOD_LABELS[method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}｜"
              f"Top-10重叠 {row.top10_overlap:.4f}｜Spearman {row.spearman:.4f}｜Regret {row.regret:.4f}", flush=True)
    gain = primary.loc["expression_minus_prior"]
    print(f"【配对增益】ΔNDCG {gain.ndcg_at_10_gain:+.4f}｜Δ命中率 {gain.selective_precision_at_10_gain:+.4f}｜"
          f"ΔSpearman {gain.spearman_gain:+.4f}｜Regret降低 {gain.regret_reduction:+.4f}", flush=True)
    if len(ccrcc_summary):
        view = ccrcc_summary.set_index("method")
        change = view.loc["expression_residual", "ndcg_at_10"] - view.loc["training_selectivity_prior", "ndcg_at_10"]
        print(f"【ccRCC探索性】模型 {ccrcc_metrics.ModelID.nunique()}｜ΔNDCG {change:+.4f}｜不可作独立验证", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】只有表达残差模型稳定优于训练选择性先验，才支持进入患者域适配。", flush=True)


if __name__ == "__main__":
    main()
