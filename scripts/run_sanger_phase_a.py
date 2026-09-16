"""GPU validation of the frozen expression model on Sanger-screened 769-P.

The model and evaluation rules come from the pre-evaluation frozen protocol.
No hyperparameter, feature, threshold, or gene mapping is selected here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_paths import source_project_root

import run_depmap_baseline as baseline


TARGET_ID = "ACH-000411"
TARGET_NAME = "769-P"
ALPHA = 100000.0
TOP_K = 10
DRIVERS = baseline.DRIVERS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_row(path, model_id):
    """Read one model from a wide CSV without materializing the full matrix."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        matches = [row for row in reader if row and row[0] == model_id]
    if len(matches) != 1:
        raise ValueError(f"{path.name} 中 {model_id} 行数不是 1：{len(matches)}")
    return header, matches[0]


def aligned_raw_values(raw_path, mapping, genes, model_id, aggregation):
    header, row = read_raw_row(raw_path, model_id)
    positions = {name: index for index, name in enumerate(header)}
    grouped = {}
    for raw_column, canonical in mapping.items():
        if canonical not in genes or raw_column not in positions:
            continue
        text = row[positions[raw_column]]
        value = float(text) if text != "" else np.nan
        grouped.setdefault(canonical, []).append(value)
    result = np.full(len(genes), np.nan, dtype=np.float32)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    for gene, values in grouped.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            result[gene_index[gene]] = np.mean(finite) if aggregation == "mean" else np.max(finite)
    return result


def target_features(raw_dir, baseline_dir, genes):
    mapping = pd.read_csv(baseline_dir / "gene_mapping.csv.gz")
    mappings = {}
    for modality in ("expression", "mutation"):
        selected = mapping[(mapping.modality == modality) & mapping.included_gene.astype(bool)]
        mappings[modality] = dict(zip(selected.raw_column, selected.canonical_symbol))
    expression = aligned_raw_values(
        raw_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        mappings["expression"], genes, TARGET_ID, "mean")
    mutation = aligned_raw_values(
        raw_dir / "OmicsSomaticMutationsMatrixDamaging.csv",
        mappings["mutation"], genes, TARGET_ID, "max")
    mutation = np.where(np.isnan(mutation), np.nan, (mutation > 0).astype(np.float32))
    driver_index = [np.flatnonzero(genes == gene)[0] for gene in DRIVERS]
    if not np.isfinite(expression).any() or not np.isfinite(mutation[driver_index]).all():
        raise ValueError("769-P 表达或驱动基因突变特征不完整")
    return expression, mutation


def expression_kernel(models, genes, matrices, train, expression, mutation):
    torch = baseline.TORCH
    kernel = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    held_kernel = torch.zeros((1, len(train)), dtype=torch.float64, device="cuda")
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    train_lineage = (labels[train, None] == categories[None, :]).astype(float)
    held_lineage = np.zeros((1, len(categories)), dtype=float)  # Kidney is wholly held out.
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in DRIVERS]
    blocks = [
        (train_lineage, held_lineage),
        (matrices["mutation"][np.ix_(train, driver_columns)], mutation[None, driver_columns]),
        (baseline.observed_mean(matrices["expression"][train], axis=1)[:, None],
         np.asarray([[baseline.observed_mean(expression, axis=0)]])),
        (matrices["expression"][train], expression[None, :]),
    ]
    used = 0
    for train_values, held_values in blocks:
        used += baseline.add_kernel(kernel, held_kernel, train_values, held_values)
    return kernel, held_kernel, used


def score(truth, binary, prediction, genes):
    valid = np.isfinite(truth) & np.isfinite(binary) & np.isfinite(prediction)
    indices = np.flatnonzero(valid)
    if len(indices) < TOP_K:
        raise ValueError("可评价基因少于 Top-K")
    selected = indices[np.argsort(prediction[indices], kind="stable")[:TOP_K]]
    oracle = indices[np.argsort(truth[indices], kind="stable")[:TOP_K]]
    discounts = 1 / np.log2(np.arange(2, TOP_K + 2))
    relevance = np.maximum(0, -truth)
    ideal = float((relevance[oracle] * discounts).sum())
    ranks_truth = pd.Series(truth[indices]).rank(method="average").to_numpy()
    ranks_prediction = pd.Series(prediction[indices]).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(ranks_truth, ranks_prediction)[0, 1])
    metrics = {
        "gene_n": len(indices),
        "ndcg_at_10": float((relevance[selected] * discounts).sum() / ideal) if ideal > 0 else np.nan,
        "binary_dependency_precision_at_10": float(binary[selected].mean()),
        "top10_overlap": len(set(selected) & set(oracle)) / TOP_K,
        "spearman": spearman,
    }
    return metrics, selected, oracle


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = source_project_root(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--raw-dir", type=Path,
                        default=source / "data/raw/depmap_24q4")
    parser.add_argument("--frozen-dir", type=Path, default=root / "outputs/sanger_validation_frozen_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_phase_a_769p_v1")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="只核对输入、冻结规则、显卡和样本隔离")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device(args.device)
    print("【阶段 1/4】校验冻结协议和输入文件", flush=True)
    protocol = json.loads((args.frozen_dir / "protocol.json").read_text())
    if protocol["status"] != "frozen_before_prediction_evaluation":
        raise ValueError("外部验证协议未处于预测前冻结状态")
    evaluation = protocol["evaluation"]
    if (evaluation["hyperparameter_rule"] != "reuse alpha=100000; no tuning on the three validation outcomes" or
            evaluation["phase_a"].split()[0] != TARGET_NAME):
        raise ValueError("冻结的阶段 A 模型或超参数发生变化")
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    raw_models = pd.read_csv(args.raw_dir / "Model.csv", low_memory=False).set_index("ModelID")
    target = raw_models.loc[TARGET_ID]
    if target.CellLineName != TARGET_NAME or target.OncotreeLineage != "Kidney":
        raise ValueError("769-P 身份或癌系注释不一致")
    train = np.flatnonzero((models.OncotreeLineage.to_numpy() != "Kidney") &
                           (models.PatientID.to_numpy() != target.PatientID))
    if len(train) == 0 or (models.iloc[train].OncotreeLineage == "Kidney").any():
        raise ValueError("非 Kidney 训练集合无效")
    if target.PatientID in set(models.iloc[train].PatientID):
        raise ValueError("目标患者泄漏到训练集合")
    expression, mutation = target_features(args.raw_dir, args.baseline_dir, genes)
    sanger_continuous = pd.read_csv(
        args.frozen_dir / "renal_scaled_bayesian_factor.csv.gz", index_col="Gene")[TARGET_NAME]
    sanger_binary = pd.read_csv(
        args.frozen_dir / "renal_binary_dependency.csv.gz", index_col="Gene")[TARGET_NAME]
    overlap = pd.Index(genes).intersection(sanger_continuous.index)
    primary = overlap[~pd.Series(common_essential, index=genes).loc[overlap].to_numpy()]
    if len(overlap) != protocol["gene_universe"]["exact_overlap_n"] or len(primary) != protocol["gene_universe"]["primary_non_common_essential_n"]:
        raise ValueError("冻结后的评价基因集合发生变化")
    print(f"【隔离检查】训练 {len(train)} 个非 Kidney 模型｜目标 769-P｜患者无重叠", flush=True)
    print(f"【固定规则】α={ALPHA:g}｜主评价基因 {len(primary)}｜不调参", flush=True)
    if args.dry_run:
        print("【检查通过】CUDA、输入、特征、标签及隔离规则正常；未训练、未计算结果。", flush=True)
        return

    print("【阶段 2/4】构建表达模型 GPU 核矩阵", flush=True)
    kernel, held_kernel, feature_n = expression_kernel(models, genes, matrices, train, expression, mutation)
    print(f"【阶段 3/4】GPU 拟合 {len(genes)} 个靶点并预测 769-P", flush=True)
    ridge = baseline.masked_ridge(kernel, held_kernel, matrices["dependency"][train], [ALPHA])[ALPHA][0]
    reference = baseline.observed_mean(matrices["dependency"][train], axis=0)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    columns = np.asarray([gene_index[gene] for gene in primary])
    truth = -sanger_continuous.loc[primary].to_numpy(dtype=float)
    binary = sanger_binary.loc[primary].to_numpy(dtype=float)
    predictions = {"training_gene_mean": reference[columns], "expression_only": ridge[columns]}

    rows, ranking_rows = [], []
    oracle_saved = False
    for method, prediction in predictions.items():
        metrics, selected, oracle = score(truth, binary, prediction, primary.to_numpy())
        rows.append({"model_id": TARGET_ID, "model_name": TARGET_NAME, "method": method,
                     "alpha": ALPHA if method == "expression_only" else None, **metrics})
        for rank, index in enumerate(selected, 1):
            ranking_rows.append({"method": method, "list": "predicted_top10", "rank": rank,
                                 "Gene": primary[index], "prediction": prediction[index],
                                 "sanger_loss": truth[index], "sanger_binary_dependency": int(binary[index])})
        if not oracle_saved:
            for rank, index in enumerate(oracle, 1):
                ranking_rows.append({"method": "oracle", "list": "sanger_top10", "rank": rank,
                                     "Gene": primary[index], "prediction": np.nan,
                                     "sanger_loss": truth[index], "sanger_binary_dependency": int(binary[index])})
            oracle_saved = True

    print("【阶段 4/4】保存固定口径结果与逐基因审计表", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(temporary / "metrics.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv", index=False)
    pd.DataFrame({"Gene": primary, "sanger_loss": truth, "sanger_binary_dependency": binary.astype(int),
                  "training_gene_mean": predictions["training_gene_mean"],
                  "expression_only": predictions["expression_only"]}).to_csv(
                      temporary / "per_gene_predictions.csv.gz", index=False, compression="gzip")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "status": "phase_a_completed_single_model_feasibility_only",
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64",
        "target": {"ModelID": TARGET_ID, "CellLineName": TARGET_NAME,
                   "PatientID": target.PatientID, "OncotreeLineage": target.OncotreeLineage},
        "training": {"model_n": len(train), "kidney_model_n": 0,
                     "patient_overlap_n": 0, "alpha": ALPHA, "feature_n": feature_n,
                     "feature_blocks": ["lineage", "VHL/PBRM1/SETD2/BAP1/MTOR mutation indicators",
                                        "expression mean", "gene expression"]},
        "evaluation": {"gene_n": len(primary), "top_k": TOP_K,
                       "scope": evaluation["primary_scope"], "metrics": rows},
        "source_sha256": {
            "frozen_protocol": sha256(args.frozen_dir / "protocol.json"),
            "frozen_continuous": sha256(args.frozen_dir / "renal_scaled_bayesian_factor.csv.gz"),
            "frozen_binary": sha256(args.frozen_dir / "renal_binary_dependency.csv.gz"),
            "baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
            "script": sha256(Path(__file__))},
        "limitations": [
            "This is one cell line and cannot estimate ccRCC-wide generalization or uncertainty.",
            "769-P was screened by both Broad and Sanger, although neither label entered training.",
            "The endpoint is an older BAGEL-processed assay and is not numerically calibrated to Chronos.",
            "Ranking agreement is evidence of cross-platform feasibility, not patient functional dependency."]}
    (temporary / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    print("【验证结果】", flush=True)
    for row in rows:
        name = "训练均值" if row["method"] == "training_gene_mean" else "仅表达模型"
        print(f"  {name}｜NDCG@10 {row['ndcg_at_10']:.4f}｜依赖命中率 {row['binary_dependency_precision_at_10']:.4f}｜Top-10 重叠 {row['top10_overlap']:.4f}｜Spearman {row['spearman']:.4f}", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f} 秒｜结果 {args.output_dir}", flush=True)
    print("【结论边界】单模型流程验证，不能外推为 ccRCC 特异功能依赖。", flush=True)


if __name__ == "__main__":
    main()
