"""GPU validation of two frozen renal models after expression-domain mapping."""
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
from run_sanger_phase_a import ALPHA, DRIVERS, TOP_K, aligned_raw_values, score


TARGET_NAMES = ("LB1047-RCC", "RCC-FG2")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_mutations(raw_dir, baseline_dir, genes, broad_ids):
    mapping = pd.read_csv(baseline_dir / "gene_mapping.csv.gz")
    selected = mapping[(mapping.modality == "mutation") & mapping.included_gene.astype(bool)]
    raw_to_gene = dict(zip(selected.raw_column, selected.canonical_symbol))
    rows = []
    for model_id in broad_ids:
        values = aligned_raw_values(raw_dir / "OmicsSomaticMutationsMatrixDamaging.csv",
                                    raw_to_gene, genes, model_id, "max")
        rows.append(np.where(np.isnan(values), np.nan, (values > 0).astype(np.float32)))
    return np.vstack(rows)


def expression_kernel(models, genes, matrices, train, held_expression, held_mutation):
    torch = baseline.TORCH
    kernel = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    held_kernel = torch.zeros((len(held_expression), len(train)), dtype=torch.float64, device="cuda")
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    train_lineage = (labels[train, None] == categories[None, :]).astype(float)
    held_lineage = np.zeros((len(held_expression), len(categories)), dtype=float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in DRIVERS]
    held_means = np.asarray([baseline.observed_mean(row, axis=0) for row in held_expression])[:, None]
    blocks = [
        (train_lineage, held_lineage),
        (matrices["mutation"][np.ix_(train, driver_columns)], held_mutation[:, driver_columns]),
        (baseline.observed_mean(matrices["expression"][train], axis=1)[:, None], held_means),
        (matrices["expression"][train], held_expression),
    ]
    used = 0
    for train_values, held_values in blocks:
        used += baseline.add_kernel(kernel, held_kernel, train_values, held_values)
    return kernel, held_kernel, used


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("/mnt/e/projects/rl-genrisk-main/data/raw/depmap_24q4"))
    parser.add_argument("--frozen-dir", type=Path, default=root / "outputs/sanger_validation_frozen_v1")
    parser.add_argument("--preparation-dir", type=Path,
                        default=root / "outputs/sanger_phase_b_preparation_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_phase_b_v1")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入、映射冻结状态、显卡和训练隔离")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device(args.device)
    print("【阶段 1/4】校验冻结映射、目标身份和输入哈希", flush=True)
    preparation = json.loads((args.preparation_dir / "run.json").read_text())
    if (preparation["status"] != "phase_b_expression_mapping_frozen_before_target_crispr_evaluation" or
            preparation["mapping"]["selected"] != "gene_mean_shift_clip0" or
            preparation["mapping"]["target_outcomes_opened"] is not False):
        raise ValueError("阶段 B 表达映射没有按预测前规则冻结")
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    frozen = pd.read_csv(args.frozen_dir / "frozen_models.csv")
    target_table = frozen[frozen.model_name.isin(TARGET_NAMES)].copy()
    target_table = target_table.set_index("model_name").loc[list(TARGET_NAMES)].reset_index()
    raw_models = pd.read_csv(args.raw_dir / "Model.csv", low_memory=False).set_index("ModelID")
    target_patients = [raw_models.at[model_id, "PatientID"] for model_id in target_table.matched_broad_id]
    if set(target_table.matched_broad_id) & set(models.index):
        raise ValueError("阶段 B 目标出现在当前训练队列")
    train = np.flatnonzero((models.OncotreeLineage.to_numpy() != "Kidney") &
                           ~np.isin(models.PatientID.to_numpy(), target_patients))
    if set(target_patients) & set(models.iloc[train].PatientID):
        raise ValueError("目标患者泄漏到训练集合")

    with np.load(args.preparation_dir / "mapped_target_expression.npz", allow_pickle=False) as archive:
        mapped_names = archive["model_names"].astype(str)
        mapped_genes = archive["genes"].astype(str)
        mapped_expression = archive["expression"]
    if mapped_genes.tolist() != genes.tolist() or set(mapped_names) != set(TARGET_NAMES):
        raise ValueError("映射表达与当前基因或冻结目标不一致")
    mapped_expression = np.vstack([mapped_expression[np.flatnonzero(mapped_names == name)[0]]
                                   for name in TARGET_NAMES])
    mutations = target_mutations(args.raw_dir, args.baseline_dir, genes,
                                 target_table.matched_broad_id.tolist())
    driver_index = [np.flatnonzero(genes == gene)[0] for gene in DRIVERS]
    if not np.isfinite(mutations[:, driver_index]).all():
        raise ValueError("阶段 B 目标驱动基因突变特征不完整")

    continuous = pd.read_csv(args.frozen_dir / "renal_scaled_bayesian_factor.csv.gz", index_col="Gene")
    binary_frame = pd.read_csv(args.frozen_dir / "renal_binary_dependency.csv.gz", index_col="Gene")
    overlap = pd.Index(genes).intersection(continuous.index)
    primary = overlap[~pd.Series(common_essential, index=genes).loc[overlap].to_numpy()]
    frozen_protocol = json.loads((args.frozen_dir / "protocol.json").read_text())
    if len(primary) != frozen_protocol["gene_universe"]["primary_non_common_essential_n"]:
        raise ValueError("阶段 B 评价基因集合偏离冻结协议")
    print(f"【隔离检查】训练 {len(train)} 个非 Kidney 模型｜目标 2｜患者重叠 0", flush=True)
    print(f"【固定规则】映射 gene_mean_shift_clip0｜α={ALPHA:g}｜主评价基因 {len(primary)}｜不调参", flush=True)
    if args.dry_run:
        print("【检查通过】CUDA、输入、映射、目标标签及隔离规则正常；未训练、未计算结果。", flush=True)
        return

    print("【阶段 2/4】构建映射表达的 GPU 核矩阵", flush=True)
    kernel, held_kernel, feature_n = expression_kernel(
        models, genes, matrices, train, mapped_expression, mutations)
    print(f"【阶段 3/4】GPU 拟合 {len(genes)} 个靶点并预测 2 个肾癌模型", flush=True)
    ridge = baseline.masked_ridge(kernel, held_kernel, matrices["dependency"][train], [ALPHA])[ALPHA]
    reference = baseline.observed_mean(matrices["dependency"][train], axis=0)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    columns = np.asarray([gene_index[gene] for gene in primary])

    metric_rows, ranking_rows, gene_frames = [], [], []
    for target_index, name in enumerate(TARGET_NAMES):
        truth = -continuous.loc[primary, name].to_numpy(dtype=float)
        binary = binary_frame.loc[primary, name].to_numpy(dtype=float)
        predictions = {"training_gene_mean": reference[columns],
                       "mapped_expression_only": ridge[target_index, columns]}
        oracle_saved = False
        for method, prediction in predictions.items():
            metrics, selected, oracle = score(truth, binary, prediction, primary.to_numpy())
            metric_rows.append({"model_id": target_table.loc[target_index, "matched_broad_id"],
                                "model_name": name, "strict_ccrcc": bool(target_table.loc[target_index, "strict_ccrcc"]),
                                "method": method, "alpha": ALPHA if method == "mapped_expression_only" else None,
                                **metrics})
            for rank, index in enumerate(selected, 1):
                ranking_rows.append({"model_name": name, "method": method, "list": "predicted_top10",
                                     "rank": rank, "Gene": primary[index], "prediction": prediction[index],
                                     "sanger_loss": truth[index],
                                     "sanger_binary_dependency": int(binary[index])})
            if not oracle_saved:
                for rank, index in enumerate(oracle, 1):
                    ranking_rows.append({"model_name": name, "method": "oracle", "list": "sanger_top10",
                                         "rank": rank, "Gene": primary[index], "prediction": None,
                                         "sanger_loss": truth[index],
                                         "sanger_binary_dependency": int(binary[index])})
                oracle_saved = True
        gene_frames.append(pd.DataFrame({"model_name": name, "Gene": primary,
                                         "sanger_loss": truth,
                                         "sanger_binary_dependency": binary.astype(int),
                                         "training_gene_mean": predictions["training_gene_mean"],
                                         "mapped_expression_only": predictions["mapped_expression_only"]}))

    print("【阶段 4/4】保存逐模型结果与审计记录", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    pd.DataFrame(metric_rows).to_csv(temporary / "metrics.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv", index=False)
    pd.concat(gene_frames, ignore_index=True).to_csv(
        temporary / "per_gene_predictions.csv.gz", index=False, compression="gzip")
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "phase_b_completed_two_model_descriptive_validation",
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64",
        "training": {"model_n": len(train), "kidney_model_n": 0, "patient_overlap_n": 0,
                     "alpha": ALPHA, "feature_n": feature_n,
                     "feature_blocks": ["lineage", "VHL/PBRM1/SETD2/BAP1/MTOR mutation indicators",
                                        "expression mean", "mapped gene expression"]},
        "mapping": {"method": preparation["mapping"]["selected"],
                    "calibration_model_n": preparation["calibration"]["model_n"],
                    "target_crispr_used_for_mapping": False},
        "evaluation": {"model_n": len(TARGET_NAMES), "gene_n": len(primary), "top_k": TOP_K,
                       "aggregation": "none; report each model separately", "metrics": metric_rows},
        "source_sha256": {
            "preparation_run": sha256(args.preparation_dir / "run.json"),
            "mapped_expression": sha256(args.preparation_dir / "mapped_target_expression.npz"),
            "frozen_protocol": sha256(args.frozen_dir / "protocol.json"),
            "frozen_continuous": sha256(args.frozen_dir / "renal_scaled_bayesian_factor.csv.gz"),
            "frozen_binary": sha256(args.frozen_dir / "renal_binary_dependency.csv.gz"),
            "baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
            "script": sha256(Path(__file__))},
        "limitations": [
            "Only RCC-FG2 is explicitly ccRCC; LB1047-RCC has generic RCC annotation.",
            "Two models cannot support an inferential estimate of ccRCC generalization.",
            "Expression mapping has no Kidney calibration model and retains substantial residual error.",
            "Both biological models also have Broad CRISPR screens, though those outcomes did not enter training.",
            "Cell-line cross-platform ranking does not establish patient functional dependency."]}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    print("【验证结果】", flush=True)
    for row in metric_rows:
        label = "训练均值" if row["method"] == "training_gene_mean" else "映射表达模型"
        subtype = "明确 ccRCC" if row["strict_ccrcc"] else "RCC未细分"
        print(f"  {row['model_name']}（{subtype}）｜{label}｜NDCG@10 {row['ndcg_at_10']:.4f}｜依赖命中率 {row['binary_dependency_precision_at_10']:.4f}｜Top-10重叠 {row['top10_overlap']:.4f}｜Spearman {row['spearman']:.4f}", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f} 秒｜结果 {args.output_dir}", flush=True)
    print("【结论边界】两模型描述性验证，不能外推为 ccRCC 特异功能依赖。", flush=True)


if __name__ == "__main__":
    main()
