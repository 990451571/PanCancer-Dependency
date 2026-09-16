#!/usr/bin/env python3
"""Run the once-only, protocol-frozen TCGA-KIRC Test confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
from run_selective_dependency import average_ranks
from run_tcga_patient_transfer import build_kernel, rank_agreement


ALPHA = 100000.0
TOP_K = 10
PRIMARY = "nonkidney_shift_driver_neutral"
SENSITIVITY = "kidney_shift_driver_neutral"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return float("nan")
    lr, rr = average_ranks(left[valid]), average_ranks(right[valid])
    lr, rr = lr - lr.mean(), rr - rr.mean()
    denominator = np.sqrt(np.dot(lr, lr) * np.dot(rr, rr))
    return float(np.dot(lr, rr) / denominator) if denominator > 1e-12 else float("nan")


def bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    means = tensor[index].mean(dim=1)
    interval = torch.quantile(
        means, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return tuple(float(x) for x in interval.cpu())


def bootstrap_median(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    medians = torch.quantile(tensor[index], .5, dim=1)
    interval = torch.quantile(
        medians, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return tuple(float(x) for x in interval.cpu())


def sample_type(sample: str) -> str:
    fields = str(sample).split("-")
    if len(fields) < 4:
        return ""
    return fields[3][:2]


def patient_id(sample: str) -> str:
    return "-".join(str(sample).split("-")[:3])


def load_alignment_helpers(source_root: Path, hgnc_path: Path):
    sys.path.insert(0, str(source_root / "scripts"))
    from build_context_module_stage0 import GeneCanonicalizer, collapse_duplicate_rows

    def align_expression(path: Path, sample_ids: list[str], genes: list[str]):
        header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
        missing = sorted(set(sample_ids) - set(header))
        if missing:
            raise ValueError(f"原始表达缺少 Test 样本：{missing[:5]}")
        raw = pd.read_csv(path, sep="\t", usecols=[header[0], *sample_ids], low_memory=False)
        canonicalizer = GeneCanonicalizer(hgnc_path)
        mapped_genes = raw.iloc[:, 0].map(canonicalizer.map)
        numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        numeric.index = mapped_genes
        collapsed = collapse_duplicate_rows(
            numeric, pd.Series(numeric.index, index=numeric.index), "mean")
        aligned = collapsed.reindex(genes).T
        aligned.index = list(numeric.columns)
        return aligned.reindex(sample_ids).astype(np.float32)

    return align_expression


def frequencies(values: np.ndarray, universe: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    result = np.zeros(values.shape[1], dtype=float)
    selected_rows = []
    columns = np.flatnonzero(universe)
    for row in values:
        valid = columns[np.isfinite(row[columns])]
        selected = valid[np.argsort(row[valid], kind="stable")[:TOP_K]]
        result[selected] += 1
        selected_rows.append(selected)
    result /= len(values)
    return result, selected_rows


def active_stability(left: np.ndarray, right: np.ndarray, genes: np.ndarray) -> dict:
    active = (left > 0) | (right > 0)
    left_top = set(genes[np.argsort(-left, kind="stable")[:100]])
    right_top = set(genes[np.argsort(-right, kind="stable")[:100]])
    return {
        "active_gene_n": int(active.sum()),
        "active_frequency_spearman": spearman(left[active], right[active]),
        "top100_frequency_overlap": len(left_top & right_top) / 100,
    }


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path,
                        default=root / "configs/tcga_locked_test_protocol_20260916.json")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--baseline-dir", type=Path,
                        default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--bridge-dir", type=Path,
                        default=root / "data/processed/tcga_kirc_expression_bridge_v1")
    parser.add_argument("--prior-dir", type=Path, default=root / "outputs/tcga_patient_transfer_v1")
    parser.add_argument("--frozen", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1/candidates.csv")
    parser.add_argument("--patients", type=Path,
                        default=source / "data/processed/context_module_stage0/patients.csv")
    parser.add_argument("--expression", type=Path, default=source / "data/raw/HiSeqV2")
    parser.add_argument("--hgnc", type=Path,
                        default=source / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--source-root", type=Path, default=source)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/tcga_locked_test_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    actual_protocol_sha = sha256(args.protocol)
    if args.protocol_sha256 != actual_protocol_sha:
        raise ValueError("命令中的协议 SHA256 与冻结协议不一致")
    if not args.unlock_test:
        raise RuntimeError("Test仍锁定；正式运行必须显式传入 --unlock-test")
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖一次性 Test 结果：{args.output_dir.resolve()}")
    baseline.configure_device("cuda")
    protocol = json.loads(args.protocol.read_text())
    expected = protocol["expected_input_sha256"]
    paths = {
        "baseline_matrices": args.baseline_dir / "matrices.npz",
        "bridge_gene_mapping": args.bridge_dir / "gene_mapping.csv.gz",
        "bridge_run": args.bridge_dir / "run.json",
        "prior_candidate_summary": args.prior_dir / "candidate_summary.csv.gz",
        "prior_patient_transfer_run": args.prior_dir / "run.json",
        "frozen_candidates": args.frozen,
        "patients": args.patients,
        "expression": args.expression,
        "hgnc": args.hgnc,
    }
    print("【阶段 1/5】核验已提交协议、输入哈希和一次性解锁条件", flush=True)
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"冻结输入哈希不一致：{name}")
    patients = pd.read_csv(args.patients, index_col=0)
    if patients.split.value_counts().to_dict() != {"train": 156, "validation": 53, "test": 51}:
        raise ValueError("TCGA历史分组发生变化")
    frozen = pd.read_csv(args.frozen).sort_values("discovery_rank")
    if frozen.Gene.tolist() != protocol["frozen_candidates"]:
        raise ValueError("冻结候选顺序与协议不一致")
    print(f"【解锁记录】协议 {actual_protocol_sha[:12]}｜Test 51｜候选20｜不调参、不重排", flush=True)

    print("【阶段 2/5】读取 Test 表达并应用 Train-only 冻结映射", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    mapping = pd.read_csv(args.bridge_dir / "gene_mapping.csv.gz")
    if mapping.Gene.tolist() != genes.tolist():
        raise ValueError("桥接基因顺序与DepMap不一致")
    test = patients[patients.split.eq("test")]
    test_ids = test.expression_sample.astype(str).tolist()
    align_expression = load_alignment_helpers(args.source_root, args.hgnc)
    raw_test = align_expression(args.expression, test_ids, genes.tolist()).to_numpy(dtype=float)
    nonkidney_expression = np.maximum(raw_test + mapping.nonkidney_shift.to_numpy(float), 0)
    kidney_expression = np.maximum(raw_test + mapping.kidney_shift.to_numpy(float), 0)
    covered = mapping.tcga_covered.to_numpy(dtype=bool)
    universe = covered & ~common_essential
    if int(universe.sum()) != 14349:
        raise ValueError("冻结主评价基因数量发生变化")

    print("【阶段 3/5】GPU生成主映射与敏感性映射的 Test 预测", flush=True)
    train = np.flatnonzero(models.OncotreeLineage.to_numpy() != "Kidney")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in baseline.DRIVERS]
    train_driver = matrices["mutation"][np.ix_(train, driver_columns)]
    neutral = np.broadcast_to(
        baseline.observed_mean(train_driver, axis=0), (len(test_ids), len(baseline.DRIVERS)))
    held_expression = np.vstack([nonkidney_expression, kidney_expression])
    held_drivers = np.vstack([neutral, neutral])
    kernel, held_kernel, feature_counts = build_kernel(
        models, genes, matrices, train, held_expression, held_drivers)
    raw_prediction = baseline.masked_ridge(
        kernel, held_kernel, matrices["dependency"][train], [ALPHA])[ALPHA]
    gene_mean = baseline.observed_mean(matrices["dependency"][train], axis=0)
    residual = raw_prediction - gene_mean
    n = len(test_ids)
    predictions = {PRIMARY: residual[:n], SENSITIVITY: residual[n:]}

    print("【阶段 4/5】计算冻结端点和 Test 配对肿瘤-正常表达", flush=True)
    primary_frequency, primary_selected = frequencies(predictions[PRIMARY], universe)
    sensitivity_frequency, sensitivity_selected = frequencies(predictions[SENSITIVITY], universe)
    sensitivity_rows, ranking_rows = [], []
    for row, current_id in enumerate(test.index.astype(str)):
        overlap, correlation = rank_agreement(
            predictions[PRIMARY][row], predictions[SENSITIVITY][row], universe)
        sensitivity_rows.append({"patient_id": current_id, "top10_overlap": overlap, "spearman": correlation})
        for method, selected, values in (
                (PRIMARY, primary_selected[row], predictions[PRIMARY][row]),
                (SENSITIVITY, sensitivity_selected[row], predictions[SENSITIVITY][row])):
            for rank, column in enumerate(selected, 1):
                ranking_rows.append({"patient_id": current_id, "method": method, "rank": rank,
                                     "Gene": genes[column], "predicted_residual": values[column]})
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    prior = pd.read_csv(args.prior_dir / "candidate_summary.csv.gz")
    prior_primary = prior[prior.method.eq(PRIMARY)].pivot(
        index="Gene", columns="split", values="top10_selection_frequency").reindex(genes)
    stability_rows = []
    for earlier in ("train", "validation"):
        stats = active_stability(prior_primary[earlier].to_numpy(float), primary_frequency, genes)
        stability_rows.append({"comparison": f"{earlier}_vs_test", **stats})
    stability = pd.DataFrame(stability_rows)

    header = pd.read_csv(args.expression, sep="\t", nrows=0).columns.astype(str).tolist()[1:]
    test_patient_set = set(test.index.astype(str))
    normal_by_patient = {
        patient_id(sample): sample for sample in header
        if sample_type(sample) == "11" and patient_id(sample) in test_patient_set
    }
    if len(normal_by_patient) != 14:
        raise ValueError(f"Test配对正常样本数量变化：{len(normal_by_patient)}")
    paired_patients = sorted(normal_by_patient)
    paired_tumor_ids = test.loc[paired_patients, "expression_sample"].astype(str).tolist()
    paired_normal_ids = [normal_by_patient[x] for x in paired_patients]
    candidate_genes = frozen.Gene.tolist()
    paired_expression = align_expression(
        args.expression, paired_tumor_ids + paired_normal_ids, candidate_genes)
    candidate_rows = []
    priority_directions = protocol["priority_expression_directions"]
    for rank, gene in frozen[["discovery_rank", "Gene"]].itertuples(index=False):
        column = gene_index[gene]
        differences = (paired_expression.loc[paired_tumor_ids, gene].to_numpy(float)
                       - paired_expression.loc[paired_normal_ids, gene].to_numpy(float))
        low, median, high = bootstrap_median(
            differences, protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"] + int(rank))
        expected_direction = priority_directions.get(gene, "not_prespecified")
        observed_median = float(np.median(differences))
        direction_replicated = (
            (expected_direction.endswith("negative") and observed_median < 0)
            or (expected_direction.endswith("positive") and observed_median > 0)
        ) if expected_direction != "not_prespecified" else False
        candidate_rows.append({
            "Gene": gene, "discovery_rank": int(rank),
            "train_top10_frequency": float(prior_primary.loc[gene, "train"]),
            "validation_top10_frequency": float(prior_primary.loc[gene, "validation"]),
            "test_top10_frequency": float(primary_frequency[column]),
            "test_sensitivity_top10_frequency": float(sensitivity_frequency[column]),
            "test_paired_n": len(differences), "test_paired_tumor_minus_normal_median": observed_median,
            "test_paired_median_bootstrap_low": low, "test_paired_median_bootstrap_median": median,
            "test_paired_median_bootstrap_high": high,
            "prespecified_expression_direction": expected_direction,
            "prespecified_direction_replicated": direction_replicated,
        })
    candidates = pd.DataFrame(candidate_rows)
    frozen20_spearman = spearman(
        candidates.validation_top10_frequency.to_numpy(), candidates.test_top10_frequency.to_numpy())
    overlap_ci = bootstrap_mean(
        sensitivity_frame.top10_overlap.to_numpy(), protocol["bootstrap"]["draws"],
        protocol["bootstrap"]["seed"] + 1000)
    spearman_ci = bootstrap_mean(
        sensitivity_frame.spearman.to_numpy(), protocol["bootstrap"]["draws"],
        protocol["bootstrap"]["seed"] + 2000)

    print("【阶段 5/5】原样保存一次性 Test 结果和访问审计", flush=True)
    output = args.output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.savez_compressed(
        temporary / "test_predicted_residuals.npz", patient_ids=test.index.astype(str).to_numpy(),
        expression_sample=np.asarray(test_ids), genes=genes,
        **{method: values.astype(np.float32) for method, values in predictions.items()})
    sensitivity_frame.to_csv(temporary / "test_input_sensitivity.csv", index=False)
    stability.to_csv(temporary / "frequency_stability.csv", index=False)
    candidates.to_csv(temporary / "frozen_candidate_test.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(temporary / "test_patient_top10.csv.gz", index=False)
    run = {
        "status": "tcga_locked_test_confirmation_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(0), "dtype": "float64_fit_bootstrap_float32_storage",
        "protocol_sha256": actual_protocol_sha,
        "test_access": {"formal_access_n": 1, "patient_n": len(test), "paired_normal_n": len(paired_patients)},
        "design": {"depmap_train_n": len(train), "kidney_train_n": 0, "alpha": ALPHA,
                   "ranking_gene_n": int(universe.sum()), "top_k": TOP_K,
                   "candidate_n": len(candidates), "candidate_reranking": False,
                   "model_or_mapping_tuning_on_test": False, "feature_counts": feature_counts},
        "primary_results": {
            "train_test_active_frequency_spearman": float(stability.loc[stability.comparison.eq("train_vs_test"), "active_frequency_spearman"].iloc[0]),
            "train_test_top100_overlap": float(stability.loc[stability.comparison.eq("train_vs_test"), "top100_frequency_overlap"].iloc[0]),
            "validation_test_active_frequency_spearman": float(stability.loc[stability.comparison.eq("validation_vs_test"), "active_frequency_spearman"].iloc[0]),
            "validation_test_top100_overlap": float(stability.loc[stability.comparison.eq("validation_vs_test"), "top100_frequency_overlap"].iloc[0]),
            "frozen20_validation_test_frequency_spearman": frozen20_spearman,
            "test_mapping_top10_overlap_mean": float(sensitivity_frame.top10_overlap.mean()),
            "test_mapping_top10_overlap_mean_ci_low": overlap_ci[0],
            "test_mapping_top10_overlap_mean_ci_high": overlap_ci[2],
            "test_mapping_spearman_mean": float(sensitivity_frame.spearman.mean()),
            "test_mapping_spearman_mean_ci_low": spearman_ci[0],
            "test_mapping_spearman_mean_ci_high": spearman_ci[2],
        },
        "rules": protocol,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "TCGA Test has no functional dependency labels; agreement is not accuracy.",
            "Test was accessed after protocol freeze, but the broader project contains earlier exploratory model development.",
            "Mean-shift mappings cannot separate platform effects from tumor-versus-cell-line biology.",
            "Paired tumor-normal expression has only fourteen Test participants and is not a toxicity assay.",
            "Bootstrap intervals are descriptive and no multiplicity-adjusted gene-level confirmation is claimed.",
        ],
    }
    output_names = ["test_predicted_residuals.npz", "test_input_sensitivity.csv",
                    "frequency_stability.csv", "frozen_candidate_test.csv", "test_patient_top10.csv.gz"]
    run["output_sha256"] = {name: sha256(temporary / name) for name in output_names}
    (temporary / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    temporary.rename(output)

    primary = run["primary_results"]
    print("【Test队列稳定性】", flush=True)
    print(f"  Train↔Test｜频率Spearman {primary['train_test_active_frequency_spearman']:.4f}｜Top-100重叠 {primary['train_test_top100_overlap']:.4f}", flush=True)
    print(f"  Validation↔Test｜频率Spearman {primary['validation_test_active_frequency_spearman']:.4f}｜Top-100重叠 {primary['validation_test_top100_overlap']:.4f}", flush=True)
    print(f"  冻结20 Validation↔Test｜频率Spearman {primary['frozen20_validation_test_frequency_spearman']:.4f}", flush=True)
    print("【Test映射敏感性】", flush=True)
    print(f"  Top-10重叠 {primary['test_mapping_top10_overlap_mean']:.4f}｜95%区间 [{primary['test_mapping_top10_overlap_mean_ci_low']:.4f}, {primary['test_mapping_top10_overlap_mean_ci_high']:.4f}]", flush=True)
    print(f"  Spearman {primary['test_mapping_spearman_mean']:.4f}｜95%区间 [{primary['test_mapping_spearman_mean_ci_low']:.4f}, {primary['test_mapping_spearman_mean_ci_high']:.4f}]", flush=True)
    for gene in priority_directions:
        row = candidates[candidates.Gene.eq(gene)].iloc[0]
        print(f"【配对表达】{gene}｜中位差 {row.test_paired_tumor_minus_normal_median:+.4f}｜"
              f"区间 [{row.test_paired_median_bootstrap_low:+.4f}, {row.test_paired_median_bootstrap_high:+.4f}]｜"
              f"预设方向 {'复现' if row.prespecified_direction_replicated else '未复现'}", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】Test只确认稳定性和表达方向；没有患者功能标签，不能验证依赖准确率。", flush=True)


if __name__ == "__main__":
    main()
