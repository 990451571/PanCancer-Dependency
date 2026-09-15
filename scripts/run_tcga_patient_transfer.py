"""Generate TCGA-KIRC selective-dependency hypotheses with mapping sensitivities.

The fixed expression ridge is trained on non-Kidney DepMap models.  TCGA test
patients remain absent.  Non-Kidney mean-shift expression with neutral driver
indicators is primary; Kidney mean shift and historical TCGA driver calls are
sensitivity inputs.  With no patient functional labels, outputs are hypotheses
and stability diagnostics rather than accuracy estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
from run_selective_dependency import average_ranks


ALPHA = 100000.0
TOP_K = 10
DRIVERS = baseline.DRIVERS
METHODS = (
    "nonkidney_shift_driver_neutral",
    "kidney_shift_driver_neutral",
    "nonkidney_shift_stage0_drivers",
    "kidney_shift_stage0_drivers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_kernel(models, genes, matrices, train, held_expression, held_drivers):
    torch = baseline.TORCH
    kernel = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    held_kernel = torch.zeros((len(held_expression), len(train)), dtype=torch.float64, device="cuda")
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    train_lineage = (labels[train, None] == categories[None, :]).astype(float)
    held_lineage = np.zeros((len(held_expression), len(categories)), dtype=float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in DRIVERS]
    blocks = (
        (train_lineage, held_lineage),
        (matrices["mutation"][np.ix_(train, driver_columns)], held_drivers),
        (baseline.observed_mean(matrices["expression"][train], axis=1)[:, None],
         baseline.observed_mean(held_expression, axis=1)[:, None]),
        (matrices["expression"][train], held_expression),
    )
    counts = []
    for train_values, held_values in blocks:
        counts.append(baseline.add_kernel(kernel, held_kernel, train_values, held_values))
    return kernel, held_kernel, counts


def rank_agreement(left, right, universe, top_k=TOP_K):
    valid = universe & np.isfinite(left) & np.isfinite(right)
    index = np.flatnonzero(valid)
    left_top = index[np.argsort(left[index], kind="stable")[:top_k]]
    right_top = index[np.argsort(right[index], kind="stable")[:top_k]]
    lr, rr = average_ranks(left[index]), average_ranks(right[index])
    lr, rr = lr - lr.mean(), rr - rr.mean()
    denominator = np.sqrt(np.dot(lr, lr) * np.dot(rr, rr))
    return len(set(left_top) & set(right_top)) / top_k, float(np.dot(lr, rr) / denominator)


def summarize_candidates(predictions, patient_ids, split, genes, universe):
    rows, rankings = [], []
    columns = np.flatnonzero(universe)
    for method, values in predictions.items():
        selected_mask = np.zeros((len(values), len(genes)), dtype=bool)
        for row, patient in enumerate(patient_ids):
            valid = columns[np.isfinite(values[row, columns])]
            selected = valid[np.argsort(values[row, valid], kind="stable")[:TOP_K]]
            selected_mask[row, selected] = True
            for rank, column in enumerate(selected, 1):
                rankings.append({"patient_id": patient, "split": split[row], "method": method,
                                 "rank": rank, "Gene": genes[column],
                                 "predicted_residual": values[row, column]})
        for subset in ("train", "validation"):
            use = split == subset
            tensor = torch.as_tensor(values[use][:, columns], dtype=torch.float64, device="cuda")
            mean = torch.nanmean(tensor, dim=0).cpu().numpy()
            q10 = torch.nanquantile(tensor, .10, dim=0).cpu().numpy()
            frequency = selected_mask[use][:, columns].mean(axis=0)
            for local, column in enumerate(columns):
                rows.append({"method": method, "split": subset, "Gene": genes[column],
                             "patient_n": int(use.sum()), "mean_predicted_residual": mean[local],
                             "q10_predicted_residual": q10[local],
                             "top10_selection_frequency": frequency[local]})
    return pd.DataFrame(rows), pd.DataFrame(rankings)


def cohort_stability(candidates):
    rows = []
    for method, group in candidates.groupby("method"):
        wide = group.pivot(index="Gene", columns="split", values="top10_selection_frequency")
        active = (wide.train > 0) | (wide.validation > 0)
        left, right = wide.loc[active, "train"].to_numpy(), wide.loc[active, "validation"].to_numpy()
        lr, rr = average_ranks(left), average_ranks(right)
        lr, rr = lr - lr.mean(), rr - rr.mean()
        denominator = np.sqrt(np.dot(lr, lr) * np.dot(rr, rr))
        correlation = float(np.dot(lr, rr) / denominator) if denominator > 1e-12 else np.nan
        train_top = set(wide.train.sort_values(ascending=False, kind="stable").head(100).index)
        validation_top = set(wide.validation.sort_values(ascending=False, kind="stable").head(100).index)
        rows.append({"method": method, "active_gene_n": int(active.sum()),
                     "active_frequency_spearman": correlation,
                     "top100_frequency_overlap": len(train_top & validation_top) / 100})
    return pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main/data/processed/context_module_stage0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--bridge-dir", type=Path, default=root / "data/processed/tcga_kirc_expression_bridge_v1")
    parser.add_argument("--patients", type=Path, default=source / "patients.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/tcga_patient_transfer_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    baseline.configure_device("cuda")
    print("【阶段 1/4】校验桥接输入、冻结模型和 TCGA Test 隔离", flush=True)
    bridge_run = json.loads((args.bridge_dir / "run.json").read_text())
    for name, expected in bridge_run["output_sha256"].items():
        if sha256(args.bridge_dir / name) != expected:
            raise ValueError(f"桥接输出哈希不一致：{name}")
    with np.load(args.bridge_dir / "expression_inputs.npz", allow_pickle=False) as archive:
        bridge = {name: archive[name] for name in archive.files}
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    if bridge["genes"].tolist() != genes.tolist():
        raise ValueError("患者桥接与 DepMap 基因顺序不一致")
    patient_table = pd.read_csv(args.patients, index_col=0)
    patient_ids, split = bridge["patient_ids"].astype(str), bridge["split"].astype(str)
    locked = set(patient_table.index[patient_table.split.eq("test")].astype(str))
    if set(patient_ids) & locked or set(split) != {"train", "validation"}:
        raise ValueError("锁定 Test 混入患者预测")
    covered = pd.read_csv(args.bridge_dir / "gene_mapping.csv.gz").tcga_covered.to_numpy(dtype=bool)
    universe = covered & ~common_essential
    train = np.flatnonzero(models.OncotreeLineage.to_numpy() != "Kidney")
    print(f"【固定队列】DepMap非 Kidney训练 {len(train)}｜患者 Train {(split=='train').sum()}｜Validation {(split=='validation').sum()}｜Test重叠 0", flush=True)
    print(f"【固定规则】α={ALPHA:g}｜主基因 {universe.sum()}｜双映射×双driver定义｜不使用患者功能标签", flush=True)
    if args.dry_run:
        print("【检查通过】未拟合模型、未生成患者预测、未写入结果。", flush=True)
        return

    print("【阶段 2/4】构建四种患者输入的 GPU 表达核", flush=True)
    nonkidney_expression = bridge["nonkidney_mean_shift_expression"].astype(float)
    kidney_expression = bridge["kidney_mean_shift_expression"].astype(float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    driver_columns = [gene_index[gene] for gene in DRIVERS]
    train_driver = matrices["mutation"][np.ix_(train, driver_columns)]
    neutral = np.broadcast_to(baseline.observed_mean(train_driver, axis=0), (len(patient_ids), len(DRIVERS)))
    stage0 = patient_table.loc[patient_ids, [f"{gene}_mut" for gene in DRIVERS]].to_numpy(dtype=float)
    held_expression = np.vstack([nonkidney_expression, kidney_expression,
                                 nonkidney_expression, kidney_expression])
    held_drivers = np.vstack([neutral, neutral, stage0, stage0])
    kernel, held_kernel, feature_counts = build_kernel(
        models, genes, matrices, train, held_expression, held_drivers)

    print(f"【阶段 3/4】GPU拟合 {len(genes)} 个靶点并生成患者选择性残差", flush=True)
    raw_prediction = baseline.masked_ridge(
        kernel, held_kernel, matrices["dependency"][train], [ALPHA])[ALPHA]
    gene_mean = baseline.observed_mean(matrices["dependency"][train], axis=0)
    residual = raw_prediction - gene_mean
    n = len(patient_ids)
    predictions = {method: residual[index*n:(index+1)*n] for index, method in enumerate(METHODS)}

    agreement_rows = []
    comparisons = {
        "mapping_neutral": (METHODS[0], METHODS[1]),
        "driver_nonkidney": (METHODS[0], METHODS[2]),
        "driver_kidney": (METHODS[1], METHODS[3]),
    }
    for comparison, (left, right) in comparisons.items():
        for row, patient in enumerate(patient_ids):
            overlap, correlation = rank_agreement(predictions[left][row], predictions[right][row], universe)
            agreement_rows.append({"patient_id": patient, "split": split[row],
                                   "comparison": comparison, "top10_overlap": overlap,
                                   "spearman": correlation})
    candidates, rankings = summarize_candidates(predictions, patient_ids, split, genes, universe)
    stability = cohort_stability(candidates)

    print("【阶段 4/4】保存预测、排名稳定性和审计记录", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.savez_compressed(temporary / "predicted_residuals.npz", patient_ids=patient_ids,
                        split=split, genes=genes,
                        **{method: values.astype(np.float32) for method, values in predictions.items()})
    pd.DataFrame(agreement_rows).to_csv(temporary / "input_sensitivity.csv", index=False)
    stability.to_csv(temporary / "cohort_stability.csv", index=False)
    candidates.to_csv(temporary / "candidate_summary.csv.gz", index=False)
    rankings.to_csv(temporary / "patient_top10.csv.gz", index=False)
    run = {
        "status": "patient_selective_dependency_hypothesis_generation",
        "elapsed_seconds": time.monotonic() - started,
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64_fit_float32_storage",
        "design": {"depmap_train_n": len(train), "kidney_train_n": 0,
                   "patient_train_n": int((split == "train").sum()),
                   "patient_validation_n": int((split == "validation").sum()),
                   "locked_test_overlap_n": 0, "target_gene_n": len(genes),
                   "ranking_gene_n": int(universe.sum()), "alpha": ALPHA, "top_k": TOP_K},
        "methods": list(METHODS), "feature_counts": feature_counts,
        "rules": {
            "primary": METHODS[0],
            "mapping_sensitivity": METHODS[1],
            "driver_sensitivity": [METHODS[2], METHODS[3]],
            "driver_neutral": "Held driver indicators set to their non-Kidney training means, giving zero centered kernel contribution",
            "stage0_driver_warning": "TCGA functional mutation calls are broader than the DepMap damaging-mutation definition",
            "target": "Prediction minus non-Kidney training gene mean",
            "ranking": "Lower predicted residual ranks first; TCGA-covered non-common-essential genes only",
            "test": "Locked TCGA test patients absent from fitting, prediction, summaries and saved matrices",
        },
        "source_sha256": {"baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "bridge_run": sha256(args.bridge_dir / "run.json"),
                          "bridge_inputs": sha256(args.bridge_dir / "expression_inputs.npz"),
                          "patients": sha256(args.patients), "script": sha256(Path(__file__))},
        "limitations": [
            "TCGA has no functional dependency labels; these predictions are unvalidated hypotheses.",
            "Bulk-tumor expression includes non-cancer cells and differs biologically from cell-line expression.",
            "Mean-shift mappings cannot separate platform effects from tumor-versus-cell-line biology.",
            "The external Sanger model improved over a prior but had low absolute Top-10 accuracy.",
            "Validation here measures input sensitivity and cohort stability, not dependency accuracy.",
            "No normal-tissue safety, druggability, survival or perturbation evidence is applied yet.",
        ],
    }
    run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(output)

    agreement = pd.DataFrame(agreement_rows)
    print("【输入敏感性】", flush=True)
    for comparison, group in agreement.groupby("comparison"):
        print(f"  {comparison}｜Top-10重叠 {group.top10_overlap.mean():.4f}｜Spearman {group.spearman.mean():.4f}", flush=True)
    stable = stability.set_index("method").loc[METHODS[0]]
    print(f"【队列稳定性｜主输入】活跃基因 {int(stable.active_gene_n)}｜频率Spearman {stable.active_frequency_spearman:.4f}｜Top-100重叠 {stable.top100_frequency_overlap:.4f}", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】患者预测没有功能真值；下一步只能筛选跨输入稳定候选并进行独立生物学验证。", flush=True)


if __name__ == "__main__":
    main()
