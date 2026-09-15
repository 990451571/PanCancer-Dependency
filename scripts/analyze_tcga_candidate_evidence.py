"""Assemble local evidence for stable TCGA-KIRC dependency hypotheses.

The discovery ordering uses TCGA-train patient consensus across the two neutral
expression mappings.  Validation stability, observed DepMap ccRCC dependency,
three Sanger renal screens, and TCGA tumor-normal expression are reported as
separate evidence columns.  No composite biological score or final target list
is produced.
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
from build_context_module_stage0 import GeneCanonicalizer, sample_type
from prepare_tcga_expression_bridge import align_expression


MAPPINGS = ("nonkidney_shift_driver_neutral", "kidney_shift_driver_neutral")
SANGER_MODELS = ("769-P", "LB1047-RCC", "RCC-FG2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_summary(values):
    count = np.isfinite(values).sum(axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(total, count, out=np.full(values.shape[1], np.nan), where=count > 0)
    median = np.nanmedian(values, axis=0)
    return count, mean, median


def leave_one_out_worst_mean(values):
    result = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        current = values[:, column]
        current = current[np.isfinite(current)]
        if len(current) >= 3:
            result[column] = np.max((current.sum() - current) / (len(current) - 1))
    return result


def consensus_frequency(top10, patient_ids, split):
    current = top10[top10.method.isin(MAPPINGS)]
    counts = current.groupby(["patient_id", "Gene"]).method.nunique()
    shared = counts[counts.eq(len(MAPPINGS))].reset_index()[["patient_id", "Gene"]]
    patient_split = pd.Series(split, index=patient_ids)
    shared["split"] = shared.patient_id.map(patient_split)
    rows = []
    for subset in ("train", "validation"):
        denominator = int(np.sum(split == subset))
        frequency = shared[shared.split.eq(subset)].Gene.value_counts() / denominator
        rows.append(frequency.rename(f"{subset}_patient_consensus_top10_frequency"))
    return pd.concat(rows, axis=1).fillna(0)


def sanger_binary_table(phase_a, phase_b):
    a = pd.read_csv(phase_a)[["Gene", "sanger_binary_dependency"]].copy()
    a["model_name"] = "769-P"
    b = pd.read_csv(phase_b)[["model_name", "Gene", "sanger_binary_dependency"]]
    long = pd.concat([a, b], ignore_index=True)
    if set(long.model_name) != set(SANGER_MODELS):
        raise ValueError("Sanger 肾癌模型集合变化")
    wide = long.pivot(index="Gene", columns="model_name", values="sanger_binary_dependency")
    wide = wide.rename(columns={name: f"sanger_{name}_binary_dependency" for name in SANGER_MODELS})
    columns = list(wide.columns)
    wide["sanger_renal_observed_n"] = wide[columns].notna().sum(axis=1)
    wide["sanger_renal_dependent_n"] = wide[columns].sum(axis=1, skipna=True)
    return wide


def shared_external_top10(phase_a_top, phase_b_top):
    a = pd.read_csv(phase_a_top)
    a = set(a[(a.method == "expression_only") & (a.list == "predicted_top10")].Gene)
    b = pd.read_csv(phase_b_top)
    sets = [set(b[(b.model_name == name) & (b.method == "mapped_expression_only") &
                  (b.list == "predicted_top10")].Gene) for name in ("LB1047-RCC", "RCC-FG2")]
    return a.intersection(*sets)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--transfer-dir", type=Path, default=root / "outputs/tcga_patient_transfer_v1")
    parser.add_argument("--bridge-dir", type=Path, default=root / "data/processed/tcga_kirc_expression_bridge_v1")
    parser.add_argument("--expression", type=Path, default=source / "data/raw/HiSeqV2")
    parser.add_argument("--hgnc", type=Path, default=source / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--phase-a", type=Path, default=root / "outputs/sanger_phase_a_769p_v1/per_gene_predictions.csv.gz")
    parser.add_argument("--phase-b", type=Path, default=root / "outputs/sanger_phase_b_v1/per_gene_predictions.csv.gz")
    parser.add_argument("--phase-a-top", type=Path, default=root / "outputs/sanger_phase_a_769p_v1/top10_rankings.csv")
    parser.add_argument("--phase-b-top", type=Path, default=root / "outputs/sanger_phase_b_v1/top10_rankings.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/tcga_candidate_evidence_v2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    baseline.configure_device("cuda")
    print("【阶段 1/4】校验患者预测、功能数据和锁定分组", flush=True)
    transfer_run = json.loads((args.transfer_dir / "run.json").read_text())
    for name, expected in transfer_run["output_sha256"].items():
        if sha256(args.transfer_dir / name) != expected:
            raise ValueError(f"患者预测输出哈希不一致：{name}")
    with np.load(args.transfer_dir / "predicted_residuals.npz", allow_pickle=False) as archive:
        patient_ids = archive["patient_ids"].astype(str)
        split = archive["split"].astype(str)
        prediction_genes = archive["genes"].astype(str)
    with np.load(args.bridge_dir / "expression_inputs.npz", allow_pickle=False) as archive:
        bridge_ids = archive["patient_ids"].astype(str)
        raw_tumor_expression = archive["raw_expression"].astype(float)
    if bridge_ids.tolist() != patient_ids.tolist():
        raise ValueError("患者预测与表达桥接顺序不一致")
    if set(split) != {"train", "validation"} or len(patient_ids) != 209:
        raise ValueError("患者队列或锁定 Test 隔离变化")
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    if genes.tolist() != prediction_genes.tolist():
        raise ValueError("患者预测与 DepMap 基因顺序不一致")
    top10 = pd.read_csv(args.transfer_dir / "patient_top10.csv.gz")
    candidates = pd.read_csv(args.transfer_dir / "candidate_summary.csv.gz")
    if set(top10.patient_id) != set(patient_ids):
        raise ValueError("患者排名覆盖不完整")
    print(f"【固定口径】患者 {len(patient_ids)}｜Train {(split=='train').sum()}｜Validation {(split=='validation').sum()}｜Test 0｜不合成总分", flush=True)
    if args.dry_run:
        print("【检查通过】未计算候选证据、未写入结果。", flush=True)
        return

    print("【阶段 2/4】汇总双映射患者共识与 Train/Validation 稳定性", flush=True)
    evidence = pd.DataFrame(index=pd.Index(genes, name="Gene"))
    evidence = evidence.join(consensus_frequency(top10, patient_ids, split))
    neutral = candidates[candidates.method.isin(MAPPINGS)]
    for subset in ("train", "validation"):
        current = neutral[neutral.split.eq(subset)].pivot(
            index="Gene", columns="method", values="mean_predicted_residual")
        for method in MAPPINGS:
            evidence[f"{subset}_{method}_mean_residual"] = current[method]
        evidence[f"{subset}_worst_mapping_mean_residual"] = current[list(MAPPINGS)].max(axis=1)
        evidence[f"{subset}_mapping_mean_residual_gap"] = (
            current[MAPPINGS[0]] - current[MAPPINGS[1]]).abs()
    evidence[[column for column in evidence if "frequency" in column]] = evidence[
        [column for column in evidence if "frequency" in column]].fillna(0)
    eligible = evidence.index.isin(candidates.Gene.unique())
    evidence["discovery_rank"] = np.nan
    evidence.loc[eligible, "discovery_rank"] = evidence.loc[eligible].sort_values(
        ["train_patient_consensus_top10_frequency", "train_worst_mapping_mean_residual"],
        ascending=[False, True], kind="stable").reset_index().reset_index().set_index("Gene")["index"] + 1
    evidence["validation_rank"] = np.nan
    evidence.loc[eligible, "validation_rank"] = evidence.loc[eligible].sort_values(
        ["validation_patient_consensus_top10_frequency", "validation_worst_mapping_mean_residual"],
        ascending=[False, True], kind="stable").reset_index().reset_index().set_index("Gene")["index"] + 1

    print("【阶段 3/4】加入真实细胞系依赖、Sanger肾癌和肿瘤表达证据", flush=True)
    labels = models.OncotreeLineage.to_numpy()
    nonkidney = labels != "Kidney"
    ccrcc = models.clear_cell_renal_cell_carcinoma.to_numpy(dtype=bool)
    kidney_other = (labels == "Kidney") & ~ccrcc
    mean_non = baseline.observed_mean(matrices["dependency"][nonkidney], axis=0)
    ccrcc_residual = matrices["dependency"][ccrcc] - mean_non
    other_residual = matrices["dependency"][kidney_other] - mean_non
    c_n, c_mean, c_median = observed_summary(ccrcc_residual)
    o_n, o_mean, o_median = observed_summary(other_residual)
    evidence["depmap_ccrcc_n"] = c_n
    evidence["depmap_ccrcc_mean_residual"] = c_mean
    evidence["depmap_ccrcc_median_residual"] = c_median
    evidence["depmap_ccrcc_fraction_residual_le_m0_5"] = np.divide(
        np.sum(np.isfinite(ccrcc_residual) & (ccrcc_residual <= -.5), axis=0), c_n,
        out=np.full(len(genes), np.nan), where=c_n > 0)
    evidence["depmap_ccrcc_leave1out_worst_mean_residual"] = leave_one_out_worst_mean(ccrcc_residual)
    evidence["depmap_other_kidney_n"] = o_n
    evidence["depmap_other_kidney_mean_residual"] = o_mean
    evidence["depmap_ccrcc_minus_other_kidney_mean"] = c_mean - o_mean
    tensor = torch.as_tensor(matrices["dependency"][nonkidney] - mean_non,
                             dtype=torch.float64, device="cuda")
    evidence["depmap_nonkidney_residual_q10"] = torch.nanquantile(tensor, .10, dim=0).cpu().numpy()
    evidence = evidence.join(sanger_binary_table(args.phase_a, args.phase_b))
    shared = shared_external_top10(args.phase_a_top, args.phase_b_top)
    evidence["shared_three_renal_absolute_prediction_top10"] = evidence.index.isin(shared)

    header = pd.read_csv(args.expression, sep="\t", nrows=0).columns.astype(str).tolist()
    normal_samples = [sample for sample in header[1:] if sample_type(sample) == "11"]
    if len(normal_samples) != 72:
        raise ValueError(f"TCGA 正常表达样本数量变化：{len(normal_samples)}")
    normal_expression, normal_audit = align_expression(
        args.expression, normal_samples, genes, GeneCanonicalizer(args.hgnc))
    normal_mean = np.nanmean(normal_expression.to_numpy(dtype=float), axis=0)
    expression = pd.DataFrame(raw_tumor_expression - normal_mean, index=patient_ids, columns=genes)
    for subset in ("train", "validation"):
        values = expression.loc[patient_ids[split == subset]]
        evidence[f"tcga_{subset}_expression_delta_median"] = values.median(axis=0)
        evidence[f"tcga_{subset}_expression_delta_positive_fraction"] = (values > 0).mean(axis=0)
    evidence["common_essential"] = common_essential

    print("【阶段 4/4】保存分层证据表与训练期前100候选", flush=True)
    full = evidence.reset_index().sort_values("discovery_rank", na_position="last")
    top100 = full[full.discovery_rank.notna()].head(100).copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    full.to_csv(temporary / "candidate_evidence.csv.gz", index=False)
    top100.to_csv(temporary / "train_discovery_top100.csv", index=False)
    audit = {
        "status": "exploratory_patient_candidate_evidence_no_final_selection",
        "elapsed_seconds": time.monotonic() - started,
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64",
        "design": {"patient_n": len(patient_ids), "train_n": int((split == "train").sum()),
                   "validation_n": int((split == "validation").sum()), "locked_test_n_used": 0,
                   "depmap_ccrcc_n": int(ccrcc.sum()), "sanger_renal_n": len(SANGER_MODELS),
                   "tcga_normal_expression_n": len(normal_samples),
                   "tcga_normal_aligned_gene_n": normal_audit["aligned_finite_gene_n"]},
        "rules": {
            "discovery_order": "TCGA train cross-mapping patient Top-10 consensus frequency, then worst-map mean residual",
            "validation": "Reported separately; not included in discovery rank",
            "functional": "Observed DepMap ccRCC residual versus non-Kidney gene mean; leave-one-cell-line-out worst mean retained",
            "sanger": "Binary Project Score dependency for 769-P, LB1047-RCC and RCC-FG2",
            "expression": "TCGA raw tumor log expression minus the 72-sample normal-tissue mean, Train and Validation separate",
            "aggregation": "No composite biological score and no final candidate threshold",
        },
        "source_sha256": {"transfer_run": sha256(args.transfer_dir / "run.json"),
                          "candidate_summary": sha256(args.transfer_dir / "candidate_summary.csv.gz"),
                          "patient_top10": sha256(args.transfer_dir / "patient_top10.csv.gz"),
                          "baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "bridge_inputs": sha256(args.bridge_dir / "expression_inputs.npz"),
                          "expression": sha256(args.expression), "hgnc": sha256(args.hgnc),
                          "phase_a": sha256(args.phase_a), "phase_b": sha256(args.phase_b),
                          "script": sha256(Path(__file__))},
        "limitations": [
            "Patient predictions have no functional truth labels.",
            "DepMap ccRCC and the three renal Sanger screens were examined in earlier analyses and are not independent confirmation.",
            "TCGA Validation is a stability split, not a functional validation set.",
            "Sanger binary calls come from an older BAGEL pipeline and only three renal models are available.",
            "Tumor overexpression does not establish tumor-specific essentiality or normal-tissue safety.",
            "No multiplicity-adjusted inferential target selection is performed.",
        ],
    }
    audit["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(output)
    print("【训练期共识前15｜仅供后续验证】", flush=True)
    for row in top100.head(15).itertuples():
        print(f"  {int(row.discovery_rank):02d} {row.Gene}｜Train共识 {row.train_patient_consensus_top10_frequency:.3f}｜"
              f"Validation共识 {row.validation_patient_consensus_top10_frequency:.3f}｜"
              f"ccRCC残差 {row.depmap_ccrcc_mean_residual:+.3f}｜Sanger依赖 {row.sanger_renal_dependent_n:.0f}/{row.sanger_renal_observed_n:.0f}", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】前100只是训练期排序；必须逐层排除泛癌依赖、单模型驱动和正常组织风险。", flush=True)


if __name__ == "__main__":
    main()
