#!/usr/bin/env python3
"""Freeze the final evidence matrix and project-level claim boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段：{missing}")


def finite_lt_zero(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & values.lt(0)


def evidence_class(row: pd.Series) -> tuple[str, str, str]:
    if row["direct_peer_reviewed_opposition"]:
        return (
            "E4_直接反向证据",
            "存在同行评议的直接 loss-of-function 反向结果，当前预测不能解释为稳定治疗脆弱性。",
            "在患者来源 ccRCC 模型中用同一扰动终点解决方向冲突。",
        )
    if (row["orthogonal_rnai_support"] and row["rnai_ccrcc_specific_support"]
            and row["direct_peer_reviewed_support"] and row["normal_organoid_liability"]):
        return (
            "E1_功能最一致但无治疗窗",
            "CRISPR、RNAi和直接文献方向一致，且RNAi支持相对其他Kidney更强；正常肾类器官存在直接功能风险。",
            "患者来源 ccRCC 与匹配正常肾模型的剂量化扰动实验。",
        )
    if row["orthogonal_rnai_support"] and row["normal_organoid_liability"]:
        return (
            "E2_肾谱系功能复现伴正常肾风险",
            "RNAi绝对依赖区间低于0，但ccRCC相对其他Kidney的区间跨0；正常肾类器官存在直接功能风险。",
            "证明ccRCC亚型选择性，并与正常成人肾细胞进行平行扰动。",
        )
    if row["orthogonal_rnai_support"]:
        return (
            "E2_肾谱系功能复现但特异性不足",
            "RNAi绝对依赖区间低于0，但未证明ccRCC相对其他Kidney更强，且缺少患者来源功能真值。",
            "患者来源ccRCC独立扰动、其他Kidney对照和正常肾安全窗实验。",
        )
    if (row["clinical_stage_tractability"] and row["direct_peer_reviewed_support"]
            and row["paired_tumor_upregulation"]):
        return (
            "E2_可成药与肿瘤表达支持但功能复现不足",
            "存在直接文献、临床阶段tractability和肿瘤高表达，但DRIVE绝对依赖区间跨0。",
            "在独立患者来源ccRCC模型中复现基础依赖，而非仅依赖表达或药物记录。",
        )
    if row["sanger_all_three_support"]:
        return (
            "E3_细胞系跨平台信号",
            "三个Sanger肾癌模型均显示依赖，但RNAi或直接文献未形成独立一致证据。",
            "正交扰动复现并检验ccRCC相对其他Kidney的特异性。",
        )
    return (
        "E5_计算候选证据不足",
        "候选来自冻结计算排序，但现有正交功能证据不足或未覆盖。",
        "先获得独立功能扰动证据，再讨论患者迁移或治疗价值。",
    )


def build_matrix(celltype: pd.DataFrame, frozen: pd.DataFrame, liability: pd.DataFrame,
                 locked_test: pd.DataFrame, eligible_patient_resource_n: int) -> pd.DataFrame:
    needed = [
        "Gene", "discovery_rank", "train_patient_consensus_top10_frequency",
        "validation_patient_consensus_top10_frequency",
        "depmap_ccrcc_leave1out_worst_mean_residual",
        "depmap_ccrcc_minus_other_kidney_mean", "sanger_renal_dependent_n",
        "sanger_renal_observed_n", "peer_reviewed_direct_support_n",
        "peer_reviewed_direct_opposition_n", "non_peer_reviewed_direct_support_n",
        "clinical_stage_tractability_true_n", "drug_and_clinical_candidate_n",
        "drive_ccrcc_n", "drive_ccrcc_residual_mean", "drive_ccrcc_residual_ci_low",
        "drive_ccrcc_residual_ci_high", "drive_ccrcc_minus_other_kidney_mean",
        "drive_ccrcc_minus_other_kidney_ci_low", "drive_ccrcc_minus_other_kidney_ci_high",
        "gtex_kidney_max_median_tpm", "hpa_renal_cell_type_positive_n",
        "hpa_renal_cell_type_max_ncpm", "tcga_train_paired_median",
        "tcga_train_paired_median_ci_low", "tcga_train_paired_median_ci_high",
        "tcga_validation_paired_median",
    ]
    require_columns(celltype, needed, "candidate_celltype_window_summary")
    require_columns(frozen, ["Gene", "validation_rank", "common_essential"], "frozen_candidates")
    if len(celltype) != 20 or celltype["Gene"].nunique() != 20:
        raise ValueError("最终候选表不是唯一的20个冻结基因")
    if set(celltype["Gene"]) != set(frozen["Gene"]):
        raise ValueError("细胞类型审计与冻结候选集合不一致")
    if frozen["common_essential"].fillna(False).astype(bool).any():
        raise ValueError("冻结候选中意外包含 common-essential 基因")
    require_columns(locked_test, [
        "Gene", "discovery_rank", "test_top10_frequency", "test_sensitivity_top10_frequency",
        "test_paired_n", "test_paired_tumor_minus_normal_median",
        "test_paired_median_bootstrap_low", "test_paired_median_bootstrap_median",
        "test_paired_median_bootstrap_high", "prespecified_expression_direction",
        "prespecified_direction_replicated",
    ], "locked_test_candidates")
    if len(locked_test) != 20 or set(locked_test["Gene"]) != set(celltype["Gene"]):
        raise ValueError("Locked Test候选与冻结20候选不一致")

    out = celltype[needed].merge(
        frozen[["Gene", "validation_rank", "common_essential"]],
        on="Gene", how="left", validate="one_to_one")
    test_columns = [column for column in locked_test.columns if column not in {"discovery_rank"}]
    out = out.merge(locked_test[test_columns], on="Gene", how="left", validate="one_to_one")
    out["validation_top100_retained"] = out["validation_rank"].le(100)
    out["depmap_kidney_lineage_direction"] = (
        finite_lt_zero(out["depmap_ccrcc_leave1out_worst_mean_residual"])
        & finite_lt_zero(out["depmap_ccrcc_minus_other_kidney_mean"]))
    out["sanger_all_three_support"] = (
        out["sanger_renal_observed_n"].eq(3) & out["sanger_renal_dependent_n"].eq(3))
    out["orthogonal_rnai_support"] = finite_lt_zero(out["drive_ccrcc_residual_ci_high"])
    out["rnai_ccrcc_specific_support"] = (
        out["orthogonal_rnai_support"]
        & finite_lt_zero(out["drive_ccrcc_minus_other_kidney_ci_high"]))
    out["direct_peer_reviewed_support"] = out["peer_reviewed_direct_support_n"].gt(0)
    out["direct_peer_reviewed_opposition"] = out["peer_reviewed_direct_opposition_n"].gt(0)
    liability_counts = liability.groupby("gene").size()
    out["normal_organoid_liability_record_n"] = out["Gene"].map(liability_counts).fillna(0).astype(int)
    out["normal_organoid_liability"] = out["normal_organoid_liability_record_n"].gt(0)
    out["paired_tumor_upregulation"] = (
        out["tcga_train_paired_median_ci_low"].gt(0)
        & out["tcga_validation_paired_median"].gt(0))
    out["clinical_stage_tractability"] = out["clinical_stage_tractability_true_n"].gt(0)
    out["patient_derived_functional_truth_available"] = eligible_patient_resource_n > 0
    out["locked_test_used_for_reranking"] = False
    out["locked_test_functional_truth"] = False

    classified = out.apply(evidence_class, axis=1, result_type="expand")
    classified.columns = ["evidence_class", "final_interpretation", "required_next_evidence"]
    out = pd.concat([out, classified], axis=1)
    out["claim_ceiling"] = np.where(
        out["patient_derived_functional_truth_available"],
        "取决于患者来源功能结果",
        "最高为细胞系/表达驱动的计算假设；不能称患者特异依赖或临床靶点",
    )
    return out.sort_values("discovery_rank").reset_index(drop=True)


def select_metric(frame: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"指标未唯一匹配：{conditions}，数量={len(selected)}")
    return selected.iloc[0]


def build_claims(internal_bootstrap: pd.DataFrame, external_bootstrap: pd.DataFrame,
                 cohort: pd.DataFrame, sensitivity: pd.DataFrame, renal_context: pd.DataFrame,
                 benchmark_summary: pd.DataFrame, benchmark_bootstrap: pd.DataFrame,
                 locked_test_run: dict, eligible_patient_resource_n: int) -> pd.DataFrame:
    internal = select_metric(
        internal_bootstrap, metric="ndcg_at_10_gain", estimand="patient_equal")
    external = select_metric(
        external_bootstrap,
        comparison="direct_expression_z_residual_vs_training_standardized_prior",
        metric="ndcg_at_10_gain", estimand="model_weighted")
    mapped = select_metric(
        external_bootstrap,
        comparison="mapped_expression_z_residual_vs_training_standardized_prior",
        metric="ndcg_at_10_gain", estimand="model_weighted")
    renal = select_metric(renal_context, metric="ndcg_at_10_gain")
    primary_cohort = select_metric(cohort, method="nonkidney_shift_driver_neutral")
    mapping = sensitivity.loc[sensitivity["comparison"].eq("mapping_neutral")]
    if len(mapping) == 0:
        raise ValueError("缺少患者表达映射敏感性结果")
    benchmark = benchmark_summary.loc[benchmark_summary["estimand"].eq("lineage_equal")].copy()
    frozen_method = "expression_kernel_ridge_frozen"
    competitors = benchmark.loc[benchmark["method"].isin(
        ["annotation_ridge", "expression_knn", "expression_pcr_ridge"])]
    if len(competitors) != 3 or not benchmark["method"].eq(frozen_method).any():
        raise ValueError("baseline benchmark方法集合不完整")
    strongest = competitors.sort_values("ndcg_at_10", ascending=False).iloc[0]
    frozen_row = benchmark.loc[benchmark["method"].eq(frozen_method)].iloc[0]
    head_to_head = select_metric(
        benchmark_bootstrap,
        comparison=f"{frozen_method}_vs_{strongest['method']}",
        metric="ndcg_at_10", estimand="patient_equal")
    test_primary = locked_test_run["primary_results"]
    baseline_verdict = "优于最强baseline" if head_to_head["ci_low"] > 0 else "未显示优于最强baseline"

    rows = [
        {
            "claim": "表达残差模型可预测泛癌选择性依赖排序",
            "verdict": "支持",
            "evidence_type": "事实",
            "basis": (f"内部完整癌系留出患者等权 ΔNDCG={internal['mean']:+.4f}，"
                      f"95%区间[{internal['ci_low']:+.4f},{internal['ci_high']:+.4f}]") ,
            "boundary": "目标是细胞系选择性残差排序，不是患者功能依赖。",
        },
        {
            "claim": "冻结表达核岭模型优于公平比较中的最强baseline",
            "verdict": baseline_verdict,
            "evidence_type": "事实",
            "basis": (f"癌系等权NDCG：冻结核岭={frozen_row['ndcg_at_10']:.4f}，"
                      f"最强baseline {strongest['method']}={strongest['ndcg_at_10']:.4f}；"
                      f"患者等权配对差={head_to_head['mean']:+.4f}，"
                      f"95%区间[{head_to_head['ci_low']:+.4f},{head_to_head['ci_high']:+.4f}]") ,
            "boundary": "比较限于同一DepMap 24Q4完整癌系留出；不是对Nature Cancer或DeepDEP论文结果的直接胜负判断。",
        },
        {
            "claim": "该泛癌信号可跨CRISPR平台复现",
            "verdict": "支持但绝对准确率有限",
            "evidence_type": "事实",
            "basis": (f"66个纯Sanger模型上直接表达 ΔNDCG={external['mean']:+.4f}，"
                      f"95%区间[{external['ci_low']:+.4f},{external['ci_high']:+.4f}]") ,
            "boundary": "跨平台校准为事后提出，且Top-10绝对重叠较低。",
        },
        {
            "claim": "表达映射足以稳定支持患者级候选",
            "verdict": "仅部分支持",
            "evidence_type": "事实加推断",
            "basis": (f"Train/Validation Top-100重叠={primary_cohort['top100_frequency_overlap']:.2f}；"
                      f"映射敏感性Top-10重叠均值={mapping['top10_overlap'].mean():.4f}，"
                      f"Spearman均值={mapping['spearman'].mean():.4f}") ,
            "boundary": "稳定性不是准确率，TCGA没有依赖标签，且映射选择显著改变头部候选。",
        },
        {
            "claim": "冻结候选在未见TCGA Test患者中保持稳定",
            "verdict": "支持相对稳定性，不支持功能准确率",
            "evidence_type": "事实",
            "basis": (f"Train/Test Top-100重叠={test_primary['train_test_top100_overlap']:.2f}；"
                      f"冻结20 Validation/Test频率Spearman="
                      f"{test_primary['frozen20_validation_test_frequency_spearman']:.4f}；"
                      f"Test映射Top-10重叠={test_primary['test_mapping_top10_overlap_mean']:.4f}"),
            "boundary": "Test没有基因扰动标签，且映射敏感性仍高；新baseline不得把已访问Test当作前瞻确认集。",
        },
        {
            "claim": "模型已证明ccRCC相对其他Kidney具有额外预测优势",
            "verdict": "未建立",
            "evidence_type": "事实",
            "basis": (f"ccRCC减其他Kidney的ΔNDCG={renal['ccrcc_minus_kidney_other']:+.4f}，"
                      f"95%区间[{renal['difference_ci_low']:+.4f},{renal['difference_ci_high']:+.4f}]") ,
            "boundary": "区间跨0，样本仅12个ccRCC患者，且分析是探索性的。",
        },
        {
            "claim": "模型已证明患者特异ccRCC功能依赖",
            "verdict": "当前公开数据不可检验",
            "evidence_type": "事实",
            "basis": f"冻结审计中合格患者来源ccRCC基因扰动资源={eligible_patient_resource_n}",
            "boundary": "无标签TCGA预测不能替代患者来源扰动真值。",
        },
        {
            "claim": "当前候选具有已建立的正常肾治疗窗",
            "verdict": "未建立",
            "evidence_type": "事实加推断",
            "basis": "PAX8和HNF1B存在正常肾类器官直接扰动表型；主要候选同时有正常肾表达暴露。",
            "boundary": "表达与发育类器官表型不能量化成人药物毒性，但足以否定已有安全窗的表述。",
        },
        {
            "claim": "当前结果足以提出临床治疗靶点",
            "verdict": "不支持",
            "evidence_type": "推断",
            "basis": ("缺少患者来源功能真值、正常肾剂量化安全窗和体内疗效链；"
                      f"映射表达外部17模型 ΔNDCG区间[{mapped['ci_low']:+.4f},{mapped['ci_high']:+.4f}]跨0"),
            "boundary": "可以提出实验优先级，不能宣称临床有效性或安全性。",
        },
    ]
    return pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1/candidates.csv")
    parser.add_argument("--celltype", type=Path,
                        default=root / "outputs/candidate_celltype_window_v1/candidate_celltype_window_summary.csv")
    parser.add_argument("--liability", type=Path,
                        default=root / "results/historical/ccrcc_primary_functional_audit_v1/normal_kidney_organoid_liability.csv")
    parser.add_argument("--resource-run", type=Path,
                        default=root / "results/historical/ccrcc_primary_functional_audit_v1/run.json")
    parser.add_argument("--internal-bootstrap", type=Path,
                        default=root / "outputs/selective_dependency_analysis_v1/bootstrap.csv")
    parser.add_argument("--renal-context", type=Path,
                        default=root / "outputs/selective_dependency_analysis_v1/renal_context.csv")
    parser.add_argument("--external-bootstrap", type=Path,
                        default=root / "outputs/sanger_selective_dependency_v1/bootstrap.csv")
    parser.add_argument("--cohort-stability", type=Path,
                        default=root / "outputs/tcga_patient_transfer_v1/cohort_stability.csv")
    parser.add_argument("--input-sensitivity", type=Path,
                        default=root / "outputs/tcga_patient_transfer_v1/input_sensitivity.csv")
    parser.add_argument("--locked-test-candidates", type=Path,
                        default=root / "results/historical/tcga_locked_test_v1/frozen_candidate_test.csv")
    parser.add_argument("--locked-test-run", type=Path,
                        default=root / "results/historical/tcga_locked_test_v1/run.json")
    parser.add_argument("--benchmark-summary", type=Path,
                        default=root / "outputs/selective_dependency_benchmark_v1/overall_summary.csv")
    parser.add_argument("--benchmark-bootstrap", type=Path,
                        default=root / "outputs/selective_dependency_benchmark_v1/bootstrap.csv")
    parser.add_argument("--benchmark-run", type=Path,
                        default=root / "outputs/selective_dependency_benchmark_v1/run.json")
    parser.add_argument("--prior-work", type=Path,
                        default=root / "configs/prior_work_comparison_20260916.csv")
    parser.add_argument("--benchmark-protocol", type=Path,
                        default=root / "configs/dependency_benchmark_protocol_20260916.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/final_evidence_synthesis_v2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    inputs = {
        "frozen_candidates": args.frozen,
        "celltype_window": args.celltype,
        "normal_organoid_liability": args.liability,
        "primary_resource_audit": args.resource_run,
        "internal_bootstrap": args.internal_bootstrap,
        "renal_context": args.renal_context,
        "external_bootstrap": args.external_bootstrap,
        "cohort_stability": args.cohort_stability,
        "input_sensitivity": args.input_sensitivity,
        "locked_test_candidates": args.locked_test_candidates,
        "locked_test_run": args.locked_test_run,
        "baseline_benchmark_summary": args.benchmark_summary,
        "baseline_benchmark_bootstrap": args.benchmark_bootstrap,
        "baseline_benchmark_run": args.benchmark_run,
        "prior_work_comparison": args.prior_work,
        "baseline_benchmark_protocol": args.benchmark_protocol,
    }
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")

    print("【阶段 1/4】校验冻结候选和全部上游证据文件", flush=True)
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if args.dry_run:
        print("【检查通过】候选 20｜不重排｜不生成综合分数｜未写入结果", flush=True)
        return

    frozen = pd.read_csv(args.frozen)
    celltype = pd.read_csv(args.celltype)
    liability = pd.read_csv(args.liability)
    locked_test = pd.read_csv(args.locked_test_candidates)
    locked_test_run = json.loads(args.locked_test_run.read_text())
    resource_run = json.loads(args.resource_run.read_text())
    eligible_n = int(resource_run["counts"]["eligible_patient_derived_ccrcc_gene_perturbation_resource_n"])

    print("【阶段 2/4】构建逐候选功能、特异性、暴露和可成药证据矩阵", flush=True)
    matrix = build_matrix(celltype, frozen, liability, locked_test, eligible_n)
    print("【冻结规则】保留 discovery rank｜证据分类不改变排名｜患者功能真值不可用", flush=True)

    print("【阶段 3/4】审计项目级可支持与不可支持结论", flush=True)
    claims = build_claims(
        pd.read_csv(args.internal_bootstrap), pd.read_csv(args.external_bootstrap),
        pd.read_csv(args.cohort_stability), pd.read_csv(args.input_sensitivity),
        pd.read_csv(args.renal_context), pd.read_csv(args.benchmark_summary),
        pd.read_csv(args.benchmark_bootstrap), locked_test_run, eligible_n)
    class_summary = (matrix.groupby("evidence_class", sort=False)
                     .agg(candidate_n=("Gene", "size"), genes=("Gene", lambda x: ";".join(x)))
                     .reset_index())

    print("【阶段 4/4】保存最终证据矩阵、结论边界和审计记录", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    matrix.to_csv(args.output_dir / "candidate_evidence_matrix.csv", index=False)
    class_summary.to_csv(args.output_dir / "evidence_class_summary.csv", index=False)
    claims.to_csv(args.output_dir / "project_claims.csv", index=False)
    pd.read_csv(args.prior_work).to_csv(args.output_dir / "prior_work_comparison.csv", index=False)
    output_names = ["candidate_evidence_matrix.csv", "evidence_class_summary.csv",
                    "project_claims.csv", "prior_work_comparison.csv"]
    run = {
        "status": "final_evidence_synthesis_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "compute": "CPU evidence-table integration; GPU acceleration is not applicable",
        "design": {
            "frozen_candidate_n": int(len(matrix)),
            "candidate_reranking": False,
            "composite_score": False,
            "locked_tcga_test_used": True,
            "locked_tcga_test_use": "evidence integration only; no reranking, model selection or evidence-class change",
            "baseline_benchmark_used": True,
            "evidence_class_changes_from_test": False,
            "eligible_patient_functional_resource_n": eligible_n,
        },
        "classification_rules": {
            "orthogonal_rnai_support": "DRIVE ccRCC residual 95% bootstrap upper bound < 0",
            "rnai_ccrcc_specific_support": "absolute RNAi support and ccRCC-minus-other-Kidney upper bound < 0",
            "sanger_all_three_support": "dependent in all three frozen Sanger renal models",
            "direct_support_or_opposition": "manually adjudicated peer-reviewed direct loss-of-function evidence",
            "normal_organoid_liability": "at least one frozen direct normal-kidney organoid perturbation record",
            "paired_tumor_upregulation": "Train paired-median lower bound > 0 and Validation paired median > 0",
            "locked_test_expression": "Descriptive replication of four prespecified expression directions; never functional truth",
            "patient_truth": "requires an eligible patient-derived ccRCC gene-perturbation resource",
        },
        "class_counts": dict(zip(class_summary["evidence_class"], class_summary["candidate_n"].astype(int))),
        "source_sha256": {name: sha256(path) for name, path in inputs.items()},
        "output_sha256": {name: sha256(args.output_dir / name) for name in output_names},
        "script_sha256": sha256(Path(__file__)),
        "final_boundary": (
            "The project supports a cross-platform expression-based selective-dependency prioritization workflow "
            "and a frozen experimental shortlist. It does not establish patient-specific ccRCC dependencies, "
            "a normal-kidney therapeutic window, clinical efficacy or clinical safety."
        ),
    }
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)

    for _, row in class_summary.iterrows():
        print(f"【证据分层】{row['evidence_class']}｜{row['candidate_n']}｜{row['genes']}", flush=True)
    print("【项目结论】泛癌内部增益支持｜跨平台有限支持｜患者特异依赖不可检验｜临床靶点不支持", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f}秒｜结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】这是实验优先级矩阵，不是患者功能验证、疗效证明或安全性证明。", flush=True)


if __name__ == "__main__":
    main()
