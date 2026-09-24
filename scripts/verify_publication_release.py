#!/usr/bin/env python3
"""One-command integrity and environment audit for the publication snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(directory: Path) -> int:
    run_path = directory / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    run = json.loads(run_path.read_text())
    checked = 0
    for name, expected in run.get("output_sha256", {}).items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"版本化结果缺失：{path}")
        if sha256(path) != expected:
            raise ValueError(f"版本化结果哈希不一致：{path}")
        checked += 1
    if checked == 0:
        raise ValueError(f"运行记录没有输出哈希：{run_path}")
    return checked


def verify_environment(root: Path, strict: bool) -> None:
    lock = json.loads((root / "configs/environment_lock.json").read_text())
    mismatches = []
    observed_python = ".".join(map(str, sys.version_info[:3]))
    if observed_python != lock["python"]:
        mismatches.append(f"python {observed_python} != {lock['python']}")
    for package, expected in lock["packages"].items():
        if package == "torch":
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package} 未安装")
            continue
        if actual != expected:
            mismatches.append(f"{package} {actual} != {expected}")
    import torch
    if torch.__version__ != lock["packages"]["torch"]:
        mismatches.append(f"torch {torch.__version__} != {lock['packages']['torch']}")
    if not torch.cuda.is_available():
        mismatches.append("CUDA 不可用")
    elif str(torch.version.cuda) != lock["cuda_runtime_reported_by_torch"]:
        mismatches.append(
            f"torch CUDA {torch.version.cuda} != {lock['cuda_runtime_reported_by_torch']}")
    if strict and mismatches:
        raise RuntimeError("环境锁定不一致：" + "；".join(mismatches))
    print("【环境审计】" + ("完全匹配" if not mismatches else "存在差异：" + "；".join(mismatches)), flush=True)


def verify_claims(root: Path) -> None:
    evidence_dir = root / "results/historical/final_evidence_synthesis_v2"
    matrix = pd.read_csv(evidence_dir / "candidate_evidence_matrix.csv")
    if len(matrix) != 20 or matrix.Gene.nunique() != 20:
        raise ValueError("最终证据矩阵不是唯一的20个候选")
    if matrix.common_essential.astype(bool).any():
        raise ValueError("最终候选意外包含 common-essential")
    if set(matrix.loc[matrix.orthogonal_rnai_support.astype(bool), "Gene"]) != {"PAX8", "HNF1B", "FERMT2"}:
        raise ValueError("RNAi复现候选集合发生变化")
    if set(matrix.loc[matrix.rnai_ccrcc_specific_support.astype(bool), "Gene"]) != {"PAX8"}:
        raise ValueError("RNAi ccRCC特异方向候选集合发生变化")
    if matrix.patient_derived_functional_truth_available.astype(bool).any():
        raise ValueError("患者功能真值状态被意外升级")
    if matrix.locked_test_used_for_reranking.astype(bool).any():
        raise ValueError("Locked Test被意外用于候选重排")
    if set(matrix.loc[matrix.prespecified_direction_replicated.astype(bool), "Gene"]) != {
            "PAX8", "HNF1B", "FERMT2", "CCND1"}:
        raise ValueError("Locked Test预设表达方向集合发生变化")

    benchmark_dir = root / "results/historical/selective_dependency_benchmark_v1"
    benchmark = pd.read_csv(benchmark_dir / "overall_summary.csv")
    benchmark = benchmark.loc[benchmark.estimand.eq("lineage_equal")].set_index("method")
    expected_internal = {
        "training_selectivity_prior": 0.224758,
        "expression_pcr_ridge": 0.441074,
        "expression_kernel_ridge_frozen": 0.417736,
        "expression_kernel_ridge_tuned": 0.440145,
    }
    for method, expected_value in expected_internal.items():
        if abs(float(benchmark.loc[method, "ndcg_at_10"]) - expected_value) > 1e-6:
            raise ValueError(f"内部baseline关键结果发生变化：{method}")
    internal_bootstrap = pd.read_csv(benchmark_dir / "bootstrap.csv")
    comparison = internal_bootstrap.loc[
        internal_bootstrap.comparison.eq(
            "expression_kernel_ridge_frozen_vs_expression_pcr_ridge")
        & internal_bootstrap.metric.eq("ndcg_at_10")
        & internal_bootstrap.estimand.eq("patient_equal")].iloc[0]
    if not (comparison.ci_high < 0):
        raise ValueError("冻结核岭相对PCR的内部结论发生变化")

    external_dir = root / "results/historical/sanger_baseline_benchmark_v1"
    external = pd.read_csv(external_dir / "overall_summary.csv")
    external = external.loc[external.estimand.eq("lineage_equal")].set_index("method")
    if not (external.loc["expression_pcr_ridge", "ndcg_at_10"]
            > external.loc["expression_kernel_ridge_frozen", "ndcg_at_10"]):
        raise ValueError("PCR相对冻结核岭的Sanger顺序发生变化")

    test_dir = root / "results/historical/tcga_locked_test_v1"
    test_run = json.loads((test_dir / "run.json").read_text())
    protocol = root / "configs/tcga_locked_test_protocol_20260916.json"
    if sha256(protocol) != test_run["protocol_sha256"]:
        raise ValueError("锁定Test协议哈希与正式运行不一致")
    if test_run["test_access"] != {"formal_access_n": 1, "patient_n": 51, "paired_normal_n": 14}:
        raise ValueError("锁定Test访问审计发生变化")
    primary = test_run["primary_results"]
    expected = {
        "train_test_top100_overlap": 0.75,
        "frozen20_validation_test_frequency_spearman": 0.7984154302092542,
        "test_mapping_top10_overlap_mean": 0.36078431372549025,
    }
    for name, value in expected.items():
        if abs(float(primary[name]) - value) > 1e-12:
            raise ValueError(f"锁定Test关键结果发生变化：{name}")


def verify_advanced_comparison(root: Path) -> None:
    internal = root / "results/historical/advanced_model_benchmark_v1"
    external = root / "results/historical/sanger_advanced_benchmark_v1"
    audit = json.loads((root / "configs/advanced_model_benchmark_audit_20260920.json").read_text())
    if sha256(internal / "run.json") != audit["internal_run_sha256"]:
        raise ValueError("高级模型审计与内部运行不匹配")
    external_run = json.loads((external / "run.json").read_text())
    if external_run["internal_run_sha256"] != sha256(internal / "run.json"):
        raise ValueError("外部比较不是对应内部选参运行")
    for directory, expected_n, expected_methods in ((internal, 805, 6), (external, 64, 5)):
        metrics = pd.read_csv(directory / "per_model_metrics.csv.gz")
        counts = metrics.groupby("method").ModelID.nunique()
        if len(counts) != expected_methods or not counts.eq(expected_n).all():
            raise ValueError("方法覆盖不一致")
        if metrics.duplicated(["ModelID", "method"]).any():
            raise ValueError("逐模型结果重复")
        observed = metrics.groupby(["heldout_lineage", "method"]).ndcg_at_10.mean().groupby("method").mean()
        summary = pd.read_csv(directory / "overall_summary.csv")
        stored = summary.loc[summary.estimand.eq("lineage_equal")].set_index("method").ndcg_at_10
        if ((observed - stored).abs() > 1e-12).any():
            raise ValueError("癌系等权结果不能从逐模型记录复算")
        convergence = pd.read_csv(directory / "elastic_convergence.csv")
        if not convergence.converged.all():
            raise ValueError("存在未达到停止条件的Elastic Net路径")
    intervals = pd.read_csv(internal / "bootstrap.csv")
    primary = intervals.loc[intervals.estimand.eq("lineage_equal") & intervals.metric.eq("ndcg_at_10")]
    if (primary.ci_low > 0).any():
        raise ValueError("高级模型主结论与区间不一致")
    if external_run["sanger_labels_used_for_tuning"]:
        raise ValueError("外部标签用于调参")
    print("【高级模型审计】内部805｜外部64｜覆盖及均值一致｜无可靠主指标增益", flush=True)


def verify_elastic_numerical_audit(root: Path) -> None:
    directory = root / "results/historical/elastic_net_numerical_audit_v1"
    protocol_path = root / "configs/elastic_net_numerical_audit_protocol_20260920.json"
    protocol = json.loads(protocol_path.read_text())
    run = json.loads((directory / "run.json").read_text())
    if sha256(protocol_path) != run["protocol_sha256"]:
        raise ValueError("数值审计协议不一致")
    historical = root / "results/historical/advanced_model_benchmark_v1/run.json"
    if sha256(historical) != run["historical_run_sha256"]:
        raise ValueError("数值审计历史运行不一致")
    certificate = pd.read_csv(directory / "per_target_certificate.csv.gz")
    if len(certificate) != 19 * 1204 or certificate.duplicated(["heldout_lineage", "Gene"]).any():
        raise ValueError("数值证书覆盖不完整")
    fields = ["refined_kkt", "refined_relative_bound", "old_objective", "refined_objective"]
    if certificate[fields].isna().any().any():
        raise ValueError("数值证书缺失")
    if (certificate.refined_kkt > protocol["kkt_absolute_tolerance"]).any():
        raise ValueError("精化解未达到KKT阈值")
    if (certificate.refined_relative_bound > protocol["relative_suboptimality_bound_tolerance"]).any():
        raise ValueError("精化解误差上界超阈值")
    if (certificate.refined_objective > certificate.old_objective + 1e-10).any():
        raise ValueError("精化目标函数增大")
    folds = pd.read_csv(directory / "folds.csv")
    if len(folds) != 19 or (folds.historical_metric_max_error > 1e-6).any():
        raise ValueError("历史指标复算不一致")
    print("【数值审计】19癌系×1204靶点精化证书通过｜历史指标复算一致", flush=True)


def verify_elastic_upper_grid(root: Path) -> None:
    directory = root / "results/historical/elastic_net_upper_grid_v1"
    protocol_path = root / "configs/elastic_net_upper_grid_protocol_20260920.json"
    protocol = json.loads(protocol_path.read_text())
    run = json.loads((directory / "run.json").read_text())
    if sha256(protocol_path) != run["signature_sha256"][str(protocol_path.relative_to(root))]:
        raise ValueError("扩展网格协议哈希不一致")
    cert = pd.read_csv(directory / "certificates.csv")
    if not cert.target_n.eq(1204).all() or cert[["maximum_kkt", "maximum_relative_bound"]].isna().any().any():
        raise ValueError("扩展网格数值证书不完整")
    if (cert.maximum_kkt > protocol["kkt_absolute_tolerance"]).any() or (cert.maximum_relative_bound > protocol["relative_suboptimality_bound_tolerance"]).any():
        raise ValueError("扩展网格存在未认证拟合")
    inner = cert.loc[cert.scope.eq("inner")]
    if len(inner) != 19 * 5 * 4 or inner.duplicated(["heldout_lineage", "inner_fold", "alpha"]).any():
        raise ValueError("内层路径覆盖错误")
    tuning = pd.read_csv(directory / "tuning.csv")
    if len(tuning) != 19 * 4:
        raise ValueError("扩展网格选参记录不完整")
    for lineage, frame in tuning.groupby("heldout_lineage"):
        expected = frame.sort_values(["inner_lineage_equal_ndcg", "alpha"], ascending=[False, False]).iloc[0].alpha
        if run["selected_alphas"][lineage] != expected or frame.selected.sum() != 1:
            raise ValueError("参数未按固定内层规则选择")
    print("【上界审计】19癌系×5内层折×4个α｜逐靶点数值条件全部通过", flush=True)


def verify_deepdep_expression_recovery(root: Path) -> None:
    directory = root / "results/historical/deepdep_expression_recovery_v1"
    run = json.loads((directory / "run.json").read_text())
    frame = pd.read_csv(directory / "feature_recovery.csv.gz")
    if len(frame) != 6016 or frame.feature_index.tolist() != list(range(6016)):
        raise ValueError("DeepDEP特征顺序或数量改变")
    expected = {"retained_observed": 4678, "recovered_from_expression_only": 706,
                "approved_absent_from_raw_expression": 380, "unresolved_hgnc_symbol": 248,
                "ambiguous_hgnc_alias": 4}
    if frame.status.value_counts().to_dict() != expected:
        raise ValueError("DeepDEP特征恢复分类不一致")
    if int(frame.now_observed.sum()) != 5384 or run["retained_expression_max_abs_difference"] != 0:
        raise ValueError("DeepDEP实测覆盖或历史一致性改变")
    if run["tcga_kirc_test_read"] or not run["non_expression_arrays_unchanged"]:
        raise ValueError("DeepDEP输入修复超出许可范围")
    protocol = json.loads((root / "configs/deepdep_recovered_comparison_protocol_20260921.json").read_text())
    if protocol["inputs"]["matrix_sha256"] != run["recovered_input_sha256"]:
        raise ValueError("下一轮方案未绑定恢复后的输入")
    print("【DeepDEP输入】恢复706｜实测5384｜填补632｜无新性能结论", flush=True)


def verify_deepdep_recovered_results(root: Path) -> None:
    directory = root / "results/historical/deepdep_recovered_comparison_v1"
    run = json.loads((directory / "run.json").read_text())
    protocol_path = root / "configs/deepdep_recovered_comparison_protocol_20260921.json"
    protocol = json.loads(protocol_path.read_text())
    if sha256(protocol_path) != run["signature"]["files"][str(protocol_path.relative_to(root))]:
        raise ValueError("DeepDEP训练协议与结果不匹配")
    metrics = pd.read_csv(directory / "per_model_metrics.csv.gz")
    training = pd.read_csv(directory / "training.csv.gz")
    tuning = pd.read_csv(directory / "tuning.csv.gz")
    variants = ["exp_deepdep_residual_primary", "exp_deepdep_absolute_secondary"]
    methods = ["expression_pcr_ridge", *variants]
    if set(metrics.method) != set(methods) or not metrics.groupby("method").ModelID.nunique().eq(805).all():
        raise ValueError("DeepDEP比较的三方法覆盖不一致")
    if metrics.duplicated(["ModelID", "method"]).any() or metrics.heldout_lineage.nunique() != 19:
        raise ValueError("DeepDEP模型重复或癌系缺失")
    fields = ["ndcg_at_10", "selective_precision_at_10", "top10_overlap", "spearman", "regret"]
    if metrics[fields].isna().any().any() or metrics[fields].isin([float("inf"), -float("inf")]).any().any():
        raise ValueError("DeepDEP评价存在非有限值")
    groups = training.groupby(["outer_lineage", "scope", "inner_fold", "variant", "seed"])
    expected = {(lineage, scope, fold, variant, seed)
                for lineage in metrics.heldout_lineage.unique()
                for scope in ("inner", "outer")
                for fold in (range(1, 6) if scope == "inner" else [0])
                for variant in variants for seed in protocol["deep_training"]["seeds"]}
    if set(groups.groups) != expected or len(expected) != 684:
        raise ValueError("DeepDEP五折或三种子拟合不完整")
    for (lineage, scope, fold, variant, seed), frame in groups:
        end = 100 if scope == "inner" else run["selected_parameters"][lineage][variant]
        if frame.epoch.tolist() != list(range(1, end + 1)):
            raise ValueError("DeepDEP训练epoch不连续或外层重拟合轮数错误")
    for lineage, frame in tuning.groupby("outer_lineage"):
        for method, current in frame.groupby("method"):
            if method == "expression_pcr_ridge":
                values = current.groupby(["rank", "alpha"]).ndcg.mean()
                selected = list(sorted(values.index, key=lambda c: (-values[c], c[0], -c[1]))[0])
            else:
                values = current.groupby("epoch").ndcg.mean()
                selected = int(sorted(values.index, key=lambda e: (-values[e], e))[0])
            if selected != run["selected_parameters"][lineage][method]:
                raise ValueError("DeepDEP/PCR参数不符合训练内选择规则")
    summary = pd.read_csv(directory / "overall_summary.csv")
    recomputed = metrics.groupby(["heldout_lineage", "method"])[fields].mean().groupby("method").mean()
    stored = summary.loc[summary.estimand.eq("lineage_equal")].set_index("method")[fields]
    if ((recomputed-stored).abs() > 1e-12).any().any():
        raise ValueError("DeepDEP总体指标不能从逐模型记录复算")
    if run["tcga_kirc_locked_test_read"]:
        raise ValueError("DeepDEP比较访问了患者Locked Test")
    print("【DeepDEP正式结果】684次拟合完整｜三方法805模型｜选参与汇总复算一致", flush=True)


def verify_portability(root: Path) -> None:
    offenders = []
    for base in (root / "scripts", root / "configs"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            linux_prefix = "/mnt/e/" + "projects/"
            windows_prefix = "E:\\" + "Projects\\"
            if linux_prefix in text or windows_prefix in text:
                offenders.append(str(path.relative_to(root)))
    if offenders:
        raise ValueError("仍含机器绝对路径：" + ", ".join(offenders))


def verify_artifacts(root: Path) -> None:
    figures = root / "results/historical/final_figures_v2"
    for path in figures.glob("*.png"):
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"PNG签名无效：{path}")
    for path in figures.glob("*.pdf"):
        if path.read_bytes()[:4] != b"%PDF":
            raise ValueError(f"PDF签名无效：{path}")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "configs/publication_release_manifest.json")
    parser.add_argument("--allow-environment-drift", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text())
    print("【阶段 1/4】核验发布快照文件和内部哈希", flush=True)
    for relative in manifest["required_scripts"] + manifest["frozen_protocols"] + manifest.get("methodological_audits", []):
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    checked = 0
    for relative in manifest["snapshot_directories"]:
        checked += verify_run(root / relative)
    print(f"【快照完整】目录 {len(manifest['snapshot_directories'])}｜哈希文件 {checked}", flush=True)

    print("【阶段 2/4】核验冻结候选、结论边界和一次性Test协议", flush=True)
    verify_claims(root)
    verify_advanced_comparison(root)
    if "results/historical/elastic_net_numerical_audit_v1" in manifest["snapshot_directories"]:
        verify_elastic_numerical_audit(root)
    if "results/historical/elastic_net_upper_grid_v1" in manifest["snapshot_directories"]:
        verify_elastic_upper_grid(root)
    if "results/historical/deepdep_expression_recovery_v1" in manifest["snapshot_directories"]:
        verify_deepdep_expression_recovery(root)
    if "results/historical/deepdep_recovered_comparison_v1" in manifest["snapshot_directories"]:
        verify_deepdep_recovered_results(root)
    print("【科学边界】候选20｜RNAi复现3｜亚型特异方向1｜患者功能真值0", flush=True)

    print("【阶段 3/4】核验环境与机器路径可移植性", flush=True)
    verify_environment(root, strict=not args.allow_environment_drift)
    verify_portability(root)
    print("【路径审计】scripts/ 与 configs/ 无机器绝对路径", flush=True)

    print("【阶段 4/4】核验PNG/PDF发布图件", flush=True)
    verify_artifacts(root)
    print("【复现通过】版本化结果、协议、环境、路径和图件全部一致", flush=True)
    print("【结论边界】该命令验证发布快照完整性；完整原始数据重跑仍需按来源许可准备大文件。", flush=True)


if __name__ == "__main__":
    main()
