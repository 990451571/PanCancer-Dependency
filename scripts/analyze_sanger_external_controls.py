"""GPU uncertainty and renal-context analysis for frozen Sanger controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


METRICS = ("ndcg_at_10", "binary_dependency_precision_at_10", "top10_overlap", "spearman")
COMPARISONS = {
    "direct_vs_mean": ("direct_expression_only", "training_gene_mean"),
    "mapped_vs_mean": ("mapped_expression_only", "training_gene_mean"),
    "mapped_vs_direct": ("mapped_expression_only", "direct_expression_only"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(values, draws, generator):
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    means = tensor[indices].mean(dim=1)
    quantiles = torch.quantile(
        means, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64, device="cuda"))
    return [float(value) for value in quantiles.cpu()]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-results", type=Path, default=root / "outputs/sanger_external_controls_v1")
    parser.add_argument("--phase-a-results", type=Path, default=root / "outputs/sanger_phase_a_769p_v1")
    parser.add_argument("--phase-b-results", type=Path, default=root / "outputs/sanger_phase_b_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/sanger_external_controls_analysis_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.draws < 1000 or args.seed < 0:
        parser.error("draws至少1000，seed必须非负")
    return args


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；程序不会自动回退CPU")
    print(f"【计算设备】{torch.cuda.get_device_name(0)}｜CUDA｜双精度", flush=True)
    print("【阶段 1/3】读取逐模型指标并构造配对差值", flush=True)
    controls = pd.read_csv(args.control_results / "metrics.csv")
    pivot = controls.pivot(index=["model_id", "model_name", "lineage"], columns="method", values=list(METRICS))
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = []
    delta_tables = {}
    for comparison, (left, right) in COMPARISONS.items():
        table = pivot.dropna(subset=[(metric, left) for metric in METRICS] +
                                  [(metric, right) for metric in METRICS]).copy()
        delta = pd.DataFrame(index=table.index)
        for metric in METRICS:
            delta[metric] = table[(metric, left)] - table[(metric, right)]
            model_ci = bootstrap(delta[metric].to_numpy(), args.draws, generator)
            lineage_means = delta.reset_index().groupby("lineage")[metric].mean().to_numpy()
            lineage_ci = bootstrap(lineage_means, args.draws, generator)
            rows.extend([
                {"comparison": comparison, "metric": metric, "estimand": "model_weighted",
                 "unit_n": len(delta), "mean": float(delta[metric].mean()),
                 "ci_low": model_ci[0], "bootstrap_median": model_ci[1], "ci_high": model_ci[2]},
                {"comparison": comparison, "metric": metric, "estimand": "lineage_equal",
                 "unit_n": len(lineage_means), "mean": float(lineage_means.mean()),
                 "ci_low": lineage_ci[0], "bootstrap_median": lineage_ci[1], "ci_high": lineage_ci[2]},
            ])
        delta_tables[comparison] = delta
    bootstrap_frame = pd.DataFrame(rows)

    print("【阶段 2/3】计算肾癌结果在泛癌分布中的经验位置", flush=True)
    phase_a = pd.read_csv(args.phase_a_results / "metrics.csv").set_index("method")
    phase_b = pd.read_csv(args.phase_b_results / "metrics.csv")
    context_rows = []
    targets = [("769-P", "direct_vs_mean",
                {metric: phase_a.loc["expression_only", metric] - phase_a.loc["training_gene_mean", metric]
                 for metric in METRICS})]
    for name in ("LB1047-RCC", "RCC-FG2"):
        current = phase_b[phase_b.model_name.eq(name)].set_index("method")
        targets.append((name, "mapped_vs_mean",
                        {metric: current.loc["mapped_expression_only", metric] -
                                 current.loc["training_gene_mean", metric] for metric in METRICS}))
    for name, comparison, values in targets:
        reference = delta_tables[comparison]
        if name in reference.index.get_level_values("model_name"):
            reference = reference[reference.index.get_level_values("model_name") != name]
        for metric, value in values.items():
            distribution = reference[metric].to_numpy()
            context_rows.append({"model_name": name, "comparison": comparison, "metric": metric,
                                 "delta": float(value), "reference_n": len(distribution),
                                 "empirical_percentile": float(100 * np.mean(distribution <= value)),
                                 "reference_ge_n": int(np.sum(distribution >= value))})
    context = pd.DataFrame(context_rows)

    print("【阶段 3/3】保存不确定性与泛癌参照", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    bootstrap_frame.to_csv(temporary / "bootstrap.csv", index=False)
    context.to_csv(temporary / "renal_context.csv", index=False)
    run = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "status": "external_control_uncertainty_completed",
           "device": torch.cuda.get_device_name(0), "dtype": "float64",
           "draws": args.draws, "seed": args.seed,
           "rules": {"model_weighted": "paired model bootstrap",
                     "lineage_equal": "bootstrap over lineage-level mean paired deltas",
                     "renal_percentile": "empirical position; excludes the target itself when present",
                     "multiple_comparison_adjustment": "none; descriptive analysis"},
           "source_sha256": {"controls": sha256(args.control_results / "metrics.csv"),
                             "phase_a": sha256(args.phase_a_results / "metrics.csv"),
                             "phase_b": sha256(args.phase_b_results / "metrics.csv"),
                             "script": sha256(Path(__file__))},
           "limitations": ["Bootstrap does not fix availability selection bias.",
                           "Lineage counts are small and unequal.",
                           "Renal empirical percentiles are descriptive and were computed after renal outcomes were known.",
                           "No multiplicity correction is applied across metrics and comparisons."]}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    for comparison in ("direct_vs_mean", "mapped_vs_mean", "mapped_vs_direct"):
        row = bootstrap_frame[(bootstrap_frame.comparison == comparison) &
                              (bootstrap_frame.metric == "ndcg_at_10") &
                              (bootstrap_frame.estimand == "model_weighted")].iloc[0]
        print(f"【NDCG】{comparison}｜均值 {row['mean']:+.4f}｜95%区间 [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]", flush=True)
    for name in ("769-P", "LB1047-RCC", "RCC-FG2"):
        row = context[(context.model_name == name) & (context.metric == "ndcg_at_10")].iloc[0]
        print(f"【泛癌位置】{name}｜ΔNDCG {row['delta']:+.4f}｜经验百分位 {row['empirical_percentile']:.1f}%", flush=True)
    print(f"【完成】结果 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
