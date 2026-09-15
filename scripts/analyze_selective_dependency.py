"""GPU bootstrap and renal-context audit for selective-dependency LOLO results."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch


GAINS = (
    "ndcg_at_10_gain",
    "selective_precision_at_10_gain",
    "top10_overlap_gain",
    "spearman_gain",
    "regret_reduction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values, draws, generator):
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    sampled = tensor[indices].mean(dim=1)
    quantiles = torch.quantile(
        sampled, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64, device="cuda"))
    return [float(value) for value in quantiles.cpu()]


def bootstrap_difference(left, right, draws, generator):
    left = torch.as_tensor(left, dtype=torch.float64, device="cuda")
    right = torch.as_tensor(right, dtype=torch.float64, device="cuda")
    li = torch.randint(len(left), (draws, len(left)), generator=generator, device="cuda")
    ri = torch.randint(len(right), (draws, len(right)), generator=generator, device="cuda")
    sampled = left[li].mean(dim=1) - right[ri].mean(dim=1)
    quantiles = torch.quantile(
        sampled, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64, device="cuda"))
    return [float(value) for value in quantiles.cpu()]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "outputs/selective_dependency_lolo_v1")
    parser.add_argument("--models", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1/models.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/selective_dependency_analysis_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.draws < 1000 or args.seed < 0:
        parser.error("draws至少1000且seed必须非负")
    return args


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；程序不会自动回退CPU")
    print(f"【计算设备】{torch.cuda.get_device_name(0)}｜CUDA｜双精度", flush=True)
    print("【阶段 1/3】校验结果并合并患者、癌系与 ccRCC 注释", flush=True)
    run = json.loads((args.results_dir / "run.json").read_text())
    for name, expected in run["output_sha256"].items():
        if sha256(args.results_dir / name) != expected:
            raise ValueError(f"结果哈希不一致：{name}")
    deltas = pd.read_csv(args.results_dir / "paired_deltas.csv")
    deltas = deltas[deltas.stratum.eq("non_common_essential")].copy()
    models = pd.read_csv(args.models, index_col=0)
    annotations = models[["PatientID", "clear_cell_renal_cell_carcinoma"]]
    deltas = deltas.join(annotations, on="ModelID", validate="one_to_one")
    if deltas[list(GAINS) + ["PatientID"]].isna().any().any():
        raise ValueError("主评价配对差值或患者注释缺失")
    if deltas.ModelID.duplicated().any():
        raise ValueError("主评价模型重复")
    print(f"【评价队列】模型 {len(deltas)}｜患者 {deltas.PatientID.nunique()}｜癌系 {deltas.heldout_lineage.nunique()}｜ccRCC {int(deltas.clear_cell_renal_cell_carcinoma.sum())}", flush=True)

    print("【阶段 2/3】GPU执行患者等权和癌系等权配对bootstrap", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    patient_means = deltas.groupby("PatientID")[list(GAINS)].mean()
    lineage_means = deltas.groupby("heldout_lineage")[list(GAINS)].mean()
    bootstrap_rows, direction_rows = [], []
    for metric in GAINS:
        for estimand, values in (("patient_equal", patient_means[metric].to_numpy()),
                                 ("lineage_equal", lineage_means[metric].to_numpy())):
            interval = bootstrap_mean(values, args.draws, generator)
            bootstrap_rows.append({"metric": metric, "estimand": estimand, "unit_n": len(values),
                                   "mean": float(values.mean()), "ci_low": interval[0],
                                   "bootstrap_median": interval[1], "ci_high": interval[2]})
        direction_rows.append({
            "metric": metric,
            "model_positive_n": int((deltas[metric] > 0).sum()),
            "model_zero_n": int((deltas[metric] == 0).sum()),
            "model_n": len(deltas),
            "patient_positive_n": int((patient_means[metric] > 0).sum()),
            "patient_zero_n": int((patient_means[metric] == 0).sum()),
            "patient_n": len(patient_means),
            "lineage_positive_n": int((lineage_means[metric] > 0).sum()),
            "lineage_zero_n": int((lineage_means[metric] == 0).sum()),
            "lineage_n": len(lineage_means),
        })

    renal_rows = []
    ccrcc = deltas[deltas.clear_cell_renal_cell_carcinoma].groupby("PatientID")[list(GAINS)].mean()
    kidney_other = deltas[(deltas.heldout_lineage == "Kidney") &
                          ~deltas.clear_cell_renal_cell_carcinoma].groupby("PatientID")[list(GAINS)].mean()
    other_lineages = lineage_means.drop(index="Kidney")
    for metric in GAINS:
        interval = bootstrap_difference(ccrcc[metric].to_numpy(), kidney_other[metric].to_numpy(),
                                        args.draws, generator)
        cc_mean = float(ccrcc[metric].mean())
        renal_rows.append({
            "metric": metric,
            "ccrcc_patient_n": len(ccrcc),
            "kidney_other_patient_n": len(kidney_other),
            "ccrcc_mean_gain": cc_mean,
            "kidney_other_mean_gain": float(kidney_other[metric].mean()),
            "ccrcc_minus_kidney_other": cc_mean - float(kidney_other[metric].mean()),
            "difference_ci_low": interval[0],
            "difference_bootstrap_median": interval[1],
            "difference_ci_high": interval[2],
            "ccrcc_percentile_vs_18_non_kidney_lineages": 100 * float(np.mean(other_lineages[metric] <= cc_mean)),
        })

    print("【阶段 3/3】保存不确定性、方向一致性与肾癌参照", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    pd.DataFrame(bootstrap_rows).to_csv(temporary / "bootstrap.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(temporary / "direction.csv", index=False)
    pd.DataFrame(renal_rows).to_csv(temporary / "renal_context.csv", index=False)
    audit = {
        "status": "selective_dependency_uncertainty_completed",
        "device": torch.cuda.get_device_name(0), "dtype": "float64",
        "draws": args.draws, "seed": args.seed,
        "source_sha256": {"paired_deltas.csv": sha256(args.results_dir / "paired_deltas.csv"),
                          "source_run.json": sha256(args.results_dir / "run.json"),
                          "models.csv": sha256(args.models), "script": sha256(Path(__file__))},
        "rules": {
            "primary_metric": "ndcg_at_10_gain",
            "gain_direction": "Positive favors expression residual; regret_reduction is prior regret minus expression regret",
            "patient_equal": "Average duplicate models within PatientID, then bootstrap patients",
            "lineage_equal": "Average models within heldout lineage, then bootstrap lineages",
            "renal_difference": "Independent patient bootstrap of ccRCC minus other Kidney models",
        },
        "limitations": [
            "Bootstrap intervals describe sampling instability and do not remove dataset or model-selection bias.",
            "The residual task and selectivity prior were specified after earlier absolute-dependency analyses.",
            "Only NDCG gain is primary; auxiliary metrics are unadjusted for multiplicity.",
            "The ccRCC comparison is post-hoc, small, and not an independent validation.",
        ],
    }
    audit["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)

    boot = pd.DataFrame(bootstrap_rows)
    for estimand in ("patient_equal", "lineage_equal"):
        row = boot[(boot.metric == "ndcg_at_10_gain") & (boot.estimand == estimand)].iloc[0]
        label = "患者等权" if estimand == "patient_equal" else "癌系等权"
        print(f"【主指标｜{label}】ΔNDCG {row['mean']:+.4f}｜95%区间 [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]", flush=True)
    direction = pd.DataFrame(direction_rows).set_index("metric").loc["ndcg_at_10_gain"]
    print(f"【方向一致性】模型 {direction.model_positive_n:.0f}/{direction.model_n:.0f}｜患者 {direction.patient_positive_n:.0f}/{direction.patient_n:.0f}｜癌系 {direction.lineage_positive_n:.0f}/{direction.lineage_n:.0f}", flush=True)
    renal = pd.DataFrame(renal_rows).set_index("metric").loc["ndcg_at_10_gain"]
    print(f"【ccRCC探索性对照】ccRCC ΔNDCG {renal.ccrcc_mean_gain:+.4f}｜其他 Kidney {renal.kidney_other_mean_gain:+.4f}｜差 {renal.ccrcc_minus_kidney_other:+.4f}｜95%区间 [{renal.difference_ci_low:+.4f}, {renal.difference_ci_high:+.4f}]", flush=True)
    print(f"【完成】结果 {args.output_dir}", flush=True)
    print("【结论边界】区间若稳定为正，只支持细胞系选择性预测；患者迁移仍需外部功能标签校验。", flush=True)


if __name__ == "__main__":
    main()
