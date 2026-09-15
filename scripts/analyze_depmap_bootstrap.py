"""GPU paired patient-cluster bootstrap of existing held-out ranking metrics.

Primary estimand: equal patient weights within each lineage, then equal lineage
weights. Intervals condition on fitted predictions; no model is refitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch


CONTRASTS = (
    ("shared_multiomics", "only_expression"),
    ("drop_mutation", "shared_multiomics"),
    ("drop_mutation", "only_expression"),
)
METRICS = ("ndcg", "dependency_precision")
NAMES = {"shared_multiomics": "完整多组学", "only_expression": "仅表达", "drop_mutation": "去突变"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seeded_generator(seed, key):
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    return torch.Generator(device="cuda").manual_seed((seed + offset) % (2**63 - 1))


def bootstrap_means(values, repeats, seed, key):
    """Each value is a paired difference averaged within one patient."""
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("每个分析组至少需要两位有完整配对指标的患者。")
    vector = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    generator = seeded_generator(seed, key)
    draws = torch.empty(repeats, dtype=torch.float64, device="cuda")
    for start in range(0, repeats, 1024):
        size = min(1024, repeats - start)
        indices = torch.randint(len(vector), (size, len(vector)), generator=generator, device="cuda")
        draws[start:start + size] = vector[indices].mean(dim=1)
    return draws


def interval(draws):
    return torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=torch.float64, device="cuda")).cpu().tolist()


def paired_patients(frame, left, right, metric):
    keys = ["heldout_lineage", "ModelID", "PatientID"]
    columns = keys + [metric, "observed_n"]
    a = frame.loc[frame.method.eq(left), columns]
    b = frame.loc[frame.method.eq(right), columns]
    merged = a.merge(b, on=keys, how="outer", suffixes=("_left", "_right"),
                     validate="one_to_one", indicator=True)
    if set(merged.heldout_lineage) != set(frame.heldout_lineage):
        raise ValueError("某癌系缺少两个比较方法，不能静默省略。")
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{left} 和 {right} 的模型身份或患者配对不完整。")
    if not merged.observed_n_left.eq(merged.observed_n_right).all():
        raise ValueError("比较方法的已观测靶点数量不同；不能直接配对评价。")
    valid = np.isfinite(merged[f"{metric}_left"]) & np.isfinite(merged[f"{metric}_right"])
    missing = merged.loc[~valid].groupby("heldout_lineage").size().to_dict()
    paired = merged.loc[valid].copy()
    paired["difference"] = paired[f"{metric}_left"] - paired[f"{metric}_right"]
    patients = paired.groupby(["heldout_lineage", "PatientID"], sort=True).agg(
        left_mean=(f"{metric}_left", "mean"), right_mean=(f"{metric}_right", "mean"),
        difference=("difference", "mean"), model_n=("ModelID", "size"),
    ).reset_index()
    expected = set(merged.heldout_lineage)
    if set(patients.heldout_lineage) != expected:
        raise ValueError("某癌系没有可配对指标，不能静默从跨癌系汇总中删除。")
    return patients, paired.groupby("heldout_lineage").difference.mean().to_dict(), missing


def validate_input(source):
    manifest = json.loads((source / "run.json").read_text())
    if manifest.get("status") != "nested_lolo_baseline" or manifest["arguments"].get("smoke_test"):
        raise ValueError("需要正式 LOLO 结果，不能把程序检查当作实验结果。")
    hashes = {"run.json": sha256(source / "run.json")}
    for name, expected in manifest["output_sha256"].items():
        actual = sha256(source / name)
        if actual != expected:
            raise ValueError(f"来源文件校验失败：{name}")
        hashes[name] = actual
    columns = ["ModelID", "PatientID", "heldout_lineage", "scope", "stratum", "method",
               "observed_n", "predicted_topk_observed_fraction", *METRICS]
    frame = pd.read_csv(source / "per_model_metrics.csv.gz", usecols=columns)
    identity = ["heldout_lineage", "scope", "stratum", "method", "ModelID"]
    if frame[identity + ["PatientID"]].isna().any().any() or frame.duplicated(identity).any():
        raise ValueError("评价结果存在缺失身份或重复模型行。")
    if not {m for pair in CONTRASTS for m in pair}.issubset(set(frame.method)):
        raise ValueError("缺少完整多组学、仅表达或去突变的消融结果。")
    lineage_rows = frame.loc[frame.scope.eq("lineage")]
    if (lineage_rows.groupby("PatientID").heldout_lineage.nunique() > 1).any():
        raise ValueError("存在跨癌系共享患者；当前分层重采样不适用。")
    splits = pd.read_csv(source / "splits.csv")
    if set(lineage_rows.heldout_lineage) != set(splits.heldout_lineage):
        raise ValueError("评价结果与分组记录的癌系范围不同。")
    for lineage, split in splits.groupby("heldout_lineage"):
        train, held = split.loc[split.role.eq("train")], split.loc[split.role.eq("test")]
        if set(train.PatientID) & set(held.PatientID):
            raise ValueError(f"{lineage} 外层患者泄漏。")
        for fold, valid in train.groupby("inner_validation_fold"):
            if set(valid.PatientID) & set(train.loc[train.inner_validation_fold.ne(fold), "PatientID"]):
                raise ValueError(f"{lineage} 内层患者泄漏。")
        actual = lineage_rows.loc[lineage_rows.heldout_lineage.eq(lineage), ["ModelID", "PatientID"]].drop_duplicates()
        if set(map(tuple, actual.to_numpy())) != set(map(tuple, held[["ModelID", "PatientID"]].to_numpy())):
            raise ValueError(f"{lineage} 评价身份与外层划分不一致。")
    return manifest, frame, hashes


def analyze(frame, repeats, seed, stratum):
    rows, patient_tables = [], []
    frame = frame.loc[frame.stratum.eq(stratum)]
    if frame.empty:
        raise ValueError(f"缺少目标分层：{stratum}")
    for left, right in CONTRASTS:
        print(f"【配对比较】{NAMES[left]} − {NAMES[right]}", flush=True)
        for metric in METRICS:
            for scope in ("lineage", "ccRCC"):
                selected = frame.loc[frame.scope.eq(scope)]
                if selected.empty:
                    continue
                patients, model_weighted, missing = paired_patients(selected, left, right, metric)
                patient_tables.append(patients.assign(left=left, right=right, metric=metric, scope=scope, stratum=stratum))
                draws, point_estimates, reference_points = [], [], []
                for lineage, group in patients.groupby("heldout_lineage", sort=True):
                    values = group.difference.to_numpy()
                    boot = bootstrap_means(values, repeats, seed, f"{stratum}/{scope}/{lineage}")
                    lower, upper = interval(boot)
                    point = float(values.mean())
                    rows.append({"left": left, "right": right, "metric": metric, "scope": scope,
                                 "heldout_lineage": lineage, "resampling": "patients_within_lineage",
                                 "stratum": stratum, "lineage_n": 1, "patient_n": len(group),
                                 "model_n": int(group.model_n.sum()), "excluded_model_n": missing.get(lineage, 0),
                                 "difference": point, "ci_lower": lower, "ci_upper": upper,
                                 "model_weighted_difference": model_weighted[lineage]})
                    draws.append(boot)
                    point_estimates.append(point)
                    reference_points.append(model_weighted[lineage])
                if scope == "lineage":
                    matrix = torch.stack(draws, dim=1)
                    fixed = matrix.mean(dim=1)
                    generator = seeded_generator(seed, f"{stratum}/hierarchical")
                    # Repeated sampled lineages receive independent patient draws.
                    lineage_indices = torch.randint(len(draws), (repeats, len(draws)), generator=generator, device="cuda")
                    bootstrap_indices = torch.randint(repeats, (repeats, len(draws)), generator=generator, device="cuda")
                    hierarchical = matrix[bootstrap_indices, lineage_indices].mean(dim=1)
                    for design, samples in (("fixed_lineages_patient_bootstrap", fixed),
                                            ("hierarchical_lineage_patient_bootstrap", hierarchical)):
                        lower, upper = interval(samples)
                        rows.append({"left": left, "right": right, "metric": metric,
                                     "scope": "cross_lineage", "heldout_lineage": "ALL", "resampling": design,
                                     "stratum": stratum, "lineage_n": len(draws), "patient_n": len(patients),
                                     "model_n": int(patients.model_n.sum()), "excluded_model_n": int(sum(missing.values())),
                                     "difference": float(np.mean(point_estimates)), "ci_lower": lower, "ci_upper": upper,
                                     "model_weighted_difference": float(np.mean(reference_points))})
    result = pd.DataFrame(rows)
    result["interpretation"] = np.select(
        [result.ci_lower.gt(0), result.ci_upper.lt(0)],
        ["区间完全大于零", "区间完全小于零"], default="区间跨零，不能判定差异方向；不代表等价")
    return result, pd.concat(patient_tables, ignore_index=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "outputs/depmap_ablation_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/depmap_ablation_bootstrap_v1")
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--stratum", choices=("non_common_essential", "all", "common_essential"), default="non_common_essential")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.repeats < 1000 or args.seed < 0:
        parser.error("重采样次数至少 1000，seed 必须非负。")
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请使用新目录：{args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不会自动切换到 CPU。")
    started = time.monotonic()
    print(f"【启动】GPU：{torch.cuda.get_device_name(0)}｜配对重采样 {args.repeats} 次｜不重新训练模型", flush=True)
    source_manifest, frame, input_hashes = validate_input(args.input_dir)
    print("【校验通过】来源哈希、配对身份和患者划分正常。", flush=True)
    results, patients = analyze(frame, args.repeats, args.seed, args.stratum)
    headline = results.loc[results.resampling.eq("fixed_lineages_patient_bootstrap") | results.scope.eq("ccRCC")]
    print("\n【主要结果】患者等权；差值为前者减后者；95% 区间为逐项探索性区间。", flush=True)
    for row in headline.itertuples():
        scope = "跨癌系平均" if row.scope == "cross_lineage" else "明确 ccRCC"
        metric = "NDCG" if row.metric == "ndcg" else "命中率"
        print(f"  {scope}｜{NAMES[row.left]}−{NAMES[row.right]}｜{metric} {row.difference:+.4f} [{row.ci_lower:+.4f}, {row.ci_upper:+.4f}]", flush=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        results.to_csv(temporary / "paired_summary.csv", index=False)
        patients.to_csv(temporary / "patient_pairs.csv.gz", index=False)
        manifest = {
            "status": "exploratory_paired_patient_bootstrap", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "input_sha256": input_hashes, "source_script_sha256": source_manifest.get("script_sha256"),
            "script_sha256": sha256(__file__), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "primary_estimand": "Within-lineage mean of paired patient means, then equal-weight mean across the observed lineages",
            "primary_resampling": "Paired patients resampled independently within fixed observed lineages",
            "secondary_resampling": "Resample lineages and independent within-lineage patient means; descriptive sensitivity analysis",
            "confidence": "Pointwise percentile 95%; no multiplicity adjustment, no equivalence margin, no confirmatory testing",
            "model_weighted_difference": "Reference estimand averaging models within lineage; matches original reporting when no metrics are missing",
            "ranking_label_coverage": frame.loc[frame.stratum.eq(args.stratum)].groupby("method").predicted_topk_observed_fraction.mean().to_dict(),
            "limitations": [
                "Intervals condition on existing fitted predictions and tuning; no model/split/seed uncertainty or retraining is included.",
                "Training sets overlap across LOLO folds; secondary lineage bootstrap does not establish independent biological cohorts.",
                "Contrasts were chosen after viewing ablation results; intervals are exploratory and not confirmatory evidence.",
                "Zero-crossing intervals do not establish equivalence; nonzero intervals do not establish practical importance.",
                "ccRCC sample size is small and historical models have been explored; no external validation or specificity claim follows.",
                "Ranking metrics are conditional on observed target labels, not a fully observed dependency screen.",
            ],
            "elapsed_seconds": time.monotonic() - started,
            "output_sha256": {p.name: sha256(p) for p in sorted(temporary.iterdir())},
        }
        (temporary / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"【完成】耗时 {time.monotonic() - started:.1f} 秒｜保留 3 个文件｜{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
