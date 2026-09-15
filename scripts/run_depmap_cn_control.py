"""CUDA copy-number correspondence controls with frozen LOLO splits.

Shuffle whole patient blocks only within partition, lineage and patient size.
Row-permuting a training-standardized linear kernel is exactly equivalent to
permuting the original CN rows: column means, variances and covariance remain.
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

import run_depmap_baseline as base


def patient_permutation(models, indices, seed, key):
    """Return local row donors, using a bijection of equal-size patient blocks."""
    local = models.iloc[indices].copy()
    local["position"] = np.arange(len(indices))
    groups = {}
    for patient, group in local.groupby("PatientID", sort=True):
        if group.OncotreeLineage.nunique() != 1:
            raise ValueError("同一分区内存在跨癌系患者，当前置换规则不适用。")
        positions = group.sort_index().position.to_numpy()
        groups.setdefault((group.OncotreeLineage.iloc[0], len(group)), []).append(positions)
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng((seed + offset) % (2**63 - 1))
    donors = np.arange(len(indices))
    patient_n = moved_patients = blocked_patients = 0
    for blocks in groups.values():
        patient_n += len(blocks)
        if len(blocks) == 1:
            blocked_patients += 1
        order = rng.permutation(len(blocks))
        for destination, source in enumerate(order):
            donors[blocks[destination]] = blocks[source]
            moved_patients += int(destination != source)
    if sorted(donors.tolist()) != list(range(len(indices))):
        raise ValueError("置换不是行索引的双射。")
    return donors, {"model_n": len(indices), "patient_n": patient_n,
                    "moved_model_n": int(np.sum(donors != np.arange(len(indices)))),
                    "moved_patient_n": moved_patients, "structurally_blocked_patient_n": blocked_patients,
                    "donor_index_sha256": hashlib.sha256(donors.astype("<i8").tobytes()).hexdigest()}


def method_kernels(models, genes, matrices, train, held, features, seeds, key, audits):
    """Calculate real blocks once, yielding each control without retaining it."""
    if set(models.iloc[train].PatientID) & set(models.iloc[held].PatientID):
        raise ValueError("训练与验证分区存在患者重叠。")
    configs = {"expression": ("lineage", "drivers", "expression_mean", "expression"),
               "cn": ("copy_number_mean", "copy_number")}
    blocks, _ = base.kernels(models, genes, matrices, train, held, features, configs)
    ex, eh = blocks["expression"]
    cn, ch = blocks["cn"]
    yield "only_expression", -1, ex, eh
    yield "expression_copy_number", -1, ex + cn, eh + ch
    for seed in seeds:
        train_donors, ta = patient_permutation(models, train, seed, key + "/train")
        held_donors, ha = patient_permutation(models, held, seed, key + "/held")
        audits.extend([{**ta, "split": key, "partition": "train", "permutation_seed": seed},
                       {**ha, "split": key, "partition": "held", "permutation_seed": seed}])
        t = base.TORCH.as_tensor(train_donors, device="cuda")
        h = base.TORCH.as_tensor(held_donors, device="cuda")
        yield "permuted_copy_number", seed, ex + cn[t[:, None], t[None, :]], eh + ch[h[:, None], t[None, :]]


def load_reference(path, source, models, genes):
    manifest = json.loads((path / "run.json").read_text())
    if manifest.get("status") != "nested_lolo_baseline" or not manifest["arguments"].get("ablation"):
        raise ValueError("需要正式消融实验作为冻结参考。")
    expected = base.config_blocks(True)
    for name in ("only_expression", "drop_mutation"):
        if manifest.get("feature_configs", {}).get(name) != list(expected[name]):
            raise ValueError("参考消融的特征定义不匹配。")
    for name, digest in manifest["input_sha256"].items():
        if base.sha256(source / name) != digest:
            raise ValueError(f"输入与参考实验不一致：{name}")
    for name in ("splits.csv", "metrics.csv", "tuning.csv"):
        if base.sha256(path / name) != manifest["output_sha256"][name]:
            raise ValueError(f"参考文件校验失败：{name}")
    if manifest["target_genes"] != genes.tolist() or manifest["feature_genes"] != genes.tolist():
        raise ValueError("参考实验必须使用相同的完整靶点及特征基因集合。")
    splits = pd.read_csv(path / "splits.csv")
    index = {model: i for i, model in enumerate(models.index)}
    result = {}
    for lineage, group in splits.groupby("heldout_lineage", sort=True):
        if set(group.ModelID) != set(index) or group.ModelID.duplicated().any():
            raise ValueError("参考分组模型身份不完整或重复。")
        ids = np.array([index[name] for name in group.ModelID])
        if not np.array_equal(models.iloc[ids].PatientID.to_numpy(), group.PatientID.to_numpy()):
            raise ValueError("参考患者身份不匹配。")
        train = np.sort(ids[group.role.eq("train")])
        held = np.sort(ids[group.role.eq("test")])
        if set(models.iloc[train].PatientID) & set(models.iloc[held].PatientID):
            raise ValueError("参考外层存在患者泄漏。")
        if not np.all(models.iloc[held].OncotreeLineage == lineage) or (models.iloc[train].OncotreeLineage == lineage).any():
            raise ValueError("参考癌系留出错误。")
        folds = [np.sort(ids[group.role.eq("train") & group.inner_validation_fold.eq(f)])
                 for f in sorted(group.loc[group.role.eq("train"), "inner_validation_fold"].unique())]
        if len(folds) < 2 or sorted(np.concatenate(folds).tolist()) != train.tolist():
            raise ValueError("参考内层分组不完整。")
        for valid in folds:
            fit = np.setdiff1d(train, valid)
            for column in ("PatientID", "OncotreeLineage"):
                if set(models.iloc[fit][column]) & set(models.iloc[valid][column]):
                    raise ValueError("参考内层患者或癌系泄漏。")
        result[lineage] = train, held, folds
    return manifest, result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--reference-dir", type=Path, default=root / "outputs/depmap_ablation_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/depmap_cn_control_v1")
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=list(range(20260915, 20260935)))
    parser.add_argument("--lineages", nargs="+")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if len(set(args.permutation_seeds)) != len(args.permutation_seeds) or min(args.permutation_seeds) < 0:
        parser.error("置换 seed 必须非负且不重复。")
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在，请使用新目录：{args.output_dir}")
    started = time.monotonic()
    base.configure_device("cuda")
    models, all_genes, matrices, all_essential = base.load_data(args.input_dir)
    reference, splits = load_reference(args.reference_dir, args.input_dir, models, all_genes)
    reference_metrics = pd.read_csv(args.reference_dir / "metrics.csv")
    requested = args.lineages or (["Kidney"] if args.smoke_test else list(splits))
    if len(set(requested)) != len(requested) or set(requested) - set(splits):
        raise ValueError("请求的癌系不在冻结参考中，或存在重复。")
    alphas = reference["arguments"]["alphas"]
    top_k = reference["arguments"]["top_k"]
    threshold = reference["arguments"]["dependency_threshold"]
    seeds = args.permutation_seeds[:2] if args.smoke_test else args.permutation_seeds
    target_indices = features = np.arange(len(all_genes))
    if args.smoke_test:
        rng = np.random.default_rng(reference["arguments"]["seed"])
        target_indices = np.sort(rng.choice(len(all_genes), min(128, len(all_genes)), replace=False))
        features = np.sort(rng.choice(len(all_genes), min(256, len(all_genes)), replace=False))
    genes = all_genes[target_indices]
    y = matrices["dependency"][:, target_indices].astype(float)
    nonessential = ~all_essential[target_indices]
    if int(nonessential.sum()) < top_k:
        raise ValueError("非普遍必需靶点少于 Top-k，无法进行该分层评价。")
    print(f"【{'程序检查' if args.smoke_test else '置换对照'}】癌系 {len(requested)}｜靶点 {len(genes)}｜置换 {len(seeds)} 次｜沿用参考的内层划分和 α 网格", flush=True)
    audits = []
    if args.dry_run:
        for lineage in requested:
            train, held, folds = splits[lineage]
            _, info = patient_permutation(models, held, seeds[0], lineage + "/outer/held")
            print(f"  {lineage}｜训练 {len(train)}｜留出 {len(held)}｜内层 {len(folds)} 折｜留出无法交换的患者 {info['structurally_blocked_patient_n']}/{info['patient_n']}", flush=True)
        print("【检查通过】未训练、未写入结果。", flush=True)
        return
    summaries, cells, tuning = [], [], []
    for number, lineage in enumerate(requested, 1):
        train, held, folds = splits[lineage]
        identities = [("only_expression", -1), ("expression_copy_number", -1)] + [("permuted_copy_number", seed) for seed in seeds]
        errors = {key: np.zeros(len(alphas)) for key in identities}
        counts = {key: np.zeros(len(alphas), dtype=np.int64) for key in identities}
        print(f"\n【癌系 {number}/{len(requested)}】{lineage}｜真实配置与每个置换版本分别调参", flush=True)
        for fold_id, valid in enumerate(folds):
            fit = np.setdiff1d(train, valid)
            print(f"  内层 {fold_id + 1}/{len(folds)}：开始拟合 {len(identities)} 个配置…", flush=True)
            for method, seed, k, h in method_kernels(models, all_genes, matrices, fit, valid, features, seeds,
                                                    f"{lineage}/inner_{fold_id}", audits):
                predictions = base.masked_ridge(k, h, y[fit], alphas)
                for a, alpha in enumerate(alphas):
                    prediction = predictions[alpha]
                    observed = np.isfinite(y[valid]) & np.isfinite(prediction)
                    errors[method, seed][a] += float(((y[valid][observed] - prediction[observed]) ** 2).sum())
                    counts[method, seed][a] += int(observed.sum())
                del predictions
            print(f"  内层 {fold_id + 1}/{len(folds)} 完成", flush=True)
        observed_counts = np.concatenate(list(counts.values()))
        if observed_counts.min() == 0 or not np.all(observed_counts == observed_counts[0]):
            raise ValueError("各方法/alpha 的内层评价覆盖不同。")
        selected = {key: min(range(len(alphas)), key=lambda i: (errors[key][i], alphas[i])) for key in identities}
        for key in identities:
            tuning.extend({"heldout_lineage": lineage, "method": key[0], "permutation_seed": key[1],
                           "alpha": alpha, "sse": errors[key][i], "observed_n": int(counts[key][i]),
                           "selected": i == selected[key]} for i, alpha in enumerate(alphas))
        mean = base.observed_mean(y[train])
        mean[np.isfinite(y[train]).sum(axis=0) < 2] = np.nan
        baseline = np.broadcast_to(mean, (len(held), len(genes)))
        expression_prediction = None
        scopes = {"lineage": np.arange(len(held))}
        if lineage == "Kidney":
            scopes["ccRCC"] = np.flatnonzero(models.iloc[held].clear_cell_renal_cell_carcinoma.to_numpy(dtype=bool))
        for method, seed, k, h in method_kernels(models, all_genes, matrices, train, held, features, seeds,
                                                f"{lineage}/outer", audits):
            alpha = alphas[selected[method, seed]]
            prediction = base.masked_ridge(k, h, y[train], [alpha])[alpha]
            if method == "only_expression":
                expression_prediction = prediction
            for scope, row_indices in scopes.items():
                if not len(row_indices):
                    continue
                ix = np.ix_(row_indices, np.flatnonzero(nonessential))
                summary, per_model, _ = base.score_scope(y[held][ix], prediction[ix], baseline[ix], expression_prediction[ix],
                    models.index.to_numpy()[held[row_indices]], models.iloc[held[row_indices]].PatientID.to_numpy(),
                    genes[nonessential], top_k, threshold)
                summary["delta_r2_vs_expression"] = summary.pop("delta_r2_vs_background")
                if not args.smoke_test and method != "permuted_copy_number":
                    original_method = "only_expression" if method == "only_expression" else "drop_mutation"
                    original = reference_metrics.loc[
                        reference_metrics.heldout_lineage.eq(lineage) & reference_metrics.scope.eq(scope)
                        & reference_metrics.stratum.eq("non_common_essential") & reference_metrics.method.eq(original_method)]
                    if len(original) != 1 or float(original.iloc[0].selected_alpha) != alpha:
                        raise ValueError("真实配置未复现参考实验的 alpha；停止，避免不公平比较。")
                    for metric in ("mse", "r2", "delta_r2_vs_global_mean", "ndcg", "dependency_precision", "topk_overlap", "regret"):
                        if not np.isclose(summary[metric], original.iloc[0][metric], rtol=1e-7, atol=1e-8, equal_nan=True):
                            raise ValueError(f"真实配置未复现参考实验：{lineage}/{scope}/{original_method}/{metric}")
                context = {"heldout_lineage": lineage, "scope": scope, "stratum": "non_common_essential",
                           "method": method, "permutation_seed": seed}
                summaries.append({**context, "selected_alpha": alpha, **summary})
                cells.append(per_model.assign(**context))
            label = "仅表达" if method == "only_expression" else "表达＋真实拷贝数" if method == "expression_copy_number" else f"表达＋置换拷贝数 seed={seed}"
            row = next(x for x in reversed(summaries) if x["scope"] == "lineage")
            print(f"  {label}｜α={alpha:g}｜NDCG {row['ndcg']:.4f}｜命中率 {row['dependency_precision']:.4f}", flush=True)
        held_audits = [a for a in audits if a["split"] == f"{lineage}/outer" and a["partition"] == "held"]
        moved = np.mean([a["moved_model_n"] / a["model_n"] for a in held_audits])
        print(f"【置换覆盖】留出模型平均实际移动 {moved:.1%}｜无法交换的患者 {held_audits[0]['structurally_blocked_patient_n']}/{held_audits[0]['patient_n']}", flush=True)
        print(f"【癌系完成】累计 {(time.monotonic() - started) / 60:.1f} 分钟", flush=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        pd.DataFrame(summaries).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(cells, ignore_index=True).to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        pd.DataFrame(tuning).to_csv(temporary / "tuning.csv", index=False)
        pd.DataFrame(audits).to_csv(temporary / "permutation_audit.csv.gz", index=False)
        manifest = {
            "status": "smoke_test_not_research_evidence" if args.smoke_test else "nested_cn_correspondence_control",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "reference_sha256": {name: base.sha256(args.reference_dir / name) for name in ("run.json", "splits.csv", "metrics.csv", "tuning.csv")},
            "input_sha256": reference["input_sha256"], "script_sha256": base.sha256(__file__),
            "baseline_helper_sha256": base.sha256(base.__file__), "device": base.TORCH.cuda.get_device_name(0),
            "torch": base.TORCH.__version__, "dtype": "float64", "alphas": alphas, "permutation_seeds": seeds,
            "target_genes": genes.tolist(), "feature_genes": all_genes[features].tolist(),
            "feature_configs": {"only_expression": "lineage + five mutation drivers + expression mean + expression",
                                "expression_copy_number": "only_expression + CN mean + CN",
                                "permuted_copy_number": "only_expression + jointly row-permuted CN mean and CN"},
            "rules": "Frozen reference outer/inner splits; independent CN permutations in every train/held partition; uniform patient-block permutation within lineage and equal patient model count; sorted ModelID pairs rows within exchanged blocks; retune each seed using observed inner SSE",
            "limitations": [
                "This is an exploratory correspondence control, not a conditional randomization test or an exact biological null.",
                "Permutation preserves marginal CN dimensions/covariance but disrupts its dependence on expression and driver annotations.",
                "Singleton patient-size/lineage strata cannot move; random permutations may also retain fixed points. Audit reports both.",
                "Permutation seeds quantify random-control variability, not independent patients/cohorts; no significance p-value is computed.",
                "All configurations retain the five damaging driver indicators; no claim of entirely mutation-free modeling.",
                "CN is shuffled within Kidney, not within RCC subtype; ccRCC subset results remain exploratory.",
                "Tuning uses all target genes; reported ranking/accuracy focuses on non-common-essential targets.",
            ],
            "elapsed_seconds": time.monotonic() - started,
            "output_sha256": {p.name: base.sha256(p) for p in sorted(temporary.iterdir())},
        }
        (temporary / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"【完成】保留 5 个文件｜{args.output_dir}｜总耗时 {(time.monotonic() - started) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
