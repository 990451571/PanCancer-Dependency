#!/usr/bin/env python3
"""Run the frozen same-data PCR, Elastic Net, reduced-rank and Exp-DeepDEP benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
import run_selective_dependency as selective
import run_selective_dependency_benchmark as benchmark
from advanced_model_methods import (
    ExpressionAutoencoder,
    elastic_net_path,
    expression_kernels,
    reduced_rank_ridge_predictions,
    standardize_expression,
    train_exp_deepdep,
)


METHOD_LABELS = {
    "training_selectivity_prior": "训练选择性先验",
    "expression_pcr_ridge": "PCR-ridge",
    "per_target_elastic_net_shared": "逐靶点Elastic Net（共享α）",
    "per_target_elastic_net_targetwise": "逐靶点Elastic Net（逐靶点α）",
    "multitask_reduced_rank_ridge": "多任务低秩岭回归",
    "exp_deepdep_adapted": "Exp-DeepDEP同数据适配",
}
METRICS = selective.METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_grid(text: str, cast):
    return tuple(cast(value) for value in text.split(",") if value.strip())


def load_inputs(directory: Path):
    audit = json.loads((directory / "audit.json").read_text())
    for name, expected in audit["output_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"高级模型输入哈希错误：{name}")
    with np.load(directory / "benchmark_inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if audit["tcga_kirc_test_read"]:
        raise ValueError("高级模型输入意外读取过Locked Test")
    return audit, arrays


def exact_outer_splits(split_path: Path, model_ids: np.ndarray, lineages: list[str] | None):
    frame = pd.read_csv(split_path)
    index = {model: i for i, model in enumerate(model_ids)}
    available = sorted(frame.heldout_lineage.unique())
    chosen = lineages or available
    if set(chosen) - set(available):
        raise ValueError("请求癌系不在冻结外层split中")
    result = {}
    for lineage in chosen:
        current = frame.loc[frame.heldout_lineage.eq(lineage)]
        train = np.asarray([index[x] for x in current.loc[current.role.eq("train"), "ModelID"]], dtype=int)
        held = np.asarray([index[x] for x in current.loc[current.role.eq("test"), "ModelID"]], dtype=int)
        result[lineage] = train, held
    return result


def score_prediction(truth, prediction, ids, lineages, genes, primary, args, method):
    rows, _ = selective.score_models(
        truth, {method: prediction}, ids, lineages, genes,
        primary, args.top_k, args.residual_threshold)
    return pd.DataFrame(rows)


def tune_score(frames: list[pd.DataFrame]) -> float:
    frame = pd.concat(frames, ignore_index=True)
    return float(frame.groupby("heldout_lineage").ndcg_at_10.mean().mean())


def select_config(scores, simplicity):
    return sorted(scores, key=lambda config: (-scores[config], simplicity(config)))[0]


def tune_classical(
    models, arrays, outer_train, inner_folds, grids, args,
):
    labels = models.OncotreeLineage.to_numpy()
    ids = models.index.to_numpy()
    y = arrays["dependency"].astype(float)
    expression = arrays["expression"].astype(float)
    genes = arrays["target_genes"]
    primary = ~arrays["target_common_essential"].astype(bool)
    accumulated = defaultdict(list)
    elastic_sse = {alpha: np.zeros(y.shape[1]) for alpha in grids["elastic_alpha"]}
    elastic_n = {alpha: np.zeros(y.shape[1], dtype=int) for alpha in grids["elastic_alpha"]}
    elastic_diagnostics = []
    for fold_number, held in enumerate(inner_folds, 1):
        fit = np.setdiff1d(outer_train, held)
        mean, _, _ = selective.training_targets(y[fit], args.prior_quantile)
        residual, truth = y[fit] - mean, y[held] - mean
        kernel, held_kernel, x64, hx64, _ = expression_kernels(expression[fit], expression[held])

        pcr = benchmark.pcr_ridge_predictions(
            kernel, held_kernel, residual, grids["pcr_rank"], grids["pcr_alpha"])
        for config, prediction in pcr.items():
            accumulated[("expression_pcr_ridge", *config)].append(
                score_prediction(truth, prediction, ids[held], labels[held], genes, primary, args, "m"))

        rrr, _ = reduced_rank_ridge_predictions(
            kernel, held_kernel, residual, grids["rrr_rank"], grids["rrr_alpha"])
        for config, prediction in rrr.items():
            accumulated[("multitask_reduced_rank_ridge", *config)].append(
                score_prediction(truth, prediction, ids[held], labels[held], genes, primary, args, "m"))

        elastic, diagnostics = elastic_net_path(
            x64.float(), residual, hx64.float(), grids["elastic_alpha"],
            args.elastic_l1_ratio, args.elastic_iterations, args.elastic_tolerance)
        for alpha, prediction in elastic.items():
            accumulated[("per_target_elastic_net_shared", alpha)].append(
                score_prediction(truth, prediction, ids[held], labels[held], genes, primary, args, "m"))
            valid = np.isfinite(truth) & np.isfinite(prediction)
            elastic_sse[alpha] += np.nansum(np.where(valid, (truth - prediction) ** 2, np.nan), axis=0)
            elastic_n[alpha] += valid.sum(0)
            elastic_diagnostics.append({"inner_fold": fold_number, "alpha": alpha, **diagnostics[alpha]})
        print(f"    内层 {fold_number}/{len(inner_folds)} 完成", flush=True)

    by_method = defaultdict(dict)
    tuning_rows = []
    for config, frames in accumulated.items():
        score = tune_score(frames)
        by_method[config[0]][config[1:]] = score
    selected = {
        "expression_pcr_ridge": select_config(
            by_method["expression_pcr_ridge"], lambda c: (c[0], -c[1])),
        "multitask_reduced_rank_ridge": select_config(
            by_method["multitask_reduced_rank_ridge"], lambda c: (c[0], -c[1])),
        "per_target_elastic_net_shared": select_config(
            by_method["per_target_elastic_net_shared"], lambda c: (-c[0],)),
    }
    for method, values in by_method.items():
        for config, score in values.items():
            tuning_rows.append({
                "method": method,
                "config": "|".join(map(str, config)),
                "inner_lineage_equal_ndcg_at_10": score,
                "selected": config == selected[method],
            })
    alpha_values = np.asarray(grids["elastic_alpha"])
    target_mse = np.vstack([
        np.divide(elastic_sse[a], elastic_n[a], out=np.full(y.shape[1], np.inf), where=elastic_n[a] > 0)
        for a in alpha_values
    ])
    target_alpha = alpha_values[np.argmin(target_mse, axis=0)]
    return selected, target_alpha, tuning_rows, elastic_diagnostics


def fit_classical(arrays, train, held, selected, target_alpha, grids, args):
    y = arrays["dependency"].astype(float)
    expression = arrays["expression"].astype(float)
    mean, prior, _ = selective.training_targets(y[train], args.prior_quantile)
    residual = y[train] - mean
    kernel, held_kernel, x64, hx64, keep = expression_kernels(expression[train], expression[held])
    predictions = {"training_selectivity_prior": np.broadcast_to(prior, (len(held), y.shape[1]))}

    rank, alpha = selected["expression_pcr_ridge"]
    predictions["expression_pcr_ridge"] = benchmark.pcr_ridge_predictions(
        kernel, held_kernel, residual, (rank,), (alpha,))[(rank, alpha)]
    rank, alpha = selected["multitask_reduced_rank_ridge"]
    predictions["multitask_reduced_rank_ridge"] = reduced_rank_ridge_predictions(
        kernel, held_kernel, residual, (rank,), (alpha,))[0][(rank, alpha)]
    elastic, diagnostics = elastic_net_path(
        x64.float(), residual, hx64.float(), grids["elastic_alpha"],
        args.elastic_l1_ratio, args.elastic_iterations, args.elastic_tolerance)
    shared_alpha = selected["per_target_elastic_net_shared"][0]
    predictions["per_target_elastic_net_shared"] = elastic[shared_alpha]
    targetwise = np.empty_like(elastic[shared_alpha])
    for alpha in grids["elastic_alpha"]:
        columns = target_alpha == alpha
        targetwise[:, columns] = elastic[alpha][:, columns]
    predictions["per_target_elastic_net_targetwise"] = targetwise
    return mean, predictions, diagnostics, int(keep.sum())


def tune_and_fit_deep(arrays, outer_train, held, inner_fold, encoder, seeds, args):
    y = arrays["dependency"].astype(float)
    expression = arrays["expression"].astype(np.float32)
    fingerprints = arrays["target_fingerprints"].astype(np.float32)
    fit = np.setdiff1d(outer_train, inner_fold)
    inner_mean, _, _ = selective.training_targets(y[fit], args.prior_quantile)
    fit_y, validation_y = y[fit] - inner_mean, y[inner_fold] - inner_mean
    outer_mean, _, _ = selective.training_targets(y[outer_train], args.prior_quantile)
    outer_y = y[outer_train] - outer_mean
    predictions, rows = [], []
    for seed in seeds:
        selection = train_exp_deepdep(
            expression[fit], fit_y, expression[inner_fold], fingerprints, encoder,
            seed, args.deep_epochs, expression[inner_fold], validation_y,
            args.deep_patience, args.deep_batch_models)
        epoch_n = max(1, selection.epoch_n)
        fitted = train_exp_deepdep(
            expression[outer_train], outer_y, expression[held], fingerprints, encoder,
            seed, epoch_n, None, None, args.deep_patience, args.deep_batch_models)
        predictions.append(fitted.prediction)
        rows.append({
            "seed": seed,
            "selected_epoch": epoch_n,
            "inner_validation_mse": selection.validation_loss,
            "refit_training_mse_before_last_update": fitted.training_loss,
        })
    return np.mean(predictions, axis=0), rows


def summarize_metrics(metrics: pd.DataFrame):
    model_weighted = metrics.groupby("method", as_index=False)[list(METRICS)].mean()
    model_weighted.insert(0, "estimand", "model_weighted")
    lineage = metrics.groupby(["heldout_lineage", "method"], as_index=False)[list(METRICS)].mean()
    lineage_equal = lineage.groupby("method", as_index=False)[list(METRICS)].mean()
    lineage_equal.insert(0, "estimand", "lineage_equal")
    return pd.concat([model_weighted, lineage_equal], ignore_index=True), lineage


def paired_deltas(metrics: pd.DataFrame):
    keys = ["ModelID", "heldout_lineage"]
    wide = metrics.pivot(index=keys, columns="method", values=list(METRICS))
    rows = []
    reference = "expression_pcr_ridge"
    for method in sorted(metrics.method.unique()):
        if method == reference:
            continue
        record = wide.index.to_frame(index=False)
        record["comparison"] = f"{method}_vs_{reference}"
        for metric in METRICS:
            left, right = wide[(metric, method)], wide[(metric, reference)]
            record[metric] = (right - left if metric == "regret" else left - right).to_numpy()
        rows.append(record)
    return pd.concat(rows, ignore_index=True)


def bootstrap_deltas(deltas, models, draws, seed):
    patient_map = models.PatientID.to_dict()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for comparison, frame in deltas.groupby("comparison"):
        current = frame.copy()
        current["PatientID"] = current.ModelID.map(patient_map)
        for metric in METRICS:
            for estimand, values in (
                ("patient_equal", current.groupby("PatientID")[metric].mean().dropna().to_numpy()),
                ("lineage_equal", current.groupby("heldout_lineage")[metric].mean().dropna().to_numpy()),
            ):
                tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
                indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
                samples = tensor[indices].mean(1)
                quantiles = torch.quantile(
                    samples, torch.tensor([0.025, 0.5, 0.975], dtype=torch.float64, device="cuda")).cpu().numpy()
                rows.append({
                    "comparison": comparison, "metric": metric, "estimand": estimand,
                    "n": len(values), "mean": float(np.mean(values)),
                    "ci_low": quantiles[0], "bootstrap_median": quantiles[1], "ci_high": quantiles[2],
                })
    return pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/advanced_model_benchmark_v1")
    parser.add_argument("--depmap-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--splits", type=Path, default=root / "results/historical/selective_dependency_benchmark_v1/splits.csv.gz")
    parser.add_argument("--pretrained-encoder", type=Path,
                        default=root / "data/processed/tcga_pancan_deepdep_pretrain_v1/expression_encoder.pt")
    parser.add_argument("--protocol", type=Path, default=root / "configs/advanced_model_benchmark_protocol_20260917.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/advanced_model_benchmark_v1")
    parser.add_argument("--lineages", nargs="+")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--prior-quantile", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--residual-threshold", type=float, default=-0.5)
    parser.add_argument("--pcr-ranks", default="32,64,128,256")
    parser.add_argument("--pcr-alphas", default="100,1000,10000,100000")
    parser.add_argument("--rrr-ranks", default="16,32,64,128")
    parser.add_argument("--rrr-alphas", default="100,1000,10000,100000")
    parser.add_argument("--elastic-alphas", default="0.0001,0.001,0.01,0.1")
    parser.add_argument("--elastic-l1-ratio", type=float, default=0.5)
    parser.add_argument("--elastic-iterations", type=int, default=5000)
    parser.add_argument("--elastic-tolerance", type=float, default=1e-4)
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--deep-patience", type=int, default=3)
    parser.add_argument("--deep-batch-models", type=int, default=32)
    parser.add_argument("--deep-seeds", default="20260917,20260918,20260919")
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；高级模型不会退回CPU")
    baseline.configure_device("cuda")
    print("【阶段 1/5】校验冻结协议、共同输入和外层split", flush=True)
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    protocol = json.loads(args.protocol.read_text())
    audit, arrays = load_inputs(args.input_dir)
    models, _, _, _ = baseline.load_data(args.depmap_dir)
    if models.index.tolist() != arrays["model_ids"].tolist():
        raise ValueError("高级模型输入与DepMap模型顺序不一致")
    splits = exact_outer_splits(args.splits, arrays["model_ids"], args.lineages)
    labels, patients = models.OncotreeLineage.to_numpy(), models.PatientID.to_numpy()
    grids = {
        "pcr_rank": parse_grid(args.pcr_ranks, int), "pcr_alpha": parse_grid(args.pcr_alphas, float),
        "rrr_rank": parse_grid(args.rrr_ranks, int), "rrr_alpha": parse_grid(args.rrr_alphas, float),
        "elastic_alpha": parse_grid(args.elastic_alphas, float),
    }
    seeds = parse_grid(args.deep_seeds, int)
    if args.smoke_test:
        lineage = "Kidney" if "Kidney" in splits else next(iter(splits))
        splits = {lineage: splits[lineage]}
        arrays = dict(arrays)
        target = np.arange(min(64, arrays["dependency"].shape[1]))
        feature = np.arange(min(256, arrays["expression"].shape[1]))
        for key in ("target_original_symbols", "target_genes", "target_common_essential"):
            arrays[key] = arrays[key][target]
        arrays["dependency"] = arrays["dependency"][:, target]
        arrays["target_fingerprints"] = arrays["target_fingerprints"][target]
        if args.skip_deep:
            arrays["expression"] = arrays["expression"][:, feature]
        args.inner_folds, args.deep_epochs, args.deep_patience = 2, 2, 1
        args.elastic_iterations, args.bootstrap_draws = 100, 100
        seeds = seeds[:1]
        smoke_features = arrays["expression"].shape[1]
        print(f"【冒烟模式】Kidney｜64靶点｜{smoke_features}表达特征｜不构成研究结果", flush=True)
    print(f"【共同空间】癌系 {len(splits)}｜模型 {len(models)}｜靶点 {len(arrays['target_genes'])}｜主评价 {(~arrays['target_common_essential']).sum()}", flush=True)
    print("【方法】PCR-ridge｜逐靶点Elastic Net｜多任务低秩岭回归｜Exp-DeepDEP适配", flush=True)
    if not args.skip_deep and not args.pretrained_encoder.is_file() and not args.dry_run:
        raise FileNotFoundError(f"缺少TCGA预训练编码器：{args.pretrained_encoder}")
    if args.dry_run:
        print("【检查通过】未拟合模型、未读取TCGA KIRC Test、未写入结果。", flush=True)
        return
    if args.skip_deep:
        encoder = None
    else:
        autoencoder = ExpressionAutoencoder(arrays["expression"].shape[1])
        autoencoder.encoder.load_state_dict(
            torch.load(args.pretrained_encoder, map_location="cpu", weights_only=True))
        encoder = autoencoder.encoder

    metric_rows, tuning_rows, deep_rows, elastic_rows = [], [], [], []
    runtime_rows, selected_parameters = [], {}
    y = arrays["dependency"].astype(float)
    print("【阶段 2/5】逐癌系进行训练内选择并拟合外层模型", flush=True)
    for number, (lineage, (train, held)) in enumerate(splits.items(), 1):
        tick = time.monotonic()
        folds = baseline.inner_folds(train, labels, patients, args.inner_folds, args.seed + number - 1)
        print(f"  【癌系 {number}/{len(splits)}】{lineage}｜训练 {len(train)}｜留出 {len(held)}", flush=True)
        selected, target_alpha, rows, diagnostics = tune_classical(models, arrays, train, folds, grids, args)
        for row in rows:
            tuning_rows.append({"heldout_lineage": lineage, **row})
        for row in diagnostics:
            elastic_rows.append({"heldout_lineage": lineage, "scope": "inner", **row})
        mean, predictions, outer_elastic, feature_n = fit_classical(
            arrays, train, held, selected, target_alpha, grids, args)
        for alpha, row in outer_elastic.items():
            elastic_rows.append({"heldout_lineage": lineage, "scope": "outer_refit", "alpha": alpha, **row})
        if not args.skip_deep:
            deep_prediction, rows = tune_and_fit_deep(
                arrays, train, held, folds[0], encoder, seeds, args)
            predictions["exp_deepdep_adapted"] = deep_prediction
            for row in rows:
                deep_rows.append({"heldout_lineage": lineage, **row})
        truth = y[held] - mean
        rows, _ = selective.score_models(
            truth, predictions, models.index.to_numpy()[held], labels[held], arrays["target_genes"],
            ~arrays["target_common_essential"].astype(bool), args.top_k, args.residual_threshold)
        metric_rows.extend(rows)
        selected_parameters[lineage] = {
            method: list(config) for method, config in selected.items()
        }
        selected_parameters[lineage]["per_target_elastic_net_targetwise_counts"] = {
            str(alpha): int((target_alpha == alpha).sum()) for alpha in grids["elastic_alpha"]
        }
        elapsed = time.monotonic() - tick
        runtime_rows.append({"heldout_lineage": lineage, "elapsed_seconds": elapsed,
                             "standardized_expression_feature_n": feature_n})
        current = pd.DataFrame(rows).groupby("method").ndcg_at_10.mean()
        print("    外层NDCG｜" + "｜".join(
            f"{METHOD_LABELS[m]} {current[m]:.4f}" for m in predictions), flush=True)
        print(f"    完成｜{elapsed / 60:.1f}分钟", flush=True)

    print("【阶段 3/5】计算同模型配对差和GPU bootstrap区间", flush=True)
    metrics = pd.DataFrame(metric_rows)
    overall, lineage_summary = summarize_metrics(metrics)
    deltas = paired_deltas(metrics)
    bootstrap = bootstrap_deltas(deltas, models, args.bootstrap_draws, args.seed)
    print("【阶段 4/5】核验覆盖、收敛和Locked Test隔离", flush=True)
    expected = sum(len(held) for _, held in splits.values())
    if metrics.ModelID.nunique() != expected:
        raise ValueError("外层模型覆盖数异常")
    if not args.smoke_test and not pd.DataFrame(elastic_rows).converged.all():
        raise ValueError("至少一个正式Elastic Net拟合未收敛")
    if protocol["data"]["locked_test_rule"].startswith("No TCGA KIRC Locked Test") is False:
        raise ValueError("Locked Test协议异常")
    print("【隔离通过】模型选择未读取Sanger或TCGA KIRC Test", flush=True)

    print("【阶段 5/5】原子保存结果和审计记录", flush=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        metrics.to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        lineage_summary.to_csv(temporary / "lineage_summary.csv", index=False)
        deltas.to_csv(temporary / "paired_deltas.csv.gz", index=False)
        bootstrap.to_csv(temporary / "bootstrap.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(temporary / "tuning.csv", index=False)
        pd.DataFrame(elastic_rows).to_csv(temporary / "elastic_convergence.csv", index=False)
        pd.DataFrame(deep_rows).to_csv(temporary / "deep_training.csv", index=False)
        pd.DataFrame(runtime_rows).to_csv(temporary / "runtime.csv", index=False)
        run = {
            "status": "advanced_same_data_benchmark_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "protocol_sha256": sha256(args.protocol),
            "input_audit_sha256": sha256(args.input_dir / "audit.json"),
            "input_matrix_sha256": sha256(args.input_dir / "benchmark_inputs.npz"),
            "pretrained_encoder_sha256": None if args.skip_deep else sha256(args.pretrained_encoder),
            "script_sha256": sha256(Path(__file__)),
            "methods_script_sha256": sha256(Path(__file__).with_name("advanced_model_methods.py")),
            "model_n": int(metrics.ModelID.nunique()),
            "lineage_n": len(splits),
            "target_n": len(arrays["target_genes"]),
            "primary_target_n": int((~arrays["target_common_essential"]).sum()),
            "selected_parameters": selected_parameters,
            "deep_seeds": list(seeds),
            "tcga_kirc_locked_test_read": False,
            "limitations": [
                "Exp-DeepDEP is a same-data adaptation, not a byte-identical rerun of the 2021 model.",
                "The shared target universe is restricted to official DeepDEP default DepOIs.",
                "Common-essential labels are release-wide rather than fold-estimated.",
                "Cell-line ranking accuracy does not establish patient functional accuracy.",
            ],
        }
        run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    primary = overall.loc[overall.estimand.eq("lineage_equal")].set_index("method")
    print("【主结果｜癌系等权、共同非common-essential靶点】", flush=True)
    for method in METHOD_LABELS:
        if method not in primary.index:
            continue
        row = primary.loc[method]
        print(f"  {METHOD_LABELS[method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}｜Spearman {row.spearman:.4f}｜Regret {row.regret:.4f}", flush=True)
    print(f"【完成】耗时 {(time.monotonic() - started) / 60:.1f}分钟｜结果 {args.output_dir}", flush=True)
    print("【结论边界】只回答同数据细胞系排序性能；不能证明患者功能依赖。", flush=True)


if __name__ == "__main__":
    main()
