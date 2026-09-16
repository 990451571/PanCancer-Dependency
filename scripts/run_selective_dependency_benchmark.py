#!/usr/bin/env python3
"""Leakage-safe GPU benchmark for selective-dependency ranking baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline
import run_selective_dependency as selective


METHOD_LABELS = {
    "training_selectivity_prior": "训练选择性先验",
    "annotation_ridge": "仅注释岭回归",
    "expression_knn": "表达近邻",
    "expression_pcr_ridge": "低秩表达PCR岭回归",
    "expression_kernel_ridge_frozen": "冻结表达核岭回归",
    "expression_kernel_ridge_tuned": "内层调参表达核岭回归",
}
METRICS = selective.METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_grid(value: str, cast):
    return tuple(cast(part) for part in value.split(",") if part.strip())


def build_kernels(models, genes, matrices, train, held, chunk=2048):
    """Return annotation and genome-wide expression kernels plus expression norms."""
    torch = baseline.TORCH
    annotation = torch.zeros((len(train), len(train)), dtype=torch.float64, device="cuda")
    annotation_held = torch.zeros((len(held), len(train)), dtype=torch.float64, device="cuda")
    annotation_counts = {}
    blocks = baseline.annotation_features(models, genes, matrices, train, held)
    for name in ("lineage", "drivers", "expression_mean"):
        if name in blocks:
            annotation_counts[name] = baseline.add_kernel(
                annotation, annotation_held, *blocks[name])

    expression = torch.zeros_like(annotation)
    expression_held = torch.zeros_like(annotation_held)
    train_norm = torch.zeros(len(train), dtype=torch.float64, device="cuda")
    held_norm = torch.zeros(len(held), dtype=torch.float64, device="cuda")
    raw_train, raw_held = matrices["expression"][train], matrices["expression"][held]
    used = 0
    for start in range(0, raw_train.shape[1], chunk):
        x = torch.as_tensor(raw_train[:, start:start + chunk], dtype=torch.float64, device="cuda")
        hx = torch.as_tensor(raw_held[:, start:start + chunk], dtype=torch.float64, device="cuda")
        mean = torch.nan_to_num(torch.nanmean(x, dim=0), nan=0.0)
        x = torch.where(torch.isfinite(x), x, mean) - mean
        hx = torch.where(torch.isfinite(hx), hx, mean) - mean
        sd = torch.sqrt((x * x).mean(dim=0))
        keep = sd > 1e-8
        x, hx = x[:, keep] / sd[keep], hx[:, keep] / sd[keep]
        expression.add_(x @ x.T)
        expression_held.add_(hx @ x.T)
        train_norm.add_((x * x).sum(dim=1))
        held_norm.add_((hx * hx).sum(dim=1))
        used += int(keep.sum())
    return {
        "annotation": (annotation, annotation_held),
        "expression": (expression, expression_held),
        "full": (annotation + expression, annotation_held + expression_held),
        "train_norm": train_norm,
        "held_norm": held_norm,
        "feature_counts": {**annotation_counts, "expression": used},
    }


def knn_predictions(kernel, held_kernel, train_norm, held_norm, residual, ks):
    """Cosine-neighbor residual predictions; labels never affect neighbor selection."""
    torch = baseline.TORCH
    denominator = torch.sqrt(held_norm[:, None].clamp_min(1e-12) * train_norm[None, :].clamp_min(1e-12))
    similarity = held_kernel / denominator
    largest = max(ks)
    neighbors = torch.topk(similarity, k=min(largest, similarity.shape[1]), dim=1).indices.cpu().numpy()
    predictions = {}
    for k in ks:
        selected = neighbors[:, :min(k, neighbors.shape[1])]
        values = residual[selected]
        count = np.isfinite(values).sum(axis=1)
        total = np.nansum(values, axis=1)
        predictions[k] = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
    return predictions


def pcr_ridge_predictions(kernel, held_kernel, residual, ranks, alphas):
    """Fit masked ridge on training-only kernel principal-component scores."""
    torch = baseline.TORCH
    n = kernel.shape[0]
    mean = kernel.mean(dim=0)
    overall = mean.mean()
    centered = kernel - mean[None, :] - mean[:, None] + overall
    held_centered = held_kernel - mean[None, :] - held_kernel.mean(dim=1, keepdim=True) + overall
    eigenvalues, eigenvectors = torch.linalg.eigh(centered)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = eigenvalues > 1e-8
    maximum = min(max(ranks), int(keep.sum()))
    if maximum < 1:
        raise ValueError("表达核没有可用主成分")
    values = eigenvalues[:maximum]
    vectors = eigenvectors[:, :maximum]
    x = vectors * torch.sqrt(values)[None, :]
    hx = (held_centered @ vectors) / torch.sqrt(values)[None, :]

    outputs = {(rank, alpha): np.full((held_kernel.shape[0], residual.shape[1]), np.nan)
               for rank in ranks for alpha in alphas}
    observed = np.isfinite(residual)
    _, groups = np.unique(np.packbits(observed.T, axis=1), axis=0, return_inverse=True)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed[:, columns[0]])
        if len(rows) < 2:
            continue
        selected = torch.as_tensor(rows, device="cuda")
        outcomes = torch.as_tensor(residual[np.ix_(rows, columns)], dtype=torch.float64, device="cuda")
        ymean = outcomes.mean(dim=0)
        outcomes = outcomes - ymean
        for requested_rank in ranks:
            rank = min(requested_rank, maximum, len(rows) - 1)
            if rank < 1:
                continue
            design = x[selected, :rank]
            design_mean = design.mean(dim=0)
            design = design - design_mean
            held_design = hx[:, :rank] - design_mean
            gram = design.T @ design
            cross = design.T @ outcomes
            eye = torch.eye(rank, dtype=torch.float64, device="cuda")
            for alpha in alphas:
                beta = torch.linalg.solve(gram + alpha * eye, cross)
                outputs[(requested_rank, alpha)][:, columns] = (held_design @ beta + ymean).cpu().numpy()
    return outputs


def metric_frame(truth, prediction, ids, lineages, genes, stratum, top_k, threshold, method):
    rows, _ = selective.score_models(
        truth, {method: prediction}, ids, lineages, genes, stratum, top_k, threshold)
    return pd.DataFrame(rows)


def tuning_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return -np.inf
    return float(frame.groupby("heldout_lineage").ndcg_at_10.mean().mean())


def select_best(scores, simplicity):
    ordered = sorted(scores, key=lambda config: (-scores[config], simplicity(config)))
    return ordered[0]


def tune_outer(models, genes, matrices, common_essential, y, outer_train, folds, grids, args):
    labels = models.OncotreeLineage.to_numpy()
    ids = models.index.to_numpy()
    accumulated = defaultdict(list)
    for fold_number, held in enumerate(folds, 1):
        fit = np.setdiff1d(outer_train, held)
        mean, prior, _ = selective.training_targets(y[fit], args.prior_quantile)
        residual = y[fit] - mean
        truth = y[held] - mean
        kernels = build_kernels(models, genes, matrices, fit, held, args.kernel_chunk)

        annotation = baseline.masked_ridge(*kernels["annotation"], y[fit], grids["annotation_alpha"])
        for alpha, raw in annotation.items():
            frame = metric_frame(truth, raw - mean, ids[held], labels[held], genes,
                                 ~common_essential, args.top_k, args.residual_threshold, "m")
            accumulated[("annotation_ridge", alpha)].append(frame)

        knn = knn_predictions(*kernels["expression"], kernels["train_norm"], kernels["held_norm"],
                              residual, grids["knn_k"])
        for k, prediction in knn.items():
            frame = metric_frame(truth, prediction, ids[held], labels[held], genes,
                                 ~common_essential, args.top_k, args.residual_threshold, "m")
            accumulated[("expression_knn", k)].append(frame)

        pcr = pcr_ridge_predictions(*kernels["expression"], residual,
                                    grids["pcr_rank"], grids["pcr_alpha"])
        for config, prediction in pcr.items():
            frame = metric_frame(truth, prediction, ids[held], labels[held], genes,
                                 ~common_essential, args.top_k, args.residual_threshold, "m")
            accumulated[("expression_pcr_ridge", *config)].append(frame)

        full = baseline.masked_ridge(*kernels["full"], y[fit], grids["kernel_alpha"])
        for alpha, raw in full.items():
            frame = metric_frame(truth, raw - mean, ids[held], labels[held], genes,
                                 ~common_essential, args.top_k, args.residual_threshold, "m")
            accumulated[("expression_kernel_ridge_tuned", alpha)].append(frame)
        print(f"    内层 {fold_number}/{len(folds)}｜训练 {len(fit)}｜验证 {len(held)}", flush=True)

    scores = {config: tuning_score(pd.concat(frames, ignore_index=True))
              for config, frames in accumulated.items()}
    selected = {}
    method_configs = defaultdict(dict)
    for config, score in scores.items():
        method_configs[config[0]][config[1:]] = score
    selected["annotation_ridge"] = select_best(
        method_configs["annotation_ridge"], lambda c: (-c[0],))
    selected["expression_knn"] = select_best(
        method_configs["expression_knn"], lambda c: (-c[0],))
    selected["expression_pcr_ridge"] = select_best(
        method_configs["expression_pcr_ridge"], lambda c: (c[0], -c[1]))
    selected["expression_kernel_ridge_tuned"] = select_best(
        method_configs["expression_kernel_ridge_tuned"], lambda c: (-c[0],))
    rows = []
    for method, values in method_configs.items():
        for config, score in values.items():
            rows.append({"method": method, "config": "|".join(map(str, config)),
                         "inner_lineage_equal_ndcg_at_10": score,
                         "selected": config == selected[method]})
    return selected, rows


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
    reference = "training_selectivity_prior"
    frozen = "expression_kernel_ridge_frozen"
    for method in sorted(metrics.method.unique()):
        if method == reference:
            continue
        record = wide.index.to_frame(index=False)
        record["comparison"] = f"{method}_vs_{reference}"
        for metric in METRICS:
            left, right = wide[(metric, method)], wide[(metric, reference)]
            record[metric] = (right - left if metric == "regret" else left - right).to_numpy()
        rows.append(record)
        if method != frozen:
            record = wide.index.to_frame(index=False)
            record["comparison"] = f"{frozen}_vs_{method}"
            for metric in METRICS:
                left, right = wide[(metric, frozen)], wide[(metric, method)]
                record[metric] = (right - left if metric == "regret" else left - right).to_numpy()
            rows.append(record)
    return pd.concat(rows, ignore_index=True)


def bootstrap_deltas(deltas, models, draws, seed):
    torch = baseline.TORCH
    patient_map = models.PatientID.to_dict()
    rows = []
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for comparison, frame in deltas.groupby("comparison"):
        frame = frame.copy()
        frame["PatientID"] = frame.ModelID.map(patient_map)
        for metric in METRICS:
            patient_values = frame.groupby("PatientID")[metric].mean().dropna().to_numpy()
            lineage_values = frame.groupby("heldout_lineage")[metric].mean().dropna().to_numpy()
            for estimand, values in (("patient_equal", patient_values), ("lineage_equal", lineage_values)):
                tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
                indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
                samples = tensor[indices].mean(dim=1)
                interval = torch.quantile(samples, torch.tensor([0.025, 0.5, 0.975],
                                         dtype=torch.float64, device="cuda")).cpu().numpy()
                rows.append({"comparison": comparison, "metric": metric, "estimand": estimand,
                             "unit_n": len(values), "mean": float(values.mean()),
                             "ci_low": float(interval[0]), "bootstrap_median": float(interval[1]),
                             "ci_high": float(interval[2])})
    return pd.DataFrame(rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--protocol", type=Path, default=root / "configs/dependency_benchmark_protocol_20260916.json")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/selective_dependency_benchmark_v1")
    parser.add_argument("--minimum-lineage-size", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--prior-quantile", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--residual-threshold", type=float, default=-0.5)
    parser.add_argument("--annotation-alphas", default="100,1000,10000,100000,1000000")
    parser.add_argument("--knn-k", default="5,10,25,50")
    parser.add_argument("--pcr-ranks", default="32,64,128,256")
    parser.add_argument("--pcr-alphas", default="100,1000,10000,100000")
    parser.add_argument("--kernel-alphas", default="1000,10000,100000,1000000")
    parser.add_argument("--frozen-alpha", type=float, default=100000.0)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--kernel-chunk", type=int, default=2048)
    parser.add_argument("--lineages", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.protocol.exists():
        parser.error("冻结 benchmark 协议不存在")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    baseline.configure_device("cuda")
    print("【阶段 1/5】校验冻结协议、输入与整癌系分组", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(source)
    labels, patients = models.OncotreeLineage.to_numpy(), models.PatientID.to_numpy()
    counts = models.OncotreeLineage.value_counts()
    eligible = sorted(counts[counts >= args.minimum_lineage_size].index)
    lineages = args.lineages or eligible
    if set(lineages) - set(eligible):
        raise ValueError("请求癌系不满足冻结的最小样本量")
    splits = {}
    for number, lineage in enumerate(lineages):
        held = np.flatnonzero(labels == lineage)
        train = np.flatnonzero((labels != lineage) & ~np.isin(patients, patients[held]))
        folds = baseline.inner_folds(train, labels, patients, args.inner_folds, args.seed + number)
        splits[lineage] = train, held, folds
    grids = {
        "annotation_alpha": parse_grid(args.annotation_alphas, float),
        "knn_k": parse_grid(args.knn_k, int),
        "pcr_rank": parse_grid(args.pcr_ranks, int),
        "pcr_alpha": parse_grid(args.pcr_alphas, float),
        "kernel_alpha": parse_grid(args.kernel_alphas, float),
    }
    print(f"【冻结比较】癌系 {len(lineages)}｜模型 {len(models)}｜基因 {len(genes)}｜主评价基因 {(~common_essential).sum()}", flush=True)
    print("【方法】选择性先验｜仅注释岭回归｜表达近邻｜低秩PCR岭回归｜完整表达核岭回归", flush=True)
    if args.dry_run:
        print("【检查通过】未读取任何外部标签、未拟合模型、未写入结果。", flush=True)
        return

    y = matrices["dependency"].astype(float)
    metric_rows, ranking_rows, tuning_rows, split_rows = [], [], [], []
    feature_counts, selected_parameters = {}, {}
    print("【阶段 2/5】训练折内选择各 baseline 参数", flush=True)
    for number, (lineage, (train, held, folds)) in enumerate(splits.items(), 1):
        tick = time.monotonic()
        print(f"  【癌系 {number}/{len(splits)}】{lineage}｜训练 {len(train)}｜留出 {len(held)}", flush=True)
        selected, rows = tune_outer(models, genes, matrices, common_essential, y, train, folds, grids, args)
        selected_parameters[lineage] = {method: list(config) for method, config in selected.items()}
        for row in rows:
            tuning_rows.append({"heldout_lineage": lineage, **row})
        for index in train:
            split_rows.append({"heldout_lineage": lineage, "ModelID": models.index[index], "role": "train"})
        for index in held:
            split_rows.append({"heldout_lineage": lineage, "ModelID": models.index[index], "role": "test"})

        print("    已选参数｜" + "｜".join(f"{METHOD_LABELS[m]}={','.join(map(str,c))}" for m,c in selected.items()), flush=True)
        mean, prior, _ = selective.training_targets(y[train], args.prior_quantile)
        residual, truth = y[train] - mean, y[held] - mean
        kernels = build_kernels(models, genes, matrices, train, held, args.kernel_chunk)
        feature_counts[lineage] = kernels["feature_counts"]
        predictions = {"training_selectivity_prior": np.broadcast_to(prior, truth.shape)}
        alpha = selected["annotation_ridge"][0]
        predictions["annotation_ridge"] = baseline.masked_ridge(
            *kernels["annotation"], y[train], [alpha])[alpha] - mean
        k = selected["expression_knn"][0]
        predictions["expression_knn"] = knn_predictions(
            *kernels["expression"], kernels["train_norm"], kernels["held_norm"], residual, [k])[k]
        rank, alpha = selected["expression_pcr_ridge"]
        predictions["expression_pcr_ridge"] = pcr_ridge_predictions(
            *kernels["expression"], residual, [rank], [alpha])[(rank, alpha)]
        selected_alpha = selected["expression_kernel_ridge_tuned"][0]
        requested = sorted(set([args.frozen_alpha, selected_alpha]))
        full = baseline.masked_ridge(*kernels["full"], y[train], requested)
        predictions["expression_kernel_ridge_frozen"] = full[args.frozen_alpha] - mean
        predictions["expression_kernel_ridge_tuned"] = full[selected_alpha] - mean

        rows, ranks = selective.score_models(
            truth, predictions, models.index.to_numpy()[held], labels[held], genes,
            ~common_essential, args.top_k, args.residual_threshold)
        metric_rows.extend(rows)
        ranking_rows.extend(ranks)
        current = pd.DataFrame(rows).groupby("method").ndcg_at_10.mean()
        print("    外层NDCG｜" + "｜".join(f"{METHOD_LABELS[m]} {current[m]:.4f}" for m in METHOD_LABELS), flush=True)
        print(f"    完成｜{time.monotonic() - tick:.1f}秒", flush=True)

    print("【阶段 3/5】计算配对增益和 GPU bootstrap 区间", flush=True)
    metrics = pd.DataFrame(metric_rows)
    overall, lineage_summary = summarize_metrics(metrics)
    deltas = paired_deltas(metrics)
    bootstrap = bootstrap_deltas(deltas, models, args.bootstrap_draws, args.seed)

    print("【阶段 4/5】核验冻结方法、覆盖率与 Test 隔离", flush=True)
    expected_model_n = sum(len(held) for _, held, _ in splits.values())
    if metrics.ModelID.nunique() != expected_model_n:
        raise ValueError("外层模型覆盖数量异常")
    protocol = json.loads(args.protocol.read_text())
    if protocol["locked_tcga_test"]["formal_access_n"] != 1:
        raise ValueError("Locked Test 协议状态异常")
    print("【隔离通过】baseline 未读取 TCGA Test 或 Sanger 标签进行选择", flush=True)

    print("【阶段 5/5】保存 benchmark 结果和审计记录", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        metrics.to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        pd.DataFrame(ranking_rows).to_csv(temporary / "top10_rankings.csv.gz", index=False)
        overall.to_csv(temporary / "overall_summary.csv", index=False)
        lineage_summary.to_csv(temporary / "lineage_summary.csv", index=False)
        deltas.to_csv(temporary / "paired_deltas.csv.gz", index=False)
        bootstrap.to_csv(temporary / "bootstrap.csv", index=False)
        pd.DataFrame(tuning_rows).to_csv(temporary / "tuning.csv", index=False)
        pd.DataFrame(split_rows).to_csv(temporary / "splits.csv.gz", index=False)
        run = {
            "status": "selective_dependency_baseline_benchmark_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "device": baseline.TORCH.cuda.get_device_name(0),
            "dtype": "float64",
            "protocol_sha256": sha256(args.protocol),
            "input_sha256": {name: sha256(source / name) for name in ("audit.json", "matrices.npz", "models.csv", "gene_coverage.csv")},
            "script_sha256": sha256(Path(__file__)),
            "model_n": int(metrics.ModelID.nunique()),
            "lineage_n": len(lineages),
            "gene_n": len(genes),
            "primary_gene_n": int((~common_essential).sum()),
            "selected_parameters": selected_parameters,
            "feature_counts": feature_counts,
            "rules": {
                "test_isolation": "No TCGA or Sanger label was read by this script",
                "outer_split": "Whole-lineage holdout with patient-overlap purging",
                "inner_split": "Whole connected lineage/patient groups",
                "selection": "Inner lineage-equal NDCG@10 on non-common-essential residuals",
                "current_frozen_model": f"Expression kernel ridge alpha={args.frozen_alpha:g}",
                "paired_universe": "All methods share the same finite prediction/truth universe within each model",
                "bootstrap": f"{args.bootstrap_draws} GPU draws by patient and by lineage",
            },
            "limitations": [
                "The benchmark compares estimators on the current DepMap release; it is not a direct rerun of the 2024 Nature Cancer or DeepDEP pipelines.",
                "The locked TCGA Test was already accessed for the historical frozen model and is not an untouched confirmation set for newly implemented baselines.",
                "Common-essential annotation is release-wide rather than fold-estimated.",
                "Cell-line benchmark superiority would not establish patient functional accuracy.",
            ],
        }
        run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise

    primary = overall[overall.estimand.eq("lineage_equal")].set_index("method")
    print("【主结果｜癌系等权、非 common-essential】", flush=True)
    for method in METHOD_LABELS:
        row = primary.loc[method]
        print(f"  {METHOD_LABELS[method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}｜Spearman {row.spearman:.4f}｜Regret {row.regret:.4f}", flush=True)
    print(f"【完成】耗时 {(time.monotonic() - started) / 60:.1f}分钟｜结果 {output}", flush=True)
    print("【结论边界】这是细胞系内的公平方法比较；不能作为患者功能依赖或临床有效性证据。", flush=True)


if __name__ == "__main__":
    main()
