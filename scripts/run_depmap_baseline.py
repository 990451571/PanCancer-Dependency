"""Nested lineage-held-out ridge baselines for the broad DepMap NPZ dataset.

Only observed outcomes enter fitting/tuning/evaluation. Patient overlap is
purged at the outer split; connected lineage/patient groups define inner folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd


METHODS = ("global_mean", "background", "shared_multiomics")
DRIVERS = ("VHL", "PBRM1", "SETD2", "BAP1", "MTOR")
TORCH = None


def configure_device(device):
    global TORCH
    TORCH = None
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请检查显卡环境。程序不会自动退回 CPU。")
        TORCH = torch
        print(f"【计算设备】{torch.cuda.get_device_name(0)}｜CUDA｜双精度，保持原岭回归口径", flush=True)
    else:
        print("【计算设备】CPU（显式指定）", flush=True)


def gpu_add_kernel(kernel, held_kernel, train_values, held_values, chunk):
    torch = TORCH
    used = 0
    for start in range(0, train_values.shape[1], chunk):
        x = torch.as_tensor(train_values[:, start:start + chunk], dtype=torch.float64, device="cuda")
        hx = torch.as_tensor(held_values[:, start:start + chunk], dtype=torch.float64, device="cuda")
        mean = torch.nan_to_num(torch.nanmean(x, dim=0), nan=0.0)
        x = torch.where(torch.isfinite(x), x, mean) - mean
        hx = torch.where(torch.isfinite(hx), hx, mean) - mean
        sd = torch.sqrt((x * x).mean(dim=0))
        keep = sd > 1e-8
        x, hx = x[:, keep] / sd[keep], hx[:, keep] / sd[keep]
        kernel.add_(x @ x.T)
        held_kernel.add_(hx @ x.T)
        used += int(keep.sum().item())
    return used


def gpu_masked_ridge(kernel, held_kernel, y, alphas):
    torch = TORCH
    predictions = {alpha: np.full((len(held_kernel), y.shape[1]), np.nan) for alpha in alphas}
    observed = np.isfinite(y)
    _, groups = np.unique(np.packbits(observed.T, axis=1), axis=0, return_inverse=True)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed[:, columns[0]])
        if len(rows) < 2:
            continue
        selected = torch.as_tensor(rows, device="cuda")
        k = kernel[selected[:, None], selected[None, :]]
        h = held_kernel[:, selected]
        mean = k.mean(dim=0)
        overall = mean.mean()
        k = k - mean[None, :] - mean[:, None] + overall
        h = h - mean[None, :] - h.mean(dim=1, keepdim=True) + overall
        eigenvalues, eigenvectors = torch.linalg.eigh(k)
        eigenvalues = eigenvalues.clamp_min(0)
        outcomes = torch.as_tensor(y[np.ix_(rows, columns)], dtype=torch.float64, device="cuda")
        ymean = outcomes.mean(dim=0)
        projected = eigenvectors.T @ (outcomes - ymean)
        held_projected = h @ eigenvectors
        for alpha in alphas:
            result = (held_projected / (eigenvalues + alpha)) @ projected + ymean
            predictions[alpha][:, columns] = result.cpu().numpy()
    return predictions


def print_results(rows, scope):
    """Compact terminal summary; full metrics remain in result files."""
    names = {"global_mean": "均值基线", "background": "背景模型", "shared_multiomics": "多组学模型"}
    current = {(row["method"], row["stratum"]): row for row in rows if row["scope"] == scope}
    print(f"  【{'癌系整体' if scope == 'lineage' else '明确 ccRCC 子集'}】", flush=True)
    print("  模型          全部靶点ΔR²   非普遍必需ΔR²   非普遍必需NDCG   非普遍必需命中率", flush=True)
    for method in METHODS:
        all_genes = current.get((method, "all"), {})
        selective = current.get((method, "non_common_essential"), {})
        values = [all_genes.get("delta_r2_vs_global_mean"), selective.get("delta_r2_vs_global_mean"),
                  selective.get("ndcg"), selective.get("dependency_precision")]
        # Preserve column positions when a stratum/metric is unavailable.
        formatted = [(f"{value:+.4f}" if i < 2 else f"{value:.4f}")
                     if value is not None and np.isfinite(value) else "不可评价" for i, value in enumerate(values)]
        print(f"  {names[method]:　<7} " + "   ".join(f"{value:>10}" for value in formatted), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_mean(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(count.shape, np.nan, dtype=float), where=count > 0)


def load_data(source):
    audit = json.loads((source / "audit.json").read_text())
    required = ("matrices.npz", "models.csv", "gene_coverage.csv")
    for name in required:
        if sha256(source / name) != audit["output_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    models = pd.read_csv(source / "models.csv", index_col=0)
    coverage = pd.read_csv(source / "gene_coverage.csv", index_col=0)
    with np.load(source / "matrices.npz", allow_pickle=False) as archive:
        ids, genes = archive["model_ids"], archive["genes"]
        matrices = {name: archive[name] for name in ("dependency", "expression", "copy_number", "mutation")}
    if models.index.tolist() != ids.tolist() or models.index.has_duplicates or len(set(genes)) != len(genes):
        raise ValueError("Misaligned or duplicate model/gene identifiers")
    if coverage.index.has_duplicates:
        raise ValueError("Duplicate coverage genes")
    for name, values in matrices.items():
        if values.shape != (len(ids), len(genes)) or np.isinf(values).any():
            raise ValueError(f"Invalid shape or infinite values: {name}")
    for column in ("PatientID", "OncotreeLineage", "OncotreePrimaryDisease"):
        if models[column].isna().any() or models[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing required annotation: {column}")
    if models.OncotreePrimaryDisease.str.casefold().eq("non-cancerous").any():
        raise ValueError("Non-cancerous models in cancer baseline")
    essential = coverage.loc[genes, "DepMap_common_essential"]
    if essential.isna().any() or not essential.isin([True, False]).all():
        raise ValueError("Invalid common-essential annotation")
    for column in ("kidney_lineage", "renal_cell_carcinoma", "clear_cell_renal_cell_carcinoma"):
        if models[column].isna().any() or not models[column].isin([True, False]).all():
            raise ValueError(f"Invalid scope annotation: {column}")
    return models, genes, matrices, essential.to_numpy(dtype=bool)


def inner_folds(indices, lineages, patients, count, seed):
    """Join lineages sharing patients, then balance whole connected groups."""
    parent = {label: label for label in np.unique(lineages[indices])}

    def root(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    seen = {}
    for index in indices:
        patient, lineage = patients[index], lineages[index]
        if patient in seen:
            parent[root(lineage)] = root(seen[patient])
        seen[patient] = lineage
    groups = {}
    for index in indices:
        groups.setdefault(root(lineages[index]), []).append(index)
    count = min(count, len(groups))
    if count < 2:
        raise ValueError("Fewer than two independent lineage/patient groups for inner CV")
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: -len(groups[key]))
    folds = [[] for _ in range(count)]
    for key in keys:
        folds[min(range(count), key=lambda i: len(folds[i]))].extend(groups[key])
    result = [np.asarray(sorted(fold), dtype=int) for fold in folds]
    for held in result:
        train = np.setdiff1d(indices, held)
        if set(patients[held]) & set(patients[train]) or set(lineages[held]) & set(lineages[train]):
            raise ValueError("Patient or lineage overlap in inner CV")
    return result


def background_features(models, genes, matrices, train, held):
    labels = models.OncotreeLineage.to_numpy()
    categories = np.unique(labels[train])
    lineage = (labels[:, None] == categories[None, :]).astype(float)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    drivers = [matrices["mutation"][:, gene_index[gene]] for gene in DRIVERS if gene in gene_index]
    summaries = [observed_mean(matrices[name], axis=1)
                 for name in ("expression", "copy_number", "mutation")]
    values = np.column_stack([lineage, *drivers, *summaries])
    return values[train], values[held]


def add_kernel(kernel, held_kernel, train_values, held_values, chunk=1024):
    """Fit imputation/scale on training rows; accumulate a linear kernel."""
    if TORCH is not None:
        return gpu_add_kernel(kernel, held_kernel, train_values, held_values, chunk)
    used = 0
    for start in range(0, train_values.shape[1], chunk):
        x = np.asarray(train_values[:, start:start + chunk], dtype=float)
        hx = np.asarray(held_values[:, start:start + chunk], dtype=float)
        mean = observed_mean(x)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        x = np.where(np.isfinite(x), x, mean) - mean
        hx = np.where(np.isfinite(hx), hx, mean) - mean
        sd = np.sqrt((x * x).mean(axis=0))
        keep = sd > 1e-8
        x, hx = x[:, keep] / sd[keep], hx[:, keep] / sd[keep]
        kernel += x @ x.T
        held_kernel += hx @ x.T
        used += int(keep.sum())
    return used


def kernels(models, genes, matrices, train, held, feature_indices):
    if TORCH is not None:
        kernel = TORCH.zeros((len(train), len(train)), dtype=TORCH.float64, device="cuda")
        held_kernel = TORCH.zeros((len(held), len(train)), dtype=TORCH.float64, device="cuda")
    else:
        kernel = np.zeros((len(train), len(train)))
        held_kernel = np.zeros((len(held), len(train)))
    x, hx = background_features(models, genes, matrices, train, held)
    background_n = add_kernel(kernel, held_kernel, x, hx)
    result = {"background": (kernel.clone(), held_kernel.clone()) if TORCH is not None
              else (kernel.copy(), held_kernel.copy())}
    feature_n = background_n
    for name in ("expression", "copy_number", "mutation"):
        # Column chunking limits temporary allocations for the wide omics input.
        for start in range(0, len(feature_indices), 1024):
            columns = feature_indices[start:start + 1024]
            feature_n += add_kernel(kernel, held_kernel,
                                    matrices[name][np.ix_(train, columns)],
                                    matrices[name][np.ix_(held, columns)])
    result["shared_multiomics"] = (kernel, held_kernel)
    return result, {"background": background_n, "shared_multiomics": feature_n}


def masked_ridge(kernel, held_kernel, y, alphas):
    """Exact intercept-bearing ridge per target, grouped by observed rows.

    Training-feature normalization is common to all targets. Each outcome mask
    gets its own centering/intercept, using only rows where that target exists.
    """
    if TORCH is not None:
        return gpu_masked_ridge(kernel, held_kernel, y, alphas)
    predictions = {alpha: np.full((len(held_kernel), y.shape[1]), np.nan) for alpha in alphas}
    observed = np.isfinite(y)
    _, groups = np.unique(np.packbits(observed.T, axis=1), axis=0, return_inverse=True)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed[:, columns[0]])
        if len(rows) < 2:
            continue
        k = kernel[np.ix_(rows, rows)]
        h = held_kernel[:, rows]
        mean = k.mean(axis=0)
        overall = float(mean.mean())
        k = k - mean[None, :] - mean[:, None] + overall
        h = h - mean[None, :] - h.mean(axis=1, keepdims=True) + overall
        eigenvalues, eigenvectors = np.linalg.eigh(k)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        outcomes = y[np.ix_(rows, columns)].astype(float)
        ymean = outcomes.mean(axis=0)
        projected = eigenvectors.T @ (outcomes - ymean)
        held_projected = h @ eigenvectors
        for alpha in alphas:
            predictions[alpha][:, columns] = (held_projected / (eigenvalues + alpha)) @ projected + ymean
    return predictions


def fit_fold(models, genes, matrices, y, train, held, feature_indices, alphas):
    kernel_map, feature_counts = kernels(models, genes, matrices, train, held, feature_indices)
    predictions = {method: masked_ridge(*kernel_map[method], y[train], alphas) for method in METHODS[1:]}
    return predictions, feature_counts


def tune(models, genes, matrices, y, train, folds, feature_indices, alphas, lineage):
    errors = {method: {alpha: 0.0 for alpha in alphas} for method in METHODS[1:]}
    counts = {method: {alpha: 0 for alpha in alphas} for method in METHODS[1:]}
    for number, held in enumerate(folds, 1):
        fold_started = time.monotonic()
        fitting = np.setdiff1d(train, held)
        print(f"  内层验证 {number}/{len(folds)}：训练 {len(fitting)}，验证 {len(held)}，正在拟合…", flush=True)
        predictions, _ = fit_fold(models, genes, matrices, y, fitting, held, feature_indices, alphas)
        for method, by_alpha in predictions.items():
            for alpha, prediction in by_alpha.items():
                valid = np.isfinite(y[held]) & np.isfinite(prediction)
                errors[method][alpha] += float(((y[held][valid] - prediction[valid]) ** 2).sum())
                counts[method][alpha] += int(valid.sum())
        print(f"  内层验证 {number}/{len(folds)} 完成｜耗时 {time.monotonic() - fold_started:.1f} 秒", flush=True)
    all_counts = {n for by_alpha in counts.values() for n in by_alpha.values()}
    if len(all_counts) != 1 or min(all_counts) == 0:
        raise ValueError("Tuning methods/alphas have inconsistent or empty evaluation coverage")
    selected = {method: min(alphas, key=lambda a: (errors[method][a], a)) for method in errors}
    rows = [{"heldout_lineage": lineage, "method": method, "alpha": alpha,
             "observed_n": counts[method][alpha], "sse": errors[method][alpha],
             "selected": alpha == selected[method]} for method in errors for alpha in alphas]
    return selected, rows


def score_scope(truth, prediction, reference, background, ids, patients, genes, k, threshold):
    """Score a common observed universe; retain top-k label coverage separately."""
    valid = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(reference) & np.isfinite(background)
    y = np.where(valid, truth, np.nan)
    n = valid.sum(axis=0)
    centered = y - observed_mean(y)
    p = np.where(valid, prediction, np.nan)
    pc = p - observed_mean(p)
    sse = np.nansum(np.where(valid, (truth - prediction) ** 2, np.nan), axis=0)
    baseline_sse = np.nansum(np.where(valid, (truth - reference) ** 2, np.nan), axis=0)
    bg_sse = np.nansum(np.where(valid, (truth - background) ** 2, np.nan), axis=0)
    sst = np.nansum(centered ** 2, axis=0)
    correlation_denominator = np.sqrt(sst * np.nansum(pc ** 2, axis=0))
    correlation = np.divide(np.nansum(centered * pc, axis=0), correlation_denominator,
                            out=np.full(len(genes), np.nan), where=(n >= 3) & (correlation_denominator > 1e-12))
    gene_rows = pd.DataFrame({"Gene": genes, "observed_n": n, "sse": sse,
                             "global_mean_sse": baseline_sse, "sst": sst, "pearson": correlation})
    cells = []
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for row, model_id in enumerate(ids):
        available = np.flatnonzero(valid[row])
        predicted_available = np.flatnonzero(np.isfinite(prediction[row]))
        full_top = predicted_available[np.argsort(prediction[row, predicted_available], kind="stable")[:k]]
        record = {"ModelID": model_id, "PatientID": patients[row], "observed_n": len(available),
                  "predicted_topk_observed_fraction": float(valid[row, full_top].mean()) if len(full_top) else np.nan,
                  "predicted_topk_genes": "|".join(genes[full_top]),
                  "ndcg": np.nan, "topk_overlap": np.nan, "dependency_precision": np.nan,
                  "regret": np.nan, "observed_universe_topk_genes": ""}
        if len(available) >= k:
            selected = available[np.argsort(prediction[row, available], kind="stable")[:k]]
            oracle = available[np.argsort(truth[row, available], kind="stable")[:k]]
            ideal = float((np.maximum(0, -truth[row, oracle]) * discounts).sum())
            record.update(ndcg=float((np.maximum(0, -truth[row, selected]) * discounts).sum()) / ideal if ideal > 0 else np.nan,
                          topk_overlap=len(set(selected) & set(oracle)) / k,
                          dependency_precision=float((truth[row, selected] <= threshold).mean()),
                          regret=float(truth[row, selected].mean() - truth[row, oracle].mean()),
                          observed_universe_topk_genes="|".join(genes[selected]))
        cells.append(record)
    cell_frame = pd.DataFrame(cells)
    denominator = float(sst.sum())
    summary = {"model_n": len(ids), "patient_n": len(set(patients)), "gene_n": len(genes),
               "observed_n": int(n.sum()), "mse": float(sse.sum() / n.sum()) if n.sum() else np.nan,
               "r2": 1 - float(sse.sum()) / denominator if denominator > 1e-12 else np.nan,
               "delta_r2_vs_global_mean": float((baseline_sse - sse).sum()) / denominator if denominator > 1e-12 else np.nan,
               "delta_r2_vs_background": float((bg_sse - sse).sum()) / denominator if denominator > 1e-12 else np.nan,
               "gene_pearson_median": float(np.median(correlation[np.isfinite(correlation)])) if np.isfinite(correlation).any() else np.nan,
               "gene_pearson_evaluable_n": int(np.isfinite(correlation).sum())}
    for metric in ("ndcg", "topk_overlap", "dependency_precision", "regret", "predicted_topk_observed_fraction"):
        summary[metric] = float(cell_frame[metric].mean())
    summary["ranking_evaluable_model_n"] = int(cell_frame.topk_overlap.notna().sum())
    return summary, cell_frame, gene_rows


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/depmap_baseline_lolo_v1")
    parser.add_argument("--lineages", nargs="+")
    parser.add_argument("--minimum-lineage-size", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[10, 100, 1000, 10000, 100000])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dependency-threshold", type=float, default=-0.5)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="默认 CUDA；CPU 仅供显式数值对照，不自动降级")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--alpha-scan", action="store_true",
                        help="额外对外层全部 α 打分并输出 ranking-vs-alpha 诊断表 alpha_scan.csv")
    parser.add_argument("--smoke-test", action="store_true", help="Kidney by default; 128 targets, 256 feature genes, two inner folds; not research evidence")
    args = parser.parse_args()
    if args.inner_folds < 2 or args.minimum_lineage_size < 2 or args.top_k < 1:
        parser.error("Require inner-folds >= 2, minimum-lineage-size >= 2, top-k >= 1")
    if not all(np.isfinite(a) and a > 0 for a in args.alphas) or len(set(args.alphas)) != len(args.alphas):
        parser.error("alphas must be distinct finite positive numbers")
    if not np.isfinite(args.dependency_threshold) or args.seed < 0:
        parser.error("Require a finite dependency threshold and nonnegative seed")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"输出目录已存在，请使用新目录，避免覆盖：{output}")
    configure_device(args.device)
    print("【启动】正在校验并读取数据…", flush=True)
    models, all_genes, matrices, all_essential = load_data(source)
    labels, patients = models.OncotreeLineage.to_numpy(), models.PatientID.to_numpy()
    counts = models.OncotreeLineage.value_counts()
    eligible = sorted(counts[counts >= args.minimum_lineage_size].index)
    requested = args.lineages or (["Kidney"] if args.smoke_test else eligible)
    if not requested or set(requested) - set(eligible) or len(set(requested)) != len(requested):
        raise ValueError("Requested lineages must be unique and meet minimum sample size")
    rng = np.random.default_rng(args.seed)
    target_indices = np.arange(len(all_genes))
    feature_indices = np.arange(len(all_genes))
    if args.smoke_test:
        target_indices = np.sort(rng.choice(len(all_genes), min(128, len(all_genes)), replace=False))
        feature_indices = np.sort(rng.choice(len(all_genes), min(256, len(all_genes)), replace=False))
    genes, essential = all_genes[target_indices], all_essential[target_indices]
    y = matrices["dependency"][:, target_indices].astype(float)
    splits, split_rows = {}, []
    for number, lineage in enumerate(requested):
        held = np.flatnonzero(labels == lineage)
        train = np.flatnonzero((labels != lineage) & ~np.isin(patients, patients[held]))
        folds = inner_folds(train, labels, patients, 2 if args.smoke_test else args.inner_folds, args.seed + number)
        splits[lineage] = train, held, folds
        assignment = np.full(len(models), -1)
        for fold_id, indices in enumerate(folds):
            assignment[indices] = fold_id
        for index, model_id in enumerate(models.index):
            split_rows.append({"heldout_lineage": lineage, "ModelID": model_id, "PatientID": patients[index],
                               "role": "test" if index in held else "train" if assignment[index] >= 0 else "patient_overlap_purged",
                               "inner_validation_fold": int(assignment[index])})
    mode = "仅检查输入" if args.dry_run else "小规模程序检查（不作为研究结果）" if args.smoke_test else "正式基线实验"
    print(f"【{mode}】模型 {len(models)}｜靶点 {len(genes)}｜特征基因 {len(feature_indices)}｜癌系 {len(requested)}", flush=True)
    if args.dry_run:
        for lineage, (train, held, folds) in splits.items():
            print(f"  {lineage}：训练 {len(train)}｜留出 {len(held)}｜留出患者 {len(set(patients[held]))}｜内层 {len(folds)} 折", flush=True)
        print("【检查通过】文件校验、数据维度、注释及分组隔离正常；未训练、未写入结果。", flush=True)
        return
    print(f"【指标】ΔR² 相对均值基线，正值更好；排序指标基于已观测靶点；命中率为预测前 {args.top_k} 个中依赖评分 ≤ {args.dependency_threshold:g} 的比例。", flush=True)
    summaries, cells, gene_frames, tuning, alpha_scan = [], [], [], [], []
    feature_counts = {}
    for number, (lineage, (train, held, folds)) in enumerate(splits.items(), 1):
        lineage_started = time.monotonic()
        summary_start = len(summaries)
        print(f"\n【癌系 {number}/{len(splits)}】{lineage}｜训练 {len(train)}｜留出 {len(held)}｜开始调参", flush=True)
        selected, rows = tune(models, all_genes, matrices, y, train, folds, feature_indices, args.alphas, lineage)
        tuning.extend(rows)
        print(f"  已选参数：背景模型 α={selected['background']:g}｜多组学模型 α={selected['shared_multiomics']:g}；正在拟合外层模型…", flush=True)
        if selected["shared_multiomics"] == max(args.alphas):
            print("  【调参边界】多组学 α 达到搜索上限；尚不能确认最佳正则化强度。", flush=True)
        # Fit each method at its chosen alpha only.
        kernel_map, feature_counts[lineage] = kernels(models, all_genes, matrices, train, held, feature_indices)
        mean = observed_mean(y[train])
        mean[np.isfinite(y[train]).sum(axis=0) < 2] = np.nan
        predictions = {"global_mean": np.broadcast_to(mean, (len(held), len(genes)))}
        for method in METHODS[1:]:
            predictions[method] = masked_ridge(*kernel_map[method], y[train], [selected[method]])[selected[method]]
        scopes = {"lineage": np.arange(len(held))}
        if lineage == "Kidney":
            scopes["ccRCC"] = np.flatnonzero(models.iloc[held].clear_cell_renal_cell_carcinoma.to_numpy(dtype=bool))
        strata = {"all": np.ones(len(genes), dtype=bool), "common_essential": essential, "non_common_essential": ~essential}
        for scope, row_indices in scopes.items():
            if not len(row_indices):
                continue
            for stratum, columns in strata.items():
                if not columns.any():
                    continue
                ix = np.ix_(row_indices, np.flatnonzero(columns))
                for method in METHODS:
                    summary, cell_frame, gene_frame = score_scope(
                        y[held][ix], predictions[method][ix], predictions["global_mean"][ix],
                        predictions["background"][ix], models.index.to_numpy()[held[row_indices]],
                        patients[held[row_indices]], genes[columns], args.top_k, args.dependency_threshold)
                    context = {"heldout_lineage": lineage, "scope": scope, "stratum": stratum, "method": method}
                    summaries.append({**context, "selected_alpha": selected.get(method), **summary})
                    cells.append(cell_frame.assign(**context))
                    # Avoid duplicating gene-level rows across overlapping strata.
                    if stratum == "all":
                        gene_frames.append(gene_frame.assign(**context, common_essential=essential))
        if args.alpha_scan:
            # 诊断：外层对全部 α 打分，比较 SSE 最优与排序最优是否一致。
            # 复用已算好的 kernel_map，α 循环仅是除法，几乎不增加耗时。
            by_alpha = {method: masked_ridge(*kernel_map[method], y[train], args.alphas)
                        for method in METHODS[1:]}
            for scope, row_indices in scopes.items():
                if not len(row_indices):
                    continue
                for stratum, columns in strata.items():
                    if not columns.any():
                        continue
                    ix = np.ix_(row_indices, np.flatnonzero(columns))
                    for method in METHODS[1:]:
                        for alpha in args.alphas:
                            summary, _, _ = score_scope(
                                y[held][ix], by_alpha[method][alpha][ix], predictions["global_mean"][ix],
                                predictions["background"][ix], models.index.to_numpy()[held[row_indices]],
                                patients[held[row_indices]], genes[columns], args.top_k, args.dependency_threshold)
                            alpha_scan.append({"heldout_lineage": lineage, "scope": scope, "stratum": stratum,
                                               "method": method, "alpha": alpha,
                                               "selected_alpha": selected[method], **summary})
        for scope, indices in scopes.items():
            if len(indices):
                print_results(summaries[summary_start:], scope)
        print(f"【本癌系完成】耗时 {time.monotonic() - lineage_started:.1f} 秒｜累计 {(time.monotonic() - started) / 60:.1f} 分钟", flush=True)
    print("【保存】正在写入评价结果与运行记录…", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        pd.DataFrame(summaries).to_csv(temporary / "metrics.csv", index=False)
        pd.concat(cells, ignore_index=True).to_csv(temporary / "per_model_metrics.csv.gz", index=False)
        pd.concat(gene_frames, ignore_index=True).to_csv(temporary / "gene_metrics.csv.gz", index=False)
        pd.DataFrame(tuning).to_csv(temporary / "tuning.csv", index=False)
        pd.DataFrame(split_rows).to_csv(temporary / "splits.csv", index=False)
        if args.alpha_scan:
            pd.DataFrame(alpha_scan).to_csv(temporary / "alpha_scan.csv", index=False)
        manifest = {
            "status": "smoke_test_not_research_evidence" if args.smoke_test else "nested_lolo_baseline",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "input_sha256": {name: sha256(source / name) for name in ("audit.json", "matrices.npz", "models.csv", "gene_coverage.csv")},
            "script_sha256": sha256(__file__), "numpy": np.__version__, "pandas": pd.__version__,
            "compute": {"device": args.device, "dtype": "float64",
                        "torch_version": TORCH.__version__ if TORCH is not None else None,
                        "gpu_name": TORCH.cuda.get_device_name(0) if TORCH is not None else None},
            "model_n": len(models), "target_genes": genes.tolist(), "feature_genes": all_genes[feature_indices].tolist(),
            "usable_feature_counts": feature_counts, "elapsed_seconds": time.monotonic() - started,
            "rules": {
                "split": "Outer whole-lineage holdout with patient-overlap purging; inner whole connected lineage/patient groups",
                "features": "Background = training lineage one-hot + five damaging driver indicators + per-model modality means; shared adds standardized expression/CN/mutation",
                "preprocessing": "Per training fold: feature mean imputation, scaling, constant-feature removal; no outcome imputation",
                "fitting": "Exact masked linear-kernel ridge with per-target observed-row intercept; targets with <2 training observations are unavailable",
                "tuning": "Minimum aggregate observed inner-validation SSE; identical coverage for all methods/alphas; ties choose smaller alpha",
                "r2": "1-SSE/SST; SST sums per-target observed heldout deviations from heldout means; delta R2=(reference SSE-model SSE)/SST",
                "ranking": "Top-k within common observed target universe per model/stratum; stable gene-order ties; separately report label coverage of unrestricted predicted top-k",
                "correlation": "Per-target Pearson across heldout models with >=3 observed outcomes and nonzero truth/prediction variance",
            },
            "limitations": [
                "Cell-line LOLO is not patient-domain or independent-platform validation.",
                "Metrics weight models/observed entries, not patients equally; patient groups prevent split leakage but no confidence intervals are estimated.",
                "Common-essential strata use release-wide annotation, not fold-estimated common essentiality.",
                "Ranking is conditional on observed labels; missing predictions/labels cannot validate an unrestricted top-k.",
                "ccRCC evaluation is exploratory; it does not select hyperparameters or prove ccRCC specificity.",
                "Expression/CN/mutation feature dimensions differ in retained variability and are not modality-balanced.",
                "No deployable final model is saved; this entry point evaluates baselines only.",
            ],
            "output_sha256": {path.name: sha256(path) for path in sorted(temporary.iterdir())},
        }
        if args.alpha_scan:
            manifest["rules"]["alpha_scan"] = ("Outer ranking/R2 scored at every alpha (not only the SSE-selected one) "
                                               "for ridge methods; output alpha_scan.csv")
        (temporary / "run.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"【完成】{len(splits)} 个癌系｜总耗时 {(time.monotonic() - started) / 60:.1f} 分钟", flush=True)
    print(f"结果目录：{output}\n主要结果：metrics.csv（含全部分层指标）", flush=True)
    if args.alpha_scan:
        print(f"排序-α 诊断：alpha_scan.csv", flush=True)


if __name__ == "__main__":
    main()
