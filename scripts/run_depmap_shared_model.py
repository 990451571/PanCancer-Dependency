"""Cross-validated pan-cancer shared model for candidate CRISPR dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


DRIVERS = ("VHL", "PBRM1", "SETD2", "BAP1", "MTOR")


def stratified_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(folds)]
    for label in sorted(set(labels)):
        indices = rng.permutation(np.flatnonzero(labels == label))
        for offset, index in enumerate(indices):
            result[offset % folds].append(int(index))
    arrays = [np.asarray(sorted(values), dtype=int) for values in result]
    if sorted(np.concatenate(arrays).tolist()) != list(range(len(labels))):
        raise ValueError("Cross-validation folds do not partition models")
    return arrays


def train_standardize(train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    sd = train.std(axis=0)
    usable = sd > 1e-8
    return (train[:, usable] - mean[usable]) / sd[usable], (held[:, usable] - mean[usable]) / sd[usable]


def kernel_ridge_predictions(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
    alphas: list[float],
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    x, hx = train_standardize(train_x, held_x)
    y_mean = train_y.mean(axis=0)
    y_sd = train_y.std(axis=0)
    if (y_sd <= 1e-8).any():
        raise ValueError("A dependency target has zero training variance")
    y, hy = (train_y - y_mean) / y_sd, (held_y - y_mean) / y_sd
    kernel = x @ x.T
    held_kernel = hx @ x.T
    identity = np.eye(len(x))
    predictions = {
        alpha: held_kernel @ np.linalg.solve(kernel + alpha * identity, y)
        for alpha in alphas
    }
    return predictions, hy


def primal_ridge_predictions(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    held_y: np.ndarray,
    alphas: list[float],
) -> tuple[dict[float, np.ndarray], np.ndarray]:
    x, hx = train_standardize(train_x, held_x)
    y_mean = train_y.mean(axis=0)
    y_sd = train_y.std(axis=0)
    if (y_sd <= 1e-8).any():
        raise ValueError("A dependency target has zero training variance")
    y, hy = (train_y - y_mean) / y_sd, (held_y - y_mean) / y_sd
    gram = x.T @ x
    cross = x.T @ y
    identity = np.eye(x.shape[1])
    predictions = {
        alpha: hx @ np.linalg.solve(gram + alpha * identity, cross)
        for alpha in alphas
    }
    return predictions, hy


def targetwise_feature_sets(
    background: np.ndarray,
    expression: np.ndarray,
    copy_number: np.ndarray,
    mutation: np.ndarray,
    vhl: np.ndarray,
    pbrm1: np.ndarray,
    kidney: np.ndarray,
    gene_index: int,
) -> dict[str, np.ndarray]:
    own = np.c_[expression[:, gene_index], copy_number[:, gene_index], mutation[:, gene_index]]
    main = np.c_[background, own]
    interactions = np.c_[own * vhl[:, None], own * pbrm1[:, None], own[:, :2] * kidney[:, None]]
    return {"targetwise_multiomics": main, "targetwise_context": np.c_[main, interactions]}


def feature_sets(
    models: pd.DataFrame,
    mutation: pd.DataFrame,
    expression: pd.DataFrame,
    copy_number: pd.DataFrame,
    genes: list[str],
) -> dict[str, np.ndarray]:
    lineage = pd.get_dummies(models["OncotreeLineage"].fillna("Unknown"), dtype=float).to_numpy()
    drivers = mutation[[gene for gene in DRIVERS]].fillna(0).to_numpy(dtype=float)
    mut = mutation[genes].fillna(0).to_numpy(dtype=float)
    expr = expression[genes].to_numpy(dtype=float)
    cn = copy_number[genes].to_numpy(dtype=float)
    mutation_burden = mut.mean(axis=1, keepdims=True)
    cn_deviation = np.abs(np.log2(np.maximum(cn, 0.01) / 2.0)).mean(axis=1, keepdims=True)
    expression_level = expr.mean(axis=1, keepdims=True)
    background = np.c_[lineage, drivers, mutation_burden, cn_deviation, expression_level]
    omics = np.c_[background, expr, cn, mut]
    vhl = drivers[:, 0, None]
    pbrm1 = drivers[:, 1, None]
    kidney = models["kidney_lineage"].astype(float).to_numpy()[:, None]
    interactions = np.c_[expr * vhl, cn * vhl, mut * vhl,
                         expr * pbrm1, cn * pbrm1, mut * pbrm1,
                         expr * kidney, cn * kidney]
    return {"background": background, "multiomics": omics, "context_interactions": np.c_[omics, interactions]}


def aggregate_r2(y: np.ndarray, prediction: np.ndarray, indices: np.ndarray) -> float:
    denominator = float((y[indices] ** 2).sum())
    return 1.0 - float(((y[indices] - prediction[indices]) ** 2).sum()) / max(denominator, 1e-12)


def bootstrap_delta(
    y: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    loss_gain = ((y - reference) ** 2 - (y - prediction) ** 2).sum(axis=1)
    scale = (y ** 2).sum(axis=1)
    scoped_labels = labels[indices]
    strata = [indices[scoped_labels == label] for label in sorted(set(scoped_labels))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
        values.append(float(loss_gain[sample].sum() / max(scale[sample].sum(), 1e-12)))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def context_candidate_ranking(
    gene_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    alphas: list[float],
) -> tuple[pd.DataFrame, list[float]]:
    stable_alphas = sorted(alphas)[-2:]
    selected = gene_metrics.loc[
        (gene_metrics["model"] == "targetwise_context")
        & gene_metrics["scope"].isin(["pan_cancer", "rcc", "ccrcc"])
        & gene_metrics["alpha"].isin(stable_alphas),
        ["Gene", "scope", "alpha", "delta_r2"],
    ].copy()
    selected["metric"] = selected.apply(
        lambda row: f"context_delta_r2_{row['scope']}_alpha_{row['alpha']:g}", axis=1
    )
    wide = selected.pivot(index="Gene", columns="metric", values="delta_r2")
    expected = [
        f"context_delta_r2_{scope}_alpha_{alpha:g}"
        for scope in ["pan_cancer", "rcc", "ccrcc"]
        for alpha in stable_alphas
    ]
    if any(column not in wide for column in expected):
        raise ValueError("Missing context-ranking metric")
    ranking = coverage.loc[wide.index].copy().join(wide[expected])
    ranking["positive_context_metric_count"] = (ranking[expected] > 0).sum(axis=1)
    ranking["minimum_context_delta_r2"] = ranking[expected].min(axis=1)
    ranking["mean_context_delta_r2"] = ranking[expected].mean(axis=1)
    ranking["stable_positive_all_scopes"] = ranking["positive_context_metric_count"] == len(expected)
    ranking["ccrcc_functional_dependency"] = ranking["ccrcc_dependency_median"] <= -0.5
    ranking["ccrcc_selective_dependency"] = ranking["ccrcc_selectivity_delta_median"] <= -0.1
    ranking["tcga_multiomics_supported"] = (
        ranking["TCGA_multiomics_support_count_abs_smd_ge_0_5"] >= 1
    )
    tier_a = (
        ranking["stable_positive_all_scopes"]
        & ranking["ccrcc_functional_dependency"]
        & ranking["ccrcc_selective_dependency"]
        & ranking["tcga_multiomics_supported"]
    )
    tier_b = (
        (ranking["positive_context_metric_count"] >= len(expected) - 1)
        & ranking["ccrcc_functional_dependency"]
        & ranking["tcga_multiomics_supported"]
        & ~tier_a
    )
    ranking["priority_tier"] = np.select([tier_a, tier_b], ["A", "B"], default="exploratory")
    tier_order = pd.Categorical(ranking["priority_tier"], ["A", "B", "exploratory"], ordered=True)
    ranking = (
        ranking.assign(_tier=tier_order)
        .sort_values(
            ["_tier", "positive_context_metric_count", "minimum_context_delta_r2",
             "TCGA_multiomics_support_count_abs_smd_ge_0_5", "ccrcc_dependency_median"],
            ascending=[True, False, False, False, True],
        )
        .drop(columns="_tier")
        .reset_index()
    )
    return ranking, stable_alphas


def repeated_candidate_stability(
    candidate_genes: list[str],
    genes: list[str],
    background: np.ndarray,
    expression: np.ndarray,
    copy_number: np.ndarray,
    mutation: np.ndarray,
    drivers: np.ndarray,
    kidney: np.ndarray,
    y_raw: np.ndarray,
    lineages: np.ndarray,
    scopes: dict[str, np.ndarray],
    alphas: list[float],
    folds: int,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    rows = []
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    all_indices = np.arange(len(lineages))
    for repeat in range(repeats):
        split_seed = seed + 1000 + repeat
        split = stratified_folds(lineages, folds, split_seed)
        for gene in candidate_genes:
            gene_index = gene_lookup[gene]
            feature_map = targetwise_feature_sets(
                background, expression, copy_number, mutation,
                drivers[:, 0], drivers[:, 1], kidney, gene_index,
            )
            outcome = y_raw[:, gene_index, None]
            standardized = np.full(len(lineages), np.nan)
            predictions = {
                name: {alpha: np.full(len(lineages), np.nan) for alpha in alphas}
                for name in feature_map
            }
            for held in split:
                train = np.setdiff1d(all_indices, held)
                for name, matrix in feature_map.items():
                    fold_predictions, held_y = primal_ridge_predictions(
                        matrix[train], matrix[held], outcome[train], outcome[held], alphas
                    )
                    for alpha, values in fold_predictions.items():
                        predictions[name][alpha][held] = values[:, 0]
                    if name == "targetwise_multiomics":
                        standardized[held] = held_y[:, 0]
            for alpha in alphas:
                reference = predictions["targetwise_multiomics"][alpha]
                current = predictions["targetwise_context"][alpha]
                for scope in ["pan_cancer", "rcc", "ccrcc"]:
                    indices = scopes[scope]
                    reference_r2 = aggregate_r2(
                        standardized[:, None], reference[:, None], indices
                    )
                    context_r2 = aggregate_r2(
                        standardized[:, None], current[:, None], indices
                    )
                    rows.append(
                        {
                            "Gene": gene,
                            "alpha": alpha,
                            "scope": scope,
                            "repeat": repeat,
                            "split_seed": split_seed,
                            "model_n": len(indices),
                            "context_r2": context_r2,
                            "delta_r2": context_r2 - reference_r2,
                        }
                    )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return (
        raw.groupby(["Gene", "alpha", "scope", "model_n"], as_index=False)
        .agg(
            repeats=("repeat", "size"),
            median_context_r2=("context_r2", "median"),
            median_delta_r2=("delta_r2", "median"),
            delta_r2_q05=("delta_r2", lambda values: values.quantile(0.05)),
            delta_r2_q95=("delta_r2", lambda values: values.quantile(0.95)),
            positive_delta_fraction=("delta_r2", lambda values: float((values > 0).mean())),
        )
    )


def kidney_holdout_evaluation(
    features: dict[str, np.ndarray],
    expression: np.ndarray,
    copy_number: np.ndarray,
    mutation: np.ndarray,
    drivers: np.ndarray,
    kidney: np.ndarray,
    y_raw: np.ndarray,
    genes: list[str],
    models: pd.DataFrame,
    alphas: list[float],
    seed: int,
    bootstrap_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = np.flatnonzero(~models["kidney_lineage"].to_numpy())
    held = np.flatnonzero(models["kidney_lineage"].to_numpy())
    if len(held) < 10:
        raise ValueError("Kidney holdout is too small")
    prediction = {
        name: {alpha: np.full_like(y_raw, np.nan) for alpha in alphas}
        for name in ["background", "multiomics", "targetwise_multiomics"]
    }
    standardized = np.full_like(y_raw, np.nan)
    for name in ["background", "multiomics"]:
        values, held_y = kernel_ridge_predictions(
            features[name][train], features[name][held], y_raw[train], y_raw[held], alphas
        )
        for alpha, current in values.items():
            prediction[name][alpha][held] = current
        if name == "background":
            standardized[held] = held_y
    for gene_index in range(len(genes)):
        matrix = targetwise_feature_sets(
            features["background"], expression, copy_number, mutation,
            drivers[:, 0], drivers[:, 1], kidney, gene_index,
        )["targetwise_multiomics"]
        values, _ = primal_ridge_predictions(
            matrix[train], matrix[held], y_raw[train, gene_index, None],
            y_raw[held, gene_index, None], alphas,
        )
        for alpha, current in values.items():
            prediction["targetwise_multiomics"][alpha][held, gene_index] = current[:, 0]
    scopes = {
        "kidney_lineage": held,
        "rcc": np.flatnonzero(models["renal_cell_carcinoma"].to_numpy()),
        "ccrcc": np.flatnonzero(models["clear_cell_renal_cell_carcinoma"].to_numpy()),
    }
    metric_rows, gene_rows = [], []
    labels = models["OncotreeSubtype"].fillna("Unknown").to_numpy(dtype=str)
    for alpha in alphas:
        reference = prediction["background"][alpha]
        for name, values in prediction.items():
            current = values[alpha]
            for scope, indices in scopes.items():
                current_r2 = aggregate_r2(standardized, current, indices)
                reference_r2 = aggregate_r2(standardized, reference, indices)
                lower, upper = bootstrap_delta(
                    standardized, reference, current, labels, indices,
                    seed + int(alpha) + len(name), bootstrap_repeats,
                )
                metric_rows.append(
                    {
                        "alpha": alpha,
                        "model": name,
                        "comparison": "background",
                        "scope": scope,
                        "model_n": len(indices),
                        "r2": current_r2,
                        "delta_r2": current_r2 - reference_r2,
                        "bootstrap_lower": lower,
                        "bootstrap_upper": upper,
                    }
                )
                current_sse = ((standardized[indices] - current[indices]) ** 2).sum(axis=0)
                reference_sse = ((standardized[indices] - reference[indices]) ** 2).sum(axis=0)
                sst = (standardized[indices] ** 2).sum(axis=0)
                gene_rows.extend(
                    {
                        "alpha": alpha,
                        "model": name,
                        "scope": scope,
                        "Gene": gene,
                        "r2": 1.0 - current_sse[index] / max(sst[index], 1e-12),
                        "delta_r2": (reference_sse[index] - current_sse[index])
                        / max(sst[index], 1e-12),
                    }
                    for index, gene in enumerate(genes)
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(gene_rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/depmap_bridge_24q4")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/depmap_shared_model")
    parser.add_argument("--alphas", type=float, nargs="+", default=[10.0, 100.0, 1000.0, 10000.0, 100000.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--stability-repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = pd.read_csv(source / "models.csv", index_col=0)
    matrices = {
        name: pd.read_csv(source / f"{name}.tsv.gz", sep="\t", index_col=0).loc[models.index]
        for name in ["dependency", "mutation", "expression", "copy_number"]
    }
    coverage = pd.read_csv(source / "gene_coverage.csv").set_index("Gene")
    genes = sorted(
        gene for gene in matrices["dependency"].columns
        if not bool(coverage.loc[gene, "DepMap_common_essential"])
        and matrices["dependency"][gene].notna().all()
    )
    if len(genes) < 100:
        raise ValueError(f"Only {len(genes)} complete non-common-essential targets")
    features = feature_sets(models, matrices["mutation"], matrices["expression"], matrices["copy_number"], genes)
    expression_values = matrices["expression"][genes].to_numpy(dtype=float)
    copy_number_values = matrices["copy_number"][genes].to_numpy(dtype=float)
    mutation_values = matrices["mutation"][genes].to_numpy(dtype=float)
    driver_values = matrices["mutation"][[gene for gene in DRIVERS]].to_numpy(dtype=float)
    kidney_values = models["kidney_lineage"].astype(float).to_numpy()
    y_raw = matrices["dependency"][genes].to_numpy(dtype=float)
    lineages = models["OncotreeLineage"].fillna("Unknown").to_numpy(dtype=str)
    folds = stratified_folds(lineages, args.folds, args.seed)
    predictions = {
        name: {alpha: np.full_like(y_raw, np.nan) for alpha in args.alphas}
        for name in [*features, "targetwise_multiomics", "targetwise_context"]
    }
    standardized_y = np.full_like(y_raw, np.nan)
    all_indices = np.arange(len(models))
    for fold_index, held in enumerate(folds):
        train = np.setdiff1d(all_indices, held)
        for name, matrix in features.items():
            fold_predictions, held_y = kernel_ridge_predictions(
                matrix[train], matrix[held], y_raw[train], y_raw[held], args.alphas
            )
            for alpha, values in fold_predictions.items():
                predictions[name][alpha][held] = values
            if name == "background":
                standardized_y[held] = held_y
        for gene_index in range(len(genes)):
            target_features = targetwise_feature_sets(
                features["background"], expression_values, copy_number_values, mutation_values,
                driver_values[:, 0], driver_values[:, 1], kidney_values, gene_index,
            )
            for name, matrix in target_features.items():
                fold_predictions, _ = primal_ridge_predictions(
                    matrix[train], matrix[held], y_raw[train, gene_index, None],
                    y_raw[held, gene_index, None], args.alphas,
                )
                for alpha, values in fold_predictions.items():
                    predictions[name][alpha][held, gene_index] = values[:, 0]
        print(f"completed fold {fold_index + 1}/{args.folds}", flush=True)
    if not np.isfinite(standardized_y).all() or any(
        not np.isfinite(values).all() for models_by_alpha in predictions.values() for values in models_by_alpha.values()
    ):
        raise ValueError("Non-finite out-of-fold predictions")

    scopes = {
        "pan_cancer": np.arange(len(models)),
        "non_kidney": np.flatnonzero(~models["kidney_lineage"].to_numpy()),
        "kidney_lineage": np.flatnonzero(models["kidney_lineage"].to_numpy()),
        "rcc": np.flatnonzero(models["renal_cell_carcinoma"].to_numpy()),
        "ccrcc": np.flatnonzero(models["clear_cell_renal_cell_carcinoma"].to_numpy()),
    }
    metric_rows = []
    cell_rows = []
    gene_rows = []
    for alpha in args.alphas:
        reference = predictions["background"][alpha]
        for name in predictions:
            current = predictions[name][alpha]
            if name == "context_interactions":
                comparison, comparison_name = predictions["multiomics"][alpha], "multiomics"
            elif name == "targetwise_context":
                comparison, comparison_name = predictions["targetwise_multiomics"][alpha], "targetwise_multiomics"
            else:
                comparison, comparison_name = reference, "background"
            for scope, indices in scopes.items():
                current_r2 = aggregate_r2(standardized_y, current, indices)
                comparison_r2 = aggregate_r2(standardized_y, comparison, indices)
                lower, upper = bootstrap_delta(
                    standardized_y, comparison, current, lineages, indices,
                    args.seed + int(alpha) + len(name), args.bootstrap,
                )
                metric_rows.append(
                    {
                        "alpha": alpha,
                        "model": name,
                        "comparison": comparison_name,
                        "scope": scope,
                        "model_n": len(indices),
                        "r2": current_r2,
                        "delta_r2": current_r2 - comparison_r2,
                        "bootstrap_lower": lower,
                        "bootstrap_upper": upper,
                    }
                )
            per_cell_sse = ((standardized_y - current) ** 2).sum(axis=1)
            per_cell_sst = (standardized_y ** 2).sum(axis=1)
            cell_rows.extend(
                {
                    "alpha": alpha,
                    "model": name,
                    "ModelID": model_id,
                    "lineage": lineages[index],
                    "sse": per_cell_sse[index],
                    "sst": per_cell_sst[index],
                }
                for index, model_id in enumerate(models.index)
            )
            for scope, indices in scopes.items():
                per_gene_sse = ((standardized_y[indices] - current[indices]) ** 2).sum(axis=0)
                per_gene_sst = (standardized_y[indices] ** 2).sum(axis=0)
                comparison_sse = ((standardized_y[indices] - comparison[indices]) ** 2).sum(axis=0)
                gene_rows.extend(
                    {
                        "alpha": alpha,
                        "model": name,
                        "comparison": comparison_name,
                        "scope": scope,
                        "model_n": len(indices),
                        "Gene": gene,
                        "r2": 1.0 - per_gene_sse[index] / max(per_gene_sst[index], 1e-12),
                        "delta_r2": (comparison_sse[index] - per_gene_sse[index])
                        / max(per_gene_sst[index], 1e-12),
                    }
                    for index, gene in enumerate(genes)
                )
    metrics = pd.DataFrame(metric_rows)
    cells = pd.DataFrame(cell_rows)
    gene_metrics = pd.DataFrame(gene_rows).merge(
        coverage.reset_index(), on="Gene", how="left", validate="many_to_one"
    )
    ranking, ranking_alphas = context_candidate_ranking(
        pd.DataFrame(gene_rows), coverage, args.alphas
    )
    candidate_genes = ranking.loc[
        ranking["priority_tier"].isin(["A", "B"]), "Gene"
    ].tolist()
    stability = repeated_candidate_stability(
        candidate_genes, genes, features["background"], expression_values, copy_number_values,
        mutation_values, driver_values, kidney_values, y_raw, lineages, scopes,
        ranking_alphas, args.folds, args.seed, args.stability_repeats,
    )
    kidney_holdout, kidney_gene_holdout = kidney_holdout_evaluation(
        features, expression_values, copy_number_values, mutation_values, driver_values,
        kidney_values, y_raw, genes, models, ranking_alphas, args.seed, args.bootstrap,
    )
    metrics.to_csv(output / "model_results.csv", index=False)
    cells.to_csv(output / "cell_line_losses.csv.gz", index=False, compression="gzip")
    gene_metrics.to_csv(output / "gene_results.csv.gz", index=False, compression="gzip")
    ranking.to_csv(output / "context_candidate_ranking.csv", index=False)
    stability.to_csv(output / "context_candidate_stability.csv", index=False)
    kidney_holdout.to_csv(output / "kidney_holdout_results.csv", index=False)
    kidney_gene_holdout.to_csv(output / "kidney_holdout_gene_results.csv.gz", index=False, compression="gzip")
    for obsolete in [output / "cell_line_losses.csv", output / "gene_results.csv"]:
        obsolete.unlink(missing_ok=True)
    input_hashes = {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
    manifest = {
        "status": "cross_validated_exploratory_model",
        "release": "DepMap 24Q4 Public CRISPRGeneEffect (Chronos)",
        "model_n": len(models),
        "target_gene_n": len(genes),
        "fold_unit": "DepMap ModelID",
        "fold_stratification": "OncotreeLineage",
        "folds": args.folds,
        "seed": args.seed,
        "alphas": args.alphas,
        "candidate_ranking_alphas": ranking_alphas,
        "candidate_stability_repeats": args.stability_repeats,
        "candidate_stability_genes": candidate_genes,
        "kidney_holdout": "all kidney-lineage models excluded from training; evaluated without kidney-lineage adaptation",
        "candidate_ranking_rule": {
            "A": "positive targetwise-context delta R2 in pan-cancer, RCC, and ccRCC at both strongest shrinkage values; ccRCC median Gene Effect <= -0.5; ccRCC selectivity delta <= -0.1; at least one TCGA context-adjusted omics |SMD| >= 0.5",
            "B": "at least five of six context delta R2 values positive; ccRCC median Gene Effect <= -0.5; at least one TCGA context-adjusted omics |SMD| >= 0.5",
        },
        "common_essential_targets_excluded": True,
        "models": {
            "background": "lineage + five driver mutations + global mutation/CN/expression summaries",
            "multiomics": "background + candidate expression, absolute CN, damaging mutation",
            "context_interactions": "multiomics + candidate omics interactions with VHL, PBRM1, and kidney lineage",
            "targetwise_multiomics": "background + each target's own expression, absolute CN, and damaging mutation",
            "targetwise_context": "targetwise multiomics + its interactions with VHL, PBRM1, and kidney lineage",
        },
        "interpretation": "More negative CRISPR Gene Effect means stronger dependency. Prediction R2 measures dependency-profile prediction, not causal patient benefit.",
        "limitations": [
            "All results are internal cell-line cross-validation; TCGA and DepMap are different biological domains.",
            "RCC and ccRCC estimates use only 20 and 12 models, respectively.",
            "Bootstrap resamples ModelIDs within lineage and does not correct gene-level selection.",
            "Alpha sensitivity is reported; this run is not a nested-CV hyperparameter estimate.",
            "Cross-gene coefficients can capture correlation and do not establish a direct regulatory edge.",
        ],
        "input_hashes": input_hashes,
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - start,
    }
    (output / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.loc[metrics["scope"].isin(["pan_cancer", "rcc", "ccrcc"])].to_string(index=False))
    print(f"completed in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
