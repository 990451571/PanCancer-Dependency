"""Nested leave-one-lineage-out validation for DepMap dependency ranking.

The outer split holds out one complete OncoTree lineage.  Ridge penalties are
chosen only with whole-lineage inner folds made from the remaining lineages.
No TCGA test labels are read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_shared_model import feature_sets


METHODS = ("global_mean", "background", "targetwise_multiomics", "shared_multiomics")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=root / "data/processed/depmap_bridge_24q4",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/depmap_lolo_validation_20260912",
    )
    parser.add_argument("--minimum-lineage-size", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[10.0, 100.0, 1000.0, 10000.0, 100000.0],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dependency-threshold", type=float, default=-0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--lineages", nargs="+", default=None,
        help="Optional exact lineage names for a smoke or targeted run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print eligible lineages without fitting models.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow replacing files in an existing output directory.",
    )
    return parser.parse_args()


def grouped_folds(
    indices: np.ndarray,
    labels: np.ndarray,
    requested_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Assign complete lineages to balanced inner folds."""
    groups = np.unique(labels[indices])
    if len(groups) < 2:
        raise ValueError("At least two training lineages are required for inner validation")
    fold_n = min(requested_folds, len(groups))
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    counts = {group: int(np.sum(labels[indices] == group)) for group in groups}
    ordered = sorted(shuffled, key=lambda group: -counts[group])
    assignments: list[list[int]] = [[] for _ in range(fold_n)]
    loads = np.zeros(fold_n, dtype=int)
    for group in ordered:
        destination = int(np.argmin(loads))
        members = indices[labels[indices] == group]
        assignments[destination].extend(members.tolist())
        loads[destination] += len(members)
    folds = [np.asarray(sorted(values), dtype=int) for values in assignments]
    if any(len(values) == 0 for values in folds):
        raise ValueError("An inner lineage fold is empty")
    if sorted(np.concatenate(folds).tolist()) != sorted(indices.tolist()):
        raise ValueError("Inner lineage folds do not partition the outer training set")
    return folds


def ridge_predictions(
    train_x: np.ndarray,
    held_x: np.ndarray,
    train_y: np.ndarray,
    alphas: list[float],
) -> dict[float, np.ndarray]:
    """Training-only standardization followed by primal or dual ridge."""
    x_mean = train_x.mean(axis=0)
    x_sd = train_x.std(axis=0)
    usable = x_sd > 1e-8
    x = (train_x[:, usable] - x_mean[usable]) / x_sd[usable]
    hx = (held_x[:, usable] - x_mean[usable]) / x_sd[usable]
    y_mean = train_y.mean(axis=0)
    y_sd = train_y.std(axis=0)
    y_scale = np.where(y_sd > 1e-8, y_sd, 1.0)
    y = (train_y - y_mean) / y_scale
    if x.shape[1] <= x.shape[0]:
        gram = x.T @ x
        cross = x.T @ y
        identity = np.eye(x.shape[1])
        standardized = {
            alpha: hx @ np.linalg.solve(gram + alpha * identity, cross)
            for alpha in alphas
        }
    else:
        kernel = x @ x.T
        held_kernel = hx @ x.T
        identity = np.eye(x.shape[0])
        standardized = {
            alpha: held_kernel @ np.linalg.solve(kernel + alpha * identity, y)
            for alpha in alphas
        }
    return {alpha: value * y_scale + y_mean for alpha, value in standardized.items()}


def targetwise_predictions(
    background: np.ndarray,
    expression: np.ndarray,
    copy_number: np.ndarray,
    mutation: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    held: np.ndarray,
    alphas: list[float],
) -> dict[float, np.ndarray]:
    predictions = {alpha: np.empty((len(held), y.shape[1])) for alpha in alphas}
    for gene_index in range(y.shape[1]):
        matrix = np.c_[
            background,
            expression[:, gene_index],
            copy_number[:, gene_index],
            mutation[:, gene_index],
        ]
        current = ridge_predictions(
            matrix[train], matrix[held], y[train, gene_index, None], alphas
        )
        for alpha in alphas:
            predictions[alpha][:, gene_index] = current[alpha][:, 0]
    return predictions


def choose_alphas(
    outer_train: np.ndarray,
    labels: np.ndarray,
    background: np.ndarray,
    shared: np.ndarray,
    expression: np.ndarray,
    copy_number: np.ndarray,
    mutation: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
    inner_folds: int,
    seed: int,
    lineage_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    squared_error = {
        method: {alpha: 0.0 for alpha in alphas}
        for method in ("background", "targetwise_multiomics", "shared_multiomics")
    }
    folds = grouped_folds(outer_train, labels, inner_folds, seed)
    for fold_index, held in enumerate(folds, start=1):
        train = np.setdiff1d(outer_train, held, assume_unique=True)
        predictions = {
            "background": ridge_predictions(background[train], background[held], y[train], alphas),
            "shared_multiomics": ridge_predictions(shared[train], shared[held], y[train], alphas),
            "targetwise_multiomics": targetwise_predictions(
                background, expression, copy_number, mutation, y, train, held, alphas
            ),
        }
        for method, by_alpha in predictions.items():
            for alpha, prediction in by_alpha.items():
                squared_error[method][alpha] += float(((y[held] - prediction) ** 2).sum())
        print(
            f"  inner {fold_index}/{len(folds)} held_models={len(held)} "
            f"held_lineages={len(np.unique(labels[held]))}",
            flush=True,
        )
    rows = []
    selected = {}
    for method in squared_error:
        selected[method] = min(alphas, key=lambda alpha: (squared_error[method][alpha], alpha))
        for alpha in alphas:
            rows.append(
                {
                    "heldout_lineage": lineage_name,
                    "method": method,
                    "alpha": alpha,
                    "inner_sse": squared_error[method][alpha],
                    "selected": alpha == selected[method],
                }
            )
    return selected, pd.DataFrame(rows)


def ranking_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    model_ids: list[str],
    genes: list[str],
    method: str,
    k: int,
    dependency_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, selections = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row_index, model_id in enumerate(model_ids):
        observed = truth[row_index]
        predicted = prediction[row_index]
        oracle = np.argsort(observed)[:k]
        selected = np.argsort(predicted)[:k]
        relevance = np.maximum(0.0, -observed)
        ideal = float((relevance[oracle] * discounts).sum())
        dcg = float((relevance[selected] * discounts).sum())
        selected_mean = float(observed[selected].mean())
        oracle_mean = float(observed[oracle].mean())
        rows.append(
            {
                "ModelID": model_id,
                "method": method,
                "ndcg_at_10": dcg / ideal if ideal > 0 else np.nan,
                "top10_overlap": len(set(oracle) & set(selected)) / k,
                "dependency_precision": float((observed[selected] <= dependency_threshold).mean()),
                "regret": selected_mean - oracle_mean,
                "selected_mean_gene_effect": selected_mean,
                "oracle_mean_gene_effect": oracle_mean,
            }
        )
        selections.append(
            {
                "ModelID": model_id,
                "method": method,
                "selected_genes": "|".join(genes[index] for index in selected),
                "oracle_genes": "|".join(genes[index] for index in oracle),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(selections)


def bootstrap_mean(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repeats)]
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def bootstrap_delta_r2(
    truth: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    center = truth.mean(axis=0, keepdims=True)
    per_model_gain = ((truth - reference) ** 2 - (truth - prediction) ** 2).sum(axis=1)
    per_model_scale = ((truth - center) ** 2).sum(axis=1)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sample = rng.integers(0, len(truth), size=len(truth))
        draws.append(
            float(per_model_gain[sample].sum() / max(per_model_scale[sample].sum(), 1e-12))
        )
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new path or pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.top_k != 10:
        raise ValueError("This frozen protocol requires --top-k 10")
    started = time.monotonic()
    source = args.input_dir.resolve()
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
    if len(genes) < args.top_k:
        raise ValueError("Fewer complete non-common-essential targets than top-k")
    labels = models["OncotreeLineage"].fillna("Unknown").to_numpy(dtype=str)
    counts = pd.Series(labels).value_counts().sort_index()
    eligible = counts[counts >= args.minimum_lineage_size].index.tolist()
    if args.lineages:
        missing = sorted(set(args.lineages) - set(counts.index))
        if missing:
            raise ValueError(f"Unknown requested lineages: {missing}")
        below = sorted(set(args.lineages) - set(eligible))
        if below:
            raise ValueError(
                f"Requested lineages below minimum size {args.minimum_lineage_size}: {below}"
            )
        eligible = [lineage for lineage in eligible if lineage in set(args.lineages)]
    print(
        f"models={len(models)} targets={len(genes)} eligible_lineages={len(eligible)} "
        f"minimum_n={args.minimum_lineage_size}",
        flush=True,
    )
    for lineage in eligible:
        print(f"  {lineage}: n={int(counts[lineage])}", flush=True)
    if args.dry_run:
        print("DRY_RUN_OK: inputs validated; no model was fitted and no output was written.", flush=True)
        return

    output = args.output_dir.resolve()
    prepare_output(output, args.overwrite)
    feature_map = feature_sets(
        models, matrices["mutation"], matrices["expression"], matrices["copy_number"], genes
    )
    background = feature_map["background"]
    shared = feature_map["multiomics"]
    expression = matrices["expression"][genes].to_numpy(dtype=float)
    copy_number = matrices["copy_number"][genes].to_numpy(dtype=float)
    mutation = matrices["mutation"][genes].fillna(0).to_numpy(dtype=float)
    y = matrices["dependency"][genes].to_numpy(dtype=float)
    all_indices = np.arange(len(models))
    lineage_rows = []
    cell_frames = []
    selection_frames = []
    tuning_frames = []

    for outer_index, lineage in enumerate(eligible, start=1):
        held = np.flatnonzero(labels == lineage)
        train = np.setdiff1d(all_indices, held, assume_unique=True)
        print(
            f"[{outer_index}/{len(eligible)}] heldout={lineage} "
            f"train_n={len(train)} held_n={len(held)}",
            flush=True,
        )
        selected_alpha, tuning = choose_alphas(
            train, labels, background, shared, expression, copy_number, mutation, y,
            args.alphas, args.inner_folds, args.seed + outer_index * 100, lineage,
        )
        tuning_frames.append(tuning)
        predictions = {
            "global_mean": np.repeat(y[train].mean(axis=0, keepdims=True), len(held), axis=0),
            "background": ridge_predictions(
                background[train], background[held], y[train], [selected_alpha["background"]]
            )[selected_alpha["background"]],
            "shared_multiomics": ridge_predictions(
                shared[train], shared[held], y[train], [selected_alpha["shared_multiomics"]]
            )[selected_alpha["shared_multiomics"]],
        }
        predictions["targetwise_multiomics"] = targetwise_predictions(
            background, expression, copy_number, mutation, y, train, held,
            [selected_alpha["targetwise_multiomics"]],
        )[selected_alpha["targetwise_multiomics"]]
        truth = y[held]
        held_ids = models.index[held].astype(str).tolist()
        global_prediction = predictions["global_mean"]
        background_prediction = predictions["background"]
        held_center = truth.mean(axis=0, keepdims=True)
        sst = max(float(((truth - held_center) ** 2).sum()), 1e-12)
        global_sse = float(((truth - global_prediction) ** 2).sum())
        background_sse = float(((truth - background_prediction) ** 2).sum())
        for method in METHODS:
            prediction = predictions[method]
            metrics, selections = ranking_metrics(
                truth, prediction, held_ids, genes, method, args.top_k,
                args.dependency_threshold,
            )
            metrics.insert(1, "heldout_lineage", lineage)
            selections.insert(1, "heldout_lineage", lineage)
            cell_frames.append(metrics)
            selection_frames.append(selections)
            sse = float(((truth - prediction) ** 2).sum())
            lower, upper = bootstrap_delta_r2(
                truth, global_prediction, prediction,
                args.seed + outer_index * 1000 + METHODS.index(method), args.bootstrap,
            )
            row = {
                "heldout_lineage": lineage,
                "model_n": len(held),
                "target_n": len(genes),
                "method": method,
                "selected_alpha": selected_alpha.get(method, np.nan),
                "r2": 1.0 - sse / sst,
                "delta_r2_vs_global_mean": (global_sse - sse) / sst,
                "delta_r2_vs_global_lower": lower,
                "delta_r2_vs_global_upper": upper,
                "delta_r2_vs_background": (background_sse - sse) / sst,
            }
            for metric in ["ndcg_at_10", "top10_overlap", "dependency_precision", "regret"]:
                values = metrics[metric].to_numpy(dtype=float)
                row[metric] = float(np.nanmean(values))
                ci = bootstrap_mean(
                    values,
                    args.seed + outer_index * 10000 + METHODS.index(method) * 100 + len(metric),
                    args.bootstrap,
                )
                row[f"{metric}_lower"] = ci[0]
                row[f"{metric}_upper"] = ci[1]
            lineage_rows.append(row)
        print(
            "  selected alpha: "
            + ", ".join(f"{method}={alpha:g}" for method, alpha in selected_alpha.items()),
            flush=True,
        )

    lineage_metrics = pd.DataFrame(lineage_rows)
    # Fill the global rows and robustly calculate all per-lineage ranking improvements.
    ranking_names = ["ndcg_at_10", "top10_overlap", "dependency_precision", "regret"]
    for lineage in eligible:
        base = lineage_metrics.loc[
            lineage_metrics.heldout_lineage.eq(lineage)
            & lineage_metrics.method.eq("global_mean")
        ].iloc[0]
        for method in METHODS:
            mask = lineage_metrics.heldout_lineage.eq(lineage) & lineage_metrics.method.eq(method)
            for metric in ranking_names:
                difference = float(lineage_metrics.loc[mask, metric].iloc[0] - base[metric])
                lineage_metrics.loc[mask, f"{metric}_improvement_vs_global"] = (
                    -difference if metric == "regret" else difference
                )

    cross_rows = []
    improvement_columns = [
        "delta_r2_vs_global_mean",
        "ndcg_at_10_improvement_vs_global",
        "top10_overlap_improvement_vs_global",
        "dependency_precision_improvement_vs_global",
        "regret_improvement_vs_global",
    ]
    for method in METHODS:
        current = lineage_metrics.loc[lineage_metrics.method.eq(method)]
        for metric in improvement_columns:
            values = current[metric].to_numpy(dtype=float)
            lower, upper = bootstrap_mean(
                values, args.seed + METHODS.index(method) * 1000 + len(metric), args.bootstrap
            )
            cross_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "lineage_n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "positive_lineage_fraction": float(np.mean(values > 0)),
                    "lineage_bootstrap_lower": lower,
                    "lineage_bootstrap_upper": upper,
                }
            )
    cross_summary = pd.DataFrame(cross_rows)

    kidney_rows = []
    if "Kidney" in eligible:
        for method in METHODS:
            current = lineage_metrics.loc[lineage_metrics.method.eq(method)]
            kidney = current.loc[current.heldout_lineage.eq("Kidney")].iloc[0]
            for metric in improvement_columns:
                values = current[metric].to_numpy(dtype=float)
                value = float(kidney[metric])
                kidney_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "kidney_value": value,
                        "eligible_lineage_n": len(values),
                        "kidney_percentile": float(np.mean(values <= value)),
                        "other_lineage_median": float(
                            current.loc[~current.heldout_lineage.eq("Kidney"), metric].median()
                        ),
                    }
                )
    kidney_context = pd.DataFrame(kidney_rows)

    per_model = pd.concat(cell_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    lineage_metrics.to_csv(output / "lineage_metrics.csv", index=False)
    cross_summary.to_csv(output / "cross_lineage_summary.csv", index=False)
    kidney_context.to_csv(output / "kidney_transfer_context.csv", index=False)
    tuning.to_csv(output / "nested_alpha_selection.csv", index=False)
    per_model.to_csv(output / "per_model_ranking_metrics.csv.gz", index=False, compression="gzip")
    selections.to_csv(output / "top10_selections.csv.gz", index=False, compression="gzip")
    manifest = {
        "status": "nested_leave_one_lineage_out_validation",
        "release": "DepMap 24Q4 Public CRISPRGeneEffect (Chronos)",
        "outer_split": "one complete OncotreeLineage held out",
        "eligible_rule": f"OncotreeLineage sample count >= {args.minimum_lineage_size}",
        "eligible_lineages": {lineage: int(counts[lineage]) for lineage in eligible},
        "model_n": len(models),
        "target_gene_n": len(genes),
        "common_essentials_excluded": True,
        "methods": {
            "global_mean": "per-target mean learned from outer-training lineages",
            "background": "lineage + five driver mutations + global omics summaries",
            "targetwise_multiomics": "background + the target's own expression/CN/mutation",
            "shared_multiomics": "background + all candidate expression/CN/mutation",
        },
        "hyperparameter_selection": {
            "rule": "minimum aggregate inner-validation SSE",
            "inner_split_unit": "complete OncotreeLineage",
            "inner_folds": args.inner_folds,
            "alphas": args.alphas,
            "heldout_lineage_used": False,
        },
        "metric_definitions": {
            "delta_r2": "heldout R2 improvement relative to outer-training per-target global mean",
            "ndcg_at_10": "relevance=max(0,-Chronos GeneEffect); higher is better",
            "top10_overlap": "fraction shared with the heldout model's true 10 strongest dependencies",
            "dependency_precision": f"fraction selected with GeneEffect <= {args.dependency_threshold:g}",
            "regret": "selected mean GeneEffect minus oracle mean; lower is better",
        },
        "kidney_interpretation": (
            "Kidney is contextualized against all eligible heldout lineages; no universal-transfer "
            "claim is emitted automatically."
        ),
        "tcga_test_read": False,
        "limitations": [
            "Lineage LOLO estimates transfer to unseen cell-line lineages, not to patients.",
            "The same DepMap release supplies training and heldout lineages; this is not an external laboratory replication.",
            "The lineage bootstrap describes across-lineage variation; cell-line bootstrap describes within-lineage sampling variation.",
            "Hyperparameters are selected for squared-error performance; ranking metrics are secondary endpoints.",
        ],
        "input_sha256": {
            name: sha256(source / name)
            for name in [
                "models.csv", "dependency.tsv.gz", "mutation.tsv.gz",
                "expression.tsv.gz", "copy_number.tsv.gz", "gene_coverage.csv",
            ]
        },
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nCross-lineage summary", flush=True)
    print(
        cross_summary.loc[cross_summary.metric.eq("delta_r2_vs_global_mean")].to_string(index=False),
        flush=True,
    )
    print(f"completed in {time.monotonic() - started:.1f}s -> {output}", flush=True)


if __name__ == "__main__":
    main()
