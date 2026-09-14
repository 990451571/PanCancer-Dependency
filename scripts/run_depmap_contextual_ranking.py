"""Test whether multi-omics supports sample-specific dependency ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from run_depmap_shared_model import DRIVERS, feature_sets, targetwise_feature_sets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ridge_raw_prediction(train_x: np.ndarray, held_x: np.ndarray, train_y: np.ndarray,
                         alpha: float) -> np.ndarray:
    mean, sd = train_x.mean(axis=0), train_x.std(axis=0)
    usable = sd > 1e-8
    x = (train_x[:, usable] - mean[usable]) / sd[usable]
    hx = (held_x[:, usable] - mean[usable]) / sd[usable]
    y_mean, y_sd = train_y.mean(axis=0), train_y.std(axis=0)
    if (y_sd <= 1e-8).any():
        raise ValueError("Zero-variance dependency target")
    y = (train_y - y_mean) / y_sd
    prediction = hx @ x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
    return prediction * y_sd + y_mean


def ranking_rows(y: np.ndarray, prediction: np.ndarray, ids: list[str], genes: list[str],
                 model: str, k: int) -> tuple[list[dict], list[dict]]:
    metric_rows, selection_rows = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for index, model_id in enumerate(ids):
        truth, predicted = y[index], prediction[index]
        oracle = np.argsort(truth)[:k]
        selected = np.argsort(predicted)[:k]
        relevance = np.maximum(0.0, -truth)
        ideal_dcg = float((relevance[oracle] * discounts).sum())
        dcg = float((relevance[selected] * discounts).sum())
        selected_mean, oracle_mean = float(truth[selected].mean()), float(truth[oracle].mean())
        metric_rows.append(
            {
                "ModelID": model_id,
                "model": model,
                "topk_overlap": len(set(oracle) & set(selected)) / k,
                "ndcg": dcg / ideal_dcg if ideal_dcg > 0 else np.nan,
                "dependent_precision": float((truth[selected] <= -0.5).mean()),
                "mean_selected_gene_effect": selected_mean,
                "oracle_mean_gene_effect": oracle_mean,
                "regret": selected_mean - oracle_mean,
            }
        )
        selection_rows.append(
            {
                "ModelID": model_id,
                "model": model,
                "selected_genes": "|".join(genes[position] for position in selected),
                "oracle_genes": "|".join(genes[position] for position in oracle),
            }
        )
    return metric_rows, selection_rows


def paired_bootstrap(values: pd.DataFrame, metric: str, scope_ids: set[str], labels: pd.Series,
                     seed: int, repeats: int) -> tuple[float, float, float]:
    table = values.loc[values["ModelID"].isin(scope_ids)].pivot(
        index="ModelID", columns="model", values=metric
    )
    if "global_mean" not in table:
        raise ValueError("Missing global ranking baseline")
    model = next(column for column in table if column != "global_mean")
    difference = table[model] - table["global_mean"]
    if metric in {"regret", "mean_selected_gene_effect"}:
        difference = -difference
    scoped_labels = labels.reindex(table.index).fillna("Unknown")
    strata = [difference.loc[scoped_labels.eq(label)].to_numpy()
              for label in sorted(scoped_labels.unique())]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(stratum, len(stratum), replace=True) for stratum in strata])
        draws.append(float(sample.mean()))
    return float(difference.mean()), *[float(value) for value in np.quantile(draws, [0.025, 0.975])]


def mean_pairwise_jaccard(selections: list[set[str]]) -> float:
    pairs = list(combinations(selections, 2))
    return float(np.mean([len(left & right) / len(left | right) for left, right in pairs])) if pairs else 1.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap", type=Path, default=root / "data/processed/depmap_bridge_24q4")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/depmap_contextual_ranking")
    parser.add_argument("--shared-alpha", type=float, default=10000.0)
    parser.add_argument("--targetwise-alpha", type=float, default=1000.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    models = pd.read_csv(args.depmap / "models.csv", index_col=0)
    matrices = {
        name: pd.read_csv(args.depmap / f"{name}.tsv.gz", sep="\t", index_col=0).loc[models.index]
        for name in ["dependency", "mutation", "expression", "copy_number"]
    }
    coverage = pd.read_csv(args.depmap / "gene_coverage.csv").set_index("Gene")
    genes = sorted(
        gene for gene in matrices["dependency"].columns
        if not bool(coverage.loc[gene, "DepMap_common_essential"])
        and matrices["dependency"][gene].notna().all()
    )
    features = feature_sets(
        models, matrices["mutation"], matrices["expression"], matrices["copy_number"], genes
    )
    expression = matrices["expression"][genes].to_numpy(dtype=float)
    copy_number = matrices["copy_number"][genes].to_numpy(dtype=float)
    mutation = matrices["mutation"][genes].fillna(0).to_numpy(dtype=float)
    drivers = matrices["mutation"][[gene for gene in DRIVERS]].fillna(0).to_numpy(dtype=float)
    kidney_flag = models["kidney_lineage"].astype(float).to_numpy()
    y = matrices["dependency"][genes].to_numpy(dtype=float)
    train = np.flatnonzero(~models["kidney_lineage"].to_numpy())
    held = np.flatnonzero(models["kidney_lineage"].to_numpy())
    held_ids = models.index[held].tolist()

    predictions = {
        "global_mean": np.repeat(y[train].mean(axis=0, keepdims=True), len(held), axis=0),
        "background": ridge_raw_prediction(
            features["background"][train], features["background"][held], y[train], args.shared_alpha
        ),
        "shared_multiomics": ridge_raw_prediction(
            features["multiomics"][train], features["multiomics"][held], y[train], args.shared_alpha
        ),
    }
    targetwise = np.full((len(held), len(genes)), np.nan)
    for gene_index in range(len(genes)):
        matrix = targetwise_feature_sets(
            features["background"], expression, copy_number, mutation,
            drivers[:, 0], drivers[:, 1], kidney_flag, gene_index,
        )["targetwise_multiomics"]
        targetwise[:, gene_index] = ridge_raw_prediction(
            matrix[train], matrix[held], y[train, gene_index, None], args.targetwise_alpha
        )[:, 0]
    predictions["targetwise_multiomics"] = targetwise

    metric_rows, selection_rows = [], []
    for name, prediction in predictions.items():
        metrics, selections = ranking_rows(y[held], prediction, held_ids, genes, name, args.top_k)
        metric_rows.extend(metrics)
        selection_rows.extend(selections)
    metrics = pd.DataFrame(metric_rows).merge(
        models[["renal_cell_carcinoma", "clear_cell_renal_cell_carcinoma", "OncotreeSubtype"]],
        left_on="ModelID", right_index=True, validate="many_to_one",
    )
    selections = pd.DataFrame(selection_rows)
    scopes = {
        "kidney_lineage": set(held_ids),
        "rcc": set(models.index[models["renal_cell_carcinoma"]]),
        "ccrcc": set(models.index[models["clear_cell_renal_cell_carcinoma"]]),
    }
    summary_rows = []
    metric_names = ["topk_overlap", "ndcg", "dependent_precision",
                    "mean_selected_gene_effect", "regret"]
    for scope, ids in scopes.items():
        scoped = metrics.loc[metrics["ModelID"].isin(ids)]
        means = scoped.groupby("model")[metric_names].mean()
        for model_name in predictions:
            for metric in metric_names:
                row = {
                    "scope": scope, "model_n": len(ids), "model": model_name, "metric": metric,
                    "mean": float(means.loc[model_name, metric]),
                    "improvement_vs_global": 0.0,
                    "bootstrap_lower": 0.0,
                    "bootstrap_upper": 0.0,
                }
                if model_name != "global_mean":
                    pair = scoped.loc[scoped["model"].isin(["global_mean", model_name])]
                    improvement, lower, upper = paired_bootstrap(
                        pair, metric, ids, models["OncotreeSubtype"],
                        args.seed + len(scope) + len(model_name) + len(metric), args.bootstrap,
                    )
                    row.update(improvement_vs_global=improvement,
                               bootstrap_lower=lower, bootstrap_upper=upper)
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    heterogeneity_rows = []
    for scope, ids in scopes.items():
        for model_name in predictions:
            selected = [set(value.split("|")) for value in selections.loc[
                selections["ModelID"].isin(ids) & selections["model"].eq(model_name), "selected_genes"
            ]]
            heterogeneity_rows.append(
                {
                    "scope": scope,
                    "model": model_name,
                    "model_n": len(selected),
                    "unique_selected_genes": len(set().union(*selected)),
                    "mean_pairwise_topk_jaccard": mean_pairwise_jaccard(selected),
                }
            )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metrics.to_csv(args.output_dir / "per_model_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "selected_targets.csv", index=False)
    heterogeneity.to_csv(args.output_dir / "selection_heterogeneity.csv", index=False)
    audit = {
        "status": "exploratory_contextual_ranking_gate",
        "training_scope": "all non-kidney DepMap models",
        "holdout_scope": "all kidney-lineage DepMap models",
        "candidate_count": len(genes),
        "common_essentials_excluded": True,
        "top_k": args.top_k,
        "models": list(predictions),
        "interpretation": "More negative GeneEffect is stronger dependency; positive improvement means better than a fixed global ranking.",
        "rl_caveat": "DepMap observes all single-gene actions, so supervised ranking is statistically more direct than offline bandit learning unless sequential non-additive utility is introduced.",
        "input_sha256": {name: sha256(args.depmap / f"{name}.tsv.gz")
                         for name in ["dependency", "mutation", "expression", "copy_number"]},
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "run.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.loc[(summary["scope"] == "ccrcc") & summary["metric"].isin(
        ["topk_overlap", "ndcg", "regret"]
    )].to_string(index=False))
    print(heterogeneity.loc[heterogeneity["scope"].eq("ccrcc")].to_string(index=False))


if __name__ == "__main__":
    main()
