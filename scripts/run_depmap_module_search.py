"""Search PPI-constrained multi-omics modules using CRISPR dependency reward."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_context_module_cv import CVReward, Search, State, ridge_fold, standardize
from run_depmap_shared_model import DRIVERS, bootstrap_delta, feature_sets, stratified_folds


DEFAULT_20_SEEDS = tuple(range(20260911, 20260931))
DEFAULT_TRACKED_GENES = ("TLN1", "AJUBA", "CTNNA1", "DSP", "DSC3", "TP53")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fold(
    raw: np.ndarray,
    background: np.ndarray,
    dependency: np.ndarray,
    fit: np.ndarray,
    held: np.ndarray,
    alpha: float,
):
    x, hx = standardize(raw[fit].reshape(len(fit), -1), raw[held].reshape(len(held), -1))
    c, hc = standardize(background[fit], background[held])
    c, hc = np.c_[np.ones(len(fit)), c], np.c_[np.ones(len(held)), hc]
    y, hy = standardize(dependency[fit], dependency[held])
    return ridge_fold(x, hx, c, hc, y, hy, alpha)


def scoped_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray | None = None,
) -> tuple[float, float]:
    if targets is None:
        targets = np.arange(y.shape[1])
    observed = y[np.ix_(rows, targets)]
    base = baseline[np.ix_(rows, targets)]
    current = prediction[np.ix_(rows, targets)]
    denominator = max(float((observed ** 2).sum()), 1e-12)
    baseline_r2 = 1.0 - float(((observed - base) ** 2).sum()) / denominator
    current_r2 = 1.0 - float(((observed - current) ** 2).sum()) / denominator
    return current_r2, current_r2 - baseline_r2


def random_connected_states(
    search: Search,
    initial: State,
    selected_count: int,
    node_count: int,
    repeats: int,
    seed: int,
) -> list[State]:
    if selected_count <= len(initial.selected):
        return [initial]
    rng = np.random.default_rng(seed)
    accepted: dict[tuple[int, ...], State] = {}
    for _ in range(max(2000, repeats * 50)):
        state = initial
        while len(state.selected) < selected_count:
            children = [child for child in search.children(state) if len(child.nodes) <= node_count]
            if not children:
                break
            state = children[int(rng.integers(len(children)))]
        if len(state.selected) == selected_count and len(state.nodes) == node_count:
            accepted[state.selected] = state
            if len(accepted) >= repeats:
                break
    return list(accepted.values())


def bootstrap_mean_interval(values: np.ndarray, seed: int, repeats: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repeats)]
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def split_genes(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {gene for gene in str(value).split("|") if gene}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap", type=Path, default=root / "data/processed/depmap_bridge_24q4")
    parser.add_argument("--ppi", type=Path, default=root / "data/processed/context_module_stage0/ppi_edges.tsv.gz")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs/depmap_module_search_20seed",
    )
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--node-budget", type=int, default=8)
    parser.add_argument("--node-penalty", type=float, default=0.00025)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_20_SEEDS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--stability-bootstrap", type=int, default=10000)
    parser.add_argument("--required-seed", default="TLN1")
    parser.add_argument("--random-controls", type=int, default=200)
    parser.add_argument(
        "--tracked-genes", nargs="+", default=list(DEFAULT_TRACKED_GENES),
        help="Genes whose event-selection and connector frequencies are reported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
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
    if args.required_seed not in genes:
        raise ValueError(f"Required seed is unavailable: {args.required_seed}")
    features = feature_sets(
        models, matrices["mutation"], matrices["expression"], matrices["copy_number"], genes
    )
    raw = np.stack(
        [
            matrices["expression"][genes].to_numpy(dtype=float),
            matrices["copy_number"][genes].to_numpy(dtype=float),
            matrices["mutation"][genes].fillna(0).to_numpy(dtype=float),
        ],
        axis=2,
    )
    dependency = matrices["dependency"][genes].to_numpy(dtype=float)
    ppi = pd.read_csv(args.ppi, sep="\t")
    non_kidney = np.flatnonzero(~models["kidney_lineage"].to_numpy())
    kidney = np.flatnonzero(models["kidney_lineage"].to_numpy())
    labels = models["OncotreeLineage"].fillna("Unknown").to_numpy(dtype=str)
    panels, singleton_rows, interaction_rows = [], [], []
    modes = {
        "unrestricted": State(),
        "tln1_seeded": State((genes.index(args.required_seed),), (args.required_seed,)),
    }
    for seed in args.seeds:
        relative_folds = stratified_folds(labels[non_kidney], args.folds, seed)
        folds = [
            prepare_fold(
                raw, features["background"], dependency,
                np.setdiff1d(non_kidney, non_kidney[relative]), non_kidney[relative], args.alpha,
            )
            for relative in relative_folds
        ]
        reward = CVReward(folds, width=3)
        search = Search(genes, ppi, reward, args.node_budget, args.node_penalty)
        singles = np.asarray([reward.delta((index,)) for index in range(len(genes))])
        singleton_rows.extend(
            {"seed": seed, "Gene": gene, "cv_delta_r2": singles[index]}
            for index, gene in enumerate(genes)
        )
        static_order = sorted(range(len(genes)), key=lambda index: (-singles[index], genes[index]))
        frequency_order = sorted(
            range(len(genes)),
            key=lambda index: (-coverage.loc[genes[index], "TCGA_train_mutation_count"], genes[index]),
        )
        for mode, initial in modes.items():
            selected = {
                "frequency_ppi": search.ranked(frequency_order, initial),
                "static_cv_ppi": search.ranked(static_order, initial),
                "greedy_ppi": search.greedy(initial),
                "beam_ppi": search.beam(args.beam_width, initial),
            }
            for method, state in selected.items():
                panels.append(
                    {
                        "seed": seed,
                        "mode": mode,
                        "method": method,
                        "state": state,
                        "cv_delta_r2": reward.delta(state.selected),
                        "cv_objective": search.objective(state),
                    }
                )
                print(
                    f"seed={seed} mode={mode} method={method} genes={len(state.selected)} "
                    f"nodes={len(state.nodes)} cv_delta_R2={reward.delta(state.selected):.6f}",
                    flush=True,
                )
        top = static_order[:30]
        rng = np.random.default_rng(seed)
        pairs = set()
        for _ in range(200):
            left, right = sorted(rng.choice(top, size=2, replace=False).tolist())
            if (left, right) in pairs:
                continue
            pairs.add((left, right))
            state = search.extend(State((left,), (genes[left],)), right)
            if state is not None:
                pair_gain = reward.delta((left, right))
                interaction_rows.append(
                    {
                        "seed": seed,
                        "gene_a": genes[left],
                        "gene_b": genes[right],
                        "nonadditivity": pair_gain - singles[left] - singles[right],
                    }
                )

    # The kidney lineage is first accessed after every module has been selected.
    final = prepare_fold(raw, features["background"], dependency, non_kidney, kidney, args.alpha)
    held_models = models.iloc[kidney]
    held_labels = held_models["OncotreeSubtype"].fillna("Unknown").to_numpy(dtype=str)
    scopes = {
        "kidney_lineage": np.arange(len(kidney)),
        "rcc": np.flatnonzero(held_models["renal_cell_carcinoma"].to_numpy()),
        "ccrcc": np.flatnonzero(held_models["clear_cell_renal_cell_carcinoma"].to_numpy()),
    }
    result_rows, target_rows = [], []
    for record in panels:
        state = record["state"]
        columns = np.asarray(
            [index * 3 + channel for index in state.selected for channel in range(3)], dtype=int
        )
        prediction = final.predict(columns)
        unselected_targets = np.asarray(
            [index for index in range(len(genes)) if index not in state.selected], dtype=int
        )
        for scope, rows in scopes.items():
            r2, delta = scoped_delta(final.y, final.baseline, prediction, rows)
            _, no_self_delta = scoped_delta(
                final.y, final.baseline, prediction, rows, unselected_targets
            )
            lower, upper = bootstrap_delta(
                final.y, final.baseline, prediction, held_labels, rows,
                record["seed"] + len(record["method"]) + len(record["mode"]), args.bootstrap,
            )
            result_rows.append(
                {
                    "seed": record["seed"],
                    "mode": record["mode"],
                    "method": record["method"],
                    "scope": scope,
                    "model_n": len(rows),
                    "event_genes": "|".join(genes[index] for index in state.selected),
                    "connectors": "|".join(sorted(set(state.nodes) - {genes[i] for i in state.selected})),
                    "total_nodes": len(state.nodes),
                    "cv_delta_r2": record["cv_delta_r2"],
                    "cv_objective": record["cv_objective"],
                    "holdout_r2": r2,
                    "holdout_delta_r2": delta,
                    "holdout_delta_r2_excluding_selected_targets": no_self_delta,
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                }
            )
        target_sst = (final.y ** 2).sum(axis=0)
        target_gain = ((final.y - final.baseline) ** 2).sum(axis=0) - ((final.y - prediction) ** 2).sum(axis=0)
        target_rows.extend(
            {
                "seed": record["seed"], "mode": record["mode"], "method": record["method"],
                "Gene": gene, "kidney_holdout_delta_r2": target_gain[index] / max(target_sst[index], 1e-12),
                "selected_as_predictor": index in state.selected,
            }
            for index, gene in enumerate(genes)
        )

    results = pd.DataFrame(result_rows)
    results["event_gene_count"] = results["event_genes"].str.count("\\|") + 1
    random_rows = []
    ccrcc_rows = scopes["ccrcc"]
    random_search = Search(genes, ppi, CVReward([final], width=3), args.node_budget, 0.0)
    structures = results[["mode", "event_gene_count", "total_nodes"]].drop_duplicates()
    for structure_index, structure in structures.iterrows():
        mode = structure["mode"]
        initial = modes[mode]
        states = random_connected_states(
            random_search, initial, int(structure["event_gene_count"]),
            int(structure["total_nodes"]), args.random_controls,
            90000 + int(structure_index),
        )
        for repeat, state in enumerate(states):
            columns = np.asarray(
                [index * 3 + channel for index in state.selected for channel in range(3)], dtype=int
            )
            prediction = final.predict(columns)
            _, delta = scoped_delta(final.y, final.baseline, prediction, ccrcc_rows)
            unselected = np.asarray(
                [index for index in range(len(genes)) if index not in state.selected], dtype=int
            )
            _, no_self = scoped_delta(
                final.y, final.baseline, prediction, ccrcc_rows, unselected
            )
            random_rows.append(
                {
                    "mode": mode,
                    "event_gene_count": len(state.selected),
                    "total_nodes": len(state.nodes),
                    "repeat": repeat,
                    "event_genes": "|".join(genes[index] for index in state.selected),
                    "ccrcc_delta_r2": delta,
                    "ccrcc_delta_r2_excluding_selected_targets": no_self,
                }
            )
    random_controls = pd.DataFrame(random_rows)
    if len(random_controls):
        random_summary = random_controls.groupby(
            ["mode", "event_gene_count", "total_nodes"]
        )["ccrcc_delta_r2"].agg(
            random_control_n="size", random_median="median",
            random_q05=lambda values: values.quantile(0.05),
            random_q95=lambda values: values.quantile(0.95),
        ).reset_index()
        results = results.merge(
            random_summary, on=["mode", "event_gene_count", "total_nodes"],
            how="left", validate="many_to_one",
        )
        lookup = {
            key: group["ccrcc_delta_r2"].to_numpy()
            for key, group in random_controls.groupby(["mode", "event_gene_count", "total_nodes"])
        }
        results["random_control_percentile"] = [
            float((lookup[(row.mode, row.event_gene_count, row.total_nodes)] <= row.holdout_delta_r2).mean())
            if row.scope == "ccrcc" and (row.mode, row.event_gene_count, row.total_nodes) in lookup
            else np.nan
            for row in results.itertuples()
        ]
        results["random_empirical_upper_p"] = [
            float(
                (1 + (lookup[(row.mode, row.event_gene_count, row.total_nodes)] >= row.holdout_delta_r2).sum())
                / (1 + len(lookup[(row.mode, row.event_gene_count, row.total_nodes)]))
            )
            if row.scope == "ccrcc"
            and len(lookup.get((row.mode, row.event_gene_count, row.total_nodes), [])) > 1
            else np.nan
            for row in results.itertuples()
        ]
    singletons = pd.DataFrame(singleton_rows)
    interactions = pd.DataFrame(interaction_rows)
    target_results = pd.DataFrame(target_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "results.csv", index=False)
    singletons.groupby("Gene").cv_delta_r2.agg(["mean", "std", "min", "max"]).reset_index().to_csv(
        args.output_dir / "singleton_stability.csv", index=False
    )
    interactions.to_csv(args.output_dir / "action_interactions.csv", index=False)
    target_results.to_csv(args.output_dir / "target_results.csv.gz", index=False, compression="gzip")
    random_controls.to_csv(
        args.output_dir / "random_module_controls.csv.gz", index=False, compression="gzip"
    )

    ccrcc = results.loc[results["scope"].eq("ccrcc")]
    method_summary = (
        ccrcc.groupby(["mode", "method"], as_index=False)
        .agg(
            seed_n=("seed", "nunique"),
            holdout_delta_r2_mean=("holdout_delta_r2", "mean"),
            holdout_delta_r2_sd=("holdout_delta_r2", "std"),
            holdout_delta_r2_min=("holdout_delta_r2", "min"),
            holdout_delta_r2_max=("holdout_delta_r2", "max"),
            cv_delta_r2_mean=("cv_delta_r2", "mean"),
            random_percentile_mean=("random_control_percentile", "mean"),
        )
    )
    method_summary.to_csv(args.output_dir / "method_stability_summary.csv", index=False)

    frequency_rows = []
    for (mode, method), group in ccrcc.groupby(["mode", "method"]):
        seed_n = group["seed"].nunique()
        for gene in args.tracked_genes:
            event_count = int(group["event_genes"].map(lambda value: gene in split_genes(value)).sum())
            node_count = int(group.apply(
                lambda row: gene in (split_genes(row["event_genes"]) | split_genes(row["connectors"])),
                axis=1,
            ).sum())
            frequency_rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "Gene": gene,
                    "seed_n": seed_n,
                    "event_selected_count": event_count,
                    "event_selected_frequency": event_count / seed_n,
                    "module_node_count": node_count,
                    "module_node_frequency": node_count / seed_n,
                }
            )
    tracked_frequency = pd.DataFrame(frequency_rows)
    tracked_frequency.to_csv(args.output_dir / "tracked_gene_selection_frequency.csv", index=False)

    gate = {}
    gate_rows = []
    for mode in modes:
        paired_rows = []
        scoped = ccrcc.loc[ccrcc["mode"].eq(mode)]
        for seed, group in scoped.groupby("seed"):
            static_row = group.loc[group["method"].eq("static_cv_ppi")].iloc[0]
            # The adaptive method is chosen using non-kidney CV objective only;
            # the ccRCC holdout is never used to pick greedy versus beam.
            adaptive_row = (
                group.loc[group["method"].isin(["greedy_ppi", "beam_ppi"])]
                .sort_values(["cv_objective", "method"], ascending=[False, True])
                .iloc[0]
            )
            paired_rows.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "adaptive_method_selected_by_cv": adaptive_row["method"],
                    "adaptive_cv_objective": float(adaptive_row["cv_objective"]),
                    "static_cv_objective": float(static_row["cv_objective"]),
                    "adaptive_holdout_delta_r2": float(adaptive_row["holdout_delta_r2"]),
                    "static_holdout_delta_r2": float(static_row["holdout_delta_r2"]),
                    "adaptive_minus_static": float(
                        adaptive_row["holdout_delta_r2"] - static_row["holdout_delta_r2"]
                    ),
                }
            )
        paired = pd.DataFrame(paired_rows)
        differences = paired["adaptive_minus_static"].to_numpy(dtype=float)
        lower, upper = bootstrap_mean_interval(
            differences, 20260912 + len(mode), args.stability_bootstrap
        )
        mean_difference = float(differences.mean())
        seed_count_ok = len(differences) >= 20
        stop_rule_met = mean_difference < 0.005 or lower <= 0.0
        decision = (
            "STOP_DDQN_ROUTE" if seed_count_ok and stop_rule_met
            else "DO_NOT_STOP_ON_THIS_GATE" if seed_count_ok
            else "INSUFFICIENT_SEEDS"
        )
        paired["bootstrap_mean_lower"] = lower
        paired["bootstrap_mean_upper"] = upper
        paired["decision"] = decision
        gate_rows.extend(paired.to_dict("records"))
        gate[mode] = {
            "seed_n": len(differences),
            "adaptive_selection": "max non-kidney CV objective among greedy_ppi and beam_ppi",
            "static_reference": "static_cv_ppi",
            "adaptive_minus_static_mean": mean_difference,
            "algorithm_seed_bootstrap_lower": lower,
            "algorithm_seed_bootstrap_upper": upper,
            "stop_threshold": 0.005,
            "stop_if": "mean < 0.005 OR bootstrap lower <= 0",
            "decision": decision,
        }
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "adaptive_vs_static_gate.csv", index=False)
    audit = {
        "status": "exploratory_kidney_lineage_holdout",
        "selection_models": len(non_kidney),
        "kidney_holdout_models": len(kidney),
        "rcc_holdout_models": int(held_models["renal_cell_carcinoma"].sum()),
        "ccrcc_holdout_models": int(held_models["clear_cell_renal_cell_carcinoma"].sum()),
        "candidate_genes": len(genes),
        "parameters": vars(args) | {"depmap": str(args.depmap), "ppi": str(args.ppi), "output_dir": str(args.output_dir)},
        "reward": "non-kidney 5-fold out-of-fold aggregate dependency delta R2 versus background ridge",
        "feature_channels": ["expression", "absolute_copy_number", "damaging_mutation"],
        "background": ["lineage", *DRIVERS, "mutation_burden", "copy_number_deviation", "expression_level"],
        "rl_necessity_gate": gate,
        "tracked_gene_frequency": {
            "genes": args.tracked_genes,
            "event_gene_and_any_module_node_reported_separately": True,
        },
        "stability_interval": (
            "95% bootstrap over algorithm/CV split seeds; this measures algorithmic instability "
            "and is not a biological-sample confidence interval"
        ),
        "action_interaction_pairs": len(interactions),
        "action_nonadditivity_abs_gt_0_001_fraction": (
            float(interactions["nonadditivity"].abs().gt(0.001).mean()) if len(interactions) else None
        ),
        "random_connected_module_controls": len(random_controls),
        "kidney_holdout_access": "after all module selections",
        "limitations": [
            "Cell lines are functional models rather than patients.",
            "The kidney holdout contains 24 models and ccRCC contains 12.",
            "Module selection optimizes an aggregate dependency endpoint across correlated target genes.",
            "The lineage holdout compares search methods exploratorily and is not a second untouched confirmation set.",
            "Random-module empirical upper-tail probabilities are unadjusted descriptive controls.",
            "Twenty CV split seeds do not create twenty independent biological cohorts.",
        ],
        "input_sha256": {
            name: sha256(args.depmap / f"{name}.tsv.gz")
            for name in ["dependency", "mutation", "expression", "copy_number"]
        } | {"models": sha256(args.depmap / "models.csv"), "ppi": sha256(args.ppi)},
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
    }
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, default=str)
    print(results.loc[results["scope"].eq("ccrcc"), [
        "seed", "mode", "method", "event_genes", "cv_delta_r2", "holdout_delta_r2",
        "holdout_delta_r2_excluding_selected_targets", "bootstrap_lower", "bootstrap_upper"
    ]].to_string(index=False))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
