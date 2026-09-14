"""Train-only CV module search against disjoint Hallmark expression endpoints.

Requires numpy/pandas only. The existing test split is excluded. Inner CV is a
search objective, not an unbiased estimate after adaptive subset selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_module_stage0 import GeneCanonicalizer

DRIVERS = ("VHL", "PBRM1", "SETD2", "BAP1", "MTOR")
GMT_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"


def read_development(path: Path, ids: pd.Index) -> pd.DataFrame:
    # Discard test rows immediately; no test-derived summaries are computed.
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", index_col=0, chunksize=64):
        pieces.append(chunk.loc[chunk.index.isin(ids)])
    frame = pd.concat(pieces)
    if not frame.index.is_unique or not ids.isin(frame.index).all():
        raise ValueError(f"Invalid patient index: {path}")
    return frame.loc[ids]


def load_absolute_purity(path: Path, patient_ids: pd.Index) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    if "array" not in frame or "purity" not in frame:
        raise ValueError("ABSOLUTE table lacks array or purity column")
    frame["patient_id"] = frame["array"].astype(str).str.slice(0, 12)
    frame["purity"] = pd.to_numeric(frame["purity"], errors="coerce")
    if "call status" in frame:
        frame.loc[~frame["call status"].eq("called"), "purity"] = np.nan
    values = frame.groupby("patient_id")["purity"].median()
    return values.reindex(patient_ids).rename("ABSOLUTE_purity")


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(train)
    count = finite.sum(axis=0)
    mean = np.divide(np.where(finite, train, 0).sum(axis=0), count,
                     out=np.zeros(train.shape[1]), where=count > 0)
    clean = np.where(finite, train, mean)
    sd = clean.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (clean - mean) / sd, (np.where(np.isfinite(other), other, mean) - mean) / sd


def make_folds(contexts: np.ndarray, seed: int, count: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for context in sorted(set(contexts)):
        indices = rng.permutation(np.flatnonzero(contexts == context))
        for i, index in enumerate(indices):
            folds[i % count].append(int(index))
    result = [np.array(sorted(fold), dtype=int) for fold in folds]
    assert sorted(np.concatenate(result).tolist()) == list(range(len(contexts)))
    return result


def feature_block(train: np.ndarray, other: np.ndarray,
                  train_context: np.ndarray, other_context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Per gene: 3 main channels, then 3 x VHL and 3 x PBRM1 interactions.
    n, genes, channels = train.shape
    a, b = standardize(train.reshape(n, -1), other.reshape(len(other), -1))
    a, b = a.reshape(n, genes, channels), b.reshape(len(other), genes, channels)
    center = train_context.mean(axis=0)
    ac, bc = train_context - center, other_context - center
    aa = [a] + [a * ac[:, i, None, None] for i in range(2)]
    bb = [b] + [b * bc[:, i, None, None] for i in range(2)]
    return np.concatenate(aa, axis=2).reshape(n, -1), np.concatenate(bb, axis=2).reshape(len(other), -1)


@dataclass
class Fold:
    gram: np.ndarray
    cross: np.ndarray
    held_design: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    alpha: float

    def predict(self, columns: np.ndarray) -> np.ndarray:
        if len(columns) == 0:
            return self.baseline.copy()
        matrix = self.gram[np.ix_(columns, columns)].copy()
        matrix.flat[::len(columns) + 1] += self.alpha
        beta = np.linalg.solve(matrix, self.cross[columns])
        return self.baseline + self.held_design[:, columns] @ beta


def ridge_fold(x: np.ndarray, held_x: np.ndarray, c: np.ndarray, held_c: np.ndarray,
               y: np.ndarray, held_y: np.ndarray, alpha: float) -> Fold:
    # Penalized Frisch-Waugh elimination is algebraically identical to fitting
    # [context, module] together; the intercept is the sole unpenalized column.
    gram_c = c.T @ c
    gram_c.flat[::len(gram_c) + 1] += np.r_[0.0, np.repeat(alpha, c.shape[1] - 1)]
    cx = np.linalg.solve(gram_c, c.T @ x)
    cy = np.linalg.solve(gram_c, c.T @ y)
    residual_x = x - c @ cx
    gram = x.T @ residual_x
    return Fold((gram + gram.T) / 2, x.T @ (y - c @ cy),
                held_x - held_c @ cx, held_y, held_c @ cy, alpha)


def prepare_fold(raw: np.ndarray, context: np.ndarray, nuisance: np.ndarray,
                 expression: np.ndarray, weights: np.ndarray,
                 fit: np.ndarray, held: np.ndarray, alpha: float) -> Fold:
    x, hx = feature_block(raw[fit], raw[held], context[fit], context[held])
    c, hc = standardize(nuisance[fit], nuisance[held])
    c, hc = np.c_[np.ones(len(fit)), c], np.c_[np.ones(len(held)), hc]
    e, he = standardize(expression[fit], expression[held])
    y, hy = standardize(e @ weights, he @ weights)
    return ridge_fold(x, hx, c, hc, y, hy, alpha)


class CVReward:
    def __init__(self, folds: list[Fold], width: int = 9):
        self.folds, self.width = folds, width
        self.sst = sum(float((f.y ** 2).sum()) for f in folds)
        self.baseline_sse = sum(float(((f.y - f.baseline) ** 2).sum()) for f in folds)
        self.cache: dict[tuple[int, ...], float] = {(): 0.0}

    def delta(self, selected: tuple[int, ...]) -> float:
        key = tuple(sorted(selected))
        if key not in self.cache:
            columns = np.array([g * self.width + j for g in key for j in range(self.width)], dtype=int)
            sse = sum(float(((f.y - f.predict(columns)) ** 2).sum()) for f in self.folds)
            self.cache[key] = (self.baseline_sse - sse) / max(self.sst, 1e-12)
        return self.cache[key]


@dataclass(frozen=True)
class State:
    selected: tuple[int, ...] = ()
    nodes: tuple[str, ...] = ()


class Search:
    def __init__(self, genes: list[str], edges: pd.DataFrame, reward: CVReward,
                 node_budget: int, penalty: float):
        self.genes, self.reward = genes, reward
        self.budget, self.penalty = node_budget, penalty
        self.adj: dict[str, set[str]] = {}
        for left, right in edges.itertuples(index=False, name=None):
            self.adj.setdefault(left, set()).add(right)
            self.adj.setdefault(right, set()).add(left)

    def objective(self, state: State) -> float:
        return self.reward.delta(state.selected) - self.penalty * len(state.nodes)

    def extend(self, state: State, index: int) -> State | None:
        if index in state.selected:
            return None
        gene, nodes = self.genes[index], set(state.nodes)
        added = {gene}
        if nodes and gene not in nodes and not (self.adj.get(gene, set()) & nodes):
            bridges = set()
            for node in nodes:
                bridges.update(self.adj.get(node, set()) & self.adj.get(gene, set()))
            if not bridges:
                return None
            # A fixed lexical tie-break avoids selecting connectors for omics gain.
            added.add(min(bridges))
        updated = nodes | added
        if len(updated) > self.budget:
            return None
        return State(tuple(sorted((*state.selected, index))), tuple(sorted(updated)))

    def children(self, state: State) -> list[State]:
        return [child for i in range(len(self.genes)) if (child := self.extend(state, i)) is not None]

    def transition(self, state: State, action: int | None) -> tuple[State, float, bool]:
        """RL-compatible transition: None is STOP, reward telescopes in CV utility."""
        if action is None:
            return state, 0.0, True
        if action < 0 or action >= len(self.genes):
            raise ValueError("Invalid action index")
        next_state = self.extend(state, action)
        if next_state is None:
            raise ValueError("Action violates selected-set, PPI, or node-budget constraints")
        return next_state, self.objective(next_state) - self.objective(state), False

    def ranked(self, order: list[int], initial: State = State()) -> State:
        # Static priority does not adapt action scores. STOP/best prefix uses CV.
        state, best = initial, initial
        while True:
            next_state = next((s for i in order if (s := self.extend(state, i)) is not None), None)
            if next_state is None:
                break
            state = next_state
            if self.objective(state) > self.objective(best):
                best = state
        return best

    def greedy(self, initial: State = State()) -> State:
        state = initial
        while children := self.children(state):
            best = max(children, key=lambda s: (self.objective(s), s.selected, s.nodes))
            if self.objective(best) <= self.objective(state):
                break
            state = best
        return state

    def beam(self, width: int, initial: State = State()) -> State:
        beam, best = [initial], initial
        for _ in range(self.budget):
            candidates = {child for state in beam for child in self.children(state)}
            if not candidates:
                break
            beam = sorted(candidates, key=lambda s: (-self.objective(s), s.selected, s.nodes))[:width]
            if self.objective(beam[0]) > self.objective(best):
                best = beam[0]
        return best


def load_inputs(args: argparse.Namespace) -> dict:
    patients = pd.read_csv(args.input_dir / "patients.csv", index_col=0)
    dev = patients.loc[patients.split.isin(["train", "validation"])].copy()
    dev["ABSOLUTE_purity"] = load_absolute_purity(args.purity, dev.index)
    train = np.flatnonzero(dev.split.to_numpy() == "train")
    val = np.flatnonzero(dev.split.to_numpy() == "validation")
    names = {"mutation": "mutation_binary", "cnv": "cnv_continuous",
             "threshold": "cnv_thresholded", "meth": "methylation_promoter_delta_beta",
             "expr": "expression_delta_log2"}
    matrices = {k: read_development(args.input_dir / (v + ".tsv.gz"), dev.index)
                for k, v in names.items()}
    mutation, cnv, meth, expression = (matrices[k] for k in ("mutation", "cnv", "meth", "expr"))
    count = mutation.iloc[train].sum()
    quality = (cnv.iloc[train].notna().mean() >= .9) & (meth.iloc[train].notna().mean() >= .9)
    maximum = int(np.floor(args.max_frequency * len(train)))
    genes = sorted(g for g in count.index if args.min_count <= count[g] <= maximum
                   and quality[g] and g not in DRIVERS)
    if not genes:
        raise ValueError("No training candidates")
    canonical = GeneCanonicalizer(args.hgnc)
    excluded = set(genes) | set(DRIVERS)
    gene_sets = {}
    rows = []
    for line in args.gmt.read_text(encoding="utf-8").splitlines():
        name, url, *symbols = line.split("\t")
        mapped = {canonical.map(g) for g in symbols}
        usable = sorted(mapped & set(expression.columns) - excluded)
        rows.append({"program": name, "original_genes": len(mapped),
                     "endpoint_genes": len(usable), "excluded_candidate_genes": len(mapped & excluded),
                     "retained": len(usable) >= 10, "genes": "|".join(usable)})
        if len(usable) >= 10:
            gene_sets[name] = usable
    target_genes = sorted(set().union(*map(set, gene_sets.values())))
    assert not set(target_genes) & excluded
    target_index = {g: i for i, g in enumerate(target_genes)}
    programs = sorted(gene_sets)
    weights = np.zeros((len(target_genes), len(programs)))
    for j, program in enumerate(programs):
        for gene in gene_sets[program]:
            weights[target_index[gene], j] = 1.0 / len(gene_sets[program])
    raw = np.stack([mutation[genes].to_numpy(), cnv[genes].to_numpy(), meth[genes].to_numpy()], axis=2)
    driver = dev[[g + "_mut" for g in DRIVERS]].to_numpy(dtype=float)
    threshold = matrices["threshold"]
    burden = np.c_[np.log1p(mutation.sum(axis=1).to_numpy()),
                   (threshold.abs().ge(2).sum(axis=1) / threshold.notna().sum(axis=1)).to_numpy(),
                   cnv.abs().mean(axis=1).to_numpy(), meth.mean(axis=1).to_numpy()]
    nuisance = np.c_[driver, driver[:, 0] * driver[:, 1], burden,
                     dev["ABSOLUTE_purity"].to_numpy(dtype=float)]
    candidate_table = pd.DataFrame({"Gene": genes, "mutation_count_train": count[genes].to_numpy(),
                                    "mutation_frequency_train": (count[genes] / len(train)).to_numpy()})
    return dict(patients=dev, train=train, val=val, genes=genes, raw=raw, context=driver[:, :2],
                nuisance=nuisance, expression=expression[target_genes].to_numpy(dtype=float),
                weights=weights, programs=programs, target_genes=target_genes,
                target_table=pd.DataFrame(rows), candidates=candidate_table,
                events=(mutation[genes].gt(0) | threshold[genes].abs().ge(2)).to_numpy(),
                edges=pd.read_csv(args.input_dir / "ppi_edges.tsv.gz", sep="\t"),
                purity_observed=int(dev["ABSOLUTE_purity"].notna().sum()))


def prepare(data: dict, fit: np.ndarray, held: np.ndarray, alpha: float) -> Fold:
    return prepare_fold(data["raw"], data["context"], data["nuisance"], data["expression"],
                        data["weights"], fit, held, alpha)


def bootstrap_delta(y: np.ndarray, baseline: np.ndarray, prediction: np.ndarray,
                    contexts: np.ndarray, seed: int) -> tuple[float, float]:
    gain = ((y - baseline)**2 - (y - prediction)**2).sum(axis=1)
    sst = (y**2).sum(axis=1)
    strata = [np.flatnonzero(contexts == c) for c in sorted(set(contexts))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        ids = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        values.append(gain[ids].sum() / max(sst[ids].sum(), 1e-12))
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_initial_state(genes: list[str], seed_gene: str | None) -> State:
    if not seed_gene:
        return State()
    normalized = seed_gene.strip().upper()
    if normalized not in genes:
        raise ValueError(f"Required seed gene is absent from the frozen candidate set: {normalized}")
    index = genes.index(normalized)
    return State((index,), (normalized,))


def selection_null_controls(data: dict, args: argparse.Namespace) -> pd.DataFrame:
    """Descriptive null stress test, rerunning all selection on scrambled RNA.

Whole expression rows move together within TRAINING context; validation stays
unused. This does not preserve every nuisance correlation and is not a formal
conditional randomization test or a confirmatory p-value.
"""
    rows = []
    train = data["train"]
    contexts = data["patients"].iloc[train].primary_context.to_numpy()
    holdouts = make_folds(contexts, args.seeds[0])
    for repeat in range(args.null_repeats):
        rng = np.random.default_rng(71000 + repeat)
        shuffled = data["expression"].copy()
        for context in sorted(set(contexts)):
            indices = train[contexts == context]
            shuffled[indices] = data["expression"][rng.permutation(indices)]
        temporary = {**data, "expression": shuffled}
        folds = [prepare(temporary, np.setdiff1d(train, train[held]), train[held], args.alpha)
                 for held in holdouts]
        reward = CVReward(folds)
        search = Search(data["genes"], data["edges"], reward, args.node_budget, args.node_penalty)
        order = sorted(range(len(data["genes"])), key=lambda i: (-reward.delta((i,)), data["genes"][i]))
        frequency = sorted(range(len(data["genes"])), key=lambda i: (-data["candidates"].iloc[i].mutation_count_train,
                                                                     data["genes"][i]))
        initial = required_initial_state(data["genes"], args.required_seed_gene)
        states = {"frequency_ppi": search.ranked(frequency, initial),
                  "static_cv_ppi": search.ranked(order, initial),
                  "greedy_ppi": search.greedy(initial),
                  "beam_ppi": search.beam(args.beam_width, initial)}
        rows.extend({"repeat": repeat, "method": method, "scrambled_search_cv_delta_r2": reward.delta(state.selected),
                     "nodes": len(state.nodes)} for method, state in states.items())
        if (repeat + 1) % 5 == 0:
            print(f"Training-only complete-search null controls {repeat + 1}/{args.null_repeats}", flush=True)
        del folds, reward, search, temporary, shuffled
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "data/processed/context_module_stage0")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/context_module_cv")
    parser.add_argument("--gmt", type=Path, default=root / "data/raw/reference/h.all.v2024.1.Hs.symbols.gmt")
    parser.add_argument("--hgnc", type=Path, default=root / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--purity", type=Path,
                        default=root / "data/raw/reference/TCGA_mastercalls.abs_tables_JSedit.fixed.txt")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-frequency", type=float, default=.05)
    parser.add_argument("--node-budget", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--node-penalty", type=float, default=.001)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260911, 20260912, 20260913])
    parser.add_argument("--required-seed-gene", type=str, default=None)
    args = parser.parse_args()
    if args.node_budget < 1 or args.alpha <= 0 or args.beam_width < 1:
        raise ValueError("Invalid budget, ridge penalty, or beam width")
    start = time.monotonic()
    data = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train={len(data['train'])}, validation={len(data['val'])}, candidates={len(data['genes'])}, "
          f"disjoint target genes={len(data['target_genes'])}, programs={len(data['programs'])}", flush=True)
    selected_panel, diagnostics = [], []
    singleton_records = []
    for seed in args.seeds:
        holdouts = make_folds(data["patients"].iloc[data["train"]].primary_context.to_numpy(), seed)
        folds = []
        for relative in holdouts:
            held = data["train"][relative]
            fit = np.setdiff1d(data["train"], held)
            assert not set(fit) & set(data["val"]) and not set(held) & set(data["val"])
            folds.append(prepare(data, fit, held, args.alpha))
        reward = CVReward(folds)
        search = Search(data["genes"], data["edges"], reward, args.node_budget, args.node_penalty)
        singles = [reward.delta((i,)) for i in range(len(data["genes"]))]
        order = sorted(range(len(singles)), key=lambda i: (-singles[i], data["genes"][i]))
        singleton_records.extend({"seed": seed, "Gene": gene, "cv_delta_r2": singles[i]}
                                 for i, gene in enumerate(data["genes"]))
        frequency = sorted(range(len(singles)), key=lambda i: (-data["candidates"].iloc[i].mutation_count_train,
                                                              data["genes"][i]))
        initial = required_initial_state(data["genes"], args.required_seed_gene)
        methods = {"frequency_ppi": search.ranked(frequency, initial),
                   "static_cv_ppi": search.ranked(order, initial),
                   "greedy_ppi": search.greedy(initial),
                   "beam_ppi": search.beam(args.beam_width, initial)}
        for method, state in methods.items():
            selected_panel.append((seed, method, state, reward.delta(state.selected), search.objective(state)))
            print(f"seed={seed} {method}: event genes={len(state.selected)} nodes={len(state.nodes)} "
                  f"inner_cv_delta_R2={reward.delta(state.selected):.5f}", flush=True)
        # Action interactions are diagnostic only; non-additivity alone is not RL evidence.
        rng = np.random.default_rng(seed)
        pool = order[:min(40, len(order))]
        pairs = set()
        for _ in range(300):
            left, right = sorted(rng.choice(pool, size=2, replace=False).tolist())
            if (left, right) in pairs:
                continue
            pairs.add((left, right))
            state = search.extend(State((left,), (data["genes"][left],)), right)
            if state:
                delta = reward.delta((left, right))
                diagnostics.append({"seed": seed, "gene_a": data["genes"][left], "gene_b": data["genes"][right],
                                    "singleton_sum": singles[left] + singles[right], "pair_delta_r2": delta,
                                    "nonadditivity": delta - singles[left] - singles[right]})
        del search, reward, folds

    # Search completes for all methods/seeds before validation targets are fitted/scored.
    final = prepare(data, data["train"], data["val"], args.alpha)
    val_context = data["patients"].iloc[data["val"]].primary_context.to_numpy()
    sst = max(float((final.y**2).sum()), 1e-12)
    baseline_sse = float(((final.y - final.baseline)**2).sum())
    result_rows, target_rows, prediction_rows = [], [], []
    for seed, method, state, cv_delta, objective in selected_panel:
        variants = {
            "full": tuple(range(9)),
            "mutation_only": (0, 3, 6),
            "no_context_interactions": (0, 1, 2),
            "no_methylation": (0, 1, 3, 4, 6, 7),
            "no_cnv": (0, 2, 3, 5, 6, 8),
        }
        for variant, channels in variants.items():
            columns = np.array([i * 9 + j for i in state.selected for j in channels], dtype=int)
            prediction = final.predict(columns)
            sse = float(((final.y - prediction)**2).sum())
            lower, upper = bootstrap_delta(final.y, final.baseline, prediction, val_context, 20260911)
            carrier = (data["events"][data["val"]][:, state.selected].any(axis=1)
                       if state.selected else np.zeros(len(data["val"]), dtype=bool))
            result_rows.append({"seed": seed, "method": method, "variant": variant,
                                "event_genes": "|".join(data["genes"][i] for i in state.selected),
                                "connectors": "|".join(sorted(set(state.nodes) - {data['genes'][i] for i in state.selected})),
                                "total_nodes": len(state.nodes), "cv_delta_r2_full": cv_delta,
                                "cv_objective_full": objective, "validation_baseline_r2": 1 - baseline_sse / sst,
                                "validation_r2": 1 - sse / sst, "validation_delta_r2": (baseline_sse - sse) / sst,
                                "bootstrap_lower": lower, "bootstrap_upper": upper,
                                "validation_genomic_carriers": int(carrier.sum()),
                                "validation_n": len(data["val"])})
            if variant == "full":
                for j, program in enumerate(data["programs"]):
                    denominator = max(float((final.y[:, j]**2).sum()), 1e-12)
                    base_error = float(((final.y[:, j] - final.baseline[:, j])**2).sum())
                    module_error = float(((final.y[:, j] - prediction[:, j])**2).sum())
                    target_rows.append({"seed": seed, "method": method, "program": program,
                                        "validation_delta_r2": (base_error - module_error) / denominator})
                for i, patient in enumerate(data["patients"].index[data["val"]]):
                    prediction_rows.append({"seed": seed, "method": method, "patient_id": patient,
                                            "baseline_sse": float(((final.y[i] - final.baseline[i])**2).sum()),
                                            "module_sse": float(((final.y[i] - prediction[i])**2).sum()),
                                            "sst": float((final.y[i]**2).sum())})
    result = pd.DataFrame(result_rows)
    singletons = pd.DataFrame(singleton_records).groupby("Gene").cv_delta_r2.agg(["mean", "std"])
    candidates = data["candidates"].join(singletons.rename(columns={"mean": "singleton_cv_mean", "std": "singleton_cv_sd"}), on="Gene")
    for name, frame in {"results": result, "targets": data["target_table"], "candidates": candidates,
                        "program_results": pd.DataFrame(target_rows),
                        "validation_losses": pd.DataFrame(prediction_rows)}.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    null_table = selection_null_controls(data, args) if args.null_repeats else pd.DataFrame()
    if not null_table.empty:
        null_table.to_csv(args.output_dir / "selection_null.csv", index=False)
    full = result.loc[result.variant == "full"]
    manifest = {"status": "exploratory_development_validation_only", "test_evaluated": False,
                "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "data": {"train_n": len(data['train']), "validation_n": len(data['val']),
                         "candidates": len(data['genes']), "programs": len(data['programs']),
                         "disjoint_endpoint_genes": len(data['target_genes']),
                         "absolute_purity_observed": data["purity_observed"]},
                "feature_channels": ["mutation", "continuous_CNV", "promoter_delta_beta"],
                "interactions": ["VHL_mut", "PBRM1_mut"],
                "nuisance": [*DRIVERS, "VHL_x_PBRM1", "log_observed_gene_mutation_burden",
                             "deep_CNV_gene_fraction", "mean_absolute_CNV", "mean_promoter_delta_beta",
                             "ABSOLUTE_purity"],
                "endpoint": "Hallmark mean fold-training-standardized expression, excluding ALL candidates and context genes",
                "reward": "inner-CV delta R2 vs identical nuisance-only model; objective subtracts node_penalty * total_nodes",
                "search_space": "same candidate set; direct edge or one connector, all connector nodes count in budget; STOP allowed",
                "required_seed_gene": args.required_seed_gene,
                "limitations": ["observational prediction, not CRISPR dependency or causal effect",
                                "validation reused from prior development; not confirmatory",
                                "CV scores optimized by search, not unbiased generalization estimates",
                                "bulk mixture/purity and broad CNV may still confound",
                                "WT labels mean no observed consequence-filtered mutation, not verified gene functionality",
                                "no per-gene callable manifest; zero observed mutation is not guaranteed callable negative",
                                "small modules may be driven by CNV/methylation, not rare mutations; inspect ablations",
                                "seed replicates share patients; not independent biological replicates",
                                "bootstrap intervals conditional on frozen modules; no post-selection or multiple-program correction"],
                "selection_null_note": "training-only RNA row permutation within context; whole search rerun, descriptive stress test only",
                "selection_null_cv_mean": (null_table.groupby('method').scrambled_search_cv_delta_r2.mean().to_dict()
                                           if not null_table.empty else {}),
                "candidate_scope": "full training partition only; inner CV conditions on this frozen candidate universe",
                "action_pair_diagnostics": {"pairs_checked": len(diagnostics),
                                            "positive_nonadditivity_above_0_005": sum(r['nonadditivity'] > .005 for r in diagnostics)},
                "reference": {"url": GMT_URL, "sha256": sha256(args.gmt)},
                "purity_reference": {
                    "url": "https://api.gdc.cancer.gov/data/4f277128-f793-4354-a13d-30cc7fe9f6b5",
                    "sha256": sha256(args.purity),
                    "source": "NCI GDC PanCanAtlas ABSOLUTE purity/ploidy supplemental table",
                },
                "hgnc_sha256": sha256(args.hgnc),
                "software": {"numpy": np.__version__, "pandas": pd.__version__},
                "input_hashes": {p.name: sha256(p) for p in args.input_dir.iterdir() if p.suffix in ['.gz', '.csv', '.json']},
                "script_sha256": sha256(Path(__file__)), "elapsed_seconds": time.monotonic() - start,
                "method_validation_delta_mean": full.groupby('method').validation_delta_r2.mean().to_dict()}
    (args.output_dir / "run.json").write_text(json.dumps(finite_json(manifest), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(full[["seed", "method", "cv_delta_r2_full", "validation_delta_r2", "bootstrap_lower", "bootstrap_upper"]].to_string(index=False))
    print(f"Finished in {time.monotonic() - start:.1f}s; test not evaluated", flush=True)


if __name__ == "__main__":
    main()
