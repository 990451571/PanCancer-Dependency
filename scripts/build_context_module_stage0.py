"""Build leakage-safe patient-level inputs for context-specific module search.

The script keeps the patient as the statistical unit.  It intersects the four
local TCGA-KIRC assays, canonicalizes gene symbols, preserves signed molecular
measurements, assigns molecular contexts, and creates a deterministic split.

No outcome is used to define a context or a split.  The generated files are the
shared input contract for static, greedy, bandit, and DDQN module searches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FUNCTIONAL_EFFECTS = {
    "missense_mutation",
    "nonsense_mutation",
    "frame_shift_del",
    "frame_shift_ins",
    "splice_site",
    "in_frame_del",
    "in_frame_ins",
    "translation_start_site",
    "nonstop_mutation",
    "large deletion",
}

PRIMARY_DRIVERS = ("VHL", "PBRM1")
SECONDARY_DRIVERS = ("SETD2", "BAP1", "MTOR")
ALL_DRIVERS = PRIMARY_DRIVERS + SECONDARY_DRIVERS


def patient_id(barcode: object) -> str:
    parts = str(barcode).strip().split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(barcode).strip()


def sample_type(barcode: object) -> str | None:
    parts = str(barcode).strip().split("-")
    return parts[3][:2] if len(parts) >= 4 and len(parts[3]) >= 2 else None


def clean_gene(value: object) -> str:
    text = str(value).strip().upper()
    if "|" in text:
        text = text.split("|", 1)[0]
    return text


def read_header(path: Path) -> list[str]:
    return [str(x) for x in pd.read_csv(path, sep="\t", nrows=0).columns]


def select_primary_samples(columns: Iterable[str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    selected: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for sample in sorted(str(x) for x in columns if sample_type(x) == "01"):
        patient = patient_id(sample)
        if patient not in selected:
            selected[patient] = sample
        else:
            duplicates.append(
                {"patient_id": patient, "kept_sample": selected[patient], "excluded_sample": sample}
            )
    return selected, duplicates


def normal_samples(columns: Iterable[str]) -> list[str]:
    return sorted(str(x) for x in columns if sample_type(x) == "11")


def split_pipe_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [clean_gene(x) for x in str(value).split("|") if str(x).strip()]


class GeneCanonicalizer:
    def __init__(self, hgnc_path: Path):
        hgnc = pd.read_csv(
            hgnc_path,
            sep="\t",
            dtype=str,
            usecols=["symbol", "alias_symbol", "prev_symbol"],
        )
        self.approved = {clean_gene(x) for x in hgnc["symbol"].dropna()}
        targets: dict[str, set[str]] = defaultdict(set)
        for row in hgnc.itertuples(index=False):
            approved = clean_gene(row.symbol)
            for alias in split_pipe_field(row.alias_symbol) + split_pipe_field(row.prev_symbol):
                if alias != approved:
                    targets[alias].add(approved)
        self.alias_targets = targets
        self.ambiguous = {key: value for key, value in targets.items() if len(value) > 1}

    def map(self, value: object) -> str:
        gene = clean_gene(value)
        if gene in self.approved:
            return gene
        targets = self.alias_targets.get(gene, set())
        return next(iter(targets)) if len(targets) == 1 else gene


def load_canonical_ppi(
    ppi_path: Path, canonicalizer: GeneCanonicalizer
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ppi_path, sep=r"\s+", names=["source_raw", "target_raw"], dtype=str)
    raw["source"] = raw["source_raw"].map(canonicalizer.map)
    raw["target"] = raw["target_raw"].map(canonicalizer.map)
    mapped = raw.loc[raw["source"] != raw["target"], ["source", "target"]].copy()
    endpoints = np.sort(mapped[["source", "target"]].to_numpy(dtype=str), axis=1)
    edges = pd.DataFrame(endpoints, columns=["source", "target"]).drop_duplicates()
    genes = sorted(set(edges["source"]) | set(edges["target"]))
    degree = pd.concat([edges["source"], edges["target"]]).value_counts().reindex(genes, fill_value=0)
    manifest = pd.DataFrame(
        {
            "Gene": genes,
            "Degree": degree.to_numpy(dtype=int),
            "HGNCApprovedSymbol": [gene in canonicalizer.approved for gene in genes],
        }
    )
    return genes, edges.sort_values(["source", "target"]), manifest


def mutation_data(
    path: Path, canonicalizer: GeneCanonicalizer
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep="\t", usecols=["sample", "gene", "effect"], dtype=str)
    raw = raw.loc[raw["sample"].map(sample_type) == "01"].copy()
    raw["patient_id"] = raw["sample"].map(patient_id)
    universe = set(raw["patient_id"])
    raw["effect_key"] = raw["effect"].str.strip().str.lower()
    functional = raw.loc[raw["effect_key"].isin(FUNCTIONAL_EFFECTS)].copy()
    functional["Gene"] = functional["gene"].map(canonicalizer.map)
    functional = functional.drop_duplicates(["patient_id", "Gene"])
    audit = (
        functional.groupby("effect_key", as_index=False)
        .agg(variant_rows=("Gene", "size"), patients=("patient_id", "nunique"), genes=("Gene", "nunique"))
        .sort_values("variant_rows", ascending=False)
    )
    matrix = pd.crosstab(functional["patient_id"], functional["Gene"]).clip(upper=1).astype(np.int8)
    return universe, matrix, audit


def collapse_duplicate_rows(frame: pd.DataFrame, genes: pd.Series, method: str = "mean") -> pd.DataFrame:
    work = frame.copy()
    work.index = genes.to_numpy(dtype=str)
    if method == "mean":
        return work.groupby(level=0, sort=True).mean()
    if method != "max_abs":
        raise ValueError(f"Unknown collapse method: {method}")
    rows: list[np.ndarray] = []
    names: list[str] = []
    for gene, group in work.groupby(level=0, sort=True):
        values = group.to_numpy(dtype=np.float32)
        if len(values) == 1:
            result = values[0]
        else:
            finite = np.isfinite(values)
            scores = np.where(finite, np.abs(values), -np.inf)
            idx = scores.argmax(axis=0)
            result = values[idx, np.arange(values.shape[1])]
            result[~finite.any(axis=0)] = np.nan
        names.append(gene)
        rows.append(result)
    return pd.DataFrame(rows, index=names, columns=work.columns)


def expression_data(
    path: Path,
    selected: dict[str, str],
    genes: list[str],
    canonicalizer: GeneCanonicalizer,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    header = read_header(path)
    normals = normal_samples(header[1:])
    usecols = [header[0], *selected.values(), *normals]
    raw = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    raw_genes = raw.iloc[:, 0].map(canonicalizer.map)
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    normal_mean = values[normals].mean(axis=1)
    normal_sd = values[normals].std(axis=1, ddof=1)
    tumor_samples = list(selected.values())
    delta = values[tumor_samples].sub(normal_mean, axis=0)
    z = delta.div(normal_sd.clip(lower=0.25), axis=0)
    delta.columns = [patient_id(x) for x in delta.columns]
    z.columns = delta.columns
    delta = collapse_duplicate_rows(delta, raw_genes, "mean").reindex(genes).T
    z = collapse_duplicate_rows(z, raw_genes, "mean").reindex(genes).T
    stats = {
        "raw_gene_rows": int(len(raw)),
        "normal_samples": int(len(normals)),
        "normal_sd_floor_log2": 0.25,
        "delta_missing_fraction": float(delta.isna().to_numpy().mean()),
        "z_missing_fraction": float(z.isna().to_numpy().mean()),
    }
    return delta.astype(np.float32), z.astype(np.float32), stats


def methylation_data(
    path: Path,
    promoter_map_path: Path,
    selected: dict[str, str],
    genes: list[str],
    canonicalizer: GeneCanonicalizer,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float | int]]:
    header = read_header(path)
    normals = normal_samples(header[1:])
    tumor_samples = list(selected.values())
    usecols = [header[0], *tumor_samples, *normals]

    mapping = pd.read_csv(promoter_map_path, dtype=str, usecols=["Probe", "Gene", "Group"])
    mapping["Probe"] = mapping["Probe"].str.strip()
    mapping["Gene"] = mapping["Gene"].map(canonicalizer.map)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    mapping = mapping.loc[mapping["Gene"].isin(gene_to_idx), ["Probe", "Gene"]].drop_duplicates()

    sums = np.zeros((len(genes), len(tumor_samples)), dtype=np.float64)
    counts = np.zeros((len(genes), len(tumor_samples)), dtype=np.int32)
    probe_counts = np.zeros(len(genes), dtype=np.int32)
    matched_probes: set[str] = set()
    raw_probe_count = 0

    reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk_no, chunk in enumerate(reader, start=1):
        chunk = chunk.reset_index(drop=True)
        chunk = chunk.rename(columns={header[0]: "Probe"})
        chunk["Probe"] = chunk["Probe"].astype(str).str.strip()
        raw_probe_count += len(chunk)
        joined = chunk[["Probe"]].reset_index(names="_row").merge(mapping, on="Probe", how="inner")
        if joined.empty:
            continue
        numeric = chunk[tumor_samples + normals].apply(pd.to_numeric, errors="coerce")
        normal_mean = numeric[normals].mean(axis=1).to_numpy(dtype=np.float32)
        delta = numeric[tumor_samples].to_numpy(dtype=np.float32) - normal_mean[:, None]
        rows = joined["_row"].to_numpy(dtype=int)
        gene_idx = joined["Gene"].map(gene_to_idx).to_numpy(dtype=int)
        selected_delta = delta[rows]
        finite = np.isfinite(selected_delta)
        np.add.at(sums, gene_idx, np.where(finite, selected_delta, 0.0))
        np.add.at(counts, gene_idx, finite.astype(np.int32))
        for idx, count in joined.groupby("Gene")["Probe"].nunique().items():
            probe_counts[gene_to_idx[idx]] += int(count)
        matched_probes.update(joined["Probe"].unique())
        if chunk_no % 20 == 0:
            print(f"methylation chunks={chunk_no}, probes={raw_probe_count}, matched={len(matched_probes)}", flush=True)

    result = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    matrix = pd.DataFrame(result.T, index=[patient_id(x) for x in tumor_samples], columns=genes)
    stats = {
        "raw_probes": int(raw_probe_count),
        "mapped_promoter_probes": int(len(matched_probes)),
        "mapped_probe_gene_pairs": int(len(mapping)),
        "normal_samples": int(len(normals)),
        "missing_fraction": float(matrix.isna().to_numpy().mean()),
        "aggregation": "mean probe delta-beta per gene; probe-level normal mean subtracted first",
    }
    return matrix.astype(np.float32), pd.Series(probe_counts, index=genes), stats


def cnv_data(
    path: Path,
    selected: dict[str, str],
    genes: list[str],
    canonicalizer: GeneCanonicalizer,
    method: str,
) -> pd.DataFrame:
    header = read_header(path)
    usecols = [header[0], *selected.values()]
    raw = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    raw_genes = raw.iloc[:, 0].map(canonicalizer.map)
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.columns = [patient_id(x) for x in values.columns]
    return collapse_duplicate_rows(values, raw_genes, method).reindex(genes).T.astype(np.float32)


def deterministic_split(contexts: pd.Series, seed: int) -> pd.Series:
    result = pd.Series(index=contexts.index, dtype="string")
    for context, patients in contexts.groupby(contexts).groups.items():
        ordered = sorted(
            patients,
            key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
        )
        n = len(ordered)
        n_train = int(round(0.60 * n))
        n_val = int(round(0.20 * n))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        result.loc[ordered[:n_train]] = "train"
        result.loc[ordered[n_train : n_train + n_val]] = "validation"
        result.loc[ordered[n_train + n_val :]] = "test"
    return result


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_matrix(frame: pd.DataFrame, path: Path) -> None:
    frame = frame.copy()
    frame.index.name = "patient_id"
    frame.to_csv(path, sep="\t", compression="gzip", float_format="%.6g")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--hgnc",
        type=Path,
        default=root / "outputs/reassessment_20260911/hgnc_complete_set.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/processed/context_module_stage0",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--methylation-chunksize", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "mutation": root / "data/raw/KIRC_mc3.txt",
        "expression": root / "data/raw/HiSeqV2",
        "methylation": root / "data/raw/HumanMethylation450",
        "promoter_map": root / "data/raw/450k_probe_gene_promoter_map.csv",
        "cnv_thresholded": root / "data/raw/cnv_kirc/KIRC_GISTIC2_thresholded.by_genes.normalized_unique_v2.tsv.gz",
        "cnv_continuous": root / "data/raw/cnv_kirc/KIRC_GISTIC2_continuous.by_genes.normalized_unique_v2.tsv.gz",
        "ppi": root / "data/HPRD.txt",
        "hgnc": args.hgnc.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    canonicalizer = GeneCanonicalizer(paths["hgnc"])
    genes, edges, gene_manifest = load_canonical_ppi(paths["ppi"], canonicalizer)
    mutation_patients, mutation_all, mutation_audit = mutation_data(paths["mutation"], canonicalizer)

    headers = {name: read_header(paths[name]) for name in ["expression", "methylation", "cnv_thresholded"]}
    selections: dict[str, dict[str, str]] = {}
    duplicate_records: list[dict[str, str]] = []
    for modality, header in headers.items():
        selected, duplicates = select_primary_samples(header[1:])
        selections[modality] = selected
        duplicate_records.extend({"modality": modality, **row} for row in duplicates)

    raw_selection_counts = {name: len(values) for name, values in selections.items()}

    shared = sorted(
        mutation_patients
        & set(selections["expression"])
        & set(selections["methylation"])
        & set(selections["cnv_thresholded"])
    )
    if len(shared) < 100:
        raise ValueError(f"Only {len(shared)} patients have all four modalities; expected at least 100.")
    selections = {name: {p: values[p] for p in shared} for name, values in selections.items()}

    mutation = mutation_all.reindex(index=shared, columns=genes, fill_value=0).astype(np.int8)
    print(f"shared patients={len(shared)}, canonical PPI genes={len(genes)}", flush=True)
    expression_delta, expression_z, expression_stats = expression_data(
        paths["expression"], selections["expression"], genes, canonicalizer
    )
    print("expression matrix complete", flush=True)
    cnv_continuous = cnv_data(
        paths["cnv_continuous"], selections["cnv_thresholded"], genes, canonicalizer, "mean"
    )
    cnv_thresholded = cnv_data(
        paths["cnv_thresholded"], selections["cnv_thresholded"], genes, canonicalizer, "max_abs"
    )
    print("CNV matrices complete", flush=True)
    methylation_delta, promoter_counts, methylation_stats = methylation_data(
        paths["methylation"],
        paths["promoter_map"],
        selections["methylation"],
        genes,
        canonicalizer,
        args.methylation_chunksize,
    )
    print("methylation matrix complete", flush=True)

    patient_table = pd.DataFrame(index=shared)
    patient_table.index.name = "patient_id"
    for driver in ALL_DRIVERS:
        patient_table[f"{driver}_mut"] = mutation[driver].astype(int) if driver in mutation else 0
    patient_table["primary_context"] = [
        f"VHL_{'MUT' if v else 'WT'}__PBRM1_{'MUT' if p else 'WT'}"
        for v, p in zip(patient_table["VHL_mut"], patient_table["PBRM1_mut"])
    ]
    patient_table["split"] = deterministic_split(patient_table["primary_context"], args.seed)
    patient_table["functional_mutation_burden"] = mutation.sum(axis=1).astype(int)
    patient_table["expression_sample"] = pd.Series(selections["expression"])
    patient_table["methylation_sample"] = pd.Series(selections["methylation"])
    patient_table["cnv_sample"] = pd.Series(selections["cnv_thresholded"])

    deep_cnv = cnv_thresholded.abs().ge(2).astype(np.int8)
    genomic_event = (mutation.astype(bool) | deep_cnv.astype(bool)).astype(np.int8)
    if not ((genomic_event >= mutation).all().all() and (genomic_event >= deep_cnv).all().all()):
        raise AssertionError("The genomic-event union does not dominate both source event matrices.")

    train_patients = patient_table.index[patient_table["split"] == "train"]
    gene_manifest["PromoterProbeCount"] = promoter_counts.reindex(genes).to_numpy(dtype=int)
    gene_manifest["MutationCountTrain"] = mutation.loc[train_patients].sum(axis=0).to_numpy(dtype=int)
    gene_manifest["DeepCNVCountTrain"] = deep_cnv.loc[train_patients].sum(axis=0).to_numpy(dtype=int)
    gene_manifest["GenomicEventCountTrain"] = genomic_event.loc[train_patients].sum(axis=0).to_numpy(dtype=int)
    gene_manifest["GenomicEventFrequencyTrain"] = gene_manifest["GenomicEventCountTrain"] / len(train_patients)

    generated = {
        "patients": output / "patients.csv",
        "genes": output / "gene_manifest.csv",
        "ppi": output / "ppi_edges.tsv.gz",
        "mutation": output / "mutation_binary.tsv.gz",
        "cnv_thresholded": output / "cnv_thresholded.tsv.gz",
        "cnv_continuous": output / "cnv_continuous.tsv.gz",
        "deep_cnv": output / "deep_cnv_binary.tsv.gz",
        "genomic_event": output / "genomic_event_binary.tsv.gz",
        "expression_delta": output / "expression_delta_log2.tsv.gz",
        "expression_z": output / "expression_normal_z.tsv.gz",
        "methylation_delta": output / "methylation_promoter_delta_beta.tsv.gz",
        "mutation_effect_audit": output / "mutation_effect_audit.csv",
        "duplicates": output / "duplicate_primary_samples.csv",
    }
    patient_table.to_csv(generated["patients"])
    gene_manifest.to_csv(generated["genes"], index=False)
    edges.to_csv(generated["ppi"], sep="\t", index=False, compression="gzip")
    write_matrix(mutation, generated["mutation"])
    write_matrix(cnv_thresholded, generated["cnv_thresholded"])
    write_matrix(cnv_continuous, generated["cnv_continuous"])
    write_matrix(deep_cnv, generated["deep_cnv"])
    write_matrix(genomic_event, generated["genomic_event"])
    write_matrix(expression_delta, generated["expression_delta"])
    write_matrix(expression_z, generated["expression_z"])
    write_matrix(methylation_delta, generated["methylation_delta"])
    mutation_audit.to_csv(generated["mutation_effect_audit"], index=False)
    pd.DataFrame(duplicate_records).to_csv(generated["duplicates"], index=False)

    context_counts = (
        patient_table.groupby(["primary_context", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    driver_counts = {driver: int(patient_table[f"{driver}_mut"].sum()) for driver in ALL_DRIVERS}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "TCGA participant; one lexicographically first primary-tumor sample per assay",
        "patient_count": len(shared),
        "gene_count": len(genes),
        "ppi_edge_count": len(edges),
        "sample_universe_counts": {
            "mutation": len(mutation_patients),
            **raw_selection_counts,
            "four_way_intersection": len(shared),
        },
        "driver_mutation_counts": driver_counts,
        "context_split_counts": context_counts,
        "split_seed": args.seed,
        "split_policy": "within primary context: 60% train, 20% validation, 20% untouched test",
        "primary_context_definition": "cross of functional VHL and PBRM1 mutation status",
        "secondary_context_covariates": list(SECONDARY_DRIVERS),
        "functional_effects": sorted(FUNCTIONAL_EFFECTS),
        "expression": expression_stats,
        "methylation": methylation_stats,
        "cnv": {
            "deep_event_definition": "absolute GISTIC threshold >= 2",
            "continuous_missing_fraction": float(cnv_continuous.isna().to_numpy().mean()),
            "thresholded_missing_fraction": float(cnv_thresholded.isna().to_numpy().mean()),
        },
        "candidate_event_note": "mutation and deep CNV are retained separately; genomic_event is their union",
        "inputs": {name: {"path": str(path), "sha256": hash_file(path)} for name, path in paths.items()},
        "outputs": {name: {"path": str(path), "sha256": hash_file(path)} for name, path in generated.items()},
    }
    manifest_path = output / "stage0_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **{k: manifest[k] for k in ["patient_count", "gene_count", "driver_mutation_counts", "context_split_counts"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
