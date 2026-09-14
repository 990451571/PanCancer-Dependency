"""Download DepMap 24Q4 files and build compact TCGA-candidate matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_module_stage0 import GeneCanonicalizer


ARTICLE = {
    "release": "DepMap 24Q4 Public",
    "doi": "10.25452/figshare.plus.27993248.v1",
    "url": "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248",
}
FILES = {
    "Model.csv": (51065297, 645696, "675210d17675f3517b0ce39a3c274f16"),
    "CRISPRGeneEffect.csv": (51064667, 428678699, "6edf7ade09b9b34199210b559d4745d3"),
    "OmicsSomaticMutationsMatrixDamaging.csv": (51065747, 147655356, "cb20fdbe1cf3b9b0d8ed4f53e1f399b6"),
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv": (51065489, 506628654, "71794802b750ce77c422dad0720a40af"),
    "OmicsAbsoluteCNGene.csv": (51065303, 238808692, "16afffb33230de2fd13e15e753e28a9b"),
    "CRISPRInferredCommonEssentials.csv": (51064916, 20795, "545edef0d377f252db6e5e675201bd77"),
}
DRIVERS = {"VHL", "PBRM1", "SETD2", "BAP1", "MTOR"}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, target: Path) -> None:
    file_id, expected_size, expected_md5 = FILES[name]
    if target.exists():
        if target.stat().st_size != expected_size or md5(target) != expected_md5:
            raise_existing = f"Existing file failed checksum: {target}"
            raise ValueError(raise_existing)
        print(f"verified {name}", flush=True)
        return
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(
        f"https://ndownloader.figshare.com/files/{file_id}",
        headers={"Range": f"bytes={offset}-"} if offset else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        if mode == "wb":
            offset = 0
        with part.open(mode) as handle:
            downloaded = offset
            next_report = downloaded + 128 * 1024 * 1024
            while block := response.read(4 * 1024 * 1024):
                handle.write(block)
                downloaded += len(block)
                if downloaded >= next_report:
                    print(f"{name}: {downloaded / 1024**2:.0f}/{expected_size / 1024**2:.0f} MiB", flush=True)
                    next_report += 128 * 1024 * 1024
    if part.stat().st_size != expected_size or md5(part) != expected_md5:
        raise ValueError(f"Downloaded file failed size/checksum: {part}")
    os.replace(part, target)
    print(f"downloaded and verified {name}", flush=True)


def gene_symbol(column: object) -> str:
    return re.sub(r"\s+\([^)]+\)$", "", str(column).strip()).upper()


def load_subset(
    path: Path,
    requested: set[str],
    canonicalizer: GeneCanonicalizer,
    aggregation: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    header = [str(x) for x in pd.read_csv(path, nrows=0).columns]
    identifier = header[0]
    mapped = {column: canonicalizer.map(gene_symbol(column)) for column in header[1:]}
    usecols = [identifier, *[column for column in header[1:] if mapped[column] in requested]]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False).set_index(identifier)
    frame.index = frame.index.astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.columns = [mapped[column] for column in frame.columns]
    transposed = frame.T
    if aggregation == "max":
        frame = transposed.groupby(level=0).max().T
    else:
        frame = transposed.groupby(level=0).mean().T
    return frame.sort_index().sort_index(axis=1), {
        "raw_rows": int(len(frame)),
        "raw_gene_columns": len(header) - 1,
        "requested_gene_columns": len(usecols) - 1,
        "canonical_genes": int(frame.shape[1]),
    }


def candidate_genes(stage0: Path, minimum: int, maximum_frequency: float) -> pd.DataFrame:
    patients = pd.read_csv(stage0 / "patients.csv", index_col=0)
    train = patients.index[patients["split"] == "train"]
    mutation = pd.read_csv(stage0 / "mutation_binary.tsv.gz", sep="\t", index_col=0).loc[train]
    count = mutation.sum(axis=0)
    maximum = int(np.floor(maximum_frequency * len(train)))
    genes = sorted(g for g in count.index if minimum <= count[g] <= maximum and g not in DRIVERS)
    return pd.DataFrame(
        {
            "Gene": genes,
            "TCGA_train_mutation_count": count[genes].astype(int).to_numpy(),
            "TCGA_train_mutation_frequency": (count[genes] / len(train)).to_numpy(),
        }
    )


def standardized_difference(values: pd.Series, carrier: pd.Series) -> float:
    altered = values[carrier].dropna().to_numpy(dtype=float)
    reference = values[~carrier].dropna().to_numpy(dtype=float)
    if len(altered) < 3 or len(reference) < 5:
        return np.nan
    denominator = len(altered) + len(reference) - 2
    pooled = np.sqrt(
        ((len(altered) - 1) * altered.var(ddof=1) + (len(reference) - 1) * reference.var(ddof=1))
        / denominator
    )
    return float((altered.mean() - reference.mean()) / pooled) if pooled > 1e-8 else np.nan


def tcga_multiomics_effects(stage0: Path, genes: list[str]) -> pd.DataFrame:
    patients = pd.read_csv(stage0 / "patients.csv", index_col=0)
    train = patients.index[patients["split"] == "train"]
    contexts = patients.loc[train, "primary_context"]
    names = {
        "expression": "expression_normal_z.tsv.gz",
        "methylation": "methylation_promoter_delta_beta.tsv.gz",
        "copy_number": "cnv_continuous.tsv.gz",
    }
    mutation = pd.read_csv(stage0 / "mutation_binary.tsv.gz", sep="\t", index_col=0).loc[train, genes]
    effects = pd.DataFrame(index=genes)
    for modality, filename in names.items():
        values = pd.read_csv(stage0 / filename, sep="\t", index_col=0).loc[train, genes]
        residual = values.copy()
        for context, ids in contexts.groupby(contexts).groups.items():
            residual.loc[ids] = values.loc[ids].sub(values.loc[ids].mean(axis=0), axis=1)
        effects[f"TCGA_{modality}_context_adjusted_smd"] = [
            standardized_difference(residual[gene], mutation[gene].gt(0)) for gene in genes
        ]
    effects["TCGA_multiomics_support_count_abs_smd_ge_0_5"] = (
        effects.abs().ge(0.5).sum(axis=1).astype(int)
    )
    effects.index.name = "Gene"
    return effects.reset_index()


def first_present(frame: pd.DataFrame, names: list[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    return next((lower[name.lower()] for name in names if name.lower() in lower), None)


def annotate_cancer_scope(annotation: pd.DataFrame) -> pd.DataFrame:
    required = ["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]
    if any(column not in annotation for column in required):
        raise ValueError("DepMap Model.csv lacks required OncoTree annotation columns")
    lineage = annotation["OncotreeLineage"].fillna("").astype(str).str.strip().str.casefold()
    disease = annotation["OncotreePrimaryDisease"].fillna("").astype(str).str.strip().str.casefold()
    subtype = annotation["OncotreeSubtype"].fillna("").astype(str).str.strip().str.casefold()
    result = annotation.copy()
    result["kidney_lineage"] = lineage.eq("kidney")
    result["renal_cell_carcinoma"] = disease.eq("renal cell carcinoma")
    result["clear_cell_renal_cell_carcinoma"] = subtype.eq("renal clear cell carcinoma")
    if (result["clear_cell_renal_cell_carcinoma"] & ~result["renal_cell_carcinoma"]).any():
        raise ValueError("ccRCC annotation is not nested inside RCC")
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/depmap_24q4")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/depmap_bridge_24q4")
    parser.add_argument("--stage0", type=Path, default=root / "data/processed/context_module_stage0")
    parser.add_argument("--hgnc", type=Path, default=root / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--minimum-count", type=int, default=3)
    parser.add_argument("--maximum-frequency", type=float, default=0.05)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        for name in FILES:
            download(name, args.raw_dir / name)
    missing = [name for name in FILES if not (args.raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DepMap files: {missing}")
    for name, (_, expected_size, expected_md5) in FILES.items():
        path = args.raw_dir / name
        if path.stat().st_size != expected_size or md5(path) != expected_md5:
            raise ValueError(f"DepMap source checksum mismatch: {path}")

    canonicalizer = GeneCanonicalizer(args.hgnc)
    candidates = candidate_genes(args.stage0, args.minimum_count, args.maximum_frequency)
    candidates = candidates.merge(
        tcga_multiomics_effects(args.stage0, candidates["Gene"].tolist()), on="Gene", how="left", validate="one_to_one"
    )
    requested = set(candidates["Gene"]) | DRIVERS
    specifications = {
        "dependency": ("CRISPRGeneEffect.csv", "mean"),
        "mutation": ("OmicsSomaticMutationsMatrixDamaging.csv", "max"),
        "expression": ("OmicsExpressionProteinCodingGenesTPMLogp1.csv", "mean"),
        "copy_number": ("OmicsAbsoluteCNGene.csv", "mean"),
    }
    matrices, matrix_audit = {}, {}
    for modality, (filename, aggregation) in specifications.items():
        matrices[modality], matrix_audit[modality] = load_subset(
            args.raw_dir / filename, requested, canonicalizer, aggregation
        )
        if modality == "mutation":
            observed = matrices[modality].notna()
            matrices[modality] = matrices[modality].gt(0).astype(np.int8).where(observed)
            matrix_audit[modality]["binarized_as_nonzero"] = True
        print(f"loaded {modality}: {matrices[modality].shape}", flush=True)

    model = pd.read_csv(args.raw_dir / "Model.csv", low_memory=False)
    model_id = first_present(model, ["ModelID", "DepMap_ID", "ModelConditionID"])
    if model_id is None:
        raise ValueError(f"Could not find model identifier: {model.columns.tolist()}")
    model[model_id] = model[model_id].astype(str)
    model = model.drop_duplicates(model_id).set_index(model_id)
    common = sorted(set(model.index).intersection(*(set(frame.index) for frame in matrices.values())))
    if not common:
        raise ValueError("No shared DepMap model IDs across required modalities")
    all_genes = sorted(set.intersection(*(set(frame.columns) for frame in matrices.values())) - DRIVERS)
    for modality, frame in matrices.items():
        frame = frame.reindex(index=common, columns=all_genes if modality == "dependency" else sorted(set(all_genes) | (DRIVERS & set(frame.columns))))
        frame.index.name = "ModelID"
        frame.to_csv(args.output_dir / f"{modality}.tsv.gz", sep="\t", compression="gzip", float_format="%.6g")
        matrices[modality] = frame

    candidates = candidates.set_index("Gene")
    candidates["DepMap_all_modalities"] = candidates.index.isin(all_genes)
    for modality, frame in matrices.items():
        candidates[f"{modality}_available"] = candidates.index.isin(frame.columns)
        candidates[f"{modality}_observed_fraction"] = [
            float(frame[gene].notna().mean()) if gene in frame else 0.0 for gene in candidates.index
        ]
    lineage_columns = ["OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]
    annotation = annotate_cancer_scope(model.reindex(common))
    for driver in sorted(DRIVERS):
        annotation[f"{driver}_damaging"] = (
            matrices["mutation"][driver].fillna(0).gt(0).astype(int) if driver in matrices["mutation"] else 0
        )
    annotation.index.name = "ModelID"
    scope_columns = ["kidney_lineage", "renal_cell_carcinoma", "clear_cell_renal_cell_carcinoma"]
    keep = [*lineage_columns, *scope_columns, *[f"{g}_damaging" for g in sorted(DRIVERS)]]
    annotation[keep].to_csv(args.output_dir / "models.csv")

    cohorts = {
        "kidney_lineage": annotation.index[annotation["kidney_lineage"]],
        "renal_cell_carcinoma": annotation.index[annotation["renal_cell_carcinoma"]],
        "clear_cell_renal_cell_carcinoma": annotation.index[annotation["clear_cell_renal_cell_carcinoma"]],
    }
    dependency = matrices["dependency"]
    common_essential_raw = pd.read_csv(args.raw_dir / "CRISPRInferredCommonEssentials.csv")
    common_essentials = {
        canonicalizer.map(gene_symbol(value)) for value in common_essential_raw.iloc[:, 0].dropna()
    }
    candidates["DepMap_common_essential"] = candidates.index.isin(common_essentials)

    def add_dependency_summary(ids: pd.Index, prefix: str) -> None:
        subset = dependency.reindex(ids)
        candidates[f"{prefix}_dependency_n"] = [
            int(subset[gene].notna().sum()) if gene in subset else 0 for gene in candidates.index
        ]
        candidates[f"{prefix}_dependency_median"] = [
            float(subset[gene].median()) if gene in subset else np.nan for gene in candidates.index
        ]
        candidates[f"{prefix}_dependent_fraction_le_m0_5"] = [
            float(subset[gene].le(-0.5).sum() / subset[gene].notna().sum())
            if gene in subset and subset[gene].notna().sum() else np.nan
            for gene in candidates.index
        ]

    add_dependency_summary(dependency.index, "pan_cancer")
    add_dependency_summary(cohorts["renal_cell_carcinoma"], "rcc")
    add_dependency_summary(cohorts["clear_cell_renal_cell_carcinoma"], "ccrcc")
    non_kidney = annotation.index[~annotation["kidney_lineage"]]
    add_dependency_summary(non_kidney, "non_kidney")
    candidates["ccrcc_selectivity_delta_median"] = (
        candidates["ccrcc_dependency_median"] - candidates["non_kidney_dependency_median"]
    )
    rcc = cohorts["renal_cell_carcinoma"]
    for driver in ["VHL", "PBRM1"]:
        mutated = annotation.index[annotation.index.isin(rcc) & annotation[f"{driver}_damaging"].eq(1)]
        wild_type = annotation.index[annotation.index.isin(rcc) & annotation[f"{driver}_damaging"].eq(0)]
        add_dependency_summary(mutated, f"rcc_{driver.lower()}_damaging")
        add_dependency_summary(wild_type, f"rcc_{driver.lower()}_wt")
        candidates[f"rcc_{driver.lower()}_dependency_delta_median"] = (
            candidates[f"rcc_{driver.lower()}_damaging_dependency_median"]
            - candidates[f"rcc_{driver.lower()}_wt_dependency_median"]
        )
    candidates.reset_index().to_csv(args.output_dir / "gene_coverage.csv", index=False)
    prioritized = candidates.loc[
        candidates["DepMap_all_modalities"]
        & ~candidates["DepMap_common_essential"]
        & candidates["ccrcc_dependency_n"].ge(10)
        & candidates["ccrcc_dependency_median"].le(-0.5)
        & candidates["ccrcc_dependent_fraction_le_m0_5"].ge(0.5)
        & candidates["ccrcc_selectivity_delta_median"].le(-0.1)
        & candidates["TCGA_multiomics_support_count_abs_smd_ge_0_5"].ge(1)
    ].copy()
    prioritized = prioritized.sort_values(
        ["ccrcc_selectivity_delta_median", "ccrcc_dependency_median", "TCGA_train_mutation_count"]
    )
    prioritized.reset_index().to_csv(args.output_dir / "prioritized_candidates.csv", index=False)
    report = {
        **ARTICLE,
        "tcga_candidate_definition": f"training mutation count {args.minimum_count}..floor({args.maximum_frequency}*156); context genes excluded",
        "tcga_candidate_count": len(candidates),
        "candidate_all_modality_count": int(candidates["DepMap_all_modalities"].sum()),
        "candidate_common_essential_count": int(candidates["DepMap_common_essential"].sum()),
        "preliminary_prioritized_candidate_count": len(prioritized),
        "preliminary_filter": {
            "all_depmap_modalities": True,
            "exclude_common_essential": True,
            "ccrcc_dependency_n_min": 10,
            "ccrcc_dependency_median_max": -0.5,
            "ccrcc_fraction_le_minus_0_5_min": 0.5,
            "ccrcc_vs_non_kidney_median_delta_max": -0.1,
            "tcga_omics_abs_smd_ge_0_5_min_modalities": 1,
        },
        "shared_pan_cancer_models": len(common),
        "cohort_model_counts": {name: len(ids) for name, ids in cohorts.items()},
        "cohort_driver_counts": {
            name: {
                driver: int(annotation.loc[ids, f"{driver}_damaging"].sum()) for driver in sorted(DRIVERS)
            }
            for name, ids in cohorts.items()
        },
        "matrix_audit": matrix_audit,
        "raw_files": {
            name: {
                "figshare_file_id": values[0],
                "bytes": values[1],
                "md5": values[2],
                "verified": True,
            }
            for name, values in FILES.items()
        },
        "limitations": [
            "Cell lines are functional evidence and are not patients.",
            "Cancer scope uses exact OncoTree lineage, primary-disease, and subtype labels.",
            "Damaging mutation is release-defined and is not equivalent to the TCGA MC3 consequence filter.",
            "Damaging-mutation counts are binarized as any value greater than zero; missing values stay missing.",
            "Methylation is absent from this DepMap bridge and remains patient-side evidence.",
            "Same-gene CRISPR dependency is a functional endpoint, not proof of a therapeutic window.",
            "The preliminary filter is descriptive and is not a multiple-testing-controlled discovery result.",
        ],
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
