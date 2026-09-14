"""Build a broad-gene DepMap 24Q4 baseline from verified local raw files.

No TCGA data, model fitting, outcome-based target selection or imputation.
Writes aligned float32 arrays with explicit IDs; missing values stay missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_context_module_stage0 import GeneCanonicalizer
from build_depmap_bridge import ARTICLE, FILES, annotate_cancer_scope, gene_symbol


MODALITIES = {
    "dependency": ("CRISPRGeneEffect.csv", "mean"),
    "expression": ("OmicsExpressionProteinCodingGenesTPMLogp1.csv", "mean"),
    "copy_number": ("OmicsAbsoluteCNGene.csv", "mean"),
    "mutation": ("OmicsSomaticMutationsMatrixDamaging.csv", "max"),
}


def hashes(path: Path) -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def build(raw: Path, hgnc: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    sources = {}
    for name, (file_id, size, expected_md5) in FILES.items():
        path = raw / name
        if path.stat().st_size != size:
            raise ValueError(f"Source size mismatch: {path}")
        digest = hashes(path)
        if digest["md5"] != expected_md5:
            raise ValueError(f"Source checksum mismatch: {path}")
        sources[name] = {"path": str(path), "bytes": size,
                         "figshare_file_id": file_id, **digest}
        print(f"verified {name}", flush=True)

    canonicalizer = GeneCanonicalizer(hgnc)
    mappings, identifiers, row_ids = {}, {}, {}
    mapping_rows = []
    for modality, (filename, _) in MODALITIES.items():
        header = pd.read_csv(raw / filename, nrows=0).columns.tolist()
        identifiers[modality] = header[0]
        mapping = {}
        for column in header[1:]:
            original = gene_symbol(column)
            mapped = canonicalizer.map(original)
            approved = mapped in canonicalizer.approved
            reason = "approved" if approved else (
                "ambiguous_alias" if original in canonicalizer.ambiguous else "unresolved_symbol"
            )
            mapping_rows.append((modality, column, mapped, reason))
            if approved:
                mapping[column] = mapped
        mappings[modality] = mapping
        ids = pd.read_csv(raw / filename, usecols=[header[0]], dtype=str).iloc[:, 0]
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"Missing or duplicate model IDs in {filename}")
        row_ids[modality] = set(ids)

    common_genes = sorted(set.intersection(*(set(m.values()) for m in mappings.values())))
    if not common_genes:
        raise ValueError("No approved genes shared across modalities")
    genes = set(common_genes)
    coverage = pd.DataFrame(index=sorted(set.union(*(set(m.values()) for m in mappings.values()))))
    coverage.index.name = "Gene"
    coverage["all_modalities"] = coverage.index.isin(genes)
    for modality, mapping in mappings.items():
        coverage[f"{modality}_available"] = coverage.index.isin(set(mapping.values()))
    essentials = pd.read_csv(raw / "CRISPRInferredCommonEssentials.csv").iloc[:, 0].dropna()
    essential_genes = {canonicalizer.map(gene_symbol(g)) for g in essentials}
    coverage["DepMap_common_essential"] = coverage.index.isin(essential_genes)

    model = pd.read_csv(raw / "Model.csv", dtype=str)
    if model.ModelID.isna().any() or model.ModelID.duplicated().any():
        raise ValueError("Missing or duplicate ModelID in Model.csv")
    model = annotate_cancer_scope(model.set_index("ModelID"))
    if (model.renal_cell_carcinoma & ~model.kidney_lineage).any():
        raise ValueError("RCC is not nested inside Kidney")
    columns = ["PatientID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease",
               "OncotreeSubtype", "OncotreeCode", "SangerModelID", "COSMICID",
               "EngineeredModel", "kidney_lineage", "renal_cell_carcinoma",
               "clear_cell_renal_cell_carcinoma"]
    model_audit = model[columns].reindex(sorted(set(model.index).union(*row_ids.values())))
    model_audit.index.name = "ModelID"
    reasons = pd.Series("", index=model_audit.index)

    def exclude(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask] = reasons.loc[mask] + reason + ";"

    exclude(pd.Series(~model_audit.index.isin(model.index), index=model_audit.index), "missing_annotation")
    disease = model_audit.OncotreePrimaryDisease.fillna("").str.strip().str.casefold()
    lineage = model_audit.OncotreeLineage.fillna("").str.strip()
    exclude(disease.eq("non-cancerous"), "non_cancerous")
    exclude(disease.isin(["", "unknown"]), "missing_disease")
    exclude(lineage.str.casefold().isin(["", "unknown"]), "missing_lineage")
    for modality, ids in row_ids.items():
        present = pd.Series(model_audit.index.isin(ids), index=model_audit.index)
        model_audit[f"{modality}_present"] = present
        exclude(~present, f"missing_{modality}")
    included = model_audit.index[reasons.eq("")]
    if len(included) == 0:
        raise ValueError("No annotated cancer models with all modalities")

    matrices, matrix_audit = {}, {}
    for modality, (filename, aggregation) in MODALITIES.items():
        mapping = mappings[modality]
        usecols = [identifiers[modality], *[col for col, gene in mapping.items() if gene in genes]]
        frame = pd.read_csv(raw / filename, usecols=usecols,
                            dtype={col: np.float32 for col in usecols[1:]})
        frame = frame.set_index(identifiers[modality]).loc[included]
        frame.columns = [mapping[col] for col in frame.columns]
        frame = frame.T.groupby(level=0).agg(aggregation).T.reindex(columns=common_genes)
        values = frame.to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError(f"Infinite values in {modality}")
        if modality == "mutation":
            values = np.where(np.isnan(values), np.nan, (values > 0).astype(np.float32))
        if np.isnan(values).all(axis=1).any():
            raise ValueError(f"Included model has no observed {modality} values")
        matrices[modality] = values
        coverage.loc[common_genes, f"{modality}_observed_fraction"] = np.isfinite(values).mean(axis=0)
        matrix_audit[modality] = {
            "raw_model_n": len(row_ids[modality]), "selected_raw_columns": len(usecols) - 1,
            "merged_columns": len(usecols) - 1 - len(common_genes),
            "duplicate_gene_aggregation": aggregation, "shape": list(values.shape),
            "missing_values": int(np.isnan(values).sum()),
        }
        print(f"loaded {modality}: {values.shape}, missing={matrix_audit[modality]['missing_values']}", flush=True)
        del frame

    model_audit["exclusion_reason"] = reasons.str.rstrip(";")
    model_audit["included"] = reasons.eq("")
    annotation = model_audit.loc[included, columns]
    scope = ["kidney_lineage", "renal_cell_carcinoma", "clear_cell_renal_cell_carcinoma"]
    mapping_frame = pd.DataFrame(mapping_rows, columns=["modality", "raw_column", "canonical_symbol", "mapping_status"])
    mapping_frame["included_gene"] = mapping_frame.canonical_symbol.isin(genes) & mapping_frame.mapping_status.eq("approved")
    patients = annotation.PatientID.dropna()
    report = {
        **ARTICLE, "status": "data_baseline_only_no_model_fitted", "raw_files": sources,
        "hgnc": {"path": str(hgnc), **hashes(hgnc)},
        "script_sha256": hashes(Path(__file__))["sha256"],
        "helper_sha256": {name: hashes(Path(__file__).with_name(name))["sha256"]
                           for name in ("build_depmap_bridge.py", "build_context_module_stage0.py")},
        "model_n": len(included), "gene_n": len(common_genes),
        "common_essential_gene_n": int(coverage.loc[common_genes, "DepMap_common_essential"].sum()),
        "scope_counts": {name: int(annotation[name].sum()) for name in scope},
        "lineage_counts": annotation.OncotreeLineage.value_counts().sort_index().to_dict(),
        "missing_patient_id_n": int(annotation.PatientID.isna().sum()),
        "patients_with_multiple_models_n": int((patients.value_counts() > 1).sum()),
        "matrix_audit": matrix_audit, "tcga_read": False,
        "rules": {
            "genes": "Intersection of approved HGNC symbols in four raw modality headers; no TCGA candidate or dependency-effect filter; drivers retained",
            "models": "All four modalities present; known lineage and disease; exclude Non-Cancerous; exact OncoTree subtype labels",
            "missing": "Preserved as NaN; no global imputation or outcome completeness filter",
            "essentiality": "Release-provided common-essential annotation retained, not used for filtering or residualization",
            "mutation": "Damaging count > 0; missing remains NaN",
        },
        "limitations": [
            "Cross-modality gene and model intersections impose availability selection.",
            "Common-essential annotation is release-wide, not estimated independently of held-out lineages.",
            "Missing outcomes require masked fitting/evaluation; missing features require training-fold imputation.",
            "PatientID grouping must be respected in future within-lineage model splits.",
            "Historical ccRCC models have already been explored and are not an untouched confirmation set.",
            "Engineered models are flagged but not automatically excluded; annotation is not biological authentication.",
            "This NPZ is not an input for the historical TSV-based LOLO runner.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        np.savez_compressed(temporary / "matrices.npz", model_ids=np.asarray(included, dtype=str),
                            genes=np.asarray(common_genes, dtype=str), **matrices)
        annotation.to_csv(temporary / "models.csv")
        model_audit.to_csv(temporary / "model_audit.csv")
        coverage.to_csv(temporary / "gene_coverage.csv")
        mapping_frame.to_csv(temporary / "gene_mapping.csv.gz", index=False, compression="gzip")
        report["output_sha256"] = {p.name: hashes(p)["sha256"] for p in sorted(temporary.iterdir())}
        (temporary / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps({key: report[key] for key in ("model_n", "gene_n", "common_essential_gene_n", "scope_counts", "patients_with_multiple_models_n")}), flush=True)
    print(f"Saved {output}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--hgnc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    args = parser.parse_args()
    build(args.raw_dir.resolve(), args.hgnc.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
