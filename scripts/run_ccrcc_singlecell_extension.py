"""Patient-level author-annotated ccRCC cell sources and RNA-mixture sensitivity.

No re-clustering is needed: annotations supplied by the original authors are
preserved, with source conflicts explicitly retained. Sparse input inspection
uses CPU; pseudobulk aggregation, normalization and PCR inference use CUDA.
"""
from __future__ import annotations

import argparse
import json
import importlib.metadata
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sparse
import torch

from run_depmap_baseline import observed_mean

from pcr_transfer_methods import (finish_run, load_model, overlap, predict_pcr,
                                  ranks_and_top, sha256, verify_manifest)


GROUPS = {
    "Tumor": "author_tumor", "Macro": "myeloid", "Macro_MKI67": "myeloid", "Mono": "myeloid",
    "Tcell": "lymphoid", "Tcell_CD8": "lymphoid", "NKcell": "lymphoid", "Plasma": "lymphoid",
    "Bcell": "lymphoid", "Mast": "mast", "Endo_PLVAP": "endothelial", "Endo_ACKR1": "endothelial",
    "AVR": "endothelial", "AEA-DVR": "endothelial", "GC": "endothelial", "vSMC": "stromal",
    "Peri": "stromal", "Mesangial": "stromal", "ua": "unassigned", "unknown": "unassigned",
    "PT-A": "normal_proximal_tubule", "PT-B": "normal_proximal_tubule", "PT-C": "normal_proximal_tubule",
    "TAL": "normal_other_epithelium", "tAL": "normal_other_epithelium", "IC-A": "normal_other_epithelium",
    "DCT": "normal_other_epithelium", "IC-PC": "normal_other_epithelium", "CNT": "normal_other_epithelium",
    "PC": "normal_other_epithelium", "DL": "normal_other_epithelium", "UC": "normal_other_epithelium",
    "IC-B": "normal_other_epithelium", "Podo": "normal_other_epithelium",
}
MARKERS = ["CA9", "EPCAM", "KRT8", "KRT18", "PTPRC", "CD3D", "CD68", "LST1", "PECAM1",
           "VWF", "ACTA2", "PDGFRB", "SLC34A1", "LRP2", "UMOD", "AQP2"]


def gpu_group_sum(counts, codes, group_n):
    """genes x groups sum, preserving raw UMIs and detection counts."""
    coo = counts.tocoo()
    index = torch.as_tensor(np.vstack([coo.col, coo.row]), dtype=torch.int64, device="cuda")
    values = torch.as_tensor(coo.data, dtype=torch.float64, device="cuda")
    matrix = torch.sparse_coo_tensor(index, values, (counts.shape[1], counts.shape[0]), check_invariants=True).coalesce()
    indicator = torch.nn.functional.one_hot(torch.as_tensor(codes, device="cuda"), group_n).double()
    sums = torch.sparse.mm(matrix, indicator).T
    detected = torch.sparse_coo_tensor(matrix.indices(), torch.ones_like(matrix.values()), matrix.shape, check_invariants=True).coalesce()
    detection = torch.sparse.mm(detected, indicator).T
    return sums.cpu().numpy(), detection.cpu().numpy()


def canonical_counts(adata):
    # Duplicate 10x gene symbols are collapsed by sum, not arbitrary suffixes.
    names = adata.var_names.astype(str).to_numpy()
    unique, codes = np.unique(names, return_inverse=True)
    mapping = sparse.csr_matrix((np.ones(len(names)), (np.arange(len(names)), codes)), shape=(len(names), len(unique)))
    counts = (adata.X.tocsr() @ mapping).tocsr()
    counts.eliminate_zeros()
    return counts, unique


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw/gse159115")
    parser.add_argument("--pcr-dir", type=Path, default=root / "outputs/pcr_patient_extension_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/ccrcc_singlecell_extension_v1")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动退回CPU")
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖结果: {output}")
    verify_manifest(args.pcr_dir)
    protocol_path = root / "configs/pcr_transfer_singlecell_protocol_20261010.json"
    protocol = json.loads(protocol_path.read_text())
    minimum_cells = protocol["singlecell"]["minimum_group_cells"]
    manifest = json.loads((args.raw_dir / "source_manifest.json").read_text())
    for name, info in manifest["files"].items():
        if sha256(args.raw_dir / name) != info["sha256"]:
            raise ValueError(f"原始来源哈希不匹配: {name}")
    samples = pd.read_csv(args.raw_dir / "samples.csv").set_index("sample_id")
    annotation = []
    for condition, name in [("author_ccrcc", "GSE159115_ccRCC_anno.csv.gz"), ("author_normal", "GSE159115_normal_anno.csv.gz")]:
        table = pd.read_csv(args.raw_dir / name, index_col=0)
        table["author_cohort"] = condition
        annotation.append(table)
    annotation = pd.concat(annotation)
    if annotation.index.has_duplicates or annotation.anno.isna().any():
        raise ValueError("作者注释条形码重复或标签缺失")
    if not set(annotation.anno).issubset(GROUPS):
        raise ValueError(f"未定义细胞类别: {set(annotation.anno)-set(GROUPS)}")
    output.mkdir(parents=True)
    sample_audit, qc_rows, cell_rows, aggregate_meta, aggregates, detection_aggregates = [], [], [], [], [], []
    author_only_meta, author_only_counts, author_only_detections = [], [], []
    all_genes = None
    for sample_number, (sample_id, table) in enumerate(annotation.groupby("sample", sort=True), 1):
        source = samples.loc[sample_id]
        author = table.author_cohort.unique()
        donor = table.patient.unique()
        if len(author) != 1 or len(donor) != 1 or "Patient " + donor[0] != source.individual:
            raise ValueError(f"患者或作者队列标识冲突: {sample_id}")
        if source.get("tumor subtype") == "Chromophobe RCC":
            raise ValueError("chRCC混入作者ccRCC/normal队列")
        files = list((args.raw_dir / "counts").glob(f"*_{sample_id}_*.h5"))
        if len(files) != 1:
            raise ValueError(f"原始矩阵匹配不唯一: {sample_id}")
        with warnings.catch_warnings():
            # Duplicates are explicitly summed below; do not suffix symbols.
            warnings.filterwarnings("ignore", message="Variable names are not unique.*")
            adata = sc.read_10x_h5(files[0])
        raw_ids = pd.Index(sample_id + "_" + adata.obs_names.astype(str))
        if raw_ids.has_duplicates or not table.index.isin(raw_ids).all():
            raise ValueError(f"注释-计数条形码匹配失败: {sample_id}")
        indices = raw_ids.get_indexer(table.index)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Variable names are not unique.*")
            adata = adata[indices].copy()
        counts, gene_names = canonical_counts(adata)
        if all_genes is None:
            all_genes = gene_names
        if not np.array_equal(all_genes, gene_names):
            raise ValueError("10x样本基因空间不一致")
        umi = np.asarray(counts.sum(axis=1)).ravel()
        detected = counts.getnnz(axis=1)
        mt = np.asarray(counts[:, np.char.startswith(gene_names.astype(str), "MT-")].sum(axis=1)).ravel()
        mt_fraction = np.divide(mt, umi, out=np.zeros_like(mt), where=umi > 0)
        umi_difference = np.max(np.abs(umi - table.total_UMI.to_numpy()))
        if umi_difference != 0:
            raise ValueError(f"作者UMI与原始计数不一致: {sample_id}, max={umi_difference}")
        author_qc = ~table.cellfilt.astype(bool).to_numpy() & ~table.doublet.astype(bool).to_numpy()
        keep = author_qc & (detected >= 200) & (mt_fraction <= .20)
        concordant = ((author[0] == "author_ccrcc" and source.tissue == "Tumor") or
                      (author[0] == "author_normal" and str(source.tissue).startswith("Benign")))
        sample_audit.append({"sample_id": sample_id, "donor": donor[0], "author_cohort": author[0],
                             "geo_tissue": source.tissue, "geo_subtype": source.get("tumor subtype"),
                             "geo_author_concordant": concordant, "annotated_cells": len(table), "retained_cells": int(keep.sum()),
                             "excluded_cells": int((~keep).sum()), "raw_umi_max_difference": float(umi_difference),
                             "raw_detected_gene_max_difference": int(np.max(np.abs(detected-table.no_genes.to_numpy()))),
                             "mitochondrial_fraction_max_difference": float(np.max(np.abs(mt_fraction-table.pct_MT.to_numpy())))})
        print(f"【单细胞 {sample_number}/13】{sample_id}｜作者{author[0]}｜GEO {source.tissue}｜保留{keep.sum()}/{len(table)}｜标签一致{concordant}", flush=True)
        enriched = table.copy()
        enriched["broad_group"] = enriched.anno.map(GROUPS)
        enriched["qc_keep"] = keep
        enriched["geo_author_concordant"] = concordant
        enriched["raw_umi"] = umi
        enriched["raw_detected_genes"] = detected
        enriched["raw_mt_fraction"] = mt_fraction
        cell_rows.append(enriched)
        for label, label_table in enriched.groupby("anno"):
            qc_rows.append({"sample_id": sample_id, "donor": donor[0], "author_cohort": author[0], "anno": label,
                            "annotated_n": len(label_table), "retained_n": int(label_table.qc_keep.sum()),
                            "retained_fraction": float(label_table.qc_keep.mean())})
        # Author-QC-only sensitivity, before imposing our mitochondrial cutoff.
        author_retained = enriched[author_qc]
        author_c = counts[author_qc]
        author_groups = sorted(author_retained.broad_group.unique())
        codes = pd.Categorical(author_retained.broad_group, categories=author_groups).codes.astype(np.int64)
        author_sums, author_detects = gpu_group_sum(author_c, codes, len(author_groups))
        for j, group_name in enumerate(author_groups):
            author_only_counts.append(author_sums[j])
            author_only_detections.append(author_detects[j])
            author_only_meta.append({"sample_id": sample_id, "donor": donor[0], "author_cohort": author[0],
                                     "group": group_name, "cell_n": int((codes == j).sum()),
                                     "geo_author_concordant": concordant})
        retained = enriched[keep]
        c = counts[keep]
        if len(retained) == 0:
            continue
        # Aggregate sample-level all/nonmalignant groups plus author labels and broad groups in one sparse CUDA pass.
        descriptors = [("all_cells", np.ones(len(retained), dtype=bool)),
                       ("nonmalignant", ~retained.anno.eq("Tumor").to_numpy())]
        descriptors += [(group, retained.broad_group.eq(group).to_numpy()) for group in sorted(retained.broad_group.unique())]
        descriptors += [("label:" + label, retained.anno.eq(label).to_numpy()) for label in sorted(retained.anno.unique())]
        # Groups overlap, so each independent membership is aggregated explicitly.
        membership = np.column_stack([mask for _, mask in descriptors]).astype(np.float64)
        coo = c.tocoo()
        matrix = torch.sparse_coo_tensor(torch.as_tensor(np.vstack([coo.col, coo.row]), dtype=torch.int64, device="cuda"),
                                         torch.as_tensor(coo.data, dtype=torch.float64, device="cuda"),
                                         (c.shape[1], c.shape[0]), check_invariants=True).coalesce()
        membership_gpu = torch.as_tensor(membership, device="cuda", dtype=torch.float64)
        sums = torch.sparse.mm(matrix, membership_gpu).T.cpu().numpy()
        detections = torch.sparse.mm(torch.sparse_coo_tensor(matrix.indices(), torch.ones_like(matrix.values()), matrix.shape, check_invariants=True).coalesce(),
                                     membership_gpu).T.cpu().numpy()
        for i, (group, mask) in enumerate(descriptors):
            if mask.sum() == 0:
                continue
            aggregates.append(sums[i])
            detection_aggregates.append(detections[i])
            aggregate_meta.append({"sample_id": sample_id, "donor": donor[0], "author_cohort": author[0],
                                   "group": group, "cell_n": int(mask.sum()), "geo_author_concordant": concordant})
    pd.DataFrame(sample_audit).to_csv(output / "sample_source_audit.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(output / "qc_by_sample_label.csv", index=False)
    pd.concat(cell_rows).to_csv(output / "cell_metadata_qc.csv.gz")
    counts = np.stack(aggregates)
    detects = np.stack(detection_aggregates)
    aggregate_meta = pd.DataFrame(aggregate_meta)
    # Combine multiple samples of the same donor/cohort (normal cortex/medulla) before summaries.
    donor_keys = ["donor", "author_cohort", "group", "geo_author_concordant"]
    combined_counts, combined_detection, combined_meta = [], [], []
    for keys, group in aggregate_meta.groupby(donor_keys, sort=True):
        index = group.index.to_numpy()
        combined_counts.append(counts[index].sum(axis=0))
        combined_detection.append(detects[index].sum(axis=0))
        combined_meta.append({**dict(zip(donor_keys, keys)), "cell_n": int(group.cell_n.sum()),
                              "sample_ids": "|".join(group.sample_id)})
    counts, detects = np.stack(combined_counts), np.stack(combined_detection)
    donors = pd.DataFrame(combined_meta)
    count_gpu = torch.as_tensor(counts, device="cuda", dtype=torch.float64)
    cpm = (count_gpu / count_gpu.sum(dim=1, keepdim=True) * 1e6).cpu().numpy()
    detection = detects / donors.cell_n.to_numpy()[:, None]
    donors.to_csv(output / "pseudobulk_groups.csv", index=False)
    np.savez_compressed(output / "pseudobulk_counts.npz", genes=all_genes, counts=counts, detection_count=detects,
                        cpm=cpm, group_row=np.arange(len(donors)))
    discovery = pd.read_csv(args.pcr_dir / "pcr_discovery_top20.csv")
    historical_path = root / "results/historical/final_evidence_synthesis_v2/candidate_evidence_matrix.csv"
    historical = pd.read_csv(historical_path)
    candidate_genes = list(dict.fromkeys([*discovery.Gene, *historical.Gene]))
    feature_path = args.pcr_dir / "pcr_candidate_input_features.csv"
    features = pd.read_csv(feature_path)
    predictor_genes = list(dict.fromkeys(features.feature_gene))
    evidence_genes = list(dict.fromkeys(candidate_genes + predictor_genes))
    flags = pd.DataFrame({"Gene": candidate_genes})
    flags["pcr_discovery_rank"] = flags.Gene.map(discovery.set_index("Gene").pcr_discovery_rank)
    flags["historical_frozen_candidate"] = flags.Gene.isin(historical.Gene)
    flags.to_csv(output / "candidate_sets.csv", index=False)
    gene_index = {g: i for i, g in enumerate(all_genes)}
    evidence_rows, marker_rows = [], []
    for i, meta in donors.iterrows():
        if meta.cell_n < minimum_cells:
            continue
        for gene in list(dict.fromkeys(evidence_genes + MARKERS)):
            j = gene_index.get(gene)
            row = {**meta.to_dict(), "Gene": gene, "gene_present": j is not None,
                   "pseudobulk_cpm": float(cpm[i, j]) if j is not None else np.nan,
                   "detection_fraction": float(detection[i, j]) if j is not None else np.nan}
            if gene in evidence_genes:
                evidence_rows.append(row)
            if gene in MARKERS:
                marker_rows.append(row)
    evidence = pd.DataFrame(evidence_rows)
    evidence[evidence.Gene.isin(candidate_genes)].to_csv(output / "candidate_donor_cell_sources.csv.gz", index=False)
    evidence[evidence.Gene.isin(predictor_genes)].to_csv(output / "predictor_feature_donor_cell_sources.csv.gz", index=False)
    pd.DataFrame(marker_rows).to_csv(output / "annotation_marker_audit.csv", index=False)
    summary = []
    # Use equal donors, never pooled cell-level p-values. Small groups remain descriptive.
    for scope in ["author_cohort", "geo_author_concordant_only"]:
        frame = evidence if scope == "author_cohort" else evidence[evidence.geo_author_concordant]
        frame = frame[~frame.group.str.startswith("label:") & ~frame.group.isin(["all_cells", "nonmalignant"])]
        for (condition, group, gene), values in frame.groupby(["author_cohort", "group", "Gene"], sort=True):
            summary.append({"scope": scope, "author_cohort": condition, "group": group, "Gene": gene,
                            "donor_n": values.donor.nunique(), "median_cpm": values.pseudobulk_cpm.median(),
                            "min_cpm": values.pseudobulk_cpm.min(), "max_cpm": values.pseudobulk_cpm.max(),
                            "mean_detection_fraction": values.detection_fraction.mean(),
                            "gene_present": bool(values.gene_present.all())})
    pd.DataFrame(summary).to_csv(output / "candidate_cell_source_summary.csv", index=False)
    # Compare renal exposure with and without our additional QC at equal-donor level.
    author_only_meta = pd.DataFrame(author_only_meta)
    author_counts, author_detection = np.stack(author_only_counts), np.stack(author_only_detections)
    qc_evidence = []
    for keys, group in author_only_meta.groupby(donor_keys, sort=True):
        indices = group.index.to_numpy()
        count_sum = author_counts[indices].sum(axis=0)
        detection_sum = author_detection[indices].sum(axis=0)
        cell_n = int(group.cell_n.sum())
        if cell_n < minimum_cells:
            continue
        normalized = torch.as_tensor(count_sum, device="cuda", dtype=torch.float64)
        normalized = (normalized / normalized.sum() * 1e6).cpu().numpy()
        for gene in candidate_genes:
            j = gene_index.get(gene)
            qc_evidence.append({**dict(zip(donor_keys, keys)), "cell_n": cell_n, "Gene": gene,
                                "pseudobulk_cpm": normalized[j] if j is not None else np.nan,
                                "detection_fraction": detection_sum[j]/cell_n if j is not None else np.nan,
                                "qc_scope": "author_qc_only"})
    extra_qc = evidence[evidence.Gene.isin(candidate_genes) & ~evidence.group.str.startswith("label:") & ~evidence.group.isin(["all_cells", "nonmalignant"])].copy()
    extra_qc["qc_scope"] = "author_qc_plus_mt20_detected200"
    pd.concat([pd.DataFrame(qc_evidence), extra_qc], ignore_index=True).to_csv(output / "candidate_qc_sensitivity.csv.gz", index=False)
    print("【阶段2/3】患者×细胞类别表达汇总与来源审计完成", flush=True)
    model, model_genes = load_model(args.pcr_dir / "pcr_model.npz")
    with np.load(args.pcr_dir / "ranking_universe.npz", allow_pickle=False) as a:
        universe = a["universe"].copy()
    model_gene_indices = np.array([gene_index.get(g, -1) for g in model_genes])
    covered = model_gene_indices >= 0
    sc_universe = universe & covered
    # A shared restricted universe is used for every mixture scenario.
    scenarios, scenario_meta = [], []
    for donor, group in donors[donors.author_cohort.eq("author_ccrcc")].groupby("donor", sort=True):
        lookup = {row.group: index for index, row in group.iterrows()}
        if not {"author_tumor", "all_cells", "nonmalignant"}.issubset(lookup):
            continue
        if min(donors.loc[lookup[g], "cell_n"] for g in ["author_tumor", "nonmalignant"]) < minimum_cells:
            continue
        tumor, nonmalignant = cpm[lookup["author_tumor"]], cpm[lookup["nonmalignant"]]
        profiles = {"tumor_only": tumor, "observed_all_cells": cpm[lookup["all_cells"]]}
        for fraction in [.25, .50, .75]:
            profiles[f"nonmalignant_rna_{int(fraction*100)}pct"] = (1-fraction)*tumor + fraction*nonmalignant
        for scenario, profile in profiles.items():
            aligned = np.full(len(model_genes), np.nan)
            aligned[covered] = np.log2(profile[model_gene_indices[covered]] + 1)
            scenarios.append(aligned)
            scenario_meta.append({"donor": donor, "scenario": scenario, "geo_author_concordant": bool(group.geo_author_concordant.all())})
    scenario_meta = pd.DataFrame(scenario_meta)
    if scenario_meta.empty:
        raise ValueError("没有足够的同患者肿瘤/非肿瘤细胞用于混合分析")
    expression = np.stack(scenarios)
    predictions, scenario_rankings, agreement_rows, sc_diagnostics = {}, [], [], []
    reference_mean = model["mean"].cpu().numpy()
    for scenario, group in scenario_meta.groupby("scenario", sort=True):
        indices = group.index.to_numpy()
        # Unlabeled, scenario-specific equal-donor mean alignment. Not fitted on TCGA/Test.
        mapped = np.maximum(expression[indices] + reference_mean - observed_mean(expression[indices], axis=0), 0)
        prediction, diagnostic = predict_pcr(model, mapped, diagnostics=True)
        rank, top = ranks_and_top(prediction, sc_universe)
        predictions[scenario] = {"donors": group.donor.tolist(), "top": top, "prediction": prediction}
        sc_diagnostics.append(pd.DataFrame({"donor": group.donor.to_numpy(), "scenario": scenario, **diagnostic}))
        for i, donor in enumerate(group.donor):
            for k, column in enumerate(top[i], 1):
                scenario_rankings.append({"donor": donor, "scenario": scenario, "rank": k, "Gene": model_genes[column],
                                          "predicted_residual": prediction[i, column], "status": "exploratory_expression_transfer"})
    tumor = predictions["tumor_only"]
    for scenario, item in predictions.items():
        if item["donors"] != tumor["donors"]:
            raise ValueError("混合场景患者顺序不一致")
        if scenario == "tumor_only":
            continue
        for donor, agreement in zip(item["donors"], overlap(tumor["top"], item["top"])):
            agreement_rows.append({"donor": donor, "scenario": scenario, "reference": "tumor_only", "top10_overlap": agreement,
                                   "geo_author_concordant": bool(scenario_meta[scenario_meta.donor.eq(donor)].geo_author_concordant.all())})
    pd.DataFrame(scenario_rankings).to_csv(output / "mixture_patient_top10.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(output / "mixture_sensitivity.csv", index=False)
    pd.concat(sc_diagnostics).to_csv(output / "mixture_domain_diagnostics.csv", index=False)
    audit = pd.DataFrame(sample_audit)
    finish_run(output, {"status": "complete_descriptive_singlecell_and_exploratory_mixture_analysis",
                        "protocol_sha256": sha256(protocol_path),
                        "source_audit_amendment_sha256": sha256(root / "configs/singlecell_source_audit_amendment_20261010.json"),
                        "input_sha256": {"source_manifest": sha256(args.raw_dir / "source_manifest.json"),
                                          "pcr_run": sha256(args.pcr_dir / "run.json"), "historical_candidates": sha256(historical_path)},
                        "script_sha256": {p.name: sha256(p) for p in [Path(__file__), root / "scripts/pcr_transfer_methods.py"]},
                        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "scanpy": importlib.metadata.version("scanpy"),
                        "numpy": np.__version__, "elapsed_seconds": time.monotonic()-started,
                        "sample_n": len(audit), "donor_n": audit.donor.nunique(),
                        "author_ccrcc_donor_n": audit[audit.author_cohort.eq("author_ccrcc")].donor.nunique(),
                        "author_normal_donor_n": audit[audit.author_cohort.eq("author_normal")].donor.nunique(),
                        "annotated_cell_n": int(audit.annotated_cells.sum()), "retained_cell_n": int(audit.retained_cells.sum()),
                        "geo_author_discordant_sample_n": int((~audit.geo_author_concordant).sum()),
                        "mixture_donor_n": len(tumor["donors"]), "mixture_ranking_gene_n": int(sc_universe.sum()),
                        "model_feature_coverage": float(covered.mean()), "candidate_gene_n": len(candidate_genes),
                        "limitations": ["Author-cohort and GEO tissue labels conflict; preserved rather than silently corrected",
                                        "Tumor labels are author-annotated, not independently CNV-confirmed",
                                        "20% mitochondrial QC may disproportionately exclude renal epithelium",
                                        "Normal source includes adjacent renal tissue and repeated samples from donors",
                                        "scRNA CPM and DepMap TPM are not directly interchangeable; mixture mapping is exploratory",
                                        "Scenario-specific mean alignment removes mean differences and does not calibrate predictions",
                                        "No dependency labels, no functional accuracy, toxicity or therapeutic window inferred"]})
    print(f"【阶段3/3 完成】细胞来源与混合敏感性已保存 {output}｜耗时{time.monotonic()-started:.1f}秒", flush=True)


if __name__ == "__main__":
    main()
