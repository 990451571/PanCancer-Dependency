"""Freeze external pan-cancer controls before evaluating their Sanger outcomes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline
from prepare_sanger_phase_b import (TARGETS as PHASE_B_TARGETS, apply_mapping,
                                    fit_mapping, read_sanger_expression)
from prepare_sanger_validation import ARCHIVE_URL, HttpRangeReader, read_member_once


SCALED_MEMBER = "Project_score_archive_data/Release1/EssentialityMatrices/03_scaledBayesianFactors.tsv"
BINARY_MEMBER = "Project_score_archive_data/Release1/EssentialityMatrices/04_binaryDepScores.tsv"
SCALED_MIRROR = "https://orcs.thebiogrid.org/uploads/processed/5cdc4514becb7/03_scaledBayesianFactors.tsv"
EXPECTED = {
    "scaled_bf": {"bytes": 104_924_382,
                  "sha256": "b789a0b28243584ac596f165c2ff5b8ad25a37169b4ecb0355cbbaf9fc711dd2"},
    "binary": {"bytes": 11_815_383,
               "sha256": "7dea7c024137cb44dfb100c9ae93e1cd9384ebb747f0fb60327ddc171650f97a"},
}
DRIVERS = baseline.DRIVERS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_uncompressed_gzip(path, expected):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"固定来源校验失败：{path}")


def save_gzip(data, path, expected):
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("下载内容与固定 Project Score Release1 成员不一致")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(data)
    temporary.rename(path)


def obtain_labels(scaled_path, binary_path):
    if scaled_path.exists():
        print("【来源复用】Project Score Release1 scaled BF", flush=True)
        verify_uncompressed_gzip(scaled_path, EXPECTED["scaled_bf"])
    else:
        print("【来源下载】scaled BF 镜像文件，并用官方归档成员哈希核验", flush=True)
        request = urllib.request.Request(SCALED_MIRROR, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        save_gzip(data, scaled_path, EXPECTED["scaled_bf"])
    if binary_path.exists():
        print("【来源复用】Project Score Release1 binary dependency", flush=True)
        verify_uncompressed_gzip(binary_path, EXPECTED["binary"])
    else:
        print("【来源下载】官方归档中的 binary dependency 成员", flush=True)
        reader = HttpRangeReader(ARCHIVE_URL)
        with zipfile.ZipFile(reader) as archive:
            data = read_member_once(reader, archive.getinfo(BINARY_MEMBER))
        save_gzip(data, binary_path, EXPECTED["binary"])


def label_header(path):
    with gzip.open(path, "rt", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))[1:]


def read_label_subset(path, model_names, genes, binary=False):
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        missing = sorted(set(model_names) - set(header))
        if missing:
            raise ValueError(f"功能矩阵缺少冻结模型：{missing}")
        selected = [header.index(name) for name in model_names]
        gene_index = {gene: index for index, gene in enumerate(genes)}
        # Preserve source precision so the pre-consumed 769-P result can be
        # replayed exactly rather than accepted under a loose tolerance.
        matrix = np.full((len(model_names), len(genes)), np.nan, dtype=float)
        seen = set()
        source_rows = 0
        for row in reader:
            source_rows += 1
            gene = row[0]
            if gene not in gene_index:
                continue
            if gene in seen:
                raise ValueError(f"功能矩阵基因重复：{gene}")
            seen.add(gene)
            values = [float(row[index]) if row[index] != "" else np.nan for index in selected]
            matrix[:, gene_index[gene]] = values
    if binary:
        finite = matrix[np.isfinite(matrix)]
        if not np.isin(finite, [0, 1]).all():
            raise ValueError("二元依赖矩阵包含非0/1值")
    return matrix, {"source_gene_n": source_rows, "exact_overlap_gene_n": len(seen)}


def read_broad_features(raw_dir, baseline_dir, model_ids, genes):
    mapping = pd.read_csv(baseline_dir / "gene_mapping.csv.gz")
    gene_index = {gene: index for index, gene in enumerate(genes)}

    expression_map = mapping[(mapping.modality == "expression") & mapping.included_gene.astype(bool)]
    expression_lookup = dict(zip(expression_map.raw_column, expression_map.canonical_symbol))
    expression = np.full((len(model_ids), len(genes)), np.nan, dtype=np.float32)
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    path = raw_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        selected = [(column, expression_lookup[name]) for column, name in enumerate(header)
                    if name in expression_lookup and expression_lookup[name] in gene_index]
        for row in reader:
            if row[0] not in model_index:
                continue
            sums, counts = {}, {}
            for column, gene in selected:
                if row[column] == "":
                    continue
                sums[gene] = sums.get(gene, 0.0) + float(row[column])
                counts[gene] = counts.get(gene, 0) + 1
            target = expression[model_index[row[0]]]
            for gene, total in sums.items():
                target[gene_index[gene]] = total / counts[gene]

    mutation_map = mapping[(mapping.modality == "mutation") & mapping.included_gene.astype(bool) &
                           mapping.canonical_symbol.isin(DRIVERS)]
    mutation_lookup = dict(zip(mutation_map.raw_column, mutation_map.canonical_symbol))
    mutations = np.full((len(model_ids), len(DRIVERS)), np.nan, dtype=np.float32)
    driver_index = {gene: index for index, gene in enumerate(DRIVERS)}
    path = raw_dir / "OmicsSomaticMutationsMatrixDamaging.csv"
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        selected = [(column, mutation_lookup[name]) for column, name in enumerate(header)
                    if name in mutation_lookup]
        for row in reader:
            if row[0] not in model_index:
                continue
            target = mutations[model_index[row[0]]]
            for column, gene in selected:
                if row[column] != "":
                    value = float(row[column])
                    index = driver_index[gene]
                    target[index] = value if np.isnan(target[index]) else max(target[index], value)
    mutations = np.where(np.isnan(mutations), np.nan, (mutations > 0).astype(np.float32))
    if np.isnan(expression).all(axis=1).any() or not np.isfinite(mutations).all():
        raise ValueError("冻结对照的 Broad 表达或驱动突变特征不完整")
    return expression, mutations


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("/mnt/e/projects/rl-genrisk-main/data/raw/depmap_24q4"))
    parser.add_argument("--external-dir", type=Path, default=root / "data/raw/external_sanger_20260914")
    parser.add_argument("--audit-dir", type=Path, default=root / "outputs/external_dependency_audit_v1")
    parser.add_argument("--phase-b-preparation", type=Path,
                        default=root / "outputs/sanger_phase_b_preparation_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/sanger_external_controls_frozen_v1")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device(args.device)
    print("【阶段 1/5】校验数据并读取纯 Sanger 矩阵表头", flush=True)
    models, genes, matrices, common_essential = baseline.load_data(args.baseline_dir)
    scaled_path = args.external_dir / "project_score_release1_scaled_bf.tsv.gz"
    binary_path = args.external_dir / "project_score_release1_binary.tsv.gz"
    obtain_labels(scaled_path, binary_path)
    screen_names = set(label_header(binary_path))

    print("【阶段 2/5】仅按身份、隔离和特征覆盖冻结泛癌对照", flush=True)
    annotation = pd.read_csv(args.external_dir / "model_list_20260814.csv")
    unique_name = annotation.groupby("model_name").model_id.agg(list)
    screen_sids = {unique_name[name][0] for name in screen_names
                   if name in unique_name and len(unique_name[name]) == 1}
    audit = pd.read_csv(args.audit_dir / "model_coverage.csv.gz")
    cohort = audit[
        audit.model_id.isin(screen_sids) & audit.identity_resolved.eq(True) &
        audit.in_current_cohort.eq(False) & audit.shares_current_patient.eq(False) &
        audit.broad_expression_row_available.eq(True) & audit.broad_mutation_row_available.eq(True)
    ].copy()
    raw_models = pd.read_csv(args.raw_dir / "Model.csv", low_memory=False).set_index("ModelID")
    valid = cohort.matched_broad_id.map(
        lambda model_id: model_id in raw_models.index and
        str(raw_models.at[model_id, "OncotreePrimaryDisease"]).strip().casefold()
        not in ("", "nan", "unknown", "non-cancerous"))
    cohort = cohort[valid].sort_values(["tissue", "model_name"], kind="stable").reset_index(drop=True)
    if len(cohort) != 66 or cohort.matched_broad_id.nunique() != 66:
        raise ValueError(f"冻结对照数量不是审计预期的66：{len(cohort)}")
    cohort["BroadPatientID"] = cohort.matched_broad_id.map(raw_models.PatientID)
    cohort["BroadLineage"] = cohort.matched_broad_id.map(raw_models.OncotreeLineage)
    current_patients = set(models.PatientID)
    if set(cohort.matched_broad_id) & set(models.index) or set(cohort.BroadPatientID) & current_patients:
        raise ValueError("泛癌外部对照与当前队列存在模型或患者重叠")
    print(f"【队列冻结】模型 {len(cohort)}｜Broad癌系 {cohort.BroadLineage.nunique()}｜患者重叠 0", flush=True)

    print("【阶段 3/5】提取 Broad 直接特征并重建冻结的 Sanger 表达映射", flush=True)
    model_ids = cohort.matched_broad_id.tolist()
    direct_expression, mutations = read_broad_features(args.raw_dir, args.baseline_dir, model_ids, genes)
    requested_sids = set(models.SangerModelID.dropna()) | set(cohort.model_id) | set(PHASE_B_TARGETS)
    sanger_sids, sanger_expression, expression_audit = read_sanger_expression(
        args.external_dir / "rnaseq_fpkm_20191101.csv.gz", requested_sids, genes)
    sid_index = {sid: index for index, sid in enumerate(sanger_sids)}
    calibration_rows = [index for index, row in models.reset_index().iterrows()
                        if row.OncotreeLineage != "Kidney" and row.SangerModelID in sid_index]
    calibration = models.iloc[calibration_rows]
    sanger_calibration = np.vstack([sanger_expression[sid_index[sid]] for sid in calibration.SangerModelID])
    preparation = json.loads((args.phase_b_preparation / "run.json").read_text())
    method = preparation["mapping"]["selected"]
    shift = fit_mapping(sanger_calibration, matrices["expression"][calibration_rows], method)
    phase_b_archive = np.load(args.phase_b_preparation / "mapped_target_expression.npz", allow_pickle=False)
    check_names = phase_b_archive["model_names"].astype(str)
    check_values = phase_b_archive["expression"]
    reconstructed = apply_mapping(
        np.vstack([sanger_expression[sid_index[sid]] for sid in PHASE_B_TARGETS]), shift, method)
    reconstructed = np.vstack([reconstructed[list(PHASE_B_TARGETS).index(
        next(sid for sid, name in PHASE_B_TARGETS.items() if name == target_name))]
                               for target_name in check_names])
    if not np.allclose(reconstructed, check_values, equal_nan=True, atol=1e-6, rtol=0):
        raise ValueError("重建的表达映射与阶段 B 冻结目标不一致")
    mapped_available = cohort.model_id.isin(sid_index)
    mapped_expression = np.full_like(direct_expression, np.nan)
    for row in np.flatnonzero(mapped_available.to_numpy()):
        mapped_expression[row] = apply_mapping(
            sanger_expression[sid_index[cohort.loc[row, "model_id"]]][None, :], shift, method)[0]
    cohort["sanger_expression_mapped"] = mapped_available
    print(f"【双输入子集】{int(mapped_available.sum())}/{len(cohort)} 个模型同时具有2019 Sanger表达", flush=True)

    print("【阶段 4/5】冻结完成后提取对应功能标签，不计算预测", flush=True)
    continuous, continuous_audit = read_label_subset(
        scaled_path, cohort.model_name.tolist(), genes, binary=False)
    binary, binary_audit = read_label_subset(
        binary_path, cohort.model_name.tolist(), genes, binary=True)
    overlap = np.isfinite(continuous).any(axis=0) & np.isfinite(binary).any(axis=0)
    primary = overlap & ~common_essential
    if int(overlap.sum()) != 14350 or int(primary.sum()) != 13120:
        raise ValueError("泛癌对照评价基因集合偏离已冻结口径")

    print("【阶段 5/5】保存冻结队列、输入和标签", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    keep = ["model_id", "model_name", "matched_broad_id", "BroadPatientID", "BroadLineage",
            "tissue", "cancer_type", "cancer_type_detail", "strict_ccrcc", "sanger_expression_mapped"]
    if "strict_ccrcc" not in cohort:
        cohort["strict_ccrcc"] = cohort.cancer_type_detail.eq("Clear Cell Renal Cell Carcinoma")
    cohort[keep].to_csv(temporary / "frozen_models.csv", index=False)
    np.savez_compressed(temporary / "matrices.npz", model_ids=np.asarray(model_ids, dtype=str),
                        model_names=cohort.model_name.astype(str).to_numpy(dtype=str), genes=genes.astype(str),
                        direct_expression=direct_expression, mapped_expression=mapped_expression,
                        driver_mutation=mutations, sanger_scaled_bf=continuous,
                        sanger_binary_dependency=binary, primary_gene_mask=primary)
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "external_controls_frozen_before_prediction_evaluation",
        "device": baseline.TORCH.cuda.get_device_name(0),
        "selection": {"functional_values_used": False,
                      "rules": ["Project Score Release1 screen column",
                                "resolved Broad identity outside current 873-model cohort",
                                "no patient overlap with current cohort",
                                "Broad expression and mutation rows available",
                                "annotated cancer model"],
                      "model_n": len(cohort), "patient_n": cohort.BroadPatientID.nunique(),
                      "lineage_n": cohort.BroadLineage.nunique(),
                      "mapped_expression_model_n": int(mapped_available.sum())},
        "mapping": {"method": method, "calibration_model_n": len(calibration),
                    "phase_b_reconstruction_check": "passed", "target_outcomes_used": False},
        "genes": {"baseline_n": len(genes), "exact_label_overlap_n": int(overlap.sum()),
                  "primary_non_common_essential_n": int(primary.sum()),
                  "continuous_label_dtype": "float64"},
        "sources": {"scaled_bf": {**EXPECTED["scaled_bf"], "local_sha256": sha256(scaled_path),
                                     "canonical_member": SCALED_MEMBER, "retrieval_url": SCALED_MIRROR},
                    "binary": {**EXPECTED["binary"], "local_sha256": sha256(binary_path),
                               "canonical_member": BINARY_MEMBER, "retrieval_url": ARCHIVE_URL},
                    "expression": expression_audit,
                    "label_parse": {"continuous": continuous_audit, "binary": binary_audit}},
        "source_sha256": {"baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "external_audit": sha256(args.audit_dir / "model_coverage.csv.gz"),
                          "phase_b_preparation": sha256(args.phase_b_preparation / "run.json"),
                          "script": sha256(Path(__file__))},
        "limitations": [
            "Only 17 of 66 controls have both direct Broad and mapped Sanger expression.",
            "The cohort is availability-selected and is not a random sample of cancer cell lines.",
            "Common-essential annotation uses the whole Broad release rather than training-only outcomes.",
            "Some Project Score names lack unique current metadata and are excluded rather than fuzzy-matched.",
            "No prediction metric or target ranking was calculated during freezing."]}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    print(f"【冻结完成】模型 {len(cohort)}｜双输入 {int(mapped_available.sum())}｜主评价基因 {int(primary.sum())}", flush=True)
    print(f"【未执行】没有生成预测或计算功能指标｜结果 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
