"""Orthogonally audit frozen candidates with DRIVE RNAi and GTEx v11.

Candidate selection is fixed before either source is read. DRIVE-only DEMETER2
scores provide a different perturbation technology, although some cell models
overlap the current DepMap cohort. GTEx normal-tissue expression is a target
exposure signal and is not interpreted as toxicity by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SOURCES = {
    "gtex_v11_median_tpm": {
        "url": "https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
        "filename": "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
        "sha256": "eb29994ed9175aa0a575aa78a2dc67a489241e11df8fc076ba8061aca2613196",
        "bytes": 10129906,
    },
    "drive_gene_scores": {
        "url": "https://ndownloader.figshare.com/files/11489693",
        "filename": "D2_DRIVE_gene_dep_scores.csv",
        "sha256": "3f863c296188be1aa8a491ef5489b135a9bfd65266f05d0690225d20fc38254b",
        "bytes": 58652780,
    },
    "drive_gene_score_sds": {
        "url": "https://ndownloader.figshare.com/files/11489690",
        "filename": "D2_DRIVE_gene_dep_score_SDs.csv",
        "sha256": "fd2afeff063a59c2de36b4a468726f9e3b8aadef18242ee5a31d455294d1dfa5",
        "bytes": 57099157,
    },
    "drive_cell_line_parameters": {
        "url": "https://ndownloader.figshare.com/files/11489684",
        "filename": "D2_DRIVE_CL_data.csv",
        "sha256": "c41be38026e570317978f89cebc28c14f803ffbf4b034e0865fa217cc71b5479",
        "bytes": 43938,
    },
    "demeter2_sample_info": {
        "url": "https://ndownloader.figshare.com/files/11489717",
        "filename": "sample_info.csv",
        "sha256": "8dcbd6da1e4858e7fa5b3910e8cf3feb1045a0c31a2e5d252eb3f2afe86dd036",
        "bytes": 76352,
    },
    "demeter2_readme": {
        "url": "https://ndownloader.figshare.com/files/13515380",
        "filename": "DEMETER2_README.txt",
        "sha256": "d0f6a52e68faedfa4190333317af5e82928ddce41a8df683a84c316375e87ca2",
        "bytes": 9476,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: dict, raw_dir: Path) -> tuple[Path, bool]:
    path = raw_dir / source["filename"]
    reused = path.exists()
    if not reused:
        temporary = path.with_suffix(path.suffix + ".tmp")
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "PanCancer-Dependency-public-audit/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
        temporary.replace(path)
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError(f"固定来源发生变化或下载损坏：{path}")
    return path, reused


def configure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("本分析的 bootstrap 要求 CUDA，不自动回退 CPU")
    torch.set_default_device("cuda")


def bootstrap_mean_ci(values, draws, generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [np.nan, np.nan, np.nan]
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    means = tensor[indices].mean(dim=1)
    quantiles = torch.quantile(
        means, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return quantiles.cpu().numpy().tolist()


def bootstrap_difference_ci(left, right, draws, generator):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left, right = left[np.isfinite(left)], right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return [np.nan, np.nan, np.nan]
    a = torch.as_tensor(left, dtype=torch.float64, device="cuda")
    b = torch.as_tensor(right, dtype=torch.float64, device="cuda")
    ia = torch.randint(len(a), (draws, len(a)), generator=generator, device="cuda")
    ib = torch.randint(len(b), (draws, len(b)), generator=generator, device="cuda")
    delta = a[ia].mean(dim=1) - b[ib].mean(dim=1)
    quantiles = torch.quantile(
        delta, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return quantiles.cpu().numpy().tolist()


def verify_candidates(frozen_dir: Path, external_dir: Path):
    frozen_path = frozen_dir / "candidates.csv"
    frozen_run = json.loads((frozen_dir / "run.json").read_text())
    if sha256(frozen_path) != frozen_run["output_sha256"]["candidates.csv"]:
        raise ValueError("冻结候选文件哈希变化")
    candidates = pd.read_csv(frozen_path).sort_values("discovery_rank")
    if candidates.Gene.tolist() != frozen_run["selection"]["genes"]:
        raise ValueError("冻结候选顺序变化")
    external_run = json.loads((external_dir / "run.json").read_text())
    summary_path = external_dir / "candidate_external_summary.csv"
    if sha256(summary_path) != external_run["output_sha256"][summary_path.name]:
        raise ValueError("既有外部候选汇总哈希变化")
    return candidates, pd.read_csv(summary_path)


def load_drive(paths: dict, model_path: Path, baseline_models_path: Path):
    scores = pd.read_csv(paths["drive_gene_scores"])
    score_sds = pd.read_csv(paths["drive_gene_score_sds"])
    first = scores.columns[0]
    if (scores.shape != (7975, 398) or score_sds.shape != scores.shape or
            score_sds.columns.tolist() != scores.columns.tolist()):
        raise ValueError("DRIVE 分数矩阵维度或列顺序变化")
    for frame in (scores, score_sds):
        parsed = frame[first].str.extract(r"^(.*) \(([^()]*)\)$")
        frame["Gene"], frame["Entrez"] = parsed[0], parsed[1]
    if scores.Gene.isna().any() or scores.Gene.duplicated().any():
        raise ValueError("DRIVE 基因标识无法唯一解析")

    parameters = pd.read_csv(paths["drive_cell_line_parameters"]).rename(
        columns={"Unnamed: 0": "CCLE_ID"})
    sample = pd.read_csv(paths["demeter2_sample_info"])
    drive_sample = sample[sample.in_DRIVE.eq(True)].copy()
    model_ids = scores.columns[1:-2].tolist()
    if (len(model_ids) != 397 or parameters.CCLE_ID.tolist() != model_ids or
            set(drive_sample.CCLE_ID) != set(model_ids)):
        raise ValueError("DRIVE 模型身份或顺序变化")

    current = pd.read_csv(model_path, low_memory=False).dropna(subset=["CCLEName"])
    if current.CCLEName.duplicated().any():
        raise ValueError("当前 DepMap 的非空 CCLEName 不唯一")
    annotation_columns = ["ModelID", "PatientID", "CCLEName", "CellLineName",
                          "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
                          "OncotreeCode"]
    audit = drive_sample.merge(current[annotation_columns], left_on="CCLE_ID",
                               right_on="CCLEName", how="left", validate="one_to_one")
    audit = pd.DataFrame({"CCLE_ID": model_ids}).merge(
        audit, on="CCLE_ID", validate="one_to_one").merge(
        parameters, on="CCLE_ID", validate="one_to_one")
    baseline = pd.read_csv(baseline_models_path)
    audit["in_current_depmap_baseline"] = audit.ModelID.isin(baseline.ModelID)
    baseline_patients = set(baseline.PatientID.dropna())
    audit["shares_current_baseline_patient"] = audit.PatientID.map(
        lambda value: value in baseline_patients if pd.notna(value) else False)
    audit["drive_kidney"] = audit.disease.eq("kidney")
    audit["mapped_ccrcc"] = audit.OncotreeCode.eq("CCRCC")
    if (audit.drive_kidney.sum() != 16 or audit.mapped_ccrcc.sum() != 8 or
            audit.loc[audit.drive_kidney, "ModelID"].notna().sum() != 16):
        raise ValueError("DRIVE Kidney/ccRCC 身份覆盖变化")
    return scores, score_sds, audit


def drive_evidence(candidates, scores, score_sds, audit, draws, seed):
    covered = scores[scores.Gene.isin(candidates.Gene)].copy()
    if len(covered) != 10:
        raise ValueError(f"DRIVE 冻结候选覆盖变化：{len(covered)}/20")
    model_ids = audit.CCLE_ID.tolist()
    values = covered.set_index("Gene")[model_ids].T.apply(pd.to_numeric, errors="coerce")
    uncertainties = score_sds[score_sds.Gene.isin(candidates.Gene)].set_index(
        "Gene")[model_ids].T.apply(pd.to_numeric, errors="coerce")
    values = values.reindex(columns=covered.Gene)
    uncertainties = uncertainties.reindex(columns=covered.Gene)
    nonkidney = ~audit.drive_kidney.to_numpy()
    ccrcc = audit.mapped_ccrcc.to_numpy()
    other_kidney = audit.drive_kidney.to_numpy() & ~ccrcc
    external_ccrcc = ccrcc & ~audit.shares_current_baseline_patient.to_numpy()
    mean_non = values.iloc[nonkidney].mean(axis=0, skipna=True)
    residual = values - mean_non
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    for gene in values.columns:
        c = residual.loc[ccrcc, gene].dropna().to_numpy()
        o = residual.loc[other_kidney, gene].dropna().to_numpy()
        e = residual.loc[external_ccrcc, gene].dropna().to_numpy()
        c_ci = bootstrap_mean_ci(c, draws, generator)
        delta_ci = bootstrap_difference_ci(c, o, draws, generator)
        raw_c = values.loc[ccrcc, gene].dropna()
        posterior_sd = uncertainties.loc[ccrcc, gene].dropna()
        leave_one_out = [np.delete(c, index).mean() for index in range(len(c))] if len(c) > 1 else []
        rows.append({
            "Gene": gene, "drive_ccrcc_n": len(c),
            "drive_ccrcc_raw_mean": raw_c.mean(),
            "drive_ccrcc_raw_fraction_le_m0_5": (raw_c <= -.5).mean(),
            "drive_ccrcc_posterior_sd_mean": posterior_sd.mean(),
            "drive_ccrcc_residual_mean": np.mean(c),
            "drive_ccrcc_residual_ci_low": c_ci[0],
            "drive_ccrcc_residual_ci_median": c_ci[1],
            "drive_ccrcc_residual_ci_high": c_ci[2],
            "drive_ccrcc_leave1out_worst_mean": max(leave_one_out) if leave_one_out else np.nan,
            "drive_other_kidney_n": len(o),
            "drive_other_kidney_residual_mean": np.mean(o),
            "drive_ccrcc_minus_other_kidney_mean": np.mean(c) - np.mean(o),
            "drive_ccrcc_minus_other_kidney_ci_low": delta_ci[0],
            "drive_ccrcc_minus_other_kidney_ci_median": delta_ci[1],
            "drive_ccrcc_minus_other_kidney_ci_high": delta_ci[2],
            "drive_external_ccrcc_n": len(e),
            "drive_external_ccrcc_residual_mean": np.mean(e) if len(e) else np.nan,
        })
    return pd.DataFrame(rows)


def gtex_evidence(candidates, hpa_path, gtex_path):
    hpa = pd.read_csv(hpa_path, usecols=["Gene", "Ensembl"])
    table = pd.read_csv(gtex_path, sep="\t", skiprows=2)
    if table.shape != (74628, 70):
        raise ValueError(f"GTEx v11 矩阵维度变化：{table.shape}")
    table["Ensembl"] = table.Name.str.split(".").str[0]
    selected_ids = set(hpa.Ensembl)
    selected = table[table.Ensembl.isin(selected_ids)].copy()
    if selected.Ensembl.duplicated().any() or len(selected) != len(candidates):
        raise ValueError("GTEx 冻结候选 Ensembl 映射不完整或不唯一")
    selected = hpa.merge(selected.drop(columns="Description"), on="Ensembl",
                         validate="one_to_one")
    tissues = [column for column in table.columns
               if column not in {"Name", "Description", "Ensembl"}]
    if len(tissues) != 68 or not {"Kidney_Cortex", "Kidney_Medulla"} <= set(tissues):
        raise ValueError("GTEx v11 组织列发生变化")
    values = selected[tissues]
    kidney = selected[["Kidney_Cortex", "Kidney_Medulla"]]
    ranks = values.rank(axis=1, method="average", pct=True)
    max_columns = values.idxmax(axis=1)
    result = selected[["Gene", "Ensembl"]].copy()
    result["gtex_tissue_n"] = len(tissues)
    result["gtex_kidney_cortex_median_tpm"] = kidney.Kidney_Cortex
    result["gtex_kidney_medulla_median_tpm"] = kidney.Kidney_Medulla
    result["gtex_kidney_max_median_tpm"] = kidney.max(axis=1)
    result["gtex_across_tissue_median_tpm"] = values.median(axis=1)
    result["gtex_across_tissue_max_tpm"] = values.max(axis=1)
    result["gtex_kidney_max_fraction_of_tissue_max"] = (
        result.gtex_kidney_max_median_tpm / result.gtex_across_tissue_max_tpm.replace(0, np.nan))
    result["gtex_kidney_max_tissue_percentile"] = ranks[
        ["Kidney_Cortex", "Kidney_Medulla"]].max(axis=1)
    result["gtex_max_expression_tissue"] = max_columns
    result["gtex_tissues_median_tpm_ge_1_n"] = values.ge(1).sum(axis=1)
    result["gtex_tissues_median_tpm_ge_10_n"] = values.ge(10).sum(axis=1)
    return result


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1")
    parser.add_argument("--external-dir", type=Path,
                        default=root / "outputs/candidate_external_adjudication_v1")
    parser.add_argument("--hpa", type=Path,
                        default=root / "outputs/candidate_external_validation_v1/hpa_normal_tissue.csv")
    parser.add_argument("--model-metadata", type=Path,
                        default=source / "data/raw/depmap_24q4/Model.csv")
    parser.add_argument("--baseline-models", type=Path,
                        default=root / "data/processed/depmap_baseline_24q4_v1/models.csv")
    parser.add_argument("--raw-dir", type=Path,
                        default=root / "data/raw/candidate_orthogonal_validation_20260915")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/candidate_orthogonal_validation_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.draws < 1000 or args.seed < 0:
        parser.error("draws 至少为1000且 seed 必须非负")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")
    configure_gpu()
    print("【阶段 1/5】校验冻结候选、既有外部证据和 GPU", flush=True)
    candidates, previous = verify_candidates(args.frozen_dir, args.external_dir)
    print(f"【冻结规则】候选 {len(candidates)}｜不按 DRIVE 或 GTEx 重排｜"
          f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    if args.dry_run:
        for path in (args.hpa, args.model_metadata, args.baseline_models):
            if not path.exists():
                raise FileNotFoundError(path)
        print("【检查通过】未联网、未读取新数据值、未写入结果。", flush=True)
        return

    print("【阶段 2/5】下载或复用固定 GTEx v11 与 DRIVE-only DEMETER2", flush=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    paths, source_audit = {}, {}
    for name, source in SOURCES.items():
        path, reused = download(source, args.raw_dir)
        paths[name] = path
        source_audit[name] = {**source, "path": str(path.resolve()), "cache_reused": reused}
    print("【来源校验】6/6 文件大小与 SHA256 一致", flush=True)

    print("【阶段 3/5】映射 DRIVE 模型并用 GPU 计算冻结候选区间", flush=True)
    scores, score_sds, model_audit = load_drive(
        paths, args.model_metadata, args.baseline_models)
    drive = drive_evidence(candidates, scores, score_sds, model_audit, args.draws, args.seed)
    kidney = model_audit[model_audit.drive_kidney]
    external_ccrcc_n = int((kidney.mapped_ccrcc & ~kidney.shares_current_baseline_patient).sum())
    print(f"【DRIVE覆盖】模型 397｜Kidney 16｜映射 ccRCC 8｜"
          f"基线患者外 ccRCC {external_ccrcc_n}｜候选基因 {len(drive)}/20", flush=True)

    print("【阶段 4/5】提取 GTEx v11 Cortex/Medulla 与跨组织定量", flush=True)
    gtex = gtex_evidence(candidates, args.hpa, paths["gtex_v11_median_tpm"])
    print(f"【GTEx覆盖】候选 {len(gtex)}/20｜组织 {int(gtex.gtex_tissue_n.iloc[0])}｜"
          "不设置安全阈值", flush=True)

    print("【阶段 5/5】保存正交功能证据、正常组织定量和审计记录", flush=True)
    summary = previous.merge(drive, on="Gene", how="left", validate="one_to_one").merge(
        gtex, on="Gene", validate="one_to_one")
    summary = candidates[["Gene", "discovery_rank"]].merge(
        summary.drop(columns="discovery_rank"), on="Gene", validate="one_to_one")
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model_audit.to_csv(temporary / "drive_model_audit.csv", index=False)
    drive.to_csv(temporary / "drive_candidate_evidence.csv", index=False)
    gtex.to_csv(temporary / "gtex_normal_tissue.csv", index=False)
    summary.to_csv(temporary / "candidate_orthogonal_summary.csv", index=False)
    run = {
        "status": "orthogonal_rnai_and_normal_tissue_validation_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(0), "dtype": "float64_bootstrap",
        "design": {
            "frozen_candidate_n": len(candidates), "drive_candidate_covered_n": len(drive),
            "drive_model_n": len(model_audit), "drive_kidney_n": int(model_audit.drive_kidney.sum()),
            "drive_mapped_ccrcc_n": int(model_audit.mapped_ccrcc.sum()),
            "drive_ccrcc_outside_baseline_patient_n": external_ccrcc_n,
            "gtex_tissue_n": int(gtex.gtex_tissue_n.iloc[0]), "bootstrap_draws": args.draws,
            "seed": args.seed,
        },
        "rules": {
            "selection": "Existing train discovery ranks 1-20 frozen before DRIVE and GTEx access",
            "drive_source": "DRIVE-only DEMETER2 posterior means; combined Achilles/DRIVE/Marcotte matrix excluded",
            "drive_residual": "Per-gene DRIVE non-Kidney mean subtracted from every model",
            "ccrcc_identity": "Unique current DepMap CCLEName mapping and OncotreeCode CCRCC",
            "strict_sensitivity": "Mapped ccRCC whose PatientID is absent from the current 873-model baseline",
            "gtex": "GTEx v11 median TPM reported quantitatively for kidney cortex, medulla and all 66 tissues",
            "aggregation": "No composite score, safety threshold, candidate removal or reranking",
        },
        "sources": source_audit,
        "source_sha256": {
            "frozen_candidates": sha256(args.frozen_dir / "candidates.csv"),
            "prior_external_summary": sha256(args.external_dir / "candidate_external_summary.csv"),
            "hpa": sha256(args.hpa), "model_metadata": sha256(args.model_metadata),
            "baseline_models": sha256(args.baseline_models), "script": sha256(Path(__file__)),
        },
        "limitations": [
            "DRIVE is an independent RNAi assay but seven of sixteen Kidney models occur in the current DepMap baseline.",
            "Only ten frozen candidates were targeted by the DRIVE library.",
            "Current-model annotations map eight DRIVE lines to ccRCC; historical DRIVE metadata only labels them Kidney carcinoma.",
            "The strict patient-nonoverlap ccRCC sensitivity subset has only three models.",
            "RNAi and CRISPR have different off-target effects and score scales; comparisons use within-platform residuals.",
            "GTEx bulk median expression does not identify the renal cell type exposed and does not prove toxicity.",
            "Bootstrap intervals are descriptive and are not multiplicity-adjusted confirmatory tests.",
        ],
    }
    run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    for row in drive.sort_values("drive_ccrcc_residual_mean").itertuples():
        kidney_tpm = float(gtex.loc[gtex.Gene.eq(row.Gene), "gtex_kidney_max_median_tpm"].iloc[0])
        print(f"  {row.Gene}｜RNAi ccRCC残差 {row.drive_ccrcc_residual_mean:+.3f} "
              f"[{row.drive_ccrcc_residual_ci_low:+.3f}, {row.drive_ccrcc_residual_ci_high:+.3f}]｜"
              f"ccRCC-其他Kidney {row.drive_ccrcc_minus_other_kidney_mean:+.3f}｜"
              f"GTEx肾脏 {kidney_tpm:.2f} TPM", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】DRIVE只提供模型层RNAi复现；GTEx只提供正常组织暴露，均不能验证患者特异依赖或临床安全。", flush=True)


if __name__ == "__main__":
    main()
