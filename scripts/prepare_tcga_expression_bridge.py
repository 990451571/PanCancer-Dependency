"""Build and audit leakage-safe TCGA-KIRC expression inputs for patient transfer.

Only the historical TCGA train split fits unsupervised gene mean shifts.  The
validation split measures distribution stability, while the locked test split
is excluded from every output matrix and calculation.  No functional outcome
exists in TCGA, so this script prepares inputs and domain diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
from build_context_module_stage0 import GeneCanonicalizer, collapse_duplicate_rows


DRIVERS = baseline.DRIVERS
METHODS = ("raw_xena_log_expression", "nonkidney_mean_shift_clip0", "kidney_mean_shift_clip0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_mean(values):
    count = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1], np.nan), where=count > 0)


def align_expression(path, sample_ids, genes, canonicalizer):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.astype(str).tolist()
    missing = sorted(set(sample_ids) - set(header))
    if missing:
        raise ValueError(f"原始表达缺少选定样本：{missing[:5]}")
    raw = pd.read_csv(path, sep="\t", usecols=[header[0], *sample_ids], low_memory=False)
    mapped_genes = raw.iloc[:, 0].map(canonicalizer.map)
    numeric = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    numeric.index = mapped_genes
    collapsed = collapse_duplicate_rows(numeric, pd.Series(numeric.index, index=numeric.index), "mean")
    # collapse_duplicate_rows expects features in rows and returns the same orientation.
    aligned = collapsed.reindex(genes).T
    aligned.index = [column for column in numeric.columns]
    aligned = aligned.reindex(sample_ids)
    return aligned.astype(np.float32), {
        "raw_gene_rows": len(raw),
        "canonical_gene_rows": int(pd.Series(mapped_genes).nunique()),
        "duplicate_canonical_rows": int(pd.Series(mapped_genes).duplicated().sum()),
        "aligned_finite_gene_n": int(np.isfinite(aligned.to_numpy()).any(axis=0).sum()),
    }


def transform(raw, tcga_train, reference):
    shift = observed_mean(reference) - observed_mean(raw[tcga_train])
    shift[~np.isfinite(shift)] = np.nan
    return np.maximum(raw + shift, 0).astype(np.float32), shift.astype(np.float32)


def diagnostics(patient_values, validation, reference, method, reference_name):
    """Descriptive GPU diagnostics; they are not dependency accuracy metrics."""
    x = torch.as_tensor(patient_values[validation], dtype=torch.float64, device="cuda")
    r = torch.as_tensor(reference, dtype=torch.float64, device="cuda")
    rmean = torch.nanmean(r, dim=0)
    centered = r - rmean
    rsd = torch.sqrt(torch.nanmean(centered * centered, dim=0))
    usable = ((torch.isfinite(x).sum(dim=0) >= max(2, int(.8 * len(x)))) &
              (torch.isfinite(r).sum(dim=0) >= max(2, int(.8 * len(r)))) & (rsd > 1e-6))
    x, r, rmean, rsd = x[:, usable], r[:, usable], rmean[usable], rsd[usable]
    x = torch.where(torch.isfinite(x), x, rmean)
    r = torch.where(torch.isfinite(r), r, rmean)
    centroid = x.mean(dim=0)
    centroid_rmse = torch.sqrt((((centroid - rmean) / rsd) ** 2).mean())
    low = torch.quantile(r, .01, dim=0)
    high = torch.quantile(r, .99, dim=0)
    outside = ((x < low) | (x > high)).to(torch.float64).mean()
    # Per-sample Pearson correlation across genes to the closest reference profile.
    xc = x - x.mean(dim=1, keepdim=True)
    rc = r - r.mean(dim=1, keepdim=True)
    xn = torch.sqrt((xc * xc).sum(dim=1, keepdim=True)).clamp_min(1e-12)
    rn = torch.sqrt((rc * rc).sum(dim=1, keepdim=True)).clamp_min(1e-12)
    nearest = ((xc / xn) @ (rc / rn).T).max(dim=1).values
    return {"method": method, "reference": reference_name,
            "validation_patient_n": len(validation), "usable_gene_n": int(usable.sum().item()),
            "validation_centroid_standardized_rmse": float(centroid_rmse.cpu()),
            "validation_outside_reference_1_99_fraction": float(outside.cpu()),
            "nearest_reference_pearson_mean": float(nearest.mean().cpu()),
            "nearest_reference_pearson_min": float(nearest.min().cpu()),
            "nearest_reference_pearson_max": float(nearest.max().cpu())}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--stage0-dir", type=Path, default=source / "data/processed/context_module_stage0")
    parser.add_argument("--expression", type=Path, default=source / "data/raw/HiSeqV2")
    parser.add_argument("--hgnc", type=Path, default=source / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/tcga_kirc_expression_bridge_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    baseline.configure_device("cuda")
    print("【阶段 1/4】校验 DepMap、TCGA 历史分组和锁定 Test", flush=True)
    models, genes, matrices, _ = baseline.load_data(args.baseline_dir)
    stage_run = json.loads((args.stage0_dir / "stage0_manifest.json").read_text())
    patients = pd.read_csv(args.stage0_dir / "patients.csv", index_col=0)
    expected = {"train": 156, "validation": 53, "test": 51}
    if patients.split.value_counts().to_dict() != expected or stage_run["split_policy"] != (
            "within primary context: 60% train, 20% validation, 20% untouched test"):
        raise ValueError("TCGA 历史锁定分组发生变化")
    selected = patients[patients.split.isin(["train", "validation"])].copy()
    if selected.index.intersection(patients.index[patients.split.eq("test")]).size:
        raise ValueError("锁定 Test 混入适配队列")
    sample_ids = selected.expression_sample.astype(str).tolist()
    print(f"【锁定规则】Train {sum(selected.split.eq('train'))}｜Validation {sum(selected.split.eq('validation'))}｜Test 51 完全排除", flush=True)
    if args.dry_run:
        header = set(pd.read_csv(args.expression, sep="\t", nrows=0).columns.astype(str))
        if set(sample_ids) - header:
            raise ValueError("Train/Validation 表达样本缺失")
        print("【检查通过】只核对样本表头；未读取患者表达数值，未拟合映射，未写入结果。", flush=True)
        return

    print("【阶段 2/4】读取 Train/Validation 原始表达并映射到 DepMap 基因", flush=True)
    canonicalizer = GeneCanonicalizer(args.hgnc)
    frame, mapping_audit = align_expression(args.expression, sample_ids, genes, canonicalizer)
    frame = frame.loc[sample_ids]
    raw = frame.to_numpy(dtype=np.float32)
    split = selected.set_index("expression_sample").loc[sample_ids, "split"].to_numpy(dtype=str)
    # Recover patient IDs without depending on row order in the source file.
    sample_to_patient = pd.Series(selected.index, index=selected.expression_sample.astype(str))
    patient_ids = sample_to_patient.loc[sample_ids].to_numpy(dtype=str)
    train_rows = np.flatnonzero(split == "train")
    validation_rows = np.flatnonzero(split == "validation")
    nonkidney = matrices["expression"][models.OncotreeLineage.ne("Kidney").to_numpy()]
    kidney = matrices["expression"][models.OncotreeLineage.eq("Kidney").to_numpy()]
    mapped_nonkidney, shift_nonkidney = transform(raw, train_rows, nonkidney)
    mapped_kidney, shift_kidney = transform(raw, train_rows, kidney)
    transformed = {"raw_xena_log_expression": raw,
                   "nonkidney_mean_shift_clip0": mapped_nonkidney,
                   "kidney_mean_shift_clip0": mapped_kidney}
    print(f"【实际覆盖】患者 {len(patient_ids)}｜映射基因 {mapping_audit['aligned_finite_gene_n']}｜DepMap Kidney模型 {len(kidney)}", flush=True)

    print("【阶段 3/4】GPU计算 Validation 分布稳定性诊断", flush=True)
    diagnostic_rows = []
    for method, values in transformed.items():
        for reference_name, reference in (("nonkidney_cell_lines", nonkidney),
                                          ("kidney_cell_lines", kidney)):
            row = diagnostics(values, validation_rows, reference, method, reference_name)
            diagnostic_rows.append(row)
            print(f"  {method} → {reference_name}｜基因 {row['usable_gene_n']}｜质心RMSE {row['validation_centroid_standardized_rmse']:.3f}｜最近相关 {row['nearest_reference_pearson_mean']:.3f}", flush=True)

    print("【阶段 4/4】保存非 Test 输入、候选映射和审计记录", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    np.savez_compressed(temporary / "expression_inputs.npz", patient_ids=patient_ids, split=split,
                        genes=genes, raw_expression=raw,
                        nonkidney_mean_shift_expression=mapped_nonkidney,
                        kidney_mean_shift_expression=mapped_kidney)
    pd.DataFrame(diagnostic_rows).to_csv(temporary / "diagnostics.csv", index=False)
    pd.DataFrame({"Gene": genes,
                  "tcga_train_mean": observed_mean(raw[train_rows]),
                  "depmap_nonkidney_mean": observed_mean(nonkidney),
                  "depmap_kidney_mean": observed_mean(kidney),
                  "nonkidney_shift": shift_nonkidney,
                  "kidney_shift": shift_kidney,
                  "tcga_covered": np.isfinite(raw).any(axis=0)}).to_csv(
                      temporary / "gene_mapping.csv.gz", index=False)
    audit = {
        "status": "patient_expression_bridge_audit_no_dependency_prediction",
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(0), "dtype": "float64_diagnostics_float32_storage",
        "patients": {"train": len(train_rows), "validation": len(validation_rows),
                     "locked_test_excluded": 51, "output_patient_n": len(patient_ids)},
        "genes": {"depmap_gene_n": len(genes), **mapping_audit},
        "methods": list(METHODS),
        "rules": {
            "fit": "Gene mean shifts use TCGA train patients only",
            "validation": "TCGA validation is used only for unlabeled distribution diagnostics",
            "test": "Locked TCGA test samples are absent from output matrices and every calculation",
            "clip": "Negative mapped log-expression values are clipped to zero",
            "selection": "No mapping is selected by this audit",
        },
        "source_sha256": {"baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "stage0_manifest": sha256(args.stage0_dir / "stage0_manifest.json"),
                          "patients": sha256(args.stage0_dir / "patients.csv"),
                          "expression": sha256(args.expression), "hgnc": sha256(args.hgnc),
                          "script": sha256(Path(__file__))},
        "limitations": [
            "TCGA Xena expression and DepMap TPM are not paired measurements, so mean shifts mix technical and biological differences.",
            "Bulk tumors include stromal and immune expression absent from cell lines.",
            "Lower domain distance does not imply more accurate dependency prediction.",
            "No patient functional dependency label is available.",
            "Only TCGA patients with the historical four-omics intersection and fixed split are included.",
        ],
    }
    audit["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(output)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】本步只审计输入域；不能用分布接近代替患者功能验证，也尚未生成依赖预测。", flush=True)


if __name__ == "__main__":
    main()
