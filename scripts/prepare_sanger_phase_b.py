"""Prepare Sanger expression-domain adaptation for the two frozen phase-B models.

Mapping selection uses only paired expression profiles. Frozen CRISPR outcomes
are neither opened nor used by this script.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_depmap_baseline as baseline
from prepare_sanger_validation import HttpRangeReader, read_member_once


URL = "https://cog.sanger.ac.uk/cmp/download/rnaseq_20191101.zip"
MEMBER = "rnaseq_fpkm_20191101.csv"
ARCHIVE_BYTES = 61_579_614
MEMBER_BYTES = 158_680_309
MEMBER_COMPRESSED_BYTES = 14_278_614
MEMBER_CRC32 = 0x453726C6
TARGETS = {"SIDM00187": "LB1047-RCC", "SIDM00819": "RCC-FG2"}
MAPPING_METHODS = ("no_mapping", "gene_mean_shift_clip0")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_source(path):
    if path.exists():
        print(f"【来源复用】{path}", flush=True)
        return
    print("【来源下载】读取2019 Sanger RNA归档中的 FPKM 成员（压缩区间约14.3 MB）", flush=True)
    reader = HttpRangeReader(URL)
    if reader.size != ARCHIVE_BYTES:
        raise ValueError(f"RNA归档大小变化：{reader.size}")
    with zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(MEMBER)
        if (info.file_size, info.compress_size, info.CRC) != (
                MEMBER_BYTES, MEMBER_COMPRESSED_BYTES, MEMBER_CRC32):
            raise ValueError("RNA归档成员大小或 CRC 与固定版本不符")
        data = read_member_once(reader, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(data)
    temporary.rename(path)
    print(f"【来源保存】{path}｜解压内容 SHA256 {hashlib.sha256(data).hexdigest()}", flush=True)


def parse_float(text):
    if text in ("", "NA", "NaN", "nan"):
        return np.nan
    return float(text)


def read_sanger_expression(path, requested_sids, genes):
    """Read Sanger-source columns and aggregate replicate analyses and symbols."""
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.reader(handle)
        model_ids, model_names, datasets, gene_header = [next(reader) for _ in range(4)]
        width = len(model_ids)
        if any(len(row) != width for row in (model_names, datasets, gene_header)):
            raise ValueError("Sanger RNA元数据行宽度不一致")
        selected_columns = [index for index in range(2, width)
                            if datasets[index] == "Sanger RNASeq" and model_ids[index] in requested_sids]
        selected_sids = [model_ids[index] for index in selected_columns]
        if not selected_columns:
            raise ValueError("没有找到所需 Sanger RNA分析列")
        gene_index = {gene: index for index, gene in enumerate(genes)}
        values = np.full((len(selected_columns), len(genes)), np.nan, dtype=np.float32)
        duplicate_symbols = set()
        seen_symbols = set()
        source_gene_rows = 0
        for row in reader:
            if len(row) != width:
                raise ValueError(f"Sanger RNA数据行宽度错误：{len(row)} != {width}")
            source_gene_rows += 1
            symbol = row[1]
            if symbol not in gene_index:
                continue
            column = gene_index[symbol]
            current = np.asarray([parse_float(row[index]) for index in selected_columns], dtype=np.float32)
            if symbol in seen_symbols:
                duplicate_symbols.add(symbol)
                previous = values[:, column]
                stacked = np.vstack([previous, current])
                valid = np.isfinite(stacked)
                count = valid.sum(axis=0)
                values[:, column] = np.divide(np.nansum(stacked, axis=0), count,
                                              out=np.full(len(current), np.nan, dtype=np.float32), where=count > 0)
            else:
                values[:, column] = current
                seen_symbols.add(symbol)
    by_sid = {}
    replicate_counts = {}
    for sid in sorted(set(selected_sids)):
        indices = np.flatnonzero(np.asarray(selected_sids) == sid)
        subset = values[indices]
        count = np.isfinite(subset).sum(axis=0)
        by_sid[sid] = np.divide(np.nansum(subset, axis=0), count,
                                out=np.full(values.shape[1], np.nan, dtype=np.float32), where=count > 0)
        replicate_counts[sid] = len(indices)
    matrix = np.vstack([by_sid[sid] for sid in sorted(by_sid)]).astype(np.float32)
    if np.nanmin(matrix) < 0:
        raise ValueError("FPKM存在负值")
    return sorted(by_sid), np.log2(matrix + 1).astype(np.float32), {
        "archive_analysis_n": width - 2,
        "selected_sanger_analysis_n": len(selected_columns),
        "selected_sanger_model_n": len(by_sid),
        "source_gene_row_n": source_gene_rows,
        "exact_symbol_gene_n": len(seen_symbols),
        "duplicate_exact_symbol_n": len(duplicate_symbols),
        "replicate_model_n": sum(count > 1 for count in replicate_counts.values()),
        "max_replicates": max(replicate_counts.values()),
    }


def fit_mapping(sanger, broad, method):
    if method == "no_mapping":
        return np.zeros(sanger.shape[1], dtype=np.float32)
    if method == "gene_mean_shift_clip0":
        shift = baseline.observed_mean(broad, axis=0) - baseline.observed_mean(sanger, axis=0)
        return np.where(np.isfinite(shift), shift, 0).astype(np.float32)
    raise ValueError(method)


def apply_mapping(sanger, shift, method):
    predicted = sanger + shift
    return np.maximum(predicted, 0) if method == "gene_mean_shift_clip0" else predicted


def evaluate_mapping(sanger, broad, shift, method, held, train, torch):
    predicted = apply_mapping(sanger[held], shift, method)
    observed = broad[held]
    broad_mean = baseline.observed_mean(broad[train], axis=0)
    centered = broad[train] - broad_mean
    broad_sd = np.sqrt(baseline.observed_mean(centered * centered, axis=0))
    valid_gene = np.isfinite(broad_sd) & (broad_sd > 1e-3)
    valid = np.isfinite(predicted) & np.isfinite(observed) & valid_gene[None, :]
    p = torch.as_tensor(predicted, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(observed, dtype=torch.float64, device="cuda")
    sd = torch.as_tensor(broad_sd, dtype=torch.float64, device="cuda")
    mask = torch.as_tensor(valid, device="cuda")
    error = ((p - y) / sd)[mask]
    sample_correlations = []
    for row in range(len(held)):
        row_mask = mask[row]
        px, py = p[row, row_mask], y[row, row_mask]
        px, py = px - px.mean(), py - py.mean()
        denominator = torch.sqrt((px * px).sum() * (py * py).sum())
        sample_correlations.append(float(((px * py).sum() / denominator).cpu()))
    return {"observed_n": int(mask.sum().item()),
            "standardized_sse": float((error * error).sum().cpu()),
            "standardized_rmse": float(torch.sqrt((error * error).mean()).cpu()),
            "standardized_mae": float(error.abs().mean().cpu()),
            "sample_pearson_mean": float(np.mean(sample_correlations))}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--frozen-dir", type=Path, default=root / "outputs/sanger_validation_frozen_v1")
    parser.add_argument("--source", type=Path,
                        default=root / "data/raw/external_sanger_20260914/rnaseq_fpkm_20191101.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/sanger_phase_b_preparation_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if args.folds < 2 or args.seed < 0:
        parser.error("folds至少为2，seed必须非负")
    return args


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    baseline.configure_device(args.device)
    print("【阶段 1/4】校验基线与阶段 B 冻结模型", flush=True)
    models, genes, matrices, _ = baseline.load_data(args.baseline_dir)
    frozen = pd.read_csv(args.frozen_dir / "frozen_models.csv")
    phase_b = frozen[frozen.model_name.isin(TARGETS.values())]
    if set(phase_b.model_id) != set(TARGETS) or len(phase_b) != 2:
        raise ValueError("阶段 B 冻结模型不是预期的两例")
    obtain_source(args.source)

    requested = set(models.SangerModelID.dropna()) | set(TARGETS)
    print("【阶段 2/4】提取 Sanger 来源表达并核实实际覆盖", flush=True)
    sids, sanger, source_audit = read_sanger_expression(args.source, requested, genes)
    sid_index = {sid: index for index, sid in enumerate(sids)}
    missing_targets = set(TARGETS) - set(sids)
    if missing_targets:
        raise ValueError(f"2019 Sanger FPKM缺少阶段 B 模型：{sorted(missing_targets)}")
    calibration_rows = [index for index, row in models.reset_index().iterrows()
                        if row.OncotreeLineage != "Kidney" and row.SangerModelID in sid_index]
    calibration = models.iloc[calibration_rows]
    if calibration.PatientID.duplicated().any():
        # Duplicated patients are allowed, but must remain joined in validation folds.
        pass
    sanger_calibration = np.vstack([sanger[sid_index[sid]] for sid in calibration.SangerModelID])
    broad_calibration = matrices["expression"][calibration_rows]
    if len(calibration) < 30:
        raise ValueError(f"实际配对模型过少：{len(calibration)}")
    folds = baseline.inner_folds(np.arange(len(calibration)), calibration.OncotreeLineage.to_numpy(),
                                 calibration.PatientID.to_numpy(), args.folds, args.seed)
    print(f"【实际覆盖】非 Kidney 配对模型 {len(calibration)}｜癌系 {calibration.OncotreeLineage.nunique()}｜内层 {len(folds)} 折", flush=True)

    print("【阶段 3/4】GPU 比较固定的无映射与逐基因均值平移", flush=True)
    cv_rows = []
    all_indices = np.arange(len(calibration))
    torch = baseline.TORCH
    for fold_id, held in enumerate(folds, 1):
        train = np.setdiff1d(all_indices, held)
        for method in MAPPING_METHODS:
            shift = fit_mapping(sanger_calibration[train], broad_calibration[train], method)
            metrics = evaluate_mapping(sanger_calibration, broad_calibration, shift, method, held, train, torch)
            cv_rows.append({"fold": fold_id, "method": method, "train_n": len(train),
                            "held_n": len(held), **metrics})
        print(f"  表达域验证 {fold_id}/{len(folds)} 完成｜留出 {len(held)}", flush=True)
    cv = pd.DataFrame(cv_rows)
    totals = cv.groupby("method", sort=False).agg(observed_n=("observed_n", "sum"),
                                                   standardized_sse=("standardized_sse", "sum"))
    totals["pooled_standardized_rmse"] = np.sqrt(totals.standardized_sse / totals.observed_n)
    selected = min(MAPPING_METHODS, key=lambda method: (totals.loc[method, "pooled_standardized_rmse"], method))
    final_shift = fit_mapping(sanger_calibration, broad_calibration, selected)
    target_sids = list(TARGETS)
    target_expression = apply_mapping(
        np.vstack([sanger[sid_index[sid]] for sid in target_sids]), final_shift, selected).astype(np.float32)

    print("【阶段 4/4】冻结映射并保存目标表达", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    cv.to_csv(temporary / "mapping_cv.csv", index=False)
    totals.reset_index().to_csv(temporary / "mapping_summary.csv", index=False)
    np.savez_compressed(temporary / "mapped_target_expression.npz",
                        model_ids=np.asarray(target_sids),
                        model_names=np.asarray([TARGETS[sid] for sid in target_sids]),
                        genes=genes, expression=target_expression)
    fold_assignment = np.full(len(calibration), -1, dtype=int)
    for fold_id, held in enumerate(folds, 1):
        fold_assignment[held] = fold_id
    if (fold_assignment < 1).any():
        raise ValueError("表达映射折分记录不完整")
    calibration_table = calibration.reset_index()[["ModelID", "PatientID", "CellLineName",
                                                    "OncotreeLineage", "SangerModelID"]]
    calibration_table["validation_fold"] = fold_assignment
    calibration_table.to_csv(temporary / "calibration_models.csv", index=False)
    summary_rows = totals.reset_index().to_dict("records")
    run = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "phase_b_expression_mapping_frozen_before_target_crispr_evaluation",
        "device": baseline.TORCH.cuda.get_device_name(0), "dtype": "float64",
        "source": {"url": URL, "archive_bytes": ARCHIVE_BYTES, "member": MEMBER,
                   "member_bytes": MEMBER_BYTES, "member_compressed_bytes": MEMBER_COMPRESSED_BYTES,
                   "member_crc32": f"{MEMBER_CRC32:08x}", "local_path": str(args.source.resolve()),
                   "local_gzip_sha256": sha256(args.source), **source_audit},
        "calibration": {"model_n": len(calibration), "patient_n": calibration.PatientID.nunique(),
                        "lineage_n": calibration.OncotreeLineage.nunique(), "kidney_model_n": 0,
                        "fold_rule": "whole connected lineage/patient groups", "fold_n": len(folds)},
        "mapping": {"candidates": list(MAPPING_METHODS),
                    "selection_metric": "minimum pooled held-out standardized RMSE",
                    "summary": summary_rows, "selected": selected,
                    "target_outcomes_opened": False},
        "targets": [{"SangerModelID": sid, "CellLineName": TARGETS[sid],
                     "finite_expression_genes": int(np.isfinite(target_expression[index]).sum()),
                     "negative_expression_values": int((target_expression[index] < 0).sum())}
                    for index, sid in enumerate(target_sids)],
        "source_sha256": {"baseline_matrices": sha256(args.baseline_dir / "matrices.npz"),
                          "frozen_models": sha256(args.frozen_dir / "frozen_models.csv"),
                          "script": sha256(Path(__file__))},
        "limitations": [
            "FPKM-to-TPM mapping is estimated from cell-line pairs and may not remove platform or culture effects.",
            "Mapping selection optimizes expression reconstruction, not dependency prediction.",
            "No Kidney calibration model is used, so transfer to renal models is an extrapolation.",
            "Only two phase-B targets remain; downstream evidence is descriptive, not inferential."]}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    print("【映射结果】", flush=True)
    for row in summary_rows:
        print(f"  {row['method']}｜配对标准化 RMSE {row['pooled_standardized_rmse']:.4f}", flush=True)
    print(f"【冻结选择】{selected}｜未读取目标 CRISPR 标签", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f} 秒｜结果 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
