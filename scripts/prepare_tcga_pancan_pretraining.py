#!/usr/bin/env python3
"""Prepare TCGA pan-cancer expression for DeepDEP autoencoder pretraining."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_URL = "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_tpm.gz"
SOURCE_BYTES = 740772247
SOURCE_SHA256 = "2ac7215f35fbe2cdc671c03c2b40b934ac1a0c908757f05f6b8267e5ba0a1b6d"
PRIMARY_TUMOR_CODES = {"01", "03", "09"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_code(sample: str) -> str:
    fields = sample.split("-")
    return fields[3][:2] if len(fields) >= 4 else ""


def patient_id(sample: str) -> str:
    return "-".join(sample.split("-")[:3])


def read_kirc_patients(path: Path) -> set[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")[1:]
    patients = {patient_id(sample) for sample in header}
    if len(patients) < 400:
        raise ValueError("KIRC排除表患者数异常")
    return patients


def build_ensembl_map(hgnc_path: Path, canonical_features: np.ndarray):
    hgnc = pd.read_csv(hgnc_path, sep="\t", low_memory=False)
    approved = hgnc.loc[hgnc.status.eq("Approved"), ["symbol", "ensembl_gene_id"]].dropna()
    symbol_to_ensembl = dict(zip(approved.symbol.astype(str), approved.ensembl_gene_id.astype(str)))
    result = {}
    for position, symbol in enumerate(canonical_features.astype(str)):
        ensembl = symbol_to_ensembl.get(symbol)
        if ensembl:
            result.setdefault(ensembl, []).append(position)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/raw/tcga_pancan/tcga_RSEM_gene_tpm.gz")
    parser.add_argument("--benchmark-input", type=Path,
                        default=root / "data/processed/advanced_model_benchmark_v1/benchmark_inputs.npz")
    parser.add_argument("--benchmark-audit", type=Path,
                        default=root / "data/processed/advanced_model_benchmark_v1/audit.json")
    parser.add_argument("--hgnc", type=Path, required=True)
    parser.add_argument("--kirc-expression", type=Path, required=True,
                        help="TCGA KIRC matrix; only its header is read to exclude the entire cohort")
    parser.add_argument("--protocol", type=Path,
                        default=root / "configs/advanced_model_benchmark_protocol_20260917.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/processed/tcga_pancan_deepdep_input_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    print("【阶段 1/4】校验TCGA泛癌来源与KIRC整队列排除表", flush=True)
    if args.source.stat().st_size != SOURCE_BYTES or sha256(args.source) != SOURCE_SHA256:
        raise ValueError("TCGA泛癌表达文件大小或哈希错误")
    for path in (args.benchmark_input, args.benchmark_audit, args.hgnc, args.kirc_expression, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(args.protocol.read_text())
    if "excluding every KIRC" not in protocol["data"]["unlabeled_pretraining"]:
        raise ValueError("冻结协议未要求排除全部KIRC")
    kirc_patients = read_kirc_patients(args.kirc_expression)
    with np.load(args.benchmark_input, allow_pickle=False) as archive:
        feature_symbols = archive["expression_feature_symbols"]
        canonical_features = archive["expression_feature_canonical"]
    feature_mapping = pd.read_csv(args.benchmark_input.with_name("expression_feature_mapping.csv.gz"))
    feature_means = feature_mapping.official_ccle_imputation_mean.to_numpy(np.float32)
    ensembl_map = build_ensembl_map(args.hgnc, canonical_features)

    print("【阶段 2/4】筛选非KIRC原发肿瘤并对齐官方6,016特征", flush=True)
    with gzip.open(args.source, "rb") as handle:
        header = handle.readline().decode().rstrip("\n\r").split("\t")
        all_samples = np.asarray(header[1:])
        selected_columns = np.asarray([
            i for i, sample in enumerate(all_samples)
            if sample_code(sample) in PRIMARY_TUMOR_CODES and patient_id(sample) not in kirc_patients
        ], dtype=int)
        selected_samples = all_samples[selected_columns]
        if len(selected_samples) < 7000:
            raise ValueError(f"非KIRC原发肿瘤数量异常：{len(selected_samples)}")
        expression = np.broadcast_to(feature_means, (len(selected_samples), len(feature_symbols))).copy()
        found = np.zeros(len(feature_symbols), dtype=bool)
        matched_rows = 0
        for line_number, line in enumerate(handle, 1):
            tab = line.find(b"\t")
            if tab < 0:
                continue
            ensembl = line[:tab].decode().split(".")[0]
            positions = ensembl_map.get(ensembl)
            if not positions:
                continue
            values = np.fromstring(line[tab + 1:].decode(), sep="\t", dtype=np.float32)
            if len(values) != len(all_samples):
                raise ValueError(f"TCGA表达行列数异常：{ensembl}")
            values = values[selected_columns]
            # Toil Xena stores log2(TPM + 0.001); align to DepMap log2(TPM + 1).
            aligned = np.log2(np.maximum(np.exp2(values) - 0.001, 0.0) + 1.0).astype(np.float32)
            for position in positions:
                expression[:, position] = aligned
                found[position] = True
            matched_rows += 1
            if matched_rows % 1000 == 0:
                print(f"  已对齐 {matched_rows} 个Ensembl表达行", flush=True)
    if not np.isfinite(expression).all():
        raise ValueError("TCGA预训练矩阵含非有限值")
    excluded_kirc_sample_n = sum(patient_id(sample) in kirc_patients for sample in all_samples)
    print(f"【预训练队列】非KIRC原发肿瘤 {len(selected_samples)}｜排除KIRC样本 {excluded_kirc_sample_n}", flush=True)
    print(f"【表达覆盖】官方特征 {len(feature_symbols)}｜TCGA实测 {found.sum()}｜官方均值填补 {(~found).sum()}", flush=True)
    if args.dry_run:
        print("【检查通过】没有保存矩阵，没有读取任何KIRC表达数值。", flush=True)
        return

    print("【阶段 3/4】保存无标签预训练矩阵", flush=True)
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        np.savez_compressed(
            temporary / "tcga_pancan_expression.npz",
            sample_ids=selected_samples.astype(str),
            feature_symbols=feature_symbols.astype(str),
            feature_observed=found,
            expression=expression,
        )
        run = {
            "status": "tcga_pancan_unlabeled_pretraining_input_complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "source_url": SOURCE_URL,
            "source_bytes": SOURCE_BYTES,
            "source_sha256": SOURCE_SHA256,
            "sample_rule": "TCGA sample-type codes 01, 03 or 09; exclude every patient present in the KIRC matrix header",
            "sample_n": len(selected_samples),
            "kirc_patient_exclusion_n": len(kirc_patients),
            "kirc_source_sample_n": excluded_kirc_sample_n,
            "feature_n": len(feature_symbols),
            "observed_feature_n": int(found.sum()),
            "imputed_feature_n": int((~found).sum()),
            "transform": "log2(TPM+0.001) converted to log2(TPM+1)",
            "dependency_labels_read": False,
            "tcga_kirc_expression_values_read": False,
            "input_sha256": {
                "benchmark_input": sha256(args.benchmark_input),
                "benchmark_audit": sha256(args.benchmark_audit),
                "hgnc": sha256(args.hgnc),
                "kirc_matrix": sha256(args.kirc_expression),
                "protocol": sha256(args.protocol),
                "script": sha256(Path(__file__)),
            },
        }
        run["output_sha256"] = {path.name: sha256(path) for path in temporary.iterdir()}
        (temporary / "audit.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print("【阶段 4/4】预训练输入审计完成", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f}秒｜结果 {args.output_dir}", flush=True)
    print("【结论边界】这是无标签表达预训练输入；不包含依赖标签，也不验证患者依赖。", flush=True)


if __name__ == "__main__":
    main()
