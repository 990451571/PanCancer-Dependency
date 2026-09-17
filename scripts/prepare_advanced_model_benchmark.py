#!/usr/bin/env python3
"""Build audited same-data inputs for the advanced model benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

import run_depmap_baseline as baseline
from build_context_module_stage0 import GeneCanonicalizer


PREP_COMMIT = "f713c84b733a90d7644db789925a2fb4f9f41a68"
REFERENCE_FILES = {
    "gene_fingerprints_CGP.RData": "9cb86a1e9683bf9bf34803276a7124c5dc3f138abadac1511f56490a4855087b",
    "ccle_exp_for_missing_value_6016.RData": "35982eba38b4e7d5673746962a8cf1aa21b18f0f15599d7ad8e730faaa5aae99",
    "default_dep_genes_1298.RData": "38e185e3bc166b683c022316768c062f10a7cd7ea1b1b6ae23dd012552c566b3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_reference(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/chenlabgccri/Prep4DeepDEP/{PREP_COMMIT}/inst/extdata"
    for name, expected in REFERENCE_FILES.items():
        path = directory / name
        if path.exists() and sha256(path) == expected:
            print(f"  已校验 {name}", flush=True)
            continue
        if path.exists():
            raise ValueError(f"已有官方参考文件哈希错误：{path}")
        part = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(f"{base}/{name}", timeout=120) as response, part.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(part) != expected:
            part.unlink(missing_ok=True)
            raise ValueError(f"官方参考文件下载哈希错误：{name}")
        os.replace(part, path)
        print(f"  已下载并校验 {name}", flush=True)


def raw_symbol(value: object) -> str:
    return re.sub(r"\s+\([^)]+\)$", "", str(value).strip()).upper()


def read_reference(reference_dir: Path):
    fingerprint_raw = pyreadr.read_r(str(reference_dir / "gene_fingerprints_CGP.RData"))["fingerprint"]
    expression_index = pyreadr.read_r(
        str(reference_dir / "ccle_exp_for_missing_value_6016.RData"))["exp.index"]
    default_targets = pyreadr.read_r(
        str(reference_dir / "default_dep_genes_1298.RData"))["dep.data"]
    feature_genes = expression_index["Gene"].astype(str).str.upper().to_numpy(str)
    feature_means = pd.to_numeric(expression_index["Mean"], errors="raise").to_numpy(np.float32)
    target_symbols = default_targets["Gene"].astype(str).str.upper().to_numpy(str)

    fingerprint_genes = fingerprint_raw.iloc[0, 1:].astype(str).str.upper().to_numpy(str)
    fingerprint_sets = fingerprint_raw.iloc[1:, 0].astype(str).to_numpy(str)
    values = fingerprint_raw.iloc[1:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(np.uint8).T
    if values.shape != (len(fingerprint_genes), 3115) or not np.isin(values, [0, 1]).all():
        raise ValueError("官方CGP指纹维度或取值异常")
    fingerprint_index = {gene: index for index, gene in enumerate(fingerprint_genes)}
    if len(fingerprint_index) != len(fingerprint_genes):
        raise ValueError("官方CGP指纹存在重复基因")
    return feature_genes, feature_means, target_symbols, fingerprint_sets, values, fingerprint_index


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depmap-dir", type=Path,
                        default=root / "data/processed/depmap_baseline_24q4_v1")
    parser.add_argument("--reference-dir", type=Path,
                        default=root / "data/raw/deepdep_official_f713c84")
    parser.add_argument("--hgnc", type=Path, required=True,
                        help="HGNC complete-set TSV used by the existing DepMap bridge")
    parser.add_argument("--splits", type=Path,
                        default=root / "results/historical/selective_dependency_benchmark_v1/splits.csv.gz")
    parser.add_argument("--protocol", type=Path,
                        default=root / "configs/advanced_model_benchmark_protocol_20260917.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data/processed/advanced_model_benchmark_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("【阶段 1/4】下载并校验官方 DeepDEP 特征与CGP指纹", flush=True)
    download_reference(args.reference_dir)
    for path in (args.hgnc, args.splits, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(args.protocol.read_text())
    if protocol["data"]["locked_test_rule"].startswith("No TCGA KIRC Locked Test") is False:
        raise ValueError("冻结协议的 Locked Test 隔离规则异常")

    print("【阶段 2/4】对齐官方6,016表达特征和1,298个默认靶点", flush=True)
    feature_symbols, feature_means, target_symbols, fingerprint_sets, fingerprints, fp_index = read_reference(
        args.reference_dir)
    models, depmap_genes, matrices, common_essential = baseline.load_data(args.depmap_dir)
    gene_index = {str(gene): index for index, gene in enumerate(depmap_genes)}
    canonicalizer = GeneCanonicalizer(args.hgnc)

    feature_canonical = np.asarray([canonicalizer.map(symbol) or "" for symbol in feature_symbols])
    feature_source = np.asarray([gene_index.get(symbol, -1) for symbol in feature_canonical], dtype=np.int32)
    expression = np.broadcast_to(feature_means, (len(models), len(feature_symbols))).copy()
    available = feature_source >= 0
    expression[:, available] = matrices["expression"][:, feature_source[available]]
    if not np.isfinite(expression).all():
        raise ValueError("DeepDEP表达输入仍包含非有限值")

    target_rows = []
    seen = set()
    for original in target_symbols:
        canonical = canonicalizer.map(original)
        if not canonical or canonical in seen or canonical not in gene_index or original not in fp_index:
            continue
        seen.add(canonical)
        target_rows.append((original, canonical, gene_index[canonical], fp_index[original]))
    if len(target_rows) != 1204:
        raise ValueError(f"官方DeepDEP靶点交集发生变化：期望1204，实际{len(target_rows)}")
    original_targets = np.asarray([row[0] for row in target_rows])
    target_genes = np.asarray([row[1] for row in target_rows])
    dependency = matrices["dependency"][:, [row[2] for row in target_rows]].astype(np.float32)
    target_fingerprints = fingerprints[[row[3] for row in target_rows]]
    target_common = common_essential[[row[2] for row in target_rows]]
    if int((~target_common).sum()) != 911:
        raise ValueError("主评价非common-essential靶点数发生变化")

    split_frame = pd.read_csv(args.splits)
    if set(split_frame.ModelID) - set(models.index):
        raise ValueError("冻结split包含未知模型")
    split_lineages = sorted(split_frame.heldout_lineage.unique())
    if len(split_lineages) != 19:
        raise ValueError("冻结split癌系数量异常")

    print(f"【共同空间】模型 {len(models)}｜靶点 {len(target_genes)}｜主评价靶点 {(~target_common).sum()}", flush=True)
    print(f"【表达覆盖】官方特征 {len(feature_symbols)}｜DepMap实测 {available.sum()}｜官方均值填补 {(~available).sum()}", flush=True)
    if args.dry_run:
        print("【检查通过】未生成矩阵、未训练模型、未读取TCGA KIRC Test。", flush=True)
        return

    print("【阶段 3/4】保存共同输入矩阵和来源映射", flush=True)
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}-", dir=args.output_dir.parent))
    try:
        np.savez_compressed(
            temporary / "benchmark_inputs.npz",
            model_ids=models.index.to_numpy(str),
            expression_feature_symbols=feature_symbols,
            expression_feature_canonical=feature_canonical,
            expression_feature_available=available,
            expression=expression.astype(np.float32),
            target_original_symbols=original_targets,
            target_genes=target_genes,
            dependency=dependency,
            target_fingerprints=target_fingerprints,
            fingerprint_sets=fingerprint_sets,
            target_common_essential=target_common,
        )
        feature_map = pd.DataFrame({
            "deepdep_feature": feature_symbols,
            "canonical_symbol": feature_canonical,
            "depmap_observed": available,
            "official_ccle_imputation_mean": feature_means,
        })
        feature_map.to_csv(temporary / "expression_feature_mapping.csv.gz", index=False)
        target_map = pd.DataFrame({
            "deepdep_target": original_targets,
            "canonical_target": target_genes,
            "common_essential": target_common,
        })
        target_map.to_csv(temporary / "target_mapping.csv", index=False)
        run = {
            "status": "advanced_model_benchmark_inputs_complete_no_model_fitted",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_n": len(models),
            "expression_feature_n": len(feature_symbols),
            "expression_observed_feature_n": int(available.sum()),
            "expression_imputed_feature_n": int((~available).sum()),
            "target_n": len(target_genes),
            "primary_non_common_essential_target_n": int((~target_common).sum()),
            "lineage_n": len(split_lineages),
            "tcga_kirc_test_read": False,
            "sources": {
                "prep4deepdep_commit": PREP_COMMIT,
                "reference_sha256": {name: sha256(args.reference_dir / name) for name in REFERENCE_FILES},
                "depmap_audit_sha256": sha256(args.depmap_dir / "audit.json"),
                "depmap_matrices_sha256": sha256(args.depmap_dir / "matrices.npz"),
                "hgnc_sha256": sha256(args.hgnc),
                "splits_sha256": sha256(args.splits),
                "protocol_sha256": sha256(args.protocol),
                "script_sha256": sha256(Path(__file__)),
            },
            "rules": {
                "expression_order": "Official DeepDEP 6016-feature order",
                "missing_expression": "Official CCLE reference mean",
                "target_universe": "Official 1298 DepOIs intersected with canonical DepMap 24Q4 targets",
                "test_isolation": "No TCGA file or patient matrix is read by this script",
            },
        }
        run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
        (temporary / "audit.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        temporary.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print("【阶段 4/4】输入审计完成", flush=True)
    print(f"【完成】结果 {args.output_dir}", flush=True)
    print("【结论边界】本步只构建同数据输入；没有模型结果，也没有读取Locked Test。", flush=True)


if __name__ == "__main__":
    main()
