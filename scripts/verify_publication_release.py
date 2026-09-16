#!/usr/bin/env python3
"""One-command integrity and environment audit for the publication snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(directory: Path) -> int:
    run_path = directory / "run.json"
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    run = json.loads(run_path.read_text())
    checked = 0
    for name, expected in run.get("output_sha256", {}).items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"版本化结果缺失：{path}")
        if sha256(path) != expected:
            raise ValueError(f"版本化结果哈希不一致：{path}")
        checked += 1
    if checked == 0:
        raise ValueError(f"运行记录没有输出哈希：{run_path}")
    return checked


def verify_environment(root: Path, strict: bool) -> None:
    lock = json.loads((root / "configs/environment_lock.json").read_text())
    mismatches = []
    observed_python = ".".join(map(str, sys.version_info[:3]))
    if observed_python != lock["python"]:
        mismatches.append(f"python {observed_python} != {lock['python']}")
    for package, expected in lock["packages"].items():
        if package == "torch":
            continue
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{package} 未安装")
            continue
        if actual != expected:
            mismatches.append(f"{package} {actual} != {expected}")
    import torch
    if torch.__version__ != lock["packages"]["torch"]:
        mismatches.append(f"torch {torch.__version__} != {lock['packages']['torch']}")
    if not torch.cuda.is_available():
        mismatches.append("CUDA 不可用")
    elif str(torch.version.cuda) != lock["cuda_runtime_reported_by_torch"]:
        mismatches.append(
            f"torch CUDA {torch.version.cuda} != {lock['cuda_runtime_reported_by_torch']}")
    if strict and mismatches:
        raise RuntimeError("环境锁定不一致：" + "；".join(mismatches))
    print("【环境审计】" + ("完全匹配" if not mismatches else "存在差异：" + "；".join(mismatches)), flush=True)


def verify_claims(root: Path) -> None:
    evidence_dir = root / "results/historical/final_evidence_synthesis_v1"
    matrix = pd.read_csv(evidence_dir / "candidate_evidence_matrix.csv")
    if len(matrix) != 20 or matrix.Gene.nunique() != 20:
        raise ValueError("最终证据矩阵不是唯一的20个候选")
    if matrix.common_essential.astype(bool).any():
        raise ValueError("最终候选意外包含 common-essential")
    if set(matrix.loc[matrix.orthogonal_rnai_support.astype(bool), "Gene"]) != {"PAX8", "HNF1B", "FERMT2"}:
        raise ValueError("RNAi复现候选集合发生变化")
    if set(matrix.loc[matrix.rnai_ccrcc_specific_support.astype(bool), "Gene"]) != {"PAX8"}:
        raise ValueError("RNAi ccRCC特异方向候选集合发生变化")
    if matrix.patient_derived_functional_truth_available.astype(bool).any():
        raise ValueError("患者功能真值状态被意外升级")

    test_dir = root / "results/historical/tcga_locked_test_v1"
    test_run = json.loads((test_dir / "run.json").read_text())
    protocol = root / "configs/tcga_locked_test_protocol_20260916.json"
    if sha256(protocol) != test_run["protocol_sha256"]:
        raise ValueError("锁定Test协议哈希与正式运行不一致")
    if test_run["test_access"] != {"formal_access_n": 1, "patient_n": 51, "paired_normal_n": 14}:
        raise ValueError("锁定Test访问审计发生变化")
    primary = test_run["primary_results"]
    expected = {
        "train_test_top100_overlap": 0.75,
        "frozen20_validation_test_frequency_spearman": 0.7984154302092542,
        "test_mapping_top10_overlap_mean": 0.36078431372549025,
    }
    for name, value in expected.items():
        if abs(float(primary[name]) - value) > 1e-12:
            raise ValueError(f"锁定Test关键结果发生变化：{name}")


def verify_portability(root: Path) -> None:
    offenders = []
    for base in (root / "scripts", root / "configs"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            linux_prefix = "/mnt/e/" + "projects/"
            windows_prefix = "E:\\" + "Projects\\"
            if linux_prefix in text or windows_prefix in text:
                offenders.append(str(path.relative_to(root)))
    if offenders:
        raise ValueError("仍含机器绝对路径：" + ", ".join(offenders))


def verify_artifacts(root: Path) -> None:
    figures = root / "results/historical/final_figures_v1"
    for path in figures.glob("*.png"):
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"PNG签名无效：{path}")
    for path in figures.glob("*.pdf"):
        if path.read_bytes()[:4] != b"%PDF":
            raise ValueError(f"PDF签名无效：{path}")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=root / "configs/publication_release_manifest.json")
    parser.add_argument("--allow-environment-drift", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text())
    print("【阶段 1/4】核验发布快照文件和内部哈希", flush=True)
    for relative in manifest["required_scripts"] + manifest["frozen_protocols"]:
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    checked = 0
    for relative in manifest["snapshot_directories"]:
        checked += verify_run(root / relative)
    print(f"【快照完整】目录 {len(manifest['snapshot_directories'])}｜哈希文件 {checked}", flush=True)

    print("【阶段 2/4】核验冻结候选、结论边界和一次性Test协议", flush=True)
    verify_claims(root)
    print("【科学边界】候选20｜RNAi复现3｜亚型特异方向1｜患者功能真值0", flush=True)

    print("【阶段 3/4】核验环境与机器路径可移植性", flush=True)
    verify_environment(root, strict=not args.allow_environment_drift)
    verify_portability(root)
    print("【路径审计】scripts/ 与 configs/ 无机器绝对路径", flush=True)

    print("【阶段 4/4】核验PNG/PDF发布图件", flush=True)
    verify_artifacts(root)
    print("【复现通过】版本化结果、协议、环境、路径和图件全部一致", flush=True)
    print("【结论边界】该命令验证发布快照完整性；完整原始数据重跑仍需按来源许可准备大文件。", flush=True)


if __name__ == "__main__":
    main()
