"""Audit renal cell-type exposure and leakage-safe TCGA tumor-normal expression.

The frozen discovery top 20 are never reranked. HPA single-cell-type nCPM is
reported only for kidney-exclusive epithelial cell-type labels. TCGA contrasts
use the historical Train normal samples as the reference, report Validation
separately, and exclude every locked-Test participant and sample.
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

from build_context_module_stage0 import GeneCanonicalizer, patient_id, sample_type
from prepare_tcga_expression_bridge import align_expression


SOURCES = {
    "hpa_cell_type": {
        "url": "https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip",
        "filename": "rna_single_cell_type.tsv.zip",
        "bytes": 16346850,
        "sha256": "2ddd3d90fb050a8efea508792ba7741c510fb5aa3d89f352aab99794ec2c9d64",
    },
    "hpa_clusters": {
        "url": "https://www.proteinatlas.org/download/tsv/rna_single_cell_clusters.tsv.zip",
        "filename": "rna_single_cell_clusters.tsv.zip",
        "bytes": 14547,
        "sha256": "bd23b3f354774658ac0bd0fafd0699feefea29078e1549514fbe3707d5ee2ac9",
    },
    "hpa_datasets": {
        "url": "https://www.proteinatlas.org/download/tsv/rna_single_cell_datasets.tsv.zip",
        "filename": "rna_single_cell_datasets.tsv.zip",
        "bytes": 1192,
        "sha256": "c37f6758c16aa3ab38fe16af479339ccf62c249f33e1e4a10bcb7df250f64bea",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: dict, raw_dir: Path) -> tuple[Path, bool]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / source["filename"]
    reused = path.exists()
    if not reused:
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(source["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.rename(path)
    if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
        raise ValueError(f"HPA 来源版本或文件内容变化：{path.name}")
    return path, reused


def verify_outputs(directory: Path) -> dict:
    run = json.loads((directory / "run.json").read_text())
    for name, expected in run["output_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"上游输出哈希不一致：{directory.name}/{name}")
    return run


def load_candidates(frozen_dir: Path, orthogonal_dir: Path) -> pd.DataFrame:
    verify_outputs(frozen_dir)
    verify_outputs(orthogonal_dir)
    frozen = pd.read_csv(frozen_dir / "candidates.csv")[["Gene", "discovery_rank"]]
    if len(frozen) != 20 or frozen.discovery_rank.astype(int).tolist() != list(range(1, 21)):
        raise ValueError("冻结候选不再是原始 discovery rank 1-20")
    prior = pd.read_csv(orthogonal_dir / "candidate_orthogonal_summary.csv")
    if prior.Gene.tolist() != frozen.Gene.tolist():
        raise ValueError("正交验证候选顺序与冻结队列不一致")
    return frozen.merge(prior.drop(columns="discovery_rank"), on="Gene", validate="one_to_one")


def hpa_evidence(candidates: pd.DataFrame, paths: dict[str, Path]):
    clusters = pd.read_csv(paths["hpa_clusters"], sep="\t")
    datasets = pd.read_csv(paths["hpa_datasets"], sep="\t")
    kidney_all = clusters[clusters.Tissue.eq("kidney")].copy()
    kidney_used = kidney_all[kidney_all["Included in aggregation"].eq("yes")].copy()
    tissue_counts = clusters.groupby("Cell type").Tissue.nunique()
    renal_types = sorted(
        cell_type for cell_type in kidney_used["Cell type"].unique()
        if tissue_counts.get(cell_type, 0) == 1
    )
    expected = {
        "distal convoluted tubule cells", "loop of henle epithelial cells",
        "papillary tip epithelial cells", "podocytes", "proximal tubule cells",
        "renal collecting duct intercalated cells",
        "renal collecting duct principal cells", "renal connecting tubule cells",
    }
    if set(renal_types) != expected:
        raise ValueError(f"HPA kidney-exclusive cell types changed: {renal_types}")
    kidney_dataset = datasets[datasets.Tissue.eq("kidney")]
    if len(kidney_dataset) != 1 or int(kidney_dataset.iloc[0]["Cell count"]) != 60929:
        raise ValueError("HPA kidney dataset metadata changed")

    expression = pd.read_csv(paths["hpa_cell_type"], sep="\t",
                             usecols=["Gene", "Gene name", "Cell type", "nCPM"])
    long = expression[
        expression["Gene name"].isin(candidates.Gene) & expression["Cell type"].isin(renal_types)
    ].copy()
    if len(long) != len(candidates) * len(renal_types):
        raise ValueError("HPA renal cell-type candidate coverage is incomplete")
    long = candidates[["Gene", "discovery_rank"]].merge(
        long.rename(columns={"Gene": "Ensembl", "Gene name": "Gene"}),
        on="Gene", validate="one_to_many")
    summary_rows = []
    for gene, current in long.groupby("Gene", sort=False):
        maximum = current.loc[current.nCPM.idxmax()]
        proximal = current.loc[current["Cell type"].eq("proximal tubule cells"), "nCPM"].iloc[0]
        summary_rows.append({
            "Gene": gene,
            "hpa_renal_cell_type_n": len(current),
            "hpa_renal_cell_type_max_ncpm": float(maximum.nCPM),
            "hpa_renal_cell_type_max_label": maximum["Cell type"],
            "hpa_proximal_tubule_ncpm": float(proximal),
            "hpa_renal_cell_type_positive_n": int(current.nCPM.gt(0).sum()),
        })
    metadata = {
        "kidney_cluster_n": len(kidney_all),
        "kidney_included_cluster_n": len(kidney_used),
        "kidney_included_high_reliability_cluster_n": int(
            kidney_used["Annotation reliability"].eq("high").sum()),
        "kidney_cell_n": int(kidney_dataset.iloc[0]["Cell count"]),
        "kidney_technique": kidney_dataset.iloc[0].Technique,
        "kidney_source": kidney_dataset.iloc[0]["Data source"],
        "kidney_pubmed_id": int(kidney_dataset.iloc[0].pubmed_id),
        "renal_cell_types": renal_types,
    }
    return long.sort_values(["discovery_rank", "Cell type"]), pd.DataFrame(summary_rows), metadata


def bootstrap_median(values: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    index = torch.randint(len(tensor), (draws, len(tensor)), generator=generator, device="cuda")
    medians = torch.quantile(tensor[index], .5, dim=1)
    interval = torch.quantile(
        medians, torch.tensor([.025, .5, .975], dtype=torch.float64, device="cuda"))
    return tuple(float(value) for value in interval.cpu())


def tcga_evidence(candidates: pd.DataFrame, patients_path: Path, expression_path: Path,
                  hgnc_path: Path, draws: int, seed: int):
    patients = pd.read_csv(patients_path, index_col=0)
    expected = {"train": 156, "validation": 53, "test": 51}
    if patients.split.value_counts().to_dict() != expected:
        raise ValueError("TCGA 历史锁定分组发生变化")
    header = pd.read_csv(expression_path, sep="\t", nrows=0).columns.astype(str).tolist()[1:]
    normal_samples = sorted(sample for sample in header if sample_type(sample) == "11")
    normal_by_patient = pd.Series(normal_samples, index=[patient_id(sample) for sample in normal_samples])
    if normal_by_patient.index.duplicated().any():
        raise ValueError("同一 TCGA 参与者存在多个邻近正常表达样本")
    normal_split = normal_by_patient.index.to_series().map(patients.split)
    split_counts = normal_split.value_counts().to_dict()
    if split_counts != {"train": 37, "validation": 14, "test": 14}:
        raise ValueError(f"TCGA 正常样本分组变化：{split_counts}")
    train_normal_ids = normal_by_patient[normal_split.eq("train")].tolist()
    validation_normal_ids = normal_by_patient[normal_split.eq("validation")].tolist()
    selected = patients[patients.split.isin(["train", "validation"])]
    tumor_ids = selected.expression_sample.astype(str).tolist()
    sample_ids = tumor_ids + train_normal_ids + validation_normal_ids
    aligned, mapping = align_expression(
        expression_path, sample_ids, candidates.Gene.tolist(), GeneCanonicalizer(hgnc_path))
    train_normal = aligned.loc[train_normal_ids]
    reference = train_normal.mean(axis=0)

    rows = []
    for rank, gene in candidates[["discovery_rank", "Gene"]].itertuples(index=False):
        row = {"Gene": gene}
        for split in ("train", "validation"):
            subset = selected[selected.split.eq(split)]
            tumors = aligned.loc[subset.expression_sample.astype(str).tolist(), gene].to_numpy(float)
            row[f"tcga_{split}_tumor_minus_train_normal_median"] = float(np.median(tumors - reference[gene]))
            row[f"tcga_{split}_tumor_above_train_normal_fraction"] = float(np.mean(tumors > reference[gene]))
            paired_patients = normal_by_patient.index.intersection(subset.index)
            paired_tumors = subset.loc[paired_patients, "expression_sample"].astype(str).tolist()
            paired_normals = normal_by_patient.loc[paired_patients].tolist()
            paired = (aligned.loc[paired_tumors, gene].to_numpy(float)
                      - aligned.loc[paired_normals, gene].to_numpy(float))
            low, median, high = bootstrap_median(paired, draws, seed + int(rank) * 10 + (split == "validation"))
            row[f"tcga_{split}_paired_n"] = len(paired)
            row[f"tcga_{split}_paired_median"] = float(np.median(paired))
            row[f"tcga_{split}_paired_median_ci_low"] = low
            row[f"tcga_{split}_paired_median_ci_high"] = high
        rows.append(row)
    audit = {
        "normal_total_n": len(normal_samples),
        "normal_train_n": len(train_normal_ids),
        "normal_validation_n": len(validation_normal_ids),
        "normal_test_excluded_n": int(normal_split.eq("test").sum()),
        "normal_outside_frozen_cohort_excluded_n": int(normal_split.isna().sum()),
        "tumor_train_n": int(selected.split.eq("train").sum()),
        "tumor_validation_n": int(selected.split.eq("validation").sum()),
        "locked_test_tumor_n_excluded": int(patients.split.eq("test").sum()),
        "aligned_candidate_gene_n": mapping["aligned_finite_gene_n"],
    }
    return pd.DataFrame(rows), audit


def parse_args():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/e/projects/rl-genrisk-main")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1")
    parser.add_argument("--orthogonal-dir", type=Path,
                        default=root / "outputs/candidate_orthogonal_validation_v1")
    parser.add_argument("--raw-dir", type=Path,
                        default=root / "data/raw/candidate_celltype_window_20260915")
    parser.add_argument("--patients", type=Path,
                        default=source / "data/processed/context_module_stage0/patients.csv")
    parser.add_argument("--expression", type=Path, default=source / "data/raw/HiSeqV2")
    parser.add_argument("--hgnc", type=Path,
                        default=source / "outputs/reassessment_20260911/hgnc_complete_set.tsv")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/candidate_celltype_window_v1")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不允许静默回退到 CPU")
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")

    print("【阶段 1/5】校验冻结候选、上游哈希与 TCGA Test 隔离", flush=True)
    candidates = load_candidates(args.frozen_dir, args.orthogonal_dir)
    print("【冻结规则】候选 20｜顺序不变｜Train正常拟合｜Validation单列｜Test完全排除", flush=True)
    if args.dry_run:
        for path in (args.patients, args.expression, args.hgnc):
            if not path.exists():
                raise FileNotFoundError(path)
        print(f"【检查通过】CUDA {torch.cuda.get_device_name(0)}｜未读取表达数值、未写入结果", flush=True)
        return

    print("【阶段 2/5】校验 HPA v25.1 单细胞类型来源", flush=True)
    paths, source_audit = {}, {}
    for name, source in SOURCES.items():
        path, reused = download(source, args.raw_dir)
        paths[name] = path
        source_audit[name] = {**source, "path": str(path.resolve()), "cache_reused": reused}
    print("【来源校验】3/3 文件大小与 SHA256 一致", flush=True)

    print("【阶段 3/5】提取 kidney-exclusive 上皮细胞类型 nCPM", flush=True)
    hpa_long, hpa_summary, hpa_audit = hpa_evidence(candidates, paths)
    print(f"【HPA覆盖】肾脏细胞核 {hpa_audit['kidney_cell_n']}｜cluster {hpa_audit['kidney_cluster_n']}｜"
          f"肾脏专属细胞类型 {len(hpa_audit['renal_cell_types'])}｜候选 20/20", flush=True)

    print("【阶段 4/5】GPU计算无 Test 的 TCGA 配对肿瘤-正常区间", flush=True)
    tcga, tcga_audit = tcga_evidence(
        candidates, args.patients, args.expression, args.hgnc, args.draws, args.seed)
    print(f"【TCGA隔离】Train正常 {tcga_audit['normal_train_n']}｜Validation正常 "
          f"{tcga_audit['normal_validation_n']}｜排除Test正常 {tcga_audit['normal_test_excluded_n']}｜"
          f"排除队列外正常 {tcga_audit['normal_outside_frozen_cohort_excluded_n']}", flush=True)

    print("【阶段 5/5】保存细胞类型暴露、配对表达和来源审计", flush=True)
    summary = candidates.merge(hpa_summary, on="Gene", validate="one_to_one").merge(
        tcga, on="Gene", validate="one_to_one")
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    hpa_long.to_csv(temporary / "hpa_renal_celltype_expression.csv", index=False)
    tcga.to_csv(temporary / "tcga_tumor_normal_expression.csv", index=False)
    summary.to_csv(temporary / "candidate_celltype_window_summary.csv", index=False)
    run = {
        "status": "celltype_exposure_and_leakage_safe_tumor_normal_audit_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "device": torch.cuda.get_device_name(0),
        "dtype": "float64_gpu_bootstrap",
        "design": {
            "frozen_candidate_n": len(candidates), "bootstrap_draws": args.draws,
            "seed": args.seed, **hpa_audit, **tcga_audit,
        },
        "rules": {
            "selection": "Existing discovery ranks 1-20 frozen; no reranking or threshold",
            "hpa": "nCPM for cell-type labels exclusive to the HPA kidney dataset",
            "tcga_reference": "Only 37 historical-Train adjacent normals define the bulk reference",
            "tcga_validation": "Validation tumors and paired normals are reported separately",
            "tcga_test": "All 51 Test tumors and 14 Test adjacent normals are excluded",
            "aggregation": "No composite efficacy, toxicity or treatment-window score",
        },
        "sources": source_audit,
        "source_sha256": {
            "frozen_candidates": sha256(args.frozen_dir / "candidates.csv"),
            "orthogonal_summary": sha256(args.orthogonal_dir / "candidate_orthogonal_summary.csv"),
            "patients": sha256(args.patients), "expression": sha256(args.expression),
            "hgnc": sha256(args.hgnc), "script": sha256(Path(__file__)),
        },
        "limitations": [
            "HPA kidney values aggregate a single 60,929-nucleus dataset containing healthy and injured states; they are not healthy-donor replicates.",
            "Kidney-exclusive labels cover renal epithelial types but omit kidney-specific stromal and immune expression because those labels also occur in other tissues.",
            "Single-nucleus nCPM and TCGA bulk log-expression units cannot be divided or directly compared.",
            "TCGA adjacent normal tissue may contain field effects and is not a healthy-organ toxicity assay.",
            "Expression exposure does not establish normal-cell essentiality, drug toxicity or a therapeutic window.",
            "Bootstrap intervals are descriptive and not adjusted for 20 candidates.",
        ],
    }
    run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    for row in summary.sort_values("discovery_rank").itertuples():
        print(f"  {int(row.discovery_rank):02d} {row.Gene}｜肾上皮最高 {row.hpa_renal_cell_type_max_ncpm:.1f} nCPM "
              f"({row.hpa_renal_cell_type_max_label})｜Validation配对差 "
              f"{row.tcga_validation_paired_median:+.3f} "
              f"[{row.tcga_validation_paired_median_ci_low:+.3f}, "
              f"{row.tcga_validation_paired_median_ci_high:+.3f}]", flush=True)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】表达只能定位暴露；不能证明正常细胞毒性、患者功能依赖或治疗窗。", flush=True)


if __name__ == "__main__":
    main()
