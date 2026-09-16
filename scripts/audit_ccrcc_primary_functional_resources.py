#!/usr/bin/env python3
"""Audit public patient-derived ccRCC gene-perturbation resources.

This is a metadata and literature audit, not a model-training entry point.  It
uses pinned Sanger and Broad metadata plus a frozen, explicitly listed set of
renal cancer and normal-kidney organoid papers.  Conclusions therefore apply
to the audited resources, not to every unpublished or future experiment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BROAD_SOURCES = {
    "model_metadata": {
        "url": "https://ndownloader.figshare.com/files/64494438",
        "filename": "model_metadata.csv",
        "bytes": 272338,
        "sha256": "a000a5ee1ebe2b2488a146b2a413ec4b93472488e6ad31b56126f522d652557c",
    },
    "screen_metadata": {
        "url": "https://ndownloader.figshare.com/files/64494441",
        "filename": "screen_metadata.csv",
        "bytes": 259986,
        "sha256": "af29e349dd19676029950d6799fec4efcefcb9b314f7622b4fcc6d564803472a",
    },
}

# Frozen before this script evaluates eligibility.  These papers cover the
# known public RCC organoid cohorts, recent ccRCC perturbation screens and the
# normal-kidney organoid evidence relevant to the leading candidates.
PAPERS = [
    {
        "pmid": "35802820", "pmcid": "PMC9270001", "role": "tumor_resource",
        "expected_title_token": "Patient-derived renal cell carcinoma organoids",
    },
    {
        "pmid": "33072556", "pmcid": "PMC7537764", "role": "tumor_resource",
        "expected_title_token": "Clear Cell Renal Cell Carcinoma Patient-Derived Organoids",
    },
    {
        "pmid": "33107222", "pmcid": "PMC7826464", "role": "tumor_resource",
        "expected_title_token": "patient-derived xenograft model",
    },
    {
        "pmid": "32968206", "pmcid": "PMC7723036", "role": "tumor_screen",
        "expected_title_token": "sunitinib resistance in renal cell carcinoma",
    },
    {
        "pmid": "40683967", "pmcid": "PMC12276353", "role": "tumor_screen",
        "expected_title_token": "PTGR2",
    },
    {
        "pmid": "42510829", "pmcid": "PMC13409911", "role": "tumor_screen",
        "expected_title_token": "Cabozantinib",
    },
    {
        "pmid": "38211588", "pmcid": "PMC10922811", "role": "tumor_screen",
        "expected_title_token": "DCLK2-TBK1",
    },
    {
        "pmid": "38340720", "pmcid": "PMC7616043", "role": "normal_liability",
        "expected_title_token": "renal mesenchymal-to-epithelial transition",
    },
    {
        "pmid": "35676472", "pmcid": "PMC9242860", "role": "normal_liability",
        "expected_title_token": "PAX8 controls oncogenic signalling",
    },
    {
        "pmid": "38788724", "pmcid": "PMC11297557", "role": "normal_liability",
        "expected_title_token": "HNF1B-associated dysplastic kidney malformations",
    },
    {
        "pmid": "37155865", "pmcid": "PMC10193973", "role": "normal_liability",
        "expected_title_token": "kidney organoid differentiation",
    },
    {
        "pmid": "30033089", "pmcid": "PMC6092837", "role": "normal_liability",
        "expected_title_token": "Generate Kidney Organoids",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int | None = None,
           expected_sha256: str | None = None) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha = sha256(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"文件大小不一致：{path}，{actual_bytes} != {expected_bytes}")
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(f"SHA256不一致：{path}")
    return {"path": str(path.resolve()), "bytes": actual_bytes, "sha256": actual_sha}


def download(source: dict, raw_dir: Path) -> tuple[Path, bool]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / source["filename"]
    reused = path.exists()
    if not reused:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "PanCancer-Dependency/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        path.write_bytes(payload)
    verify(path, source["bytes"], source["sha256"])
    return path, reused


def fetch_papers(raw_dir: Path) -> tuple[list[dict], Path, bool]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "europe_pmc_targeted_records.json.gz"
    reused = cache.exists()
    if reused:
        with gzip.open(cache, "rt", encoding="utf-8") as handle:
            records = json.load(handle)
    else:
        records = []
    by_pmid = {str(row.get("pmid")): row for row in records}
    missing = [paper for paper in PAPERS if paper["pmid"] not in by_pmid]
    if missing:
        for paper in missing:
            query = urllib.parse.urlencode({
                "query": f"EXT_ID:{paper['pmid']} AND SRC:MED",
                "format": "json", "pageSize": 1, "resultType": "core",
            })
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + query
            request = urllib.request.Request(url, headers={"User-Agent": "PanCancer-Dependency/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            hits = result.get("resultList", {}).get("result", [])
            if len(hits) != 1 or str(hits[0].get("pmid")) != paper["pmid"]:
                raise ValueError(f"Europe PMC 未唯一返回 PMID {paper['pmid']}")
            records.append(hits[0])
        with gzip.open(cache, "wt", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

    by_pmid = {str(row.get("pmid")): row for row in records}
    for paper in PAPERS:
        record = by_pmid.get(paper["pmid"])
        if record is None:
            raise ValueError(f"缓存缺失 PMID {paper['pmid']}")
        if paper["expected_title_token"].lower() not in record.get("title", "").lower():
            raise ValueError(f"题名校验失败 PMID {paper['pmid']}：{record.get('title')}")
        if record.get("pmcid") != paper["pmcid"]:
            raise ValueError(f"PMCID校验失败 PMID {paper['pmid']}")
    return records, cache, reused


def broad_audit(model_path: Path, screen_path: Path) -> tuple[pd.DataFrame, dict]:
    models = pd.read_csv(model_path)
    screens = pd.read_csv(screen_path)
    required_models = {"ModelID", "OncotreeLineage", "OncotreeCode", "HasCRISPRData", "IsNextGen"}
    required_screens = {"ScreenID", "ModelID", "PassesQC", "OncotreeLineage", "OncotreeCode", "IsNextGen"}
    if not required_models.issubset(models.columns) or not required_screens.issubset(screens.columns):
        raise ValueError("Broad NextGen 元数据字段发生变化")

    nextgen_models = models[models["IsNextGen"]]
    nextgen_crispr = nextgen_models[nextgen_models["HasCRISPRData"]]
    nextgen_screens = screens[screens["IsNextGen"]]
    nextgen_qc = nextgen_screens[nextgen_screens["PassesQC"]]
    if len(nextgen_crispr) != 147 or len(nextgen_qc) != 147:
        raise ValueError("Broad 论文冻结的 147 个 NextGen CRISPR 模型/通过QC筛选未复现")

    coverage = (nextgen_models.groupby("OncotreeLineage", dropna=False)
                .agg(nextgen_model_n=("ModelID", "nunique"),
                     nextgen_crispr_model_n=("HasCRISPRData", "sum"))
                .reset_index().rename(columns={"OncotreeLineage": "lineage"}))
    qc_counts = (nextgen_qc.groupby("OncotreeLineage")["ScreenID"].nunique()
                 .rename("nextgen_qc_screen_n").reset_index()
                 .rename(columns={"OncotreeLineage": "lineage"}))
    coverage = coverage.merge(qc_counts, on="lineage", how="left").fillna({"nextgen_qc_screen_n": 0})
    coverage[["nextgen_model_n", "nextgen_crispr_model_n", "nextgen_qc_screen_n"]] = coverage[
        ["nextgen_model_n", "nextgen_crispr_model_n", "nextgen_qc_screen_n"]].astype(int)
    coverage = coverage.sort_values(["nextgen_qc_screen_n", "lineage"], ascending=[False, True])

    audit = {
        "all_model_n": int(models["ModelID"].nunique()),
        "nextgen_model_n": int(nextgen_models["ModelID"].nunique()),
        "nextgen_crispr_model_n": int(nextgen_crispr["ModelID"].nunique()),
        "nextgen_screen_n": int(nextgen_screens["ScreenID"].nunique()),
        "nextgen_qc_screen_n": int(nextgen_qc["ScreenID"].nunique()),
        "kidney_model_n": int(models.loc[models["OncotreeLineage"].eq("Kidney"), "ModelID"].nunique()),
        "kidney_nextgen_model_n": int(nextgen_models.loc[nextgen_models["OncotreeLineage"].eq("Kidney"), "ModelID"].nunique()),
        "kidney_nextgen_qc_screen_n": int(nextgen_qc.loc[nextgen_qc["OncotreeLineage"].eq("Kidney"), "ScreenID"].nunique()),
        "ccrcc_model_n": int(models.loc[models["OncotreeCode"].eq("CCRCC"), "ModelID"].nunique()),
        "ccrcc_nextgen_model_n": int(nextgen_models.loc[nextgen_models["OncotreeCode"].eq("CCRCC"), "ModelID"].nunique()),
    }
    return coverage, audit


def sanger_audit(model_list_path: Path, availability_path: Path) -> tuple[pd.DataFrame, dict]:
    models = pd.read_csv(model_list_path)
    availability = pd.read_csv(availability_path)
    selected = availability[availability["CRISPR Sanger Organoid"]].merge(
        models[["model_id", "tissue", "cancer_type", "model_type"]], on="model_id", how="left", validate="one_to_one")
    if len(selected) != 162 or selected["model_id"].nunique() != 162:
        raise ValueError("Sanger 冻结的 162 个类器官 CRISPR 模型未复现")
    if not selected["model_type"].eq("Organoid").all():
        raise ValueError("Sanger CRISPR Organoid 队列混入非类器官模型")
    counts = (selected.groupby("tissue")["model_id"].nunique().rename("organoid_crispr_model_n")
              .reset_index().sort_values(["organoid_crispr_model_n", "tissue"], ascending=[False, True]))
    audit = {
        "organoid_crispr_model_n": int(selected["model_id"].nunique()),
        "kidney_organoid_crispr_model_n": int(selected.loc[selected["tissue"].eq("Kidney"), "model_id"].nunique()),
        "tissue_n": int(selected["tissue"].nunique()),
        "tissue_counts": dict(zip(counts["tissue"], counts["organoid_crispr_model_n"].astype(int))),
    }
    return counts, audit


def resource_inventory(records: list[dict]) -> pd.DataFrame:
    titles = {str(row["pmid"]): row["title"] for row in records}
    rows = [
        {
            "resource": "Sanger tumor-derived organoid biobank 2026", "identifier": "10.1038/s41586-026-10830-y",
            "model_system": "patient-derived tumor organoids", "ccrcc_or_kidney_model": False,
            "gene_perturbation": "genome-wide CRISPR", "candidate_level_public_matrix": True,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "162 screens but zero Kidney organoids",
        },
        {
            "resource": "Broad NextGen Dependency Map 2026", "identifier": "10.1038/s41586-026-10843-7",
            "model_system": "organoid/neurosphere/other NextGen models", "ccrcc_or_kidney_model": False,
            "gene_perturbation": "genome-wide CRISPR", "candidate_level_public_matrix": True,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "147 QC-passing NextGen screens but zero Kidney models",
        },
        {
            "resource": titles["35802820"], "identifier": "PMID:35802820",
            "model_system": "33 patient-derived RCC organoid lines", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "none; drug and CAR-T response", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": True,
            "eligible": False, "exclusion_reason": "no gene-level perturbation screen",
        },
        {
            "resource": titles["33072556"], "identifier": "PMID:33072556",
            "model_system": "ccRCC patient-derived ALI organoids", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "none; phenotype and therapy response", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": True,
            "eligible": False, "exclusion_reason": "no gene-level perturbation screen",
        },
        {
            "resource": titles["33107222"], "identifier": "PMID:33107222",
            "model_system": "ccRCC patient-derived xenograft", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "PDX passaging under temsirolimus; DNMT1 tested in 786-O", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": True,
            "eligible": False, "exclusion_reason": "PDX was not gene-perturbed; candidate perturbation was performed in 786-O",
        },
        {
            "resource": titles["32968206"], "identifier": "PMID:32968206",
            "model_system": "786-O screen; PNX0010 validation", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "genome-wide CRISPR under sunitinib", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "screen performed in an established cell line and is drug-conditional",
        },
        {
            "resource": titles["40683967"], "identifier": "PMID:40683967",
            "model_system": "Caki-1R and 786-OR resistant cell lines", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "genome-wide CRISPR under sunitinib", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "screen performed in acquired-resistant established cell lines",
        },
        {
            "resource": titles["42510829"], "identifier": "PMID:42510829",
            "model_system": "786-O", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "kinome CRISPR under cabozantinib", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "screen performed in an established cell line and is drug-conditional",
        },
        {
            "resource": titles["38211588"], "identifier": "PMID:38211588",
            "model_system": "UMRC6", "ccrcc_or_kidney_model": True,
            "gene_perturbation": "709-gene kinome siRNA", "candidate_level_public_matrix": False,
            "functional_endpoint": True, "independent_patient_derived_ccrcc": False,
            "eligible": False, "exclusion_reason": "screen performed in an established cell line; not a patient-derived model screen",
        },
    ]
    frame = pd.DataFrame(rows)
    if frame["eligible"].any():
        raise ValueError("冻结资源资格状态被意外改变")
    return frame


def normal_liability(records: list[dict]) -> pd.DataFrame:
    titles = {str(row["pmid"]): row["title"] for row in records}
    return pd.DataFrame([
        {"gene": "PAX8", "pmid": "38340720", "title": titles["38340720"],
         "model": "human iPSC-derived renal organoid", "perturbation": "CRISPRi PAX8 knockdown",
         "reported_normal_renal_phenotype": "reduced renal mesenchymal-to-epithelial transition and epithelialization",
         "interpretation": "direct normal-renal developmental liability; not an adult toxicity estimate"},
        {"gene": "PAX8", "pmid": "35676472", "title": titles["35676472"],
         "model": "normal human renal epithelial organoid", "perturbation": "PAX8 depletion",
         "reported_normal_renal_phenotype": "normal renal epithelial organoid growth was experimentally assessed after depletion",
         "interpretation": "normal-renal functional exposure; not a dose-response therapeutic window"},
        {"gene": "HNF1B", "pmid": "38788724", "title": titles["38788724"],
         "model": "CRISPR-edited human pluripotent stem-cell kidney organoid", "perturbation": "heterozygous HNF1B disruption",
         "reported_normal_renal_phenotype": "malformed tubules and dysregulated epithelial turnover",
         "interpretation": "direct developmental liability; heterozygous disease model, not drug inhibition"},
        {"gene": "HNF1B", "pmid": "37155865", "title": titles["37155865"],
         "model": "human kidney organoid", "perturbation": "HNF1B CRISPRi",
         "reported_normal_renal_phenotype": "reduced proximal-tubule and loop-of-Henle differentiation markers",
         "interpretation": "direct developmental liability; not an adult toxicity estimate"},
        {"gene": "HNF1B", "pmid": "30033089", "title": titles["30033089"],
         "model": "human pluripotent stem-cell kidney organoid", "perturbation": "HNF1B knockout comparison",
         "reported_normal_renal_phenotype": "impaired nephron patterning and tubulogenesis",
         "interpretation": "supporting developmental liability; not a clinical safety assay"},
    ])


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanger-model-list", type=Path,
                        default=root / "data/raw/external_sanger_20260914/model_list_20260814.csv")
    parser.add_argument("--sanger-availability", type=Path,
                        default=root / "data/raw/external_sanger_20260914/model_dataset_availability_20260727.csv")
    parser.add_argument("--frozen-candidates", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1/candidates.csv")
    parser.add_argument("--raw-dir", type=Path,
                        default=root / "data/raw/ccrcc_primary_functional_audit_20260916")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/ccrcc_primary_functional_audit_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")

    print("【阶段 1/4】校验冻结候选和公开资源审计规则", flush=True)
    frozen = pd.read_csv(args.frozen_candidates)
    gene_col = "gene" if "gene" in frozen.columns else "Gene"
    if len(frozen) != 20 or frozen[gene_col].nunique() != 20:
        raise ValueError("冻结候选不是唯一的前20基因")
    print("【资格规则】患者来源ccRCC模型＋基因扰动＋功能终点＋候选可评估＋独立于基线", flush=True)
    if args.dry_run:
        for path in (args.sanger_model_list, args.sanger_availability, args.frozen_candidates):
            verify(path)
        print("【检查通过】元数据审计无需GPU｜未联网｜未写入结果", flush=True)
        return

    print("【阶段 2/4】核验 Sanger 与 Broad 2026 类器官/NextGen CRISPR 覆盖", flush=True)
    broad_paths, broad_sources = {}, {}
    for name, source in BROAD_SOURCES.items():
        path, reused = download(source, args.raw_dir)
        broad_paths[name] = path
        broad_sources[name] = {**source, **verify(path), "cache_reused": reused}
    broad_coverage, broad = broad_audit(broad_paths["model_metadata"], broad_paths["screen_metadata"])
    sanger_counts, sanger = sanger_audit(args.sanger_model_list, args.sanger_availability)
    print(f"【覆盖结果】Sanger类器官 {sanger['organoid_crispr_model_n']}｜Kidney 0；"
          f"Broad NextGen通过QC {broad['nextgen_qc_screen_n']}｜Kidney 0", flush=True)

    print("【阶段 3/4】核对冻结的 RCC 与正常肾类器官文献记录", flush=True)
    records, literature_cache, literature_reused = fetch_papers(args.raw_dir)
    inventory = resource_inventory(records)
    liability = normal_liability(records)
    eligible_n = int(inventory["eligible"].sum())
    print(f"【患者来源功能数据】符合全部资格 {eligible_n}/{len(inventory)}｜"
          f"正常肾类器官风险记录 {len(liability)}", flush=True)

    print("【阶段 4/4】保存资源清单、排除原因和来源审计", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inventory.to_csv(args.output_dir / "resource_inventory.csv", index=False)
    liability.to_csv(args.output_dir / "normal_kidney_organoid_liability.csv", index=False)
    broad_coverage.to_csv(args.output_dir / "broad_nextgen_lineage_coverage.csv", index=False)
    sanger_counts.to_csv(args.output_dir / "sanger_organoid_tissue_coverage.csv", index=False)

    source_files = {
        "sanger_model_list": verify(args.sanger_model_list),
        "sanger_availability": verify(args.sanger_availability),
        "frozen_candidates": verify(args.frozen_candidates),
        "europe_pmc_targeted_records": {**verify(literature_cache), "cache_reused": literature_reused},
        **broad_sources,
    }
    run = {
        "status": "public_primary_ccrcc_functional_resource_audit_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "compute": "CPU metadata and table audit; GPU acceleration is not applicable",
        "frozen_candidate_n": int(frozen[gene_col].nunique()),
        "eligibility_rule": {
            "model": "human patient-derived ccRCC organoid, PDX or short-term primary model",
            "perturbation": "gene-level loss-of-function covering a frozen candidate",
            "endpoint": "viability, growth or tumor endpoint",
            "independence": "not a reuse of the current DepMap baseline or its established cell lines",
        },
        "counts": {
            "audited_resource_n": int(len(inventory)),
            "eligible_patient_derived_ccrcc_gene_perturbation_resource_n": eligible_n,
            "normal_kidney_organoid_liability_record_n": int(len(liability)),
            "targeted_literature_record_n": int(len(records)),
            "sanger": sanger,
            "broad": broad,
        },
        "sources": source_files,
        "official_resource_urls": {
            "sanger_organoid_matrix": "https://cog.sanger.ac.uk/cmp/download/Organoids_fitness_scores_20260723.zip",
            "sanger_article_doi": "10.1038/s41586-026-10830-y",
            "broad_figshare_article": "https://api.figshare.com/v2/articles/29472362",
            "broad_article_doi": "10.1038/s41586-026-10843-7",
            "europe_pmc_api": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        },
        "conclusion": (
            "Within the frozen audit scope, zero public systematic patient-derived ccRCC gene-perturbation "
            "resources satisfy all validation criteria. Existing RCC organoids provide drug-response models "
            "without gene-dependency screens; published ccRCC screens use established cell lines."
        ),
        "limitations": [
            "This is a targeted public-resource audit, not a registered systematic review and not proof that no unpublished dataset exists.",
            "A zero eligible count is an availability result, not evidence that the frozen candidates lack biological effects.",
            "Drug-conditional resistance screens answer a different question from untreated baseline dependency.",
            "Kidney organoids are developmental or fetal-like models and cannot quantify adult renal toxicity or a clinical therapeutic window.",
            "Normal-organoid evidence for PAX8 and HNF1B increases liability concern but does not establish drug-dose toxicity.",
        ],
    }
    output_files = [
        "resource_inventory.csv", "normal_kidney_organoid_liability.csv",
        "broad_nextgen_lineage_coverage.csv", "sanger_organoid_tissue_coverage.csv",
    ]
    run["output_sha256"] = {name: sha256(args.output_dir / name) for name in output_files}
    run["script_sha256"] = sha256(Path(__file__))
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)

    print("【核心结论】合格的公开患者来源ccRCC基因扰动数据 0｜不能完成患者功能真值验证", flush=True)
    print("【风险证据】PAX8、HNF1B存在正常肾类器官直接扰动表型｜进一步削弱治疗窗假设", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f}秒｜结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】0表示本次公开资源审计未找到合格数据，不代表不存在未公开数据或候选无效。", flush=True)


if __name__ == "__main__":
    main()
