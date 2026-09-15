"""Acquire external evidence for the frozen TCGA candidate shortlist.

The candidate order is fixed before these sources are queried. Human Protein
Atlas fields are normal-tissue screening signals, Open Targets fields describe
tractability and known liabilities, and Europe PMC results are a literature
screening queue. None of them is treated as functional validation by itself.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HPA_URL = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
HPA_VERSION = "25.1"
HPA_SHA256 = "ad401f8519ecdee5b67b0ad3bf28175d402d1ea0ad6c542b90fcab2bee39b470"
OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = "PanCancer-Dependency-public-data-audit/1.0"

HPA_COLUMNS = [
    "Gene", "Ensembl", "Gene description", "Protein class", "Evidence",
    "RNA tissue specificity", "RNA tissue distribution", "RNA tissue specific nTPM",
    "RNA tissue cell type enrichment", "Protein tissue specificity",
    "Protein tissue distribution", "Protein tissue specific Intensity",
    "Reliability (IH)", "Subcellular main location", "Secretome location",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(url: str, data: bytes | None = None, headers: dict | None = None,
                  attempts: int = 3) -> tuple[bytes, dict]:
    request_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    last_error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=data, headers=request_headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                metadata = {key.lower(): value for key, value in response.headers.items()}
                metadata.update(status=response.status, final_url=response.url)
                return payload, metadata
        except Exception as error:  # preserve the final HTTP/network exception
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"外部请求失败：{url}") from last_error


def verify_frozen(frozen_dir: Path, current_candidates: Path) -> pd.DataFrame:
    run_path = frozen_dir / "run.json"
    candidates_path = frozen_dir / "candidates.csv"
    run = json.loads(run_path.read_text())
    expected = run["output_sha256"]["candidates.csv"]
    if sha256(candidates_path) != expected:
        raise ValueError("冻结候选文件哈希变化")
    candidates = pd.read_csv(candidates_path)
    genes = candidates.Gene.astype(str).tolist()
    if (genes != run["selection"]["genes"] or len(genes) != 20 or
            candidates.discovery_rank.astype(int).tolist() != list(range(1, 21))):
        raise ValueError("冻结候选顺序或数量变化")
    current = pd.read_csv(current_candidates, usecols=["Gene", "discovery_rank"])
    current = current.sort_values("discovery_rank").head(20)
    if current.Gene.astype(str).tolist() != genes:
        raise ValueError("当前 v2 前20与外部验证冻结队列不一致")
    return candidates[["Gene", "discovery_rank"]].copy()


def acquire_hpa(cache: Path) -> tuple[pd.DataFrame, dict]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"cache_reused": cache.exists()}
    if not cache.exists():
        payload, response_metadata = request_bytes(HPA_URL)
        temporary = cache.with_suffix(cache.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache)
        metadata.update(response_metadata)
    observed = sha256(cache)
    if observed != HPA_SHA256:
        raise ValueError(
            f"HPA 下载版本变化：期望 {HPA_SHA256}，实际 {observed}；需先审计新版本")
    with zipfile.ZipFile(cache) as archive:
        members = archive.namelist()
        if members != ["proteinatlas.tsv"]:
            raise ValueError(f"HPA 压缩包成员变化：{members}")
        with archive.open(members[0]) as handle:
            table = pd.read_csv(handle, sep="\t", usecols=HPA_COLUMNS, dtype=str)
    metadata.update(url=HPA_URL, version=HPA_VERSION, sha256=observed,
                    bytes=cache.stat().st_size, zip_member=members[0])
    return table, metadata


def hpa_candidates(shortlist: pd.DataFrame, hpa: pd.DataFrame) -> pd.DataFrame:
    selected = hpa[hpa.Gene.isin(shortlist.Gene)].copy()
    if selected.Gene.duplicated().any():
        raise ValueError("HPA 候选基因存在重复行")
    result = shortlist.merge(selected, on="Gene", how="left", validate="one_to_one")
    if result.Ensembl.isna().any():
        raise ValueError(f"HPA 缺少候选：{result.loc[result.Ensembl.isna(), 'Gene'].tolist()}")
    rna = result["RNA tissue specific nTPM"].fillna("").str.lower()
    protein = result["Protein tissue specific Intensity"].fillna("").str.lower()
    result["kidney_named_in_rna_specific_ntpm"] = rna.str.contains(r"(?:^|;)kidney:", regex=True)
    result["kidney_named_in_protein_specific_intensity"] = protein.str.contains(
        r"(?:^|;)kidney:", regex=True)
    result["rna_detected_in_all_normal_tissues"] = result["RNA tissue distribution"].eq(
        "Detected in all")
    result["protein_detected_in_all_normal_tissues"] = result[
        "Protein tissue distribution"].eq("Detected in all")
    return result


def query_open_targets(hpa: pd.DataFrame) -> tuple[dict, dict]:
    fields = """id approvedSymbol approvedName biotype isEssential
      tractability { label modality value }
      chemicalProbes { id isHighQuality mechanismOfAction }
      safetyLiabilities { event eventId datasource literature url }
      drugAndClinicalCandidates { count }"""
    aliases = []
    for index, row in hpa.iterrows():
        aliases.append(f't{index}: target(ensemblId: "{row.Ensembl}") {{ {fields} }}')
    query = "query FrozenTargets {\n" + "\n".join(aliases) + "\n}"
    payload, metadata = request_bytes(
        OPEN_TARGETS_URL,
        data=json.dumps({"query": query}, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"})
    parsed = json.loads(payload)
    if parsed.get("errors"):
        raise RuntimeError(f"Open Targets GraphQL 错误：{parsed['errors']}")
    targets = parsed.get("data", {})
    if len(targets) != len(hpa) or any(value is None for value in targets.values()):
        raise ValueError("Open Targets 候选返回不完整")
    metadata.update(url=OPEN_TARGETS_URL,
                    response_sha256=hashlib.sha256(payload).hexdigest())
    return targets, metadata


def normalize_open_targets(shortlist: pd.DataFrame, raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, tractability_rows = [], []
    clinical_labels = {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}
    for index, candidate in shortlist.reset_index(drop=True).iterrows():
        target = raw[f"t{index}"]
        true_items = [item for item in target["tractability"] if item["value"]]
        clinical_items = [item for item in true_items if item["label"] in clinical_labels]
        for item in target["tractability"]:
            tractability_rows.append({"Gene": candidate.Gene,
                                      "discovery_rank": int(candidate.discovery_rank), **item})
        modalities = sorted({item["modality"] for item in true_items})
        labels = sorted(item["label"] for item in true_items)
        probes = target["chemicalProbes"]
        liabilities = target["safetyLiabilities"]
        summary_rows.append({
            "Gene": candidate.Gene, "discovery_rank": int(candidate.discovery_rank),
            "Ensembl": target["id"], "open_targets_symbol": target["approvedSymbol"],
            "biotype": target["biotype"], "open_targets_is_essential": target["isEssential"],
            "tractability_true_n": len(true_items),
            "tractability_true_modalities": ";".join(modalities),
            "tractability_true_labels": ";".join(labels),
            "clinical_stage_tractability_true_n": len(clinical_items),
            "clinical_stage_tractability_labels": ";".join(sorted(
                f'{item["modality"]}:{item["label"]}' for item in clinical_items)),
            "chemical_probe_n": len(probes),
            "high_quality_chemical_probe_n": sum(bool(item["isHighQuality"]) for item in probes),
            "safety_liability_n": len(liabilities),
            "drug_and_clinical_candidate_n": target["drugAndClinicalCandidates"]["count"],
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(tractability_rows)


def literature_query(gene: str) -> str:
    disease = ('(TITLE_ABS:"clear cell renal cell carcinoma" OR TITLE_ABS:ccRCC '
               'OR TITLE_ABS:KIRC)')
    perturbation = ('(TITLE_ABS:CRISPR OR TITLE_ABS:knockout OR TITLE_ABS:"knock out" '
                    'OR TITLE_ABS:knockdown OR TITLE_ABS:"knock down" OR TITLE_ABS:silencing '
                    'OR TITLE_ABS:depletion OR TITLE_ABS:siRNA OR TITLE_ABS:shRNA '
                    'OR TITLE_ABS:inhibitor OR TITLE_ABS:inhibition)')
    return f'TITLE_ABS:"{gene}" AND {disease} AND {perturbation}'


def query_europe_pmc(shortlist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, summaries, raw = [], [], {}
    fields = ["source", "id", "pmid", "pmcid", "doi", "title", "authorString",
              "journalTitle", "pubYear", "citedByCount", "isOpenAccess", "abstractText"]
    for candidate in shortlist.itertuples(index=False):
        query = literature_query(candidate.Gene)
        url = EUROPE_PMC_URL + "?" + urllib.parse.urlencode({
            "query": query, "format": "json", "resultType": "core", "pageSize": 100})
        payload, _ = request_bytes(url)
        parsed = json.loads(payload)
        hit_count = int(parsed["hitCount"])
        results = parsed.get("resultList", {}).get("result", [])
        raw[candidate.Gene] = {"query": query, "hitCount": hit_count, "results": results}
        summaries.append({"Gene": candidate.Gene,
                          "discovery_rank": int(candidate.discovery_rank),
                          "query_hit_n": hit_count, "retrieved_n": len(results),
                          "requires_manual_screening": True})
        for rank, result in enumerate(results, 1):
            row = {"Gene": candidate.Gene, "discovery_rank": int(candidate.discovery_rank),
                   "query_rank": rank}
            row.update({field: result.get(field) for field in fields})
            rows.append(row)
        time.sleep(.1)
    metadata = {"url": EUROPE_PMC_URL, "query_scope": "title_or_abstract",
                "result_type": "core", "page_size": 100,
                "raw_sha256": hashlib.sha256(json.dumps(
                    raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    return pd.DataFrame(summaries), pd.DataFrame(rows), {"metadata": metadata, "queries": raw}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1")
    parser.add_argument("--current-candidates", type=Path,
                        default=root / "outputs/tcga_candidate_evidence_v2/train_discovery_top100.csv")
    parser.add_argument("--hpa-cache", type=Path,
                        default=root / "data/raw/external_candidate_validation_20260915/proteinatlas_v25_1.tsv.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    print("【阶段 1/4】校验冻结前20候选和当前 v2 排名", flush=True)
    shortlist = verify_frozen(args.frozen_dir, args.current_candidates)
    print(f"【冻结队列】候选 {len(shortlist)}｜排名 1–20｜不按外部证据重排", flush=True)
    if args.dry_run:
        print("【检查通过】未联网、未写入结果。", flush=True)
        return

    print("【阶段 2/4】读取 HPA v25.1 正常组织表达筛查字段", flush=True)
    hpa_all, hpa_metadata = acquire_hpa(args.hpa_cache)
    hpa = hpa_candidates(shortlist, hpa_all)
    kidney_named = (hpa.kidney_named_in_rna_specific_ntpm |
                    hpa.kidney_named_in_protein_specific_intensity)
    print(f"【HPA筛查】肾脏出现在组织特异字段 {int(kidney_named.sum())}/{len(hpa)}｜"
          "该字段不是毒性证据", flush=True)

    print("【阶段 3/4】查询 Open Targets 靶点属性和 Europe PMC 文献队列", flush=True)
    open_targets_raw, open_targets_metadata = query_open_targets(hpa)
    ot_summary, ot_tractability = normalize_open_targets(shortlist, open_targets_raw)
    literature_summary, literature_hits, literature_raw = query_europe_pmc(shortlist)
    print(f"【靶点属性】存在临床阶段 tractability 标签 "
          f"{int(ot_summary.clinical_stage_tractability_true_n.gt(0).sum())}/{len(shortlist)}｜"
          f"存在药物或临床候选记录 "
          f"{int(ot_summary.drug_and_clinical_candidate_n.gt(0).sum())}/{len(shortlist)}", flush=True)
    print(f"【文献初筛】有检索命中 {int(literature_summary.query_hit_n.gt(0).sum())}/{len(shortlist)}｜"
          f"返回记录 {int(literature_summary.retrieved_n.sum())}｜尚未人工判定", flush=True)

    print("【阶段 4/4】保存分层外部证据和来源审计", flush=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    hpa.to_csv(temporary / "hpa_normal_tissue.csv", index=False)
    ot_summary.to_csv(temporary / "open_targets_summary.csv", index=False)
    ot_tractability.to_csv(temporary / "open_targets_tractability.csv", index=False)
    (temporary / "open_targets_raw.json").write_text(
        json.dumps(open_targets_raw, ensure_ascii=False, indent=2) + "\n")
    literature_summary.to_csv(temporary / "literature_search_summary.csv", index=False)
    literature_hits.to_csv(temporary / "literature_screening_queue.csv.gz", index=False,
                           compression="gzip")
    with gzip.open(temporary / "europe_pmc_raw.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(literature_raw, handle, ensure_ascii=False)
    run = {
        "status": "external_evidence_acquired_manual_literature_screening_pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "candidate_rule": "Frozen train discovery ranks 1-20; external evidence does not alter order",
        "candidate_n": len(shortlist),
        "sources": {"human_protein_atlas": hpa_metadata,
                    "open_targets": open_targets_metadata,
                    "europe_pmc": literature_raw["metadata"]},
        "source_sha256": {
            "frozen_candidates": sha256(args.frozen_dir / "candidates.csv"),
            "frozen_run": sha256(args.frozen_dir / "run.json"),
            "current_candidate_table": sha256(args.current_candidates),
            "script": sha256(Path(__file__)),
        },
        "interpretation_rules": {
            "hpa": "Descriptive normal-tissue expression flags; absence from tissue-specific fields is not proof of absent kidney expression or safety",
            "open_targets": "Database annotations of tractability, probes, clinical candidates and liabilities; not efficacy evidence in ccRCC",
            "europe_pmc": "Automated title/abstract retrieval queue; every hit requires manual full-text eligibility and perturbation-outcome review",
            "aggregation": "No composite score, threshold, candidate removal or reranking",
        },
        "limitations": [
            "HPA tissue-specific nTPM and intensity fields omit full per-tissue values for genes classified as broadly expressed.",
            "Normal-tissue expression is a safety signal, not direct toxicity evidence.",
            "Open Targets and Europe PMC are mutable services; retrieval time and raw responses are retained.",
            "Keyword co-occurrence does not establish an independent gene perturbation experiment or causal dependency.",
            "The frozen candidates remain exploratory because patient dependency truth labels do not exist.",
        ],
    }
    run["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(output)
    print(f"【完成】耗时 {time.monotonic()-started:.1f}秒｜结果 {output}", flush=True)
    print("【结论边界】这是外部证据采集；文献未人工核读，不能据此宣称 ccRCC 功能依赖或治疗靶点。", flush=True)


if __name__ == "__main__":
    main()
