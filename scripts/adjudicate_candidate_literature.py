"""Adjudicate the frozen Europe PMC screening queue using explicit evidence rules.

This script records a manual title/abstract/full-text review. Primary external
support requires an independent, peer-reviewed, direct loss-of-function
experiment on the candidate in a relevant renal cancer model with a growth or
tumour phenotype. Mechanistic, non-peer-reviewed and opposing evidence remain
separate and no composite candidate score is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# key -> (classification, direction, perturbation, model, rationale)
REVIEW = {
    "MED:41230160": ("not_direct", "none", "YBX3 knockdown", "ccRCC cells",
                     "HNF1B is an inferred transcription factor; only YBX3 is functionally perturbed."),
    "ETH:774617": ("direct_support_non_peer_reviewed", "support", "HNF1B CRISPR knockout and CRISPRi",
                   "ccRCC models", "Direct growth and phenotype evidence, but the source is a PhD thesis."),
    "MED:33850477": ("reused_depmap", "none", "none", "DepMap and patient databases",
                     "Reanalyzes DepMap dependency scores and is not an independent perturbation experiment."),
    "MED:36681680": ("not_direct", "none", "FOXI1 overexpression plus EPAS1 knockdown", "786-O",
                     "HNF1B is nominated for papillary RCC and is not directly perturbed."),
    "MED:27690003": ("not_direct", "none", "SRSF2 silencing", "ccRCC cell-line panel",
                     "CFLAR is a splice-product observation downstream of SRSF2 perturbation."),
    "MED:41899022": ("not_direct", "none", "none", "case report",
                     "PAX8 is used as an immunohistochemical diagnostic marker."),
    "PPR:PPR1260535": ("not_direct", "none", "none", "histopathology cohorts",
                       "PAX8 staining supervises an imaging model; PAX8 is not perturbed."),
    "PPR:PPR1153549": ("not_direct", "none", "none", "case report preprint",
                       "PAX8 is used as an immunohistochemical diagnostic marker."),
    "MED:37554444": ("direct_support_peer_reviewed", "support", "two PAX8 shRNAs",
                     "786-M1A VHL-mutant metastatic ccRCC",
                     "PAX8 suppression causes a negative proliferative phenotype; a resistance screen is also performed."),
    "MED:37511191": ("direct_support_peer_reviewed", "support", "two PAX8 siRNAs", "A498 RCC",
                     "PAX8 depletion reduces proliferation in one renal cancer line."),
    "MED:35570589": ("not_direct", "none", "none", "case report",
                     "PAX8 is a diagnostic marker and no candidate perturbation is performed."),
    "MED:35674183": ("direct_opposition_peer_reviewed", "oppose", "YPEL5 siRNA", "786-O ccRCC",
                     "YPEL5 depletion increases proliferation, migration and invasion, opposite to a dependency claim."),
    "MED:27128972": ("not_direct", "none", "none", "human ccRCC tissues",
                     "FOXA1 inhibition is inferred from proteotranscriptomics without perturbation."),
    "MED:35081978": ("not_direct", "none", "NFAT1 and pathway perturbations", "renal cancer models",
                     "FOXA1 is an upstream association and is not directly perturbed."),
    "MED:42656173": ("not_direct", "none", "AKT/beta-catenin perturbations", "ccRCC models",
                     "CCND1 is a downstream transcriptional readout."),
    "MED:41570697": ("review", "none", "none", "mini-review",
                     "Narrative review without a new candidate perturbation experiment."),
    "MED:40178040": ("mechanistic_support_gain_of_function", "support", "CCND1 CRISPRa and overexpression",
                     "OSRC2, TUHR4TKB and 786-O ccRCC",
                     "Sustained CCND1 expression confers HIF2-inhibitor resistance; this is not loss-of-function validation."),
    "MED:38890303": ("not_direct", "none", "AURKB/CDC37 knockdown", "ccRCC models",
                     "CCND1 is downstream of the perturbed AURKB/CDC37-MYC pathway."),
    "MED:37263638": ("not_direct", "none", "NSUN5 knockout", "ccRCC models",
                     "CCND1 decreases downstream of NSUN5 knockout and is not directly perturbed."),
    "MED:36500549": ("not_direct", "none", "HIF2A siRNA", "patient-derived ccRCC cells",
                     "Full text confirms CCND1 reduction is downstream of HIF2A siRNA, not direct CCND1 knockdown."),
    "MED:32905496": ("not_direct", "none", "NFYA knockdown", "ccRCC cells",
                     "CCND1 is a transcriptional target downstream of NFYA."),
    "MED:31434797": ("mechanistic_support_rescue", "support", "CCND1 re-expression rescue", "ccRCC models",
                     "CCND1 rescue reverses a miR-625 phenotype, but direct CCND1 loss-of-function is not tested."),
    "MED:34586831": ("not_direct", "none", "FTO/BRD9 perturbations", "ccRCC models",
                     "CCND1 is a downstream enhancer-associated gene."),
    "MED:32003757": ("not_direct", "none", "ISG20 knockdown", "ccRCC models",
                     "CCND1 is a downstream expression readout."),
    "MED:18809243": ("not_direct", "none", "MYC knockdown", "ccRCC models",
                     "CCND1 is a downstream MYC target and is not directly perturbed."),
    "MED:33644060": ("not_direct", "none", "POLE2 knockdown", "RCC models",
                     "CCND1 is a downstream protein readout."),
    "MED:24260413": ("direct_support_peer_reviewed", "support", "CCND1 shRNA", "786-O xenograft",
                     "Direct CCND1 suppression impairs tumour growth in an independent xenograft experiment."),
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_v1")
    parser.add_argument("--frozen-dir", type=Path,
                        default=root / "outputs/candidate_external_validation_frozen_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/candidate_external_adjudication_v1")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")
    print("【阶段 1/3】校验正式外部结果与29篇冻结文献", flush=True)
    run_path = args.external_dir / "run.json"
    queue_path = args.external_dir / "literature_screening_queue.csv.gz"
    external_run = json.loads(run_path.read_text())
    if (external_run["status"] != "external_evidence_acquired_manual_literature_screening_pending" or
            sha256(queue_path) != external_run["output_sha256"][queue_path.name]):
        raise ValueError("外部验证状态或文献队列哈希变化")
    queue = pd.read_csv(queue_path)
    queue["review_key"] = queue.source.astype(str) + ":" + queue.id.astype(str)
    actual_keys = set(queue.review_key)
    missing = actual_keys - set(REVIEW)
    unused = set(REVIEW) - actual_keys
    if missing or unused or len(queue) != 29:
        raise ValueError(f"人工判定与冻结队列不一致：未判定={missing}，多余={unused}")

    print("【阶段 2/3】写入直接性、独立性、同行评议和方向判定", flush=True)
    records = []
    for row in queue.itertuples(index=False):
        classification, direction, perturbation, model, rationale = REVIEW[row.review_key]
        direct = classification.startswith("direct_")
        peer_reviewed = row.source == "MED"
        independent = classification != "reused_depmap"
        primary = classification == "direct_support_peer_reviewed"
        opposing = classification == "direct_opposition_peer_reviewed"
        record = row._asdict()
        record.update(classification=classification, evidence_direction=direction,
                      candidate_perturbation=perturbation, experimental_model=model,
                      direct_candidate_loss_of_function=direct,
                      peer_reviewed=peer_reviewed, independent_of_depmap=independent,
                      primary_independent_support=primary,
                      primary_independent_opposition=opposing,
                      adjudication_rationale=rationale)
        records.append(record)
    adjudication = pd.DataFrame(records).drop(columns="review_key")

    frozen = pd.read_csv(args.frozen_dir / "candidates.csv").sort_values("discovery_rank")
    hpa = pd.read_csv(args.external_dir / "hpa_normal_tissue.csv")
    ot = pd.read_csv(args.external_dir / "open_targets_summary.csv")
    summary = frozen[["Gene", "discovery_rank",
                      "train_patient_consensus_top10_frequency",
                      "validation_patient_consensus_top10_frequency",
                      "depmap_ccrcc_leave1out_worst_mean_residual",
                      "depmap_ccrcc_minus_other_kidney_mean",
                      "sanger_renal_dependent_n", "sanger_renal_observed_n",
                      "shared_three_renal_absolute_prediction_top10"]].copy()
    counts = adjudication.groupby("Gene").agg(
        peer_reviewed_direct_support_n=("primary_independent_support", "sum"),
        peer_reviewed_direct_opposition_n=("primary_independent_opposition", "sum"),
        non_peer_reviewed_direct_support_n=("classification", lambda x: (x == "direct_support_non_peer_reviewed").sum()),
        mechanistic_support_n=("classification", lambda x: x.str.startswith("mechanistic_support").sum()),
        reused_depmap_article_n=("classification", lambda x: (x == "reused_depmap").sum()),
        retrieved_article_n=("classification", "size"),
    ).reset_index()
    summary = summary.merge(counts, on="Gene", how="left", validate="one_to_one")
    count_columns = [column for column in counts if column != "Gene"]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    hpa_fields = ["Gene", "kidney_named_in_rna_specific_ntpm",
                  "kidney_named_in_protein_specific_intensity",
                  "rna_detected_in_all_normal_tissues", "protein_detected_in_all_normal_tissues"]
    ot_fields = ["Gene", "clinical_stage_tractability_true_n",
                 "drug_and_clinical_candidate_n", "safety_liability_n"]
    summary = summary.merge(hpa[hpa_fields], on="Gene", validate="one_to_one")
    summary = summary.merge(ot[ot_fields], on="Gene", validate="one_to_one")

    print("【阶段 3/3】保存逐文献判定和候选分层汇总", flush=True)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    adjudication.to_csv(temporary / "literature_adjudication.csv", index=False)
    summary.to_csv(temporary / "candidate_external_summary.csv", index=False)
    audit = {
        "status": "external_literature_adjudication_complete_exploratory_candidates",
        "elapsed_seconds": time.monotonic() - started,
        "primary_rule": (
            "Independent peer-reviewed direct candidate loss-of-function in a relevant renal cancer model "
            "with growth or tumour phenotype"),
        "counts": {
            "retrieved_articles": len(adjudication),
            "genes_with_peer_reviewed_direct_support": int(summary.peer_reviewed_direct_support_n.gt(0).sum()),
            "peer_reviewed_direct_support_articles": int(adjudication.primary_independent_support.sum()),
            "genes_with_peer_reviewed_direct_opposition": int(summary.peer_reviewed_direct_opposition_n.gt(0).sum()),
            "peer_reviewed_direct_opposition_articles": int(adjudication.primary_independent_opposition.sum()),
            "genes_with_non_peer_reviewed_direct_support": int(summary.non_peer_reviewed_direct_support_n.gt(0).sum()),
        },
        "source_sha256": {
            "external_run": sha256(run_path), "literature_queue": sha256(queue_path),
            "hpa": sha256(args.external_dir / "hpa_normal_tissue.csv"),
            "open_targets": sha256(args.external_dir / "open_targets_summary.csv"),
            "frozen_candidates": sha256(args.frozen_dir / "candidates.csv"),
            "script": sha256(Path(__file__)),
        },
        "full_text_checked": ["PMC9738223", "PMC3832366", "PMC10405256",
                              "PMC10380508", "PMC9204605", "PMC12223508"],
        "limitations": [
            "The frozen keyword search can miss papers whose title and abstract omit a gene symbol or perturbation term.",
            "Direct experiments validate general model-level biology, not patient-specific dependency predictions.",
            "A single cell line, xenograft, or thesis does not establish ccRCC-wide specificity or clinical safety.",
            "HPA and Open Targets annotations are retained as separate descriptive fields and are not combined into a score.",
        ],
    }
    audit["output_sha256"] = {path.name: sha256(path) for path in sorted(temporary.iterdir())}
    (temporary / "run.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    temporary.rename(args.output_dir)
    supportive = summary.loc[summary.peer_reviewed_direct_support_n.gt(0), "Gene"].tolist()
    opposing = summary.loc[summary.peer_reviewed_direct_opposition_n.gt(0), "Gene"].tolist()
    grey = summary.loc[summary.non_peer_reviewed_direct_support_n.gt(0), "Gene"].tolist()
    print(f"【独立直接支持】同行评议基因 {len(supportive)}：{', '.join(supportive)}", flush=True)
    print(f"【独立反向证据】同行评议基因 {len(opposing)}：{', '.join(opposing)}", flush=True)
    print(f"【非同行评议支持】基因 {len(grey)}：{', '.join(grey)}", flush=True)
    print(f"【完成】结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】支持的是模型层功能，不是患者特异依赖；其余候选也不能因零命中被判定为无效。", flush=True)


if __name__ == "__main__":
    main()
