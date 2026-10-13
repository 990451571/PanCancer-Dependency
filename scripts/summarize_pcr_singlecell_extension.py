"""Integrate the completed PCR extension without changing candidate order."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcr_transfer_methods import finish_run, sha256, verify_manifest


def describe(frame, keys, column):
    rows = []
    for group_key, group in frame.groupby(keys, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group[column].dropna()
        rows.append({**dict(zip(keys, group_key)), "n": len(values), "mean": values.mean(),
                     "median": values.median(), "p025": values.quantile(.025), "p975": values.quantile(.975),
                     "min": values.min(), "max": values.max()})
    return pd.DataFrame(rows)


def save_figure(fig, directory, name):
    for suffix in ["png", "pdf"]:
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def snapshot(source, target, selected):
    if target.exists():
        raise FileExistsError(f"拒绝覆盖版本化快照: {target}")
    target.mkdir(parents=True)
    for name in selected:
        shutil.copy2(source / name, target / name)
    record = {"type": "compact_result_snapshot", "upstream_output_dir": str(source),
              "upstream_run_sha256": sha256(source / "run.json"),
              "included_sha256": {name: sha256(target / name) for name in selected},
              "note": "Upstream run.json lists large outputs retained locally; snapshot.json verifies the included subset"}
    (target / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcr-dir", type=Path, default=root / "outputs/pcr_patient_extension_v1")
    parser.add_argument("--singlecell-dir", type=Path, default=root / "outputs/ccrcc_singlecell_extension_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/pcr_singlecell_synthesis_v1")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    pcr_run, sc_run = verify_manifest(args.pcr_dir), verify_manifest(args.singlecell_dir)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖结果: {output}")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    primary = "nonkidney_mean_shift_clip0"
    mapping = pd.read_csv(args.pcr_dir / "mapping_sensitivity.csv")
    bootstrap = pd.read_csv(args.pcr_dir / "bootstrap_patient_overlap.csv.gz")
    candidates = pd.read_csv(args.pcr_dir / "candidate_summary.csv.gz")
    discovery = pd.read_csv(args.pcr_dir / "pcr_discovery_top20.csv")
    uncertainty = pd.read_csv(args.pcr_dir / "candidate_patient_uncertainty.csv.gz")
    patient_bootstrap = bootstrap.groupby(["kind", "patient_id", "split"], as_index=False).top10_overlap.mean()
    mapping_summary = describe(mapping, ["comparison", "split"], "top10_overlap")
    bootstrap_summary = describe(patient_bootstrap, ["kind", "split"], "top10_overlap")
    mapping_summary.to_csv(output / "mapping_overlap_summary.csv", index=False)
    bootstrap_summary.to_csv(output / "bootstrap_overlap_summary.csv", index=False)
    patient_bootstrap.to_csv(output / "bootstrap_per_patient_mean_overlap.csv", index=False)
    domain = pd.read_csv(args.pcr_dir / "domain_diagnostics.csv")
    domain_summary = []
    for column in ["out_of_reference_fraction", "missing_feature_fraction", "pca_reconstruction_rmse", "training_reconstruction_percentile"]:
        frame = describe(domain, ["method", "split"], column)
        frame["metric"] = column
        domain_summary.append(frame)
    pd.concat(domain_summary).to_csv(output / "domain_diagnostic_summary.csv", index=False)
    mixture = pd.read_csv(args.singlecell_dir / "mixture_sensitivity.csv")
    mixture_summaries = []
    for scope in ["author_cohort", "geo_author_concordant_only"]:
        frame = mixture if scope == "author_cohort" else mixture[mixture.geo_author_concordant]
        summary = describe(frame, ["scenario"], "top10_overlap")
        summary["scope"] = scope
        mixture_summaries.append(summary)
    mixture_summary = pd.concat(mixture_summaries, ignore_index=True)
    mixture_summary.to_csv(output / "mixture_overlap_summary.csv", index=False)
    sources = pd.read_csv(args.singlecell_dir / "candidate_cell_source_summary.csv")
    sources = sources[sources.Gene.isin(discovery.Gene)]
    source_main = sources[sources.scope.eq("author_cohort")]
    flags = pd.read_csv(args.singlecell_dir / "candidate_sets.csv").set_index("Gene")
    evidence = discovery.copy()
    evidence["historical_frozen_candidate"] = evidence.Gene.map(flags.historical_frozen_candidate)
    for method in candidates.method.unique():
        for split in ["train", "validation"]:
            lookup = candidates[(candidates.method == method) & (candidates.split == split)].set_index("Gene")
            evidence[f"{method}_{split}_top10_frequency"] = evidence.Gene.map(lookup.top10_frequency)
    for kind in uncertainty.kind.unique():
        for split in ["train", "validation"]:
            frame = uncertainty[(uncertainty.kind == kind) & (uncertainty.split == split)]
            inclusion = frame.groupby("Gene").top10_inclusion_frequency.mean()
            evidence[f"{kind}_{split}_mean_inclusion_frequency"] = evidence.Gene.map(inclusion)
    for condition, group, short in [("author_ccrcc", "author_tumor", "author_tumor"),
                                    ("author_normal", "normal_proximal_tubule", "author_normal_pt"),
                                    ("author_normal", "normal_other_epithelium", "author_normal_other")]:
        frame = source_main[(source_main.author_cohort == condition) & (source_main.group == group)].set_index("Gene")
        for metric in ["donor_n", "median_cpm", "mean_detection_fraction"]:
            evidence[f"{short}_{metric}"] = evidence.Gene.map(frame[metric])
    concordant_sources = sources[sources.scope.eq("geo_author_concordant_only")]
    for condition, group, short in [("author_ccrcc", "author_tumor", "concordant_tumor"),
                                    ("author_normal", "normal_proximal_tubule", "concordant_normal_pt")]:
        frame = concordant_sources[(concordant_sources.author_cohort == condition) & (concordant_sources.group == group)].set_index("Gene")
        for metric in ["donor_n", "median_cpm"]:
            evidence[f"{short}_{metric}"] = evidence.Gene.map(frame[metric])
    evidence["claim_ceiling"] = "PCR development hypothesis + descriptive expression-source support; no functional truth"
    evidence["source_label_caveat"] = "GEO-author tissue conflicts; concordant normal donor n<=1"
    evidence.to_csv(output / "pcr_candidate_evidence_table.csv", index=False)
    features = pd.read_csv(args.pcr_dir / "pcr_candidate_input_features.csv")
    source_features = pd.read_csv(args.singlecell_dir / "candidate_cell_source_summary.csv")
    feature_sources = features.merge(source_features, left_on="feature_gene", right_on="Gene", how="left", validate="many_to_many")
    feature_sources.to_csv(output / "pcr_input_feature_cell_sources.csv.gz", index=False)
    audit = pd.read_csv(args.singlecell_dir / "sample_source_audit.csv")
    qc = pd.read_csv(args.singlecell_dir / "qc_by_sample_label.csv")
    qc_summary = qc.groupby(["author_cohort", "anno"], as_index=False)[["annotated_n", "retained_n"]].sum()
    qc_summary["retained_fraction"] = qc_summary.retained_n / qc_summary.annotated_n
    qc_summary.to_csv(output / "qc_retention_by_label.csv", index=False)
    qc_evidence = pd.read_csv(args.singlecell_dir / "candidate_qc_sensitivity.csv.gz")
    qc_med = qc_evidence.groupby(["qc_scope", "author_cohort", "group", "Gene"], as_index=False).agg(
        donor_n=("donor", "nunique"), median_cpm=("pseudobulk_cpm", "median"), mean_detection=("detection_fraction", "mean"))
    qc_med.to_csv(output / "candidate_qc_source_summary.csv", index=False)
    # Figure 1: patient-level stability, with different resampling units kept apart.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    methods = list(mapping.comparison.unique())
    axes[0, 0].boxplot([mapping[mapping.comparison.eq(m)].top10_overlap for m in methods], tick_labels=["Kidney shift", "Raw expression"], showfliers=False)
    axes[0, 0].set(title="A  Mapping sensitivity (209 development patients)", ylabel="Top-10 overlap with non-Kidney shift", ylim=(-.03, 1.03))
    kinds = list(patient_bootstrap.kind.unique())
    axes[0, 1].boxplot([patient_bootstrap[patient_bootstrap.kind.eq(k)].top10_overlap for k in kinds], tick_labels=["Source-patient refits", "TCGA Train mapping"], showfliers=False)
    axes[0, 1].set(title="B  Mean overlap over 50 draws per patient", ylabel="Mean Top-10 overlap with fixed PCR", ylim=(-.03, 1.03))
    primary_candidates = candidates[candidates.method.eq(primary)].pivot(index="Gene", columns="split", values="top10_frequency")
    active = (primary_candidates.train > 0) | (primary_candidates.validation > 0)
    axes[1, 0].scatter(primary_candidates.loc[active, "train"], primary_candidates.loc[active, "validation"], s=12, alpha=.5, color="#0072B2")
    for gene in discovery.Gene.head(3):
        axes[1, 0].annotate(gene, (primary_candidates.loc[gene, "train"], primary_candidates.loc[gene, "validation"]), fontsize=8)
    axes[1, 0].plot([0, .5], [0, .5], ls="--", color="gray", lw=1)
    axes[1, 0].set(title="C  Cohort candidate frequency", xlabel="Train Top-10 frequency", ylabel="Validation Top-10 frequency")
    joined = mapping[mapping.comparison.eq(methods[0])].merge(domain[domain.method.eq(primary)], on=["patient_id", "split"], validate="one_to_one")
    for split, color in [("train", "#0072B2"), ("validation", "#D55E00")]:
        group = joined[joined.split.eq(split)]
        axes[1, 1].scatter(group.out_of_reference_fraction, group.top10_overlap, s=13, alpha=.5, color=color, label=split)
    axes[1, 1].set(title="D  Domain distance and mapping disagreement", xlabel="Features outside training 1%-99% range", ylabel="Top-10 overlap between two shifts", ylim=(-.03, 1.03))
    axes[1, 1].legend(frameon=False)
    fig.suptitle("PCR development transfer: stability is not functional accuracy", fontsize=13)
    save_figure(fig, output, "figure1_pcr_transfer_stability")
    # Figure 2: expression/detection, with donor counts printed beside each group.
    fig, ax = plt.subplots(figsize=(12.2, 8.5), constrained_layout=True)
    display_groups = [("author_ccrcc", "author_tumor"), ("author_ccrcc", "myeloid"), ("author_ccrcc", "lymphoid"),
                      ("author_ccrcc", "endothelial"), ("author_ccrcc", "stromal"),
                      ("author_normal", "normal_proximal_tubule"), ("author_normal", "normal_other_epithelium")]
    points = []
    labels = []
    for j, (condition, group) in enumerate(display_groups):
        frame = source_main[(source_main.author_cohort == condition) & (source_main.group == group)].set_index("Gene")
        donor_n = int(frame.donor_n.max()) if not frame.empty else 0
        labels.append(f"{group.replace('_', ' ')}\n(n={donor_n} donors)")
        for i, gene in enumerate(discovery.Gene):
            if gene in frame.index and bool(frame.loc[gene, "gene_present"]):
                row = frame.loc[gene]
                points.append((j, i, np.log2(row.median_cpm + 1), row.mean_detection_fraction))
    points = np.asarray(points)
    dots = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=15+points[:, 3]*180, cmap="viridis", vmin=0)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(discovery)), [f"{rank}. {gene}" for rank, gene in zip(discovery.pcr_discovery_rank, discovery.Gene)])
    ax.invert_yaxis()
    ax.set_title("PCR Train-discovered candidates: author-annotated cell sources\nColor: log2(median donor CPM+1); size: equal-donor mean detection fraction", fontsize=11)
    fig.colorbar(dots, ax=ax, label="log2(median donor CPM + 1)", fraction=.035, pad=.02)
    fig.text(.02, -.015, "Descriptive expression only. 8/13 GEO-author tissue labels conflict; tumor labels are not independently CNV-confirmed.", fontsize=8)
    save_figure(fig, output, "figure2_candidate_cell_sources")
    # Figure 3: each line is one donor, and the concordance audit is visible.
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    mix_scenarios = ["nonmalignant_rna_25pct", "nonmalignant_rna_50pct", "nonmalignant_rna_75pct"]
    for donor, group in mixture.groupby("donor"):
        frame = group.set_index("scenario")
        concordant = bool(group.geo_author_concordant.all())
        axes[0].plot([0, 25, 50, 75], [1]+frame.loc[mix_scenarios, "top10_overlap"].tolist(),
                     marker="o", lw=1.2, color="#0072B2" if concordant else "#999999", alpha=.75,
                     label=donor + (" (concordant)" if concordant else " (conflict)"))
    axes[0].set(xlabel="Nonmalignant RNA fraction (%)", ylabel="Top-10 overlap with tumor-only scenario",
                title="A  Within-donor mixture sensitivity", ylim=(-.03, 1.03))
    axes[0].legend(fontsize=7, frameon=False)
    x = np.arange(len(audit))
    colors = ["#0072B2" if c else "#D55E00" for c in audit.geo_author_concordant]
    axes[1].bar(x, audit.retained_cells/audit.annotated_cells, color=colors)
    axes[1].set_xticks(x, audit.sample_id, rotation=65, ha="right", fontsize=7)
    axes[1].set(ylabel="Fraction retained after extra QC", ylim=(0, 1.05), title="B  QC and tissue-label audit")
    axes[1].text(.02, .04, "Blue: GEO-author concordant\nOrange: tissue-label conflict", transform=axes[1].transAxes, fontsize=8)
    fig.suptitle("GSE159115 exploratory sensitivity (7 tumor donors)", fontsize=13)
    fig.text(.02, -.02, "Each scenario uses donor-mean expression alignment; these are RNA fractions, not measured TCGA purity or dependency accuracy.", fontsize=8)
    save_figure(fig, output, "figure3_mixture_and_source_audit")
    metrics = {"pcr": {k: pcr_run[k] for k in ["training_model_n", "training_patient_n", "patient_n", "ranking_gene_n", "pcr_rank", "alpha", "bootstrap_draws", "locked_test_overlap"]},
               "singlecell": {k: sc_run[k] for k in ["sample_n", "donor_n", "author_ccrcc_donor_n", "author_normal_donor_n", "retained_cell_n", "geo_author_discordant_sample_n", "mixture_donor_n", "model_feature_coverage"]},
               "mapping_overlap": mapping_summary.to_dict(orient="records"),
               "bootstrap_overlap": bootstrap_summary.to_dict(orient="records"),
               "mixture_overlap": mixture_summary.to_dict(orient="records"),
               "primary_cohort_stability": pd.read_csv(args.pcr_dir / "cohort_stability.csv").query("method == @primary").to_dict(orient="records"),
               "pcr_top20_historical_overlap_n": int(evidence.historical_frozen_candidate.sum()),
               "status": "development_and_descriptive_analysis_complete_not_patient_functional_validation"}
    (output / "extension_summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    finish_run(output, {"status": "complete", "pcr_run_sha256": sha256(args.pcr_dir / "run.json"),
                        "singlecell_run_sha256": sha256(args.singlecell_dir / "run.json"),
                        "script_sha256": sha256(Path(__file__)), "candidate_reranking": False,
                        "locked_test_accessed": False, "figure_n": 3,
                        "claim_ceiling": "Patient development ranking stability and descriptive independent expression sources"})
    if args.snapshot:
        snapshot(args.pcr_dir, root / "results/historical/pcr_patient_extension_v1",
                 [p.name for p in args.pcr_dir.iterdir() if p.is_file() and p.name not in ["pcr_model.npz", "predicted_residuals.npz"]])
        snapshot(args.singlecell_dir, root / "results/historical/ccrcc_singlecell_extension_v1",
                 [p.name for p in args.singlecell_dir.iterdir() if p.is_file() and p.name not in ["pseudobulk_counts.npz", "cell_metadata_qc.csv.gz"]])
        snapshot(output, root / "results/historical/pcr_singlecell_synthesis_v1", [p.name for p in output.iterdir() if p.is_file()])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"【完成】综合证据表、3张PNG/PDF和紧凑版本化快照: {output}")


if __name__ == "__main__":
    main()
