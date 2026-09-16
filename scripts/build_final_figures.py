#!/usr/bin/env python3
"""Build publication-ready figures from the frozen final evidence synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BLUE = "#2878B5"
LIGHT_BLUE = "#9ECAE1"
RED = "#D9534F"
ORANGE = "#E69F00"
GREEN = "#2A9D8F"
GRAY = "#9AA0A6"
DARK = "#25313C"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    names = [f"{stem}.png", f"{stem}.pdf"]
    fig.savefig(output_dir / names[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / names[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return names


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.edgecolor": "#4F5B66",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def select_one(frame: pd.DataFrame, **conditions) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"指标未唯一匹配：{conditions}，数量={len(selected)}")
    return selected.iloc[0]


def figure_validation(internal: pd.DataFrame, external: pd.DataFrame,
                      cohort: pd.DataFrame, sensitivity: pd.DataFrame,
                      benchmark: pd.DataFrame, locked_test_run: dict,
                      candidates: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.4),
                             gridspec_kw={"width_ratios": [1.1, 1.4, 1.1, 0.9]})

    ax = axes[0]
    labels = {
        "training_selectivity_prior": "Selective prior",
        "annotation_ridge": "Annotation ridge",
        "expression_knn": "Expression kNN",
        "expression_pcr_ridge": "Expression PCR-ridge",
        "expression_kernel_ridge_frozen": "Frozen kernel ridge",
        "expression_kernel_ridge_tuned": "Tuned kernel ridge",
    }
    current = benchmark.loc[benchmark["estimand"].eq("lineage_equal")].copy()
    order = list(labels)
    current = current.set_index("method").loc[order]
    colors = [GRAY, GRAY, LIGHT_BLUE, GREEN, BLUE, ORANGE]
    ypos = np.arange(len(order))[::-1]
    bars = ax.barh(ypos, current["ndcg_at_10"], color=colors, height=0.62)
    ax.set_yticks(ypos, [labels[x] for x in order])
    ax.set_xlim(0, max(0.5, float(current["ndcg_at_10"].max()) * 1.18))
    ax.set_xlabel("Lineage-equal NDCG@10")
    ax.set_title("A  Fair baseline benchmark", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, current["ndcg_at_10"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7.5)

    rows = [
        ("DepMap LOLO\n(patient equal)", select_one(internal, metric="ndcg_at_10_gain", estimand="patient_equal"), BLUE),
        ("DepMap LOLO\n(lineage equal)", select_one(internal, metric="ndcg_at_10_gain", estimand="lineage_equal"), BLUE),
        ("Sanger direct\n(model weighted)", select_one(
            external, comparison="direct_expression_z_residual_vs_training_standardized_prior",
            metric="ndcg_at_10_gain", estimand="model_weighted"), GREEN),
        ("Sanger direct\n(lineage equal)", select_one(
            external, comparison="direct_expression_z_residual_vs_training_standardized_prior",
            metric="ndcg_at_10_gain", estimand="lineage_equal"), GREEN),
        ("Sanger mapped\n(model weighted)", select_one(
            external, comparison="mapped_expression_z_residual_vs_training_standardized_prior",
            metric="ndcg_at_10_gain", estimand="model_weighted"), ORANGE),
    ]
    ax = axes[1]
    y = np.arange(len(rows))[::-1]
    for pos, (label, row, color) in zip(y, rows):
        mean, low, high = row["mean"], row["ci_low"], row["ci_high"]
        ax.errorbar(mean, pos, xerr=[[mean - low], [high - mean]], fmt="o", color=color,
                    ecolor=color, capsize=3, markersize=6, linewidth=1.7)
    ax.axvline(0, color="#6B7280", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, [x[0] for x in rows])
    ax.set_xlabel("NDCG@10 gain vs selective prior")
    ax.set_title("B  Internal and external gains", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    primary = select_one(cohort, method="nonkidney_shift_driver_neutral")
    mapping = sensitivity.loc[sensitivity["comparison"].eq("mapping_neutral")]
    test = locked_test_run["primary_results"]
    stability_labels = ["Train-Val\nTop-100", "Train-Test\nTop-100", "Frozen20\nVal-Test rho",
                        "Test map\nTop-10"]
    stability_values = [primary["top100_frequency_overlap"], test["train_test_top100_overlap"],
                        test["frozen20_validation_test_frequency_spearman"],
                        test["test_mapping_top10_overlap_mean"]]
    colors = [BLUE, BLUE, GREEN, ORANGE]
    bars = ax.bar(np.arange(4), stability_values, color=colors, width=0.68)
    ax.set_xticks(np.arange(4), stability_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement")
    ax.set_title("C  Locked patient stability", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, stability_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[3]
    attrition = [
        ("Frozen candidates", len(candidates), DARK),
        ("RNAi absolute support", int(candidates["orthogonal_rnai_support"].sum()), BLUE),
        ("RNAi ccRCC-specific", int(candidates["rnai_ccrcc_specific_support"].sum()), GREEN),
        ("Patient-derived truth", int(candidates["patient_derived_functional_truth_available"].sum()), RED),
    ]
    ypos = np.arange(len(attrition))[::-1]
    bars = ax.barh(ypos, [x[1] for x in attrition], color=[x[2] for x in attrition], height=0.58)
    ax.set_yticks(ypos, [x[0] for x in attrition])
    ax.set_xlim(0, 21)
    ax.set_xlabel("Candidate count")
    ax.set_title("D  Evidence attrition", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, (_, value, _) in zip(bars, attrition):
        ax.text(max(value + 0.45, 0.45), bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)

    fig.suptitle("Benchmark performance, external replication, and the patient-evidence ceiling",
                 x=0.04, ha="left", fontweight="bold")
    fig.text(0.04, 0.005,
             "Baseline models use identical whole-lineage splits. Intervals are 95% bootstrap intervals; patient agreement is not functional accuracy.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93], w_pad=2.0)
    return fig


def figure_evidence_matrix(candidates: pd.DataFrame) -> plt.Figure:
    columns = [
        ("Validation\nTop-100", "validation_top100_retained", "support"),
        ("Locked Test\ndirection", "prespecified_direction_replicated", "support"),
        ("DepMap Kidney\ndirection", "depmap_kidney_lineage_direction", "support"),
        ("Sanger\n3/3", "sanger_all_three_support", "support"),
        ("RNAi\nabsolute", "orthogonal_rnai_support", "support"),
        ("RNAi ccRCC\nspecific", "rnai_ccrcc_specific_support", "support"),
        ("Direct LoF\nsupport", "direct_peer_reviewed_support", "support"),
        ("Tumor\nupregulation", "paired_tumor_upregulation", "support"),
        ("Clinical-stage\ntractability", "clinical_stage_tractability", "support"),
        ("Direct\nopposition", "direct_peer_reviewed_opposition", "risk"),
        ("Normal organoid\nliability", "normal_organoid_liability", "risk"),
    ]
    data = np.zeros((len(candidates), len(columns)), dtype=int)
    for j, (_, field, kind) in enumerate(columns):
        present = candidates[field].fillna(False).astype(bool).to_numpy()
        data[present, j] = -1 if kind == "risk" else 1

    fig, ax = plt.subplots(figsize=(10.4, 7.3))
    cmap = ListedColormap([RED, "#F4F5F7", BLUE])
    ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(columns)), [x[0] for x in columns])
    ax.set_yticks(np.arange(len(candidates)),
                  [f"{int(r.discovery_rank):02d}  {r.Gene}" for r in candidates.itertuples()])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("left")
        label.set_rotation_mode("anchor")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(candidates), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1:
                ax.text(j, i, "●", ha="center", va="center", color="white", fontsize=9)
            elif data[i, j] == -1:
                ax.text(j, i, "!", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_title("Frozen ccRCC candidate evidence matrix", loc="left", pad=62, fontsize=13, fontweight="bold")
    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, markeredgecolor="none", markersize=9,
               label="Supporting evidence"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markeredgecolor="none", markersize=9,
               label="Opposition or normal-tissue liability"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#F4F5F7", markeredgecolor="#D1D5DB", markersize=9,
               label="Criterion not met / unavailable"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.045), ncol=3, frameon=False)
    fig.text(0.125, 0.025,
             "Rows retain discovery rank. Empty cells are not proof of no effect. No composite score or reranking was used.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    return fig


def figure_dependency_exposure(candidates: pd.DataFrame) -> plt.Figure:
    covered = candidates.loc[candidates["drive_ccrcc_n"].notna()].copy()
    covered["dependency_strength"] = -covered["drive_ccrcc_residual_mean"]
    covered["strength_low"] = -covered["drive_ccrcc_residual_ci_high"]
    covered["strength_high"] = -covered["drive_ccrcc_residual_ci_low"]
    covered["kidney_log_tpm"] = np.log10(covered["gtex_kidney_max_median_tpm"] + 1)

    class_colors = {
        "E1_功能最一致但无治疗窗": RED,
        "E2_肾谱系功能复现伴正常肾风险": ORANGE,
        "E2_肾谱系功能复现但特异性不足": BLUE,
        "E2_可成药与肿瘤表达支持但功能复现不足": GREEN,
        "E3_细胞系跨平台信号": LIGHT_BLUE,
        "E4_直接反向证据": "#8E5EA2",
        "E5_计算候选证据不足": GRAY,
    }
    colors = [class_colors.get(x, GRAY) for x in covered["evidence_class"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    x = covered["dependency_strength"].to_numpy()
    xerr = np.vstack([x - covered["strength_low"].to_numpy(), covered["strength_high"].to_numpy() - x])
    ax.errorbar(x, covered["kidney_log_tpm"], xerr=xerr, fmt="none", ecolor="#AAB2BD",
                elinewidth=1.1, capsize=2, zorder=1)
    ax.scatter(x, covered["kidney_log_tpm"], c=colors, s=64, edgecolor="white", linewidth=0.8, zorder=2)
    label_offsets = {
        "PAX8": (0.018, 0.055), "HNF1B": (0.018, -0.10),
        "CCND1": (0.020, 0.045), "CFLAR": (0.018, -0.080),
        "HSD17B12": (-0.145, 0.020),
    }
    for row in covered.itertuples():
        dx, dy = label_offsets.get(row.Gene, (0.018, 0.025))
        ax.text(row.dependency_strength + dx, row.kidney_log_tpm + dy, row.Gene, fontsize=8)
    ax.axvline(0, color="#6B7280", linestyle="--", linewidth=0.9)
    ax.set_xlabel("DRIVE ccRCC dependency strength  (− residual mean)")
    ax.set_ylabel("Normal kidney exposure  log10(GTEx median TPM + 1)")
    ax.set_title("Functional dependency versus normal-kidney exposure", loc="left", fontweight="bold")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "right = stronger dependency\nup = higher normal exposure",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#4B5563")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=7,
               label="PAX8: strongest function, normal liability"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=7,
               label="HNF1B: renal function plus liability"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7,
               label="FERMT2: function, specificity unresolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="white", markersize=7,
               label="CCND1: tractability, RNAi inconclusive"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False)
    fig.text(0.12, 0.015,
             "Horizontal bars are 95% bootstrap intervals. Expression is exposure, not toxicity; no safety threshold is defined.",
             fontsize=8, color="#4B5563")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path,
                        default=root / "outputs/final_evidence_synthesis_v2")
    parser.add_argument("--internal-bootstrap", type=Path,
                        default=root / "outputs/selective_dependency_analysis_v1/bootstrap.csv")
    parser.add_argument("--external-bootstrap", type=Path,
                        default=root / "outputs/sanger_selective_dependency_v1/bootstrap.csv")
    parser.add_argument("--cohort-stability", type=Path,
                        default=root / "outputs/tcga_patient_transfer_v1/cohort_stability.csv")
    parser.add_argument("--input-sensitivity", type=Path,
                        default=root / "outputs/tcga_patient_transfer_v1/input_sensitivity.csv")
    parser.add_argument("--benchmark-summary", type=Path,
                        default=root / "outputs/selective_dependency_benchmark_v1/overall_summary.csv")
    parser.add_argument("--locked-test-run", type=Path,
                        default=root / "results/historical/tcga_locked_test_v1/run.json")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "outputs/final_figures_v2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    inputs = {
        "candidate_evidence_matrix": args.evidence_dir / "candidate_evidence_matrix.csv",
        "evidence_run": args.evidence_dir / "run.json",
        "internal_bootstrap": args.internal_bootstrap,
        "external_bootstrap": args.external_bootstrap,
        "cohort_stability": args.cohort_stability,
        "input_sensitivity": args.input_sensitivity,
        "baseline_benchmark": args.benchmark_summary,
        "locked_test_run": args.locked_test_run,
    }
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(f"拒绝覆盖已有结果：{args.output_dir.resolve()}")
    print("【阶段 1/4】校验最终证据矩阵和绘图输入", flush=True)
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    candidates = pd.read_csv(inputs["candidate_evidence_matrix"])
    if len(candidates) != 20 or candidates["Gene"].nunique() != 20:
        raise ValueError("最终证据矩阵不是唯一的20个冻结候选")
    if args.dry_run:
        print("【检查通过】候选 20｜图 3｜PNG+PDF｜未写入结果", flush=True)
        return

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print("【阶段 2/4】绘制模型验证与证据上限图", flush=True)
    generated = save_figure(
        figure_validation(pd.read_csv(args.internal_bootstrap), pd.read_csv(args.external_bootstrap),
                          pd.read_csv(args.cohort_stability), pd.read_csv(args.input_sensitivity),
                          pd.read_csv(args.benchmark_summary), json.loads(args.locked_test_run.read_text()),
                          candidates),
        args.output_dir, "figure1_validation_and_evidence_ceiling")
    print("【阶段 3/4】绘制候选证据矩阵和功能—正常肾暴露图", flush=True)
    generated += save_figure(figure_evidence_matrix(candidates), args.output_dir, "figure2_candidate_evidence_matrix")
    generated += save_figure(figure_dependency_exposure(candidates), args.output_dir,
                             "figure3_dependency_vs_normal_kidney_exposure")

    print("【阶段 4/4】保存图注、来源哈希和运行记录", flush=True)
    manifest = pd.DataFrame([
        {
            "figure": "Figure 1", "stem": "figure1_validation_and_evidence_ceiling",
            "caption": ("Fair whole-lineage baseline performance, internal and external NDCG@10 gains, locked patient "
                        "stability, and evidence attrition. Patient stability is not functional accuracy."),
        },
        {
            "figure": "Figure 2", "stem": "figure2_candidate_evidence_matrix",
            "caption": ("Frozen top-20 candidate evidence matrix retaining discovery order and descriptive locked-Test "
                        "expression-direction replication. Blue denotes support; red denotes opposition or normal-kidney liability."),
        },
        {
            "figure": "Figure 3", "stem": "figure3_dependency_vs_normal_kidney_exposure",
            "caption": ("DRIVE ccRCC dependency strength versus GTEx normal-kidney exposure for covered candidates. "
                        "Expression indicates exposure, not toxicity, and no therapeutic-window threshold is defined."),
        },
    ])
    manifest.to_csv(args.output_dir / "figure_manifest.csv", index=False)
    generated.append("figure_manifest.csv")
    run = {
        "status": "final_publication_figures_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "compute": "CPU vector/raster rendering; GPU acceleration is not applicable",
        "design": {
            "frozen_candidate_n": 20,
            "figure_n": 3,
            "formats": ["png_300dpi", "pdf_vector"],
            "candidate_reranking": False,
            "composite_score": False,
            "locked_tcga_test_used": True,
            "locked_tcga_test_use": "descriptive stability and prespecified expression direction only",
            "baseline_benchmark_used": True,
        },
        "source_sha256": {name: sha256(path) for name, path in inputs.items()},
        "output_sha256": {name: sha256(args.output_dir / name) for name in generated},
        "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "Figures summarize frozen analyses and do not add independent biological evidence.",
            "Patient-transfer agreement is stability without functional ground truth.",
            "GTEx expression is normal-tissue exposure, not drug toxicity.",
            "The evidence matrix deliberately does not collapse heterogeneous evidence into a score.",
        ],
    }
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
    print("【图件完成】主图 3｜PNG 300dpi 3｜矢量PDF 3｜不改变候选排序", flush=True)
    print(f"【完成】耗时 {time.monotonic() - started:.1f}秒｜结果 {args.output_dir.resolve()}", flush=True)
    print("【结论边界】图件只汇总既有证据，不增加患者功能真值或临床证据。", flush=True)


if __name__ == "__main__":
    main()
