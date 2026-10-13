"""Redraw frozen results using MechDec visual conventions; no model fitting.

Default stages new images for inspection. --install replaces the authorized old
images, retires v1 images, and refreshes rendering manifests without changing
analysis tables or training records.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import build_final_figures as original
from summarize_pcr_instability_audit import VARIANTS, LABELS

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs/mechdec_redraw_20261013_review2"
BLUE, TEAL, GOLD, RED, PURPLE, GRAY = "#6599B7", "#68A69B", "#D7AD63", "#BC6765", "#A18CAF", "#A6A6A6"
PALETTE = [BLUE, TEAL, GOLD, RED, PURPLE, GRAY]
SEQUENTIAL = LinearSegmentedColormap.from_list("mechdec_expression", ["#F3F7F8", "#BFD6E3", "#6599B7", "#35657A"])
SOURCES = {}
RECORDS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = sha(path)
    return pd.read_csv(path)


def read_json(relative):
    path = ROOT / relative
    SOURCES[relative] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "normal", "axes.labelsize": 9,
        "axes.edgecolor": "#4C4C4C", "axes.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "grid.color": "#EAEAEA", "grid.linewidth": .45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    original.BLUE, original.LIGHT_BLUE, original.RED = BLUE, "#BDD4E1", RED
    original.ORANGE, original.GREEN, original.GRAY, original.DARK = GOLD, TEAL, GRAY, "#5A5A5A"


def panels(axes):
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        title = re.sub(r"^[A-D]\s+", "", ax.get_title() or ax.get_title(loc="left"))
        ax.set_title(""); ax.set_title("", loc="right")
        ax.set_title(title, loc="left", fontsize=10, fontweight="normal", pad=10)
        ax.text(-.12, 1.04, chr(97+i), transform=ax.transAxes,
                fontweight="bold", fontsize=13, ha="left", va="bottom")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#555555"); spine.set_linewidth(.7)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color("#EAEAEA"); line.set_linewidth(.45)


def save(fig, group, stem, note):
    target = STAGE / group
    target.mkdir(exist_ok=True)
    fig.canvas.draw()
    for ext in ("png", "pdf"):
        fig.savefig(target / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    RECORDS.append(dict(group=group, stem=stem, interpretation=note,
        png_sha256=sha(target/f"{stem}.png"), pdf_sha256=sha(target/f"{stem}.pdf")))


def main_figures():
    group = "final_figures_v2"
    candidate = read("results/historical/final_evidence_synthesis_v2/candidate_evidence_matrix.csv")
    internal = read("outputs/selective_dependency_benchmark_v1/overall_summary.csv")
    external = read("outputs/sanger_baseline_benchmark_v1/overall_summary.csv")
    fig = original.figure_validation(read("outputs/selective_dependency_analysis_v1/bootstrap.csv"),
        read("outputs/sanger_selective_dependency_v1/bootstrap.csv"),
        read("outputs/tcga_patient_transfer_v1/cohort_stability.csv"),
        read("outputs/tcga_patient_transfer_v1/input_sensitivity.csv"), internal,
        read_json("results/historical/tcga_locked_test_v1/run.json"), candidate)
    # Keep the original frozen statistics; use a less crowded two-row layout.
    fig.set_size_inches(12.8, 8.0)
    positions = [[.20,.59,.29,.29],[.69,.59,.28,.29],[.20,.18,.29,.29],[.69,.18,.28,.29]]
    for ax, pos in zip(fig.axes, positions): ax.set_position(pos)
    fig._suptitle.set_text("")
    for text in fig.texts:
        if text is not fig._suptitle: text.set_position((.12,.025))
    panels(fig.axes)
    save(fig,group,"figure1_validation_and_evidence_ceiling","Frozen validation statistics; original bootstrap intervals; patient stability is not accuracy.")

    columns = [("Validation\nTop-100","validation_top100_retained",False),
        ("Locked Test\ndirection","prespecified_direction_replicated",False),
        ("DepMap Kidney\ndirection","depmap_kidney_lineage_direction",False),
        ("Sanger\n3/3","sanger_all_three_support",False),
        ("RNAi\nabsolute","orthogonal_rnai_support",False),
        ("RNAi ccRCC\nspecific","rnai_ccrcc_specific_support",False),
        ("Direct LoF\nsupport","direct_peer_reviewed_support",False),
        ("Tumor\nupregulation","paired_tumor_upregulation",False),
        ("Clinical-stage\ntractability","clinical_stage_tractability",False),
        ("Direct\nopposition","direct_peer_reviewed_opposition",True),
        ("Normal organoid\nliability","normal_organoid_liability",True)]
    fig,ax=plt.subplots(figsize=(11.8,7.4))
    for i in range(len(candidate)):
        ax.axhline(i,color="#F1F1F1",lw=.5,zorder=0)
    for j,(_,field,risk) in enumerate(columns):
        values=candidate[field].fillna(False).astype(bool).to_numpy()
        ax.scatter(np.full(len(values),j),np.arange(len(values)),s=20,marker="o",facecolors="none",edgecolors="#D7D7D7",lw=.6)
        ax.scatter(np.full(values.sum(),j),np.flatnonzero(values),s=40,marker="X" if risk else "o",c=RED if risk else TEAL,edgecolors="white",lw=.3)
    ax.set_xticks(range(len(columns)),[v[0] for v in columns],rotation=35,ha="left",rotation_mode="anchor")
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=7)
    ax.set_yticks(range(len(candidate)),[f"{int(r.discovery_rank):02}  {r.Gene}" for r in candidate.itertuples()])
    ax.set_ylim(len(candidate)-.6,-.6);ax.set_xlim(-.6,len(columns)-.4)
    ax.spines[["left","bottom"]].set_visible(False)
    ax.legend(handles=[Line2D([],[],marker="o",ls="",color=TEAL,label="Support"),
        Line2D([],[],marker="X",ls="",color=RED,label="Opposition / normal liability"),
        Line2D([],[],marker="o",ls="",markerfacecolor="none",color="#BBBBBB",label="Criterion not met / unavailable")],
        bbox_to_anchor=(0,-.035),loc="upper left",ncol=3)
    fig.text(.12,.025,"Frozen discovery order; no composite score. Locked Test: prespecified median direction only; HNF1B interval crosses zero.",fontsize=8,color="#555555")
    fig.subplots_adjust(left=.15,right=.97,bottom=.13,top=.77)
    save(fig,group,"figure2_candidate_evidence_matrix","All 20 candidates and 11 criteria retained; hollow cells are not proof of absence.")
    fig=original.figure_dependency_exposure(candidate); panels([fig.axes[0]])
    save(fig,group,"figure3_dependency_vs_normal_kidney_exposure","RNAi intervals unchanged; normal expression is exposure, not toxicity.")

    labels={"training_selectivity_prior":"Selective prior","annotation_ridge":"Annotation ridge",
        "expression_knn":"Expression kNN","expression_pcr_ridge":"PCR-ridge",
        "expression_kernel_ridge_frozen":"Frozen kernel ridge","expression_kernel_ridge_tuned":"Tuned kernel ridge"}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7))
    for ax,frame,title in zip(axes,[internal,external],["DepMap: whole-lineage holdout","Sanger: external platform"]):
        data=frame[frame.estimand.eq("lineage_equal")].set_index("method")
        for i,(method,label) in enumerate(labels.items()):
            if method not in data.index: continue
            value=float(data.loc[method,"ndcg_at_10"])
            color=[GRAY,GRAY,BLUE,TEAL,PURPLE,GOLD][i]
            ax.plot([0,value],[i,i],color=color,lw=1.2,alpha=.55)
            ax.scatter(value,i,s=42,c=color,marker=["o","o","^","D","s","v"][i],zorder=3)
            ax.text(value+.01,i,f"{value:.3f}",va="center",fontsize=8,color="#555555")
        ax.set_yticks(range(6),labels.values());ax.set_ylim(5.6,-.6);ax.set_xlim(0,.51)
        ax.set_xlabel("Lineage-equal NDCG@10");ax.set_title(title);ax.grid(axis="x")
    panels(axes);fig.tight_layout(w_pad=3)
    fig.subplots_adjust(bottom=.2)
    fig.text(.04,.035,"Identical within-benchmark splits and gene spaces. Sanger: 64/66 models; two lineages lack eligible tuning cohorts.",fontsize=8,color="#555555")
    save(fig,group,"figure4_baseline_generalization","Original lineage-equal point estimates, no new uncertainty estimates or cross-dataset paired inference.")


def raincloud(ax, arrays, labels):
    # Deterministic display jitter only; every finite patient observation retained.
    rng=np.random.default_rng(20261013)
    for i,values in enumerate(arrays,1):
        values=np.asarray(values,dtype=float);values=values[np.isfinite(values)]
        color=PALETTE[(i-1)%len(PALETTE)]
        if len(np.unique(values))>1:
            violin=ax.violinplot([values],positions=[i],widths=.55,showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.18)
                verts=body.get_paths()[0].vertices;verts[:,0]=np.minimum(verts[:,0],i)
        ax.scatter(i+rng.uniform(.08,.28,len(values)),values,s=8,c=color,alpha=.4,edgecolors="none",rasterized=True)
        bp=ax.boxplot([values],positions=[i],widths=.13,showfliers=False,patch_artist=True,
            medianprops=dict(color="#444444",lw=1.1),boxprops=dict(facecolor="white",edgecolor=color,lw=.9),
            whiskerprops=dict(color=color,lw=.9),capprops=dict(color=color,lw=.9))
    ax.set_xticks(range(1,len(labels)+1),labels);ax.set_ylim(-.03,1.03);ax.grid(axis="y")


def transfer_figures():
    group="pcr_singlecell_synthesis_v1";primary="nonkidney_mean_shift_clip0"
    mapping=read("outputs/pcr_patient_extension_v1/mapping_sensitivity.csv")
    patient=read("outputs/pcr_singlecell_synthesis_v1/bootstrap_per_patient_mean_overlap.csv")
    candidates=read("outputs/pcr_patient_extension_v1/candidate_summary.csv.gz")
    discovery=read("outputs/pcr_patient_extension_v1/pcr_discovery_top20.csv")
    domain=read("outputs/pcr_patient_extension_v1/domain_diagnostics.csv")
    fig,axes=plt.subplots(2,2,figsize=(11.5,8.6));methods=list(mapping.comparison.unique())
    raincloud(axes[0,0],[mapping[mapping.comparison.eq(m)].top10_overlap for m in methods],["Kidney shift","Raw expression"])
    axes[0,0].set(title="Mapping sensitivity: 209 development patients",ylabel="Top-10 overlap with non-Kidney shift")
    kinds=list(patient.kind.unique())
    raincloud(axes[0,1],[patient[patient.kind.eq(k)].top10_overlap for k in kinds],["Source-patient refits","TCGA Train mapping"])
    axes[0,1].set(title="Patient mean across 50 draws",ylabel="Top-10 overlap with fixed PCR")
    freq=candidates[candidates.method.eq(primary)].pivot(index="Gene",columns="split",values="top10_frequency")
    active=(freq.train>0)|(freq.validation>0)
    axes[1,0].scatter(freq.loc[active,"train"],freq.loc[active,"validation"],s=10,c=BLUE,alpha=.45,rasterized=True)
    for gene in discovery.Gene.head(3):
        axes[1,0].annotate(gene,(freq.loc[gene,"train"],freq.loc[gene,"validation"]),xytext=(4,5),textcoords="offset points",fontsize=8)
    axes[1,0].plot([0,.5],[0,.5],ls="--",c=GRAY,lw=.8)
    axes[1,0].set(title="Cohort candidate frequency",xlabel="Train Top-10 frequency",ylabel="Validation Top-10 frequency")
    joined=mapping[mapping.comparison.eq(methods[0])].merge(domain[domain.method.eq(primary)],on=["patient_id","split"],validate="one_to_one")
    for split,color,marker in [("train",BLUE,"o"),("validation",RED,"^")]:
        g=joined[joined.split.eq(split)]
        axes[1,1].scatter(g.out_of_reference_fraction,g.top10_overlap,s=15,c=color,marker=marker,alpha=.55,label=split,rasterized=True)
    axes[1,1].set(title="Domain distance and mapping disagreement",xlabel="Features outside training 1%–99% range",ylabel="Top-10 overlap between shifts",ylim=(-.03,1.03));axes[1,1].legend()
    panels(axes);fig.tight_layout(h_pad=3,w_pad=3);fig.subplots_adjust(bottom=.10)
    fig.text(.04,.025,"Development cohort only. All patient observations shown; box: median / IQR, whiskers: 1.5 IQR. Stability is not accuracy.",fontsize=8,color="#555555")
    save(fig,group,"figure1_pcr_transfer_stability","209 development patients; draw averages kept separate from independent patient counts.")
    sources=read("outputs/ccrcc_singlecell_extension_v1/candidate_cell_source_summary.csv")
    sources=sources[sources.scope.eq("author_cohort")]
    source_dot_plot(discovery.Gene.tolist(),sources,"author_cohort",group,"figure2_candidate_cell_sources",old=True)
    mixture=read("outputs/ccrcc_singlecell_extension_v1/mixture_sensitivity.csv")
    audit=read("outputs/ccrcc_singlecell_extension_v1/sample_source_audit.csv")
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    scenarios=["nonmalignant_rna_25pct","nonmalignant_rna_50pct","nonmalignant_rna_75pct"]
    for i,(donor,g) in enumerate(mixture.groupby("donor")):
        s=g.set_index("scenario");concordant=bool(g.geo_author_concordant.all())
        axes[0].plot([0,25,50,75],[1]+s.loc[scenarios,"top10_overlap"].tolist(),marker=["o","s","^","D","v","P","X"][i],
            lw=1,c=BLUE if concordant else GRAY,alpha=.7,label=donor+(" (concordant)" if concordant else " (conflict)"))
    axes[0].set(title="Within-donor RNA mixture sensitivity",xlabel="Nonmalignant RNA fraction (%)",ylabel="Top-10 overlap with tumor-only",ylim=(-.03,1.03));axes[0].legend(fontsize=7)
    for i,row in enumerate(audit.itertuples()):
        color=BLUE if row.geo_author_concordant else RED
        axes[1].bar(i,row.retained_cells/row.annotated_cells,color=color,alpha=.75,width=.6,
            hatch="" if row.geo_author_concordant else "//",edgecolor="white",lw=.4)
    axes[1].set_xticks(range(len(audit)),audit.sample_id,rotation=65,ha="right",fontsize=7)
    axes[1].set(title="QC retention and tissue-label audit",ylabel="Retained cell fraction",ylim=(0,1.05))
    axes[1].legend(handles=[matplotlib.patches.Patch(facecolor=BLUE,label="Concordant"),matplotlib.patches.Patch(facecolor=RED,hatch="//",label="Tissue-label conflict")],fontsize=7)
    panels(axes);fig.tight_layout(w_pad=3);fig.subplots_adjust(bottom=.28)
    fig.text(.04,.035,"GSE159115: 7 tumor donors; author/GEO labels conflict. RNA mixture fractions are not measured patient purity or accuracy.",fontsize=8,color="#555555")
    save(fig,group,"figure3_mixture_and_source_audit","All 7 donor curves and all 13 tissue-label audit records retained.")


def source_dot_plot(genes,summary,scope,group,stem,old=False):
    conditions=[("author_ccrcc" if old else "Tumor","author_tumor"),
        ("author_ccrcc" if old else "Tumor","myeloid"),("author_ccrcc" if old else "Tumor","lymphoid"),
        ("author_ccrcc" if old else "Tumor","endothelial"),("author_ccrcc" if old else "Tumor","stromal"),
        ("author_normal" if old else "Normal","normal_proximal_tubule"),
        ("author_normal" if old else "Normal","normal_other_epithelium")]
    labels=[];points=[];missing=[]
    fig,ax=plt.subplots(figsize=(10.8,8))
    for j,(condition,cell) in enumerate(conditions):
        subset=summary[(summary.author_cohort.eq(condition) if old else summary.condition.eq(condition))&summary.group.eq(cell)]
        if not old: subset=subset[subset.scope.eq(scope)&subset.qc.eq("extra_qc")]
        subset=subset.set_index("Gene")
        n=int(subset.donor_n.max()) if len(subset) else 0
        labels.append(cell.replace("normal_","").replace("author_","").replace("_"," ")+f"\n(n={n} donors)")
        for i,gene in enumerate(genes):
            if gene not in subset.index or ("gene_present" in subset.columns and not bool(subset.loc[gene,"gene_present"])) or not np.isfinite(subset.loc[gene,"median_cpm"]):
                missing.append((j,i));continue
            row=subset.loc[gene]
            detection=float(row.mean_detection_fraction) if old else None
            points.append((j,i,np.log2(row.median_cpm+1) if old else np.log10(row.median_cpm+1),detection))
    points=np.array(points,dtype=float)
    for i in range(len(genes)): ax.axhline(i,color="#F1F1F1",lw=.5,zorder=0)
    sizes=15+points[:,3]*180 if old else np.full(len(points),65)
    dots=ax.scatter(points[:,0],points[:,1],c=points[:,2],s=sizes,cmap=SEQUENTIAL,vmin=0,lw=.3,edgecolor="#5F8494")
    if missing:
        m=np.array(missing);ax.scatter(m[:,0],m[:,1],marker="x",s=25,c=GRAY,lw=.8,label="Unmeasured gene / absent group")
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha="right",fontsize=8)
    ax.set_yticks(range(len(genes)),[f"{i+1:02}  {g}" for i,g in enumerate(genes)])
    ax.set_ylim(len(genes)-.7,-.7);ax.set_xlim(-.5,6.5);ax.tick_params(length=0)
    ax.spines[["left","bottom"]].set_visible(False)
    ax.set_title("GSE159115: author-annotated sources" if old else "GSE178481: primary-cohort cell sources",loc="left",pad=12)
    fig.colorbar(dots,ax=ax,label="log2(median donor CPM + 1)" if old else "log10(median donor CPM + 1)",fraction=.035,pad=.025)
    if old:
        # Same original area encoding, explicit guide for detection fractions.
        handles=[ax.scatter([],[],s=15+f*180,facecolors="none",edgecolors="#5F8494",label=f"{int(f*100)}% detection") for f in (.1,.5,1)]
        ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(0,-.19),ncol=3,fontsize=7)
    elif missing: ax.legend(loc="upper left",bbox_to_anchor=(0,-.17),fontsize=8)
    note="Frozen PCR order; expression is not dependency. " + ("8/13 GEO-author tissue labels conflict; dot area = donor-mean detection." if old else "6 tumor / 6 normal donors; crosses = missing, including MRPL57; not zero expression.")
    fig.subplots_adjust(left=.15,right=.91,top=.92,bottom=.24)
    fig.text(.08,.025,note,fontsize=8,color="#555555")
    save(fig,group,stem,note)


def instability_figures():
    group="pcr_instability_synthesis_v1"
    frame=read("outputs/pcr_instability_audit_v1/object_summary.csv.gz");patient=frame[frame.case.eq("TCGA")]
    association=read("outputs/pcr_instability_audit_v1/functional_associations.csv")
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),sharey=True)
    for ax,split in zip(axes,["train","validation"]):
        for color,marker,k in zip([BLUE,GOLD,TEAL],["o","s","^"],[10,20,50]):
            means=patient[(patient.split==split)&(patient.k==k)].groupby("variant").topk_overlap.mean().reindex(VARIANTS)
            ax.plot(means,np.arange(6),marker=marker,lw=1.1,c=color,label=f"Top {k}")
        ax.set_yticks(range(6),LABELS);ax.set_xlim(0,1);ax.set_ylim(5.5,-.5)
        ax.set_xlabel("Mean bootstrap overlap with original ranking");ax.set_title(f"TCGA {split}");ax.legend();ax.grid(axis="x")
    panels(axes);fig.tight_layout(w_pad=3);save(fig,group,"paired_controls","Same six conditional contrasts and three list lengths; not additive causal contributions.")
    a=association[association.k.eq(10)].set_index("variant").reindex(VARIANTS)
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.8))
    axes[0].hlines(range(6),0,a.mean_within_lineage_spearman,color=BLUE,lw=1.1,alpha=.55)
    axes[0].scatter(a.mean_within_lineage_spearman,range(6),c=BLUE,s=35,marker="s")
    axes[0].set_xlabel("Mean within-lineage Spearman");axes[0].set_title("Stability versus functional NDCG")
    axes[1].errorbar(a.lineage_equal_high_minus_low_ndcg,range(6),
        xerr=np.vstack([a.lineage_equal_high_minus_low_ndcg-a.ci_low,a.ci_high-a.lineage_equal_high_minus_low_ndcg]),
        fmt="D",c=TEAL,capsize=2,ms=5,lw=1.1)
    axes[1].set_xlabel("High minus low stability: NDCG@10");axes[1].set_title("Lineage-cluster bootstrap 95% intervals")
    for ax in axes: ax.set_yticks(range(6),LABELS);ax.set_ylim(5.5,-.5);ax.axvline(0,c=GRAY,ls="--",lw=.7);ax.grid(axis="x")
    panels(axes);fig.tight_layout(w_pad=3);fig.subplots_adjust(bottom=.19)
    fig.text(.04,.025,"19 held-out lineages; source patients are the sampling unit. Intervals are descriptive after development-data reuse.",fontsize=8,color="#555555")
    save(fig,group,"functional_stability","Original correlations and bootstrap bounds; no patient probability calibration.")
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5))
    for ax,split,color,marker in zip(axes,["train","validation"],[BLUE,RED],["o","^"]):
        g=patient[(patient.split==split)&(patient.k==10)&(patient.variant==VARIANTS[1])]
        ax.scatter(g.boundary_gap,g.topk_overlap,s=14,c=color,marker=marker,alpha=.5,rasterized=True)
        ax.set_xscale("symlog",linthresh=1e-5);ax.set_ylim(0,1)
        ax.set(xlabel="Rank 11 minus rank 10 residual (symlog)",ylabel="Mean common-reference Top-10 overlap",title=f"TCGA {split}: one point per patient")
        ax.grid(lw=.45)
    panels(axes);fig.tight_layout(w_pad=3)
    save(fig,group,"boundary_stability","Every patient observation retained; original symmetric-log axis and bounds.")


def replication_figures():
    group="ccrcc_source_replication_synthesis_v1"
    candidate=read("outputs/ccrcc_source_replication_v1/candidate_sets.csv")
    genes=candidate[candidate.pcr_discovery_rank.notna()].sort_values("pcr_discovery_rank").Gene.tolist()
    summary=read("outputs/ccrcc_source_replication_v1/cell_source_summary.csv.gz")
    source_dot_plot(genes,summary,"primary",group,"candidate_cell_sources")
    paired=read("outputs/ccrcc_source_replication_v1/paired_tumor_proximal_expression.csv")
    paired=paired[paired.qc.eq("extra_qc")&paired.Gene.isin(genes)]
    fig,ax=plt.subplots(figsize=(9.5,7));donors=sorted(paired.donor.unique())
    for i,donor in enumerate(donors):
        values=paired[paired.donor.eq(donor)].set_index("Gene").log2_cpm_ratio.reindex(genes)
        ax.scatter(values,np.arange(len(genes))+(i-2)*.10,s=20,c=PALETTE[i],marker=["o","s","^","D","v"][i],alpha=.8,label=donor)
    medians=paired.groupby("Gene").log2_cpm_ratio.median().reindex(genes)
    ax.plot(medians,range(len(genes)),"|",c="#444444",ms=11,label="Median")
    missing=medians.isna()
    for i in np.flatnonzero(missing):
        ax.text(.98,i,"unmeasured",transform=ax.get_yaxis_transform(),ha="right",va="center",c=GRAY,fontsize=8)
    ax.set_yticks(range(len(genes)),[f"{i+1:02}  {g}" for i,g in enumerate(genes)])
    ax.set_ylim(len(genes)-.6,-.6);ax.axvline(0,c=GRAY,ls="--",lw=.8);ax.grid(axis="x")
    ax.set(xlabel="log2((tumor CPM + 1) / (adjacent-normal proximal CPM + 1))",title="GSE178481: five paired donors")
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.035),ncol=6,fontsize=8)
    fig.subplots_adjust(left=.18,right=.97,top=.84,bottom=.14)
    fig.text(.12,.035,"Frozen PCR order; MRPL57 unmeasured (0 valid pairs). Expression contrasts do not establish dependency or a therapeutic window.",fontsize=8,color="#555555")
    save(fig,group,"paired_expression","Five donor-level values; missing gene explicit; no cell-level significance tests.")


def overview():
    sheet=Image.new("RGB",(1600,350*3),"white");draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(RECORDS):
        im=Image.open(STAGE/row["group"]/(row["stem"]+".png")).convert("RGB")
        im.thumbnail((390,300))
        x=(i%4)*400;y=(i//4)*350
        sheet.paste(im,(x+(390-im.width)//2,y+(300-im.height)//2))
        draw.text((x+5,y+305),row["stem"][:47],fill="#444444")
    sheet.save(STAGE/"figure_overview.jpg",quality=92)


def install():
    report=json.loads((STAGE/"render_manifest.json").read_text())
    # Check all source tables before any replacement.
    for relative,digest in report["source_sha256"].items():
        assert sha(ROOT/relative)==digest,relative
    changed=[];old=[]
    for row in report["figures"]:
        for ext in ("png","pdf"):
            source=STAGE/row["group"]/(row["stem"]+"."+ext)
            assert sha(source)==row[ext+"_sha256"]
            for prefix in ("outputs","results/historical"):
                dest=ROOT/prefix/row["group"]/source.name
                assert dest.resolve().is_relative_to(ROOT.resolve())
                old.append(dict(path=str(dest.relative_to(ROOT)),old_sha256=sha(dest)))
                shutil.copyfile(source,dest)
                changed.append(str(dest.relative_to(ROOT)))
    retired=[]
    for prefix in ("outputs","results/historical"):
        directory=ROOT/prefix/"final_figures_v1"
        assert directory.resolve().is_relative_to(ROOT.resolve())
        for path in sorted(directory.iterdir()):
            if path.suffix in (".png",".pdf"):
                retired.append(dict(path=str(path.relative_to(ROOT)),old_sha256=sha(path)))
                path.unlink()
    groups={row["group"] for row in report["figures"]}|{"final_figures_v1"}
    for group in sorted(groups):
        directory=ROOT/"outputs"/group;runpath=directory/"run.json"
        run=json.loads(runpath.read_text())
        previous=sha(runpath)
        run["output_sha256"]={name:sha(directory/name) for name in run["output_sha256"] if (directory/name).exists()}
        run["render_revision"]={"date":"2026-10-13","reference":"ppt/MechDec.pptx",
            "renderer":"scripts/redraw_mechdec_figures.py","renderer_sha256":sha(Path(__file__)),
            "previous_run_sha256":previous,"analysis_unchanged":True,
            "status":"retired_images_use_final_figures_v2" if group=="final_figures_v1" else "mechdec_style_redraw"}
        runpath.write_text(json.dumps(run,indent=2,ensure_ascii=False)+"\n")
        snapshot=ROOT/"results/historical"/group
        shutil.copyfile(runpath,snapshot/"run.json")
        snap_path=snapshot/"snapshot.json"
        if snap_path.exists():
            snap=json.loads(snap_path.read_text())
            snap["upstream_run_sha256"]=sha(runpath)
            snap["included_sha256"]={name:sha(snapshot/name) for name in snap["included_sha256"] if (snapshot/name).exists()}
            snap["render_revision"]=run["render_revision"]
            snap_path.write_text(json.dumps(snap,indent=2,ensure_ascii=False)+"\n")
    record=dict(status="installed",analysis_unchanged=True,replaced=changed,previous_images=old,deleted_obsolete_images=retired,
        source_sha256=report["source_sha256"],render_manifest_sha256=sha(STAGE/"render_manifest.json"))
    release=ROOT/"results/historical/mechdec_figure_redraw_20261013"
    release.mkdir(exist_ok=False)
    shutil.copyfile(STAGE/"render_manifest.json",release/"render_manifest.json")
    shutil.copyfile(STAGE/"figure_overview.jpg",release/"figure_overview.jpg")
    (release/"replacement_audit.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(f"新图已安装：{len(report['figures'])}张PNG+PDF；旧版本图删除{len(retired)}文件；分析表不变。",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--install",action="store_true")
    args=parser.parse_args()
    if args.install: install();return
    if STAGE.exists(): raise FileExistsError(f"拒绝覆盖待审图件：{STAGE}")
    STAGE.mkdir();style()
    for i,(name,fn) in enumerate([("主结果",main_figures),("PCR迁移",transfer_figures),("排序不稳定性",instability_figures),("独立单细胞复核",replication_figures)],1):
        print(f"【{i}/4】重绘{name}",flush=True);fn()
    overview()
    for relative,digest in SOURCES.items(): assert sha(ROOT/relative)==digest
    report=dict(status="staged_for_visual_review",date="2026-10-13",reference="ppt/MechDec.pptx",
        reference_sha256=sha(ROOT/"ppt/MechDec.pptx"),renderer_sha256=sha(Path(__file__)),
        figures=RECORDS,source_sha256=SOURCES,palette=PALETTE,formats=["PNG 300dpi","PDF vector text/lines"],
        changes=["white background","muted palette","fine axes","lowercase panel letters","patient observations retained",
            "matrix uses symbols","missing values explicitly marked"],
        analysis_unchanged=True,new_test_access=False)
    (STAGE/"render_manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(f"【完成】{len(RECORDS)}张待审图；原数据哈希通过；路径：{STAGE}",flush=True)


if __name__=="__main__": main()
