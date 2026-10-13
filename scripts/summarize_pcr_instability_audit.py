"""Make audited instability tables, export figures, and portable result snapshots."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pcr_transfer_methods import sha256, finish_run, verify_manifest
from pcr_instability_methods import case_path


LABELS=['Full / local reference','Full / common reference','Reference shift only',
        'Regression / fixed geometry','PCA + regression / fixed scaler','Scaler + regression / fixed PCA']
VARIANTS=['full_refit_local_reference','full_refit_common_reference','reference_shift_only',
          'fixed_geometry_beta_refit','fixed_scaler_pca_beta_refit','refit_scaler_fixed_pca_beta_refit']


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'outputs/pcr_instability_audit_v1'
    run=verify_manifest(source)
    for case,expected in run['cases'].items():
        assert sha256(case_path(source,case)/'run.json')==expected
        verify_manifest(case_path(source,case))
    out=root/'outputs/pcr_instability_synthesis_v1'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    frame=pd.read_csv(source/'object_summary.csv.gz')
    patient=frame[frame.case=='TCGA']
    associations=pd.read_csv(source/'functional_associations.csv')
    # Patient-equal, then lineage-equal functional aggregation; model counts retained separately.
    function=frame[frame.case!='TCGA'].groupby(['case','variant','k','patient_id'])[['topk_overlap','ndcg_at_10','ndcg_at_10_baseline','true_top10_overlap','selective_precision_at_10']].mean().reset_index()
    lineage=function.groupby(['case','variant','k']).mean(numeric_only=True).reset_index()
    lineage.to_csv(out/'functional_lineage_summary.csv',index=False)
    lineage.groupby(['variant','k']).mean(numeric_only=True).to_csv(out/'functional_lineage_equal_summary.csv')
    component=[]; geometry=[]
    for case in run['cases']:
        c=pd.read_csv(case_path(source,case)/'components.csv.gz'); c['case']=case; component.append(c)
        g=pd.read_csv(case_path(source,case)/'geometry.csv.gz'); g['case']=case; geometry.append(g)
    components=pd.concat(component)
    components.groupby(['case','left','right'])[['cross_product','total_mse','reconstruction_error']].mean().to_csv(out/'component_cross_products.csv')
    pd.concat(geometry).groupby(['case','contrast']).mean(numeric_only=True).to_csv(out/'geometry_summary.csv')
    historical=pd.read_csv(root/'results/historical/selective_dependency_benchmark_v1/per_model_metrics.csv.gz')
    historical=historical[historical.method=='expression_pcr_ridge']
    baseline=frame[(frame.case!='TCGA')&(frame.variant==VARIANTS[0])&(frame.k==10)]
    coverage=baseline.merge(historical,left_on=['case','object_id'],right_on=['heldout_lineage','ModelID'],validate='one_to_one',suffixes=('_audit','_historical'))
    coverage['gene_n_difference']=coverage.ranking_gene_n-coverage.gene_n
    coverage['ndcg_difference']=coverage.ndcg_at_10_baseline-coverage.ndcg_at_10_historical
    coverage[['case','object_id','ranking_gene_n','gene_n','gene_n_difference','ndcg_at_10_baseline','ndcg_at_10_historical','ndcg_difference']].to_csv(out/'historical_coverage_comparison.csv.gz',index=False)
    # Deduplicate six variants before descriptive boundary summaries.
    original=patient[patient.variant==VARIANTS[0]]
    original.groupby(['split','k'])[['boundary_gap','near_boundary_gene_n']].agg(['mean','median','min','max']).to_csv(out/'patient_boundary_summary.csv')
    gap_association=[]
    for (split,k,variant),g in patient.groupby(['split','k','variant']):
        gap_association.append(dict(split=split,k=k,variant=variant,spearman=spearmanr(g.boundary_gap,g.topk_overlap).statistic))
    pd.DataFrame(gap_association).to_csv(out/'patient_gap_stability.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    colors=['#0072B2','#E69F00','#009E73']
    for ax,split in zip(axes,['train','validation']):
        for color,k in zip(colors,[10,20,50]):
            means=patient[(patient.split==split)&(patient.k==k)].groupby('variant').topk_overlap.mean().reindex(VARIANTS)
            ax.plot(means.to_numpy(),np.arange(6),'o-',color=color,label=f'Top {k}')
        ax.set_yticks(np.arange(6),LABELS); ax.set_xlim(0,1); ax.set_xlabel('Mean bootstrap overlap with original ranking'); ax.set_title(f'TCGA {split}'); ax.set_ylim(5.5,-.5); ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'paired_controls.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    a=associations[associations.k==10].set_index('variant').reindex(VARIANTS)
    axes[0].barh(np.arange(6),a.mean_within_lineage_spearman,color='#0072B2'); axes[0].set_yticks(np.arange(6),LABELS); axes[0].invert_yaxis(); axes[0].axvline(0,color='gray',lw=.8); axes[0].set_xlabel('Mean within-lineage Spearman'); axes[0].set_title('Top10 stability vs original functional NDCG')
    axes[1].errorbar(a.lineage_equal_high_minus_low_ndcg,np.arange(6),xerr=np.vstack([a.lineage_equal_high_minus_low_ndcg-a.ci_low,a.ci_high-a.lineage_equal_high_minus_low_ndcg]),fmt='o',color='#009E73',capsize=3)
    axes[1].set_yticks(np.arange(6),LABELS); axes[1].invert_yaxis(); axes[1].axvline(0,color='gray',lw=.8); axes[1].set_xlabel('High minus low stability: NDCG@10'); axes[1].set_title('Descriptive lineage-cluster 95% intervals')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'functional_stability.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,['train','validation']):
        g=patient[(patient.split==split)&(patient.k==10)&(patient.variant==VARIANTS[1])]
        ax.scatter(g.boundary_gap,g.topk_overlap,s=14,alpha=.6,color='#0072B2')
        ax.set_xscale('symlog',linthresh=1e-5); ax.set_ylim(0,1)
        ax.set_xlabel('Original rank11 minus rank10 residual'); ax.set_ylabel('Mean common-reference Top10 overlap'); ax.set_title(f'TCGA {split}; each point one patient')
    fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'boundary_stability.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    finish_run(out,dict(upstream_run_sha256=sha256(source/'run.json'),script_sha256=sha256(Path(__file__)),note='Conditional contrasts; intervals descriptive; no candidate reranking or clinical probability calibration'))
    for directory in [source,out]:
        target=root/'results/historical'/directory.name
        if target.exists(): raise FileExistsError(target)
        target.mkdir()
        # Full small tables and figures; case-level draw files remain local.
        selected=[p for p in directory.iterdir() if p.is_file()]
        for p in selected: shutil.copy2(p,target/p.name)
        record={'type':'compact_result_snapshot','upstream_output_dir':f'outputs/{directory.name}',
                'upstream_run_sha256':sha256(directory/'run.json'),
                'included_sha256':{p.name:sha256(target/p.name) for p in selected}}
        (target/'snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Synthesis and portable snapshots complete',flush=True)


if __name__=='__main__': main()
