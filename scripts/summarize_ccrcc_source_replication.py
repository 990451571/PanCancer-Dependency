"""Export donor-based source replication figures and portable small snapshots."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcr_transfer_methods import verify_manifest,finish_run,sha256


def main():
    root=Path(__file__).resolve().parents[1]; source=root/'outputs/ccrcc_source_replication_v1'
    record=verify_manifest(source); out=root/'outputs/ccrcc_source_replication_synthesis_v1'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    candidates=pd.read_csv(source/'candidate_sets.csv'); genes=candidates[candidates.pcr_discovery_rank.notna()].sort_values('pcr_discovery_rank').Gene.tolist()
    sources=pd.read_csv(source/'candidate_donor_cell_sources.csv.gz')
    summary=pd.read_csv(source/'cell_source_summary.csv.gz')
    columns=[('Tumor','author_tumor'),('Tumor','myeloid'),('Tumor','lymphoid'),('Tumor','endothelial'),('Tumor','stromal'),('Normal','normal_proximal_tubule'),('Normal','normal_other_epithelium')]
    values=[]; labels=[]
    for condition,group in columns:
        s=summary[(summary.scope=='primary')&(summary.qc=='extra_qc')&(summary.condition==condition)&(summary.group==group)].set_index('Gene')
        values.append(s.median_cpm.reindex(genes).to_numpy())
        n=s.donor_n.max() if len(s) else 0
        labels.append(f'{condition}: {group}\n{int(n)} donors')
    fig,ax=plt.subplots(figsize=(10,7)); cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#d9d9d9')
    plotted=ax.imshow(np.log10(np.array(values).T+1),aspect='auto',cmap=cmap)
    ax.set_yticks(np.arange(len(genes)),genes); ax.set_xticks(np.arange(len(columns)),labels,rotation=45,ha='right')
    ax.set_title('GSE178481 primary cohort; frozen PCR candidate order\nEqual-donor median CPM; gray = missing group or gene')
    fig.colorbar(plotted,ax=ax,label='log10(median CPM + 1)'); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'candidate_cell_sources.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    paired=pd.read_csv(source/'paired_tumor_proximal_expression.csv')
    paired=paired[(paired.qc=='extra_qc')&paired.Gene.isin(genes)]
    fig,ax=plt.subplots(figsize=(10,6)); donors=sorted(paired.donor.unique()); colors=plt.get_cmap('tab10')
    for i,donor in enumerate(donors):
        s=paired[paired.donor==donor].set_index('Gene').log2_cpm_ratio.reindex(genes)
        ax.scatter(s,np.arange(len(genes))+(i-(len(donors)-1)/2)*.09,s=16,color=colors(i),label=donor,alpha=.8)
    medians=paired.groupby('Gene').log2_cpm_ratio.median().reindex(genes)
    ax.plot(medians,np.arange(len(genes)),'k|',ms=11,label='Median')
    ax.set_yticks(np.arange(len(genes)),genes); ax.set_ylim(len(genes)-.5,-.5); ax.axvline(0,color='gray',lw=.8)
    ax.set_xlabel('log2((tumor-cell CPM + 1) / (adjacent-normal proximal CPM + 1))')
    ax.set_title('Paired donor expression; descriptive, not dependency or therapeutic window'); ax.legend(loc='best',fontsize=8); fig.tight_layout()
    for ext in ['png','pdf']: fig.savefig(out/f'paired_expression.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    mix=pd.read_csv(source/'mixture_sensitivity.csv')
    mix.groupby(['scope','scenario']).top10_overlap.agg(['count','mean','min','max']).to_csv(out/'mixture_summary.csv')
    broad=sources[(sources.scope=='primary')&(sources.qc=='extra_qc')&~sources.group.isin(['all_cells','nonmalignant'])]
    exposure=[]
    for gene in candidates.Gene:
        g=broad[broad.Gene==gene]
        tumor=g[(g.condition=='Tumor')&(g.group=='author_tumor')]
        normal=g[(g.condition=='Normal')&(g.group=='normal_proximal_tubule')]
        other=g[~g.group.eq('author_tumor')]
        exposure.append(dict(Gene=gene,tumor_donor_n=tumor.donor.nunique(),median_tumor_detection=tumor.detection_fraction.median(),normal_proximal_donor_n=normal.donor.nunique(),median_normal_proximal_detection=normal.detection_fraction.median(),other_cell_groups_detected_n=other[other.detection_fraction>0].group.nunique()))
    candidates.merge(pd.DataFrame(exposure),on='Gene',validate='one_to_one').to_csv(out/'candidate_exposure_audit.csv',index=False)
    qc=sources[~sources.group.isin(['all_cells','nonmalignant'])].groupby(['scope','qc','condition','group']).agg(donor_n=('donor','nunique'),minimum_cells=('cell_n','min')).reset_index()
    qc.to_csv(out/'qc_group_coverage.csv',index=False)
    finish_run(out,dict(upstream_run_sha256=sha256(source/'run.json'),script_sha256=sha256(Path(__file__)),limitations=record.get('limitations',['Expression is not functional dependency; adjacent normal is not healthy volunteer tissue'])))
    for directory in [source,out]:
        target=root/'results/historical'/directory.name
        if target.exists(): raise FileExistsError(target)
        target.mkdir()
        selected=[p for p in directory.iterdir() if p.is_file() and p.name not in ['cell_metadata_qc.csv.gz','pseudobulk_counts.npz']]
        for p in selected: shutil.copy2(p,target/p.name)
        snap={'type':'compact_result_snapshot','upstream_output_dir':f'outputs/{directory.name}','upstream_run_sha256':sha256(directory/'run.json'),'included_sha256':{p.name:sha256(target/p.name) for p in selected}}
        (target/'snapshot.json').write_text(json.dumps(snap,indent=2)+'\n')
    print('Replication figures and snapshots complete',flush=True)


if __name__=='__main__': main()
