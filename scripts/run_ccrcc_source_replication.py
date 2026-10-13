"""Stream author-selected GSE178481 counts; CUDA pseudobulk and frozen-PCR inference."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from pcr_transfer_methods import sha256,verify_manifest,finish_run,load_model,predict_pcr,ranks_and_top,overlap,tensor


GROUPS={'Tumor':'author_tumor','Macro':'myeloid','Mono':'myeloid','mDC':'myeloid',
        'CTL':'lymphoid','CD4':'lymphoid','NK':'lymphoid','NKT':'lymphoid','Treg':'lymphoid',
        'Mature_B':'lymphoid','memBcell':'lymphoid','Proliferation Tcell':'lymphoid',
        'Pericytes':'stromal','Fibroblast':'stromal','Endothelial':'endothelial','Mast_cells':'mast',
        'Proximal T':'normal_proximal_tubule','Collecting_duct':'normal_other_epithelium',
        'Connecting T':'normal_other_epithelium','Loop of Henle':'normal_other_epithelium','Erythroid':'erythroid'}
MARKERS=['CA9','EPCAM','KRT8','PTPRC','CD3D','LST1','PECAM1','VWF','ACTA2','PDGFRB','SLC34A1','LRP2','UMOD','AQP2']


def read_sample(path,annotation):
    columns=pd.read_csv(path,nrows=0).columns[1:]
    assert not columns.has_duplicates and set(columns)==set(annotation.index)
    a=annotation.loc[columns].copy()
    assert set(a['ano.l1']).issubset(GROUPS)
    a['group']=a['ano.l1'].map(GROUPS)
    a['extra_qc']=(a['gene.num']>=200)&(a['MT.ratio']<=.20)
    descriptors=[]
    for mode in ['author_only','extra_qc']:
        keep=np.ones(len(a),bool) if mode=='author_only' else a.extra_qc.to_numpy()
        masks={'all_cells':keep,'nonmalignant':keep&~a['ano.l1'].eq('Tumor').to_numpy()}
        masks.update({g:keep&a.group.eq(g).to_numpy() for g in sorted(a.group.unique())})
        descriptors += [(mode,g,m) for g,m in masks.items() if m.any()]
    membership=tensor(np.column_stack([m for _,_,m in descriptors]))
    umi=torch.zeros(len(a),device='cuda',dtype=torch.float64); detected=umi.clone(); mt=umi.clone()
    genes=[]; summed=[]; detections=[]
    for chunk in pd.read_csv(path,index_col=0,chunksize=128):
        data=chunk.to_numpy()
        if not np.isfinite(data).all() or np.any(data<0) or np.any(data!=np.floor(data)):
            raise ValueError('Counts are not finite nonnegative integers')
        x=tensor(data); nz=(x>0).double()
        umi+=x.sum(axis=0); detected+=nz.sum(axis=0)
        mt_mask=np.array([str(g).startswith('MT-') for g in chunk.index])
        if mt_mask.any(): mt+=x[mt_mask].sum(axis=0)
        summed.append((x@membership).T.cpu().numpy()); detections.append((nz@membership).T.cpu().numpy())
        genes.extend(chunk.index.astype(str))
    genes=np.asarray(genes)
    if len(np.unique(genes))!=len(genes): raise ValueError('Duplicate gene symbols require explicit collapse')
    umi=umi.cpu().numpy(); detected=detected.cpu().numpy(); mt=mt.cpu().numpy()
    fraction=np.divide(mt,umi,out=np.zeros_like(mt),where=umi>0)
    errors={'umi_error':float(np.max(np.abs(umi-a.TotalUMI.to_numpy()))),
            'detected_error':float(np.max(np.abs(detected-a['gene.num'].to_numpy()))),
            'mt_fraction_error':float(np.max(np.abs(fraction-a['MT.ratio'].to_numpy())))}
    assert errors['umi_error']==0 and errors['detected_error']==0 and errors['mt_fraction_error']<1e-8,errors
    assert np.array_equal(a.extra_qc.to_numpy(),(detected>=200)&(fraction<=.20))
    a['raw_umi']=umi; a['raw_detected']=detected; a['raw_mt_fraction']=fraction
    meta=pd.DataFrame([dict(qc=mode,group=g,cell_n=int(m.sum())) for mode,g,m in descriptors])
    return genes,np.concatenate(summed,axis=1),np.concatenate(detections,axis=1),meta,a,errors


def mixture(model,model_genes,universe,genes,cpm,meta):
    index={g:i for i,g in enumerate(genes)}; ix=np.array([index.get(g,-1) for g in model_genes]); covered=ix>=0
    scenarios={}; donor_ids=[]
    for donor,g in meta.groupby('donor',sort=True):
        lookup={r.group:i for i,r in g.iterrows()}
        if not {'author_tumor','nonmalignant','all_cells'}.issubset(lookup): continue
        if min(meta.loc[lookup[s],'cell_n'] for s in ['author_tumor','nonmalignant'])<20: continue
        donor_ids.append(donor); t=cpm[lookup['author_tumor']]; n=cpm[lookup['nonmalignant']]
        profiles={'tumor_only':t,'observed_all_cells':cpm[lookup['all_cells']]}
        profiles.update({f'nonmalignant_rna_{int(f*100)}pct':(1-f)*t+f*n for f in [.25,.5,.75]})
        for name,profile in profiles.items():
            aligned=np.full(len(model_genes),np.nan); aligned[covered]=np.log2(profile[ix[covered]]+1)
            scenarios.setdefault(name,[]).append(aligned)
    tops={}; rankings=[]
    for name,values in scenarios.items():
        x=np.stack(values); shift=model['mean'].cpu().numpy()-np.nanmean(x,axis=0)
        mapped=np.maximum(x+shift,0); pred=predict_pcr(model,mapped)
        _,top=ranks_and_top(pred,universe&covered); tops[name]=top
        rankings += [dict(donor=d,scenario=name,rank=k+1,Gene=model_genes[col],predicted_residual=pred[i,col]) for i,d in enumerate(donor_ids) for k,col in enumerate(top[i])]
    rows=[dict(donor=d,scenario=name,top10_overlap=value) for name,top in tops.items() if name!='tumor_only' for d,value in zip(donor_ids,overlap(tops['tumor_only'],top))]
    return rows,rankings,int((universe&covered).sum()),float(covered.mean())


def main():
    root=Path(__file__).resolve().parents[1]; raw=root/'data/raw/gse178481'; out=root/'outputs/ccrcc_source_replication_v1'
    if out.exists(): raise FileExistsError(out)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4); started=time.monotonic()
    protocol=root/'configs/ccrcc_source_replication_protocol_20261010.json'
    manifest=json.loads((raw/'source_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha256(protocol)
    for name,record in manifest['files'].items(): assert sha256(raw/name)==record['sha256']
    assert sha256(raw/'samples.csv')==manifest['samples_sha256']
    pcr=root/'outputs/pcr_patient_extension_v1'; old=root/'outputs/ccrcc_singlecell_extension_v1'
    verify_manifest(pcr); verify_manifest(old)
    annotations=pd.read_csv(raw/'cell.ano.csv',index_col=0)
    assert not annotations.index.has_duplicates
    annotations['sample']=annotations.index.str.rsplit('_',n=1).str[0]
    samples=pd.read_csv(raw/'samples.csv',keep_default_na=False)
    assert set(annotations['sample'])==set(samples.sample_id)
    samples.to_csv(out/'sample_inventory.csv',index=False) if out.exists() else None
    out.mkdir(); samples.to_csv(out/'sample_inventory.csv',index=False)
    metas=[]; sums=[]; detects=[]; audits=[]; cell_rows=[]; all_genes=None
    selected=samples[samples.scope!='excluded']
    for number,(_,row) in enumerate(selected.iterrows(),1):
        a=annotations[annotations['sample']==row.sample_id]
        concordant=('Normal' in row.sample_id)==(row.condition=='Normal')
        assert concordant and row.Stage=='ccRCC' and row.treatment=='No'
        if row.condition=='Normal': assert not a['ano.l1'].eq('Tumor').any()
        genes,count,detect,meta,cells,errors=read_sample(raw/row.filename,a)
        if all_genes is None: all_genes=genes
        assert np.array_equal(all_genes,genes)
        for key in ['sample_id','donor','scope','condition']: meta[key]=row[key]; cells[key]=row[key]
        metas.append(meta); sums.append(count); detects.append(detect); cell_rows.append(cells)
        audits.append(dict(sample_id=row.sample_id,donor=row.donor,scope=row.scope,condition=row.condition,annotated_cells=len(a),retained_cells=int(cells.extra_qc.sum()),geo_author_concordant=concordant,**errors))
        print(f'Sample {number}/{len(selected)} {row.sample_id}: {cells.extra_qc.sum()}/{len(a)} retained, {time.monotonic()-started:.0f}s',flush=True)
    metadata=pd.concat(metas,ignore_index=True); counts=np.concatenate(sums); detection=np.concatenate(detects)
    donor_meta=[]; donor_counts=[]; donor_detection=[]
    keys=['scope','donor','condition','qc','group']
    for values,g in metadata.groupby(keys,sort=True):
        ix=g.index.to_numpy(); donor_counts.append(counts[ix].sum(axis=0)); donor_detection.append(detection[ix].sum(axis=0))
        donor_meta.append(dict(zip(keys,values),cell_n=int(g.cell_n.sum()),sample_ids='|'.join(g.sample_id)))
    meta=pd.DataFrame(donor_meta); counts=np.stack(donor_counts); detect=np.stack(donor_detection)
    c=tensor(counts); cpm=(c/c.sum(axis=1,keepdim=True)*1e6).cpu().numpy(); detection=detect/meta.cell_n.to_numpy()[:,None]
    meta.to_csv(out/'pseudobulk_groups.csv',index=False)
    np.savez_compressed(out/'pseudobulk_counts.npz',genes=all_genes,counts=counts,detection_count=detect,cpm=cpm)
    pd.DataFrame(audits).to_csv(out/'sample_source_audit.csv',index=False)
    pd.concat(cell_rows).to_csv(out/'cell_metadata_qc.csv.gz')
    discovery=pd.read_csv(pcr/'pcr_discovery_top20.csv'); historical=pd.read_csv(root/'results/historical/final_evidence_synthesis_v2/candidate_evidence_matrix.csv')
    features=pd.read_csv(pcr/'pcr_candidate_input_features.csv')
    candidates=list(dict.fromkeys([*discovery.Gene,*historical.Gene])); predictors=list(dict.fromkeys(features.feature_gene))
    flags=pd.DataFrame({'Gene':candidates}); flags['pcr_discovery_rank']=flags.Gene.map(discovery.set_index('Gene').pcr_discovery_rank); flags['historical_frozen_candidate']=flags.Gene.isin(historical.Gene)
    flags.to_csv(out/'candidate_sets.csv',index=False)
    index={g:i for i,g in enumerate(all_genes)}; rows=[]
    for i,r in meta.iterrows():
        if r.cell_n<20: continue
        for gene in list(dict.fromkeys(candidates+predictors+MARKERS)):
            j=index.get(gene)
            rows.append(dict(r.to_dict(),Gene=gene,gene_present=j is not None,pseudobulk_cpm=cpm[i,j] if j is not None else np.nan,detection_fraction=detection[i,j] if j is not None else np.nan))
    evidence=pd.DataFrame(rows)
    evidence[evidence.Gene.isin(candidates)].to_csv(out/'candidate_donor_cell_sources.csv.gz',index=False)
    evidence[evidence.Gene.isin(predictors)].to_csv(out/'predictor_feature_donor_cell_sources.csv.gz',index=False)
    evidence[evidence.Gene.isin(MARKERS)].to_csv(out/'annotation_marker_audit.csv.gz',index=False)
    broad=evidence[~evidence.group.isin(['all_cells','nonmalignant'])]
    summaries=broad.groupby(['scope','condition','qc','group','Gene']).agg(donor_n=('donor','nunique'),median_cpm=('pseudobulk_cpm','median'),mean_detection_fraction=('detection_fraction','mean'),gene_present=('gene_present','all')).reset_index()
    summaries.to_csv(out/'cell_source_summary.csv.gz',index=False)
    pairs=[]
    for qc in ['author_only','extra_qc']:
        tumor=evidence[(evidence.scope=='primary')&(evidence.condition=='Tumor')&(evidence.group=='author_tumor')&(evidence.qc==qc)&evidence.Gene.isin(candidates)]
        normal=evidence[(evidence.scope=='primary')&(evidence.condition=='Normal')&(evidence.group=='normal_proximal_tubule')&(evidence.qc==qc)&evidence.Gene.isin(candidates)]
        joined=tumor.merge(normal,on=['donor','Gene'],suffixes=('_tumor','_normal'),validate='one_to_one')
        for _,r in joined.iterrows(): pairs.append(dict(donor=r.donor,Gene=r.Gene,qc=qc,log2_cpm_ratio=np.log2((r.pseudobulk_cpm_tumor+1)/(r.pseudobulk_cpm_normal+1)),tumor_cpm=r.pseudobulk_cpm_tumor,normal_cpm=r.pseudobulk_cpm_normal))
    paired=pd.DataFrame(pairs); paired.to_csv(out/'paired_tumor_proximal_expression.csv',index=False)
    paired.groupby(['qc','Gene']).agg(paired_donor_n=('donor','nunique'),median_log2_cpm_ratio=('log2_cpm_ratio','median'),positive_donor_n=('log2_cpm_ratio',lambda s:int((s>0).sum()))).to_csv(out/'paired_expression_summary.csv')
    old_evidence=pd.read_csv(old/'candidate_donor_cell_sources.csv.gz')
    previous=old_evidence[(old_evidence.author_cohort=='author_ccrcc')&(old_evidence.group=='author_tumor')].groupby('Gene').pseudobulk_cpm.median()
    current=evidence[(evidence.scope=='primary')&(evidence.condition=='Tumor')&(evidence.group=='author_tumor')&(evidence.qc=='extra_qc')].groupby('Gene').pseudobulk_cpm.median()
    comparison=pd.concat([previous.rename('gse159115_median_tumor_cpm'),current.rename('gse178481_median_tumor_cpm')],axis=1).reindex(candidates).reset_index()
    comparison.to_csv(out/'cross_study_candidate_expression.csv',index=False)
    model,model_genes=load_model(pcr/'pcr_model.npz')
    with np.load(pcr/'ranking_universe.npz') as a: universe=a['universe']
    mixture_rows=[]; rankings=[]; coverage={}
    for scope in ['primary','metastatic_status']:
        selected_meta=meta[(meta.scope==scope)&(meta.condition=='Tumor')&(meta.qc=='extra_qc')]
        r,t,n,cover=mixture(model,model_genes,universe,all_genes,cpm,selected_meta)
        mixture_rows.extend([dict(x,scope=scope) for x in r]); rankings.extend([dict(x,scope=scope) for x in t]); coverage[scope]=dict(ranking_gene_n=n,feature_coverage=cover)
    pd.DataFrame(mixture_rows).to_csv(out/'mixture_sensitivity.csv',index=False)
    pd.DataFrame(rankings).to_csv(out/'mixture_patient_top10.csv',index=False)
    primary=pd.DataFrame(audits).query("scope=='primary'")
    finite=comparison.dropna()
    finish_run(out,dict(status='complete_descriptive_source_replication',protocol_sha256=sha256(protocol),input_sha256={'source_manifest':sha256(raw/'source_manifest.json'),'pcr_run':sha256(pcr/'run.json'),'old_singlecell_run':sha256(old/'run.json')},script_sha256={Path(__file__).name:sha256(Path(__file__)),'pcr_transfer_methods.py':sha256(root/'scripts/pcr_transfer_methods.py')},device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,elapsed_seconds=time.monotonic()-started,primary_sample_n=len(primary),primary_donor_n=primary.donor.nunique(),primary_annotated_cell_n=int(primary.annotated_cells.sum()),primary_retained_cell_n=int(primary.retained_cells.sum()),source_concordant=True,paired_donor_n=int(paired[paired.qc=='extra_qc'].donor.nunique()),cross_study_candidate_cpm_spearman=float(spearmanr(finite.iloc[:,1],finite.iloc[:,2]).statistic),mixture_coverage=coverage,locked_test_access=False,model_refitting=False,candidate_reranking=False))
    print('Replication complete',flush=True)


if __name__=='__main__': main()
