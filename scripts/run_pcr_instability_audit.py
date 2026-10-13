"""Run preregistered paired PCR instability diagnostics on CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import run_depmap_baseline as baseline
from pcr_transfer_methods import fit_pcr, predict_pcr, load_model, cluster_bootstrap, finish_run, sha256, verify_manifest
from pcr_instability_methods import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.train:
        parser.error('--train required')
    root = Path(__file__).resolve().parents[1]
    out = root/'outputs/pcr_instability_audit_v1'
    protocol = root/'configs/pcr_instability_audit_protocol_20261010.json'
    sources = [Path(__file__), root/'scripts/pcr_instability_methods.py', protocol]
    signature = {p.name: sha256(p) for p in sources}
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    out.mkdir(exist_ok=True)
    baseline.configure_device('cuda')
    torch.set_num_threads(4)
    models, genes, matrices, essential = baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    historical = json.loads((root/'results/historical/selective_dependency_benchmark_v1/run.json').read_text())
    signature['matrices.npz'] = sha256(root/'data/processed/depmap_baseline_24q4_v1/matrices.npz')
    if signature['matrices.npz'] != historical['input_sha256']['matrices.npz']:
        raise ValueError('Historical input changed')
    prior = root/'outputs/pcr_patient_extension_v1'
    verify_manifest(prior)
    bridge_dir = root/'data/processed/tcga_kirc_expression_bridge_v1'
    verify_manifest(bridge_dir)
    signature['patient_run.json'] = sha256(prior/'run.json')
    signature['bridge_run.json'] = sha256(bridge_dir/'run.json')
    with np.load(bridge_dir/'expression_inputs.npz') as a:
        bridge = {k:a[k] for k in a.files}
    if set(bridge['split']) != {'train','validation'} or not np.array_equal(bridge['genes'],genes):
        raise ValueError('Bridge isolation/order mismatch')
    labels = models.OncotreeLineage.to_numpy()
    patients = models.PatientID.to_numpy()
    cases = ['TCGA'] + sorted(k for k,v in pd.Series(labels).value_counts().items() if v>=20)
    if len(cases)!=20:
        raise ValueError('Expected 19 functional cases')
    started = time.monotonic()
    for case_no, case in enumerate(cases):
        directory = case_path(out,case)
        if (directory/'run.json').exists():
            record = verify_manifest(directory)
            if record['signature'] != signature:
                raise ValueError('Resume signature mismatch')
            continue
        if directory.exists():
            raise FileExistsError('Incomplete case must be preserved: '+str(directory))
        directory.mkdir()
        lineage = 'Kidney' if case=='TCGA' else case
        held = np.flatnonzero(labels==lineage)
        train = np.flatnonzero((labels!=lineage)&~np.isin(patients,patients[held]))
        lookup = np.full(len(models),-1); lookup[train]=np.arange(len(train))
        config = historical['selected_parameters'][lineage]['expression_pcr_ridge']
        y = matrices['dependency'] if case=='TCGA' else matrices['dependency'].astype(float)
        if case=='TCGA':
            base, saved_genes = load_model(prior/'pcr_model.npz')
            with np.load(prior/'pcr_model.npz') as a:
                if not np.array_equal(a['train_ids'],models.index[train].to_numpy()):
                    raise ValueError('Patient training order mismatch')
            hx=bridge['nonkidney_mean_shift_expression']; ids=bridge['patient_ids']; pids=ids; splits=bridge['split']
            with np.load(prior/'ranking_universe.npz') as a: universe=a['universe']
            draws=50; rng=np.random.default_rng(20261010)
        else:
            base=fit_pcr(matrices['expression'][train],y[train],*config)
            hx=matrices['expression'][held]; ids=models.index[held].to_numpy(); pids=patients[held]; splits=np.repeat('heldout',len(held))
            universe=~essential & (np.isfinite(y[train]).sum(axis=0)>=20)
            draws=20; rng=np.random.default_rng(20261011+case_no-1)
        bp=predict_pcr(base,hx)
        masks=np.broadcast_to(universe,bp.shape).copy()
        truth=None
        if case!='TCGA':
            truth=y[held]-base['gene_mean'].cpu().numpy()
            masks &= np.isfinite(truth)
        _, bt=masked_ranks(bp,masks)
        boundaries=boundary_diagnostics(bp,masks,bt)
        base_metrics=fast_functional_metrics(truth,bp,masks,bt) if truth is not None else {}
        base_rows=[]
        for k in (10,20,50):
            frame=pd.DataFrame(dict(case=case,object_id=ids,patient_id=pids,split=splits,k=k,ranking_gene_n=masks.sum(axis=1),**boundaries[k],**base_metrics))
            base_rows.append(frame)
        pd.concat(base_rows).to_csv(directory/'baseline.csv.gz',index=False)
        train_z=standardized(base,matrices['expression'][train]); held_z=standardized(base,hx)
        rows=[]; component_rows=[]; geometry_rows=[]; samples=[]
        inclusion=np.zeros((len(ids),int(universe.sum())),dtype=np.uint16) if case=='TCGA' else None
        for draw in range(draws):
            sampled=cluster_bootstrap(models,train,rng)
            boot=fit_pcr(matrices['expression'][sampled],y[sampled],*config)
            preds,spaces,shift=paired_predictions(base,boot,bp,train_z,held_z,lookup[sampled],matrices['expression'][sampled],hx,y[sampled])
            if case=='TCGA':
                rng.choice(np.flatnonzero(splits=='train'),size=int((splits=='train').sum()),replace=True)
            samples.append(dict(draw=draw,sampled_model_n=len(sampled),unique_patient_n=len(np.unique(patients[sampled])),sample_sha256=hashlib.sha256(sampled.astype('<i8').tobytes()).hexdigest()))
            for variant,pred in preds.items():
                rank,top=masked_ranks(pred,masks)
                if inclusion is not None and variant==VARIANTS[0]: inclusion+=(rank[:,universe]<=10)
                metrics=fast_functional_metrics(truth,pred,masks,top) if truth is not None else {}
                rmse=np.sqrt(np.sum(np.where(masks,(pred-bp)**2,0),axis=1)/masks.sum(axis=1))
                for k in (10,20,50):
                    rows.append(pd.DataFrame(dict(case=case,draw=draw,variant=variant,object_id=ids,patient_id=pids,split=splits,k=k,topk_overlap=overlap_at_k(bt,top,k),prediction_rmse=rmse,**metrics)))
            cross,error,total=prediction_components(preds,bp,masks)
            for i,a in enumerate(COMPONENTS):
                for j,b in enumerate(COMPONENTS): component_rows.append(dict(draw=draw,left=a,right=b,cross_product=cross[i,j],total_mse=total,reconstruction_error=error))
            for name,values in spaces.items(): geometry_rows.append(dict(draw=draw,contrast=name,**values))
            del boot,preds
            print(f'{case} draw {draw+1}/{draws}, total {time.monotonic()-started:.0f}s',flush=True)
        if inclusion is not None:
            with np.load(prior/'source_patient_bootstrap_inclusion.npz') as a:
                if not np.array_equal(inclusion,a['selection_count']): raise ValueError('Original bootstrap replay failed')
        pd.concat(rows).to_csv(directory/'draw_metrics.csv.gz',index=False)
        pd.DataFrame(component_rows).to_csv(directory/'components.csv.gz',index=False)
        pd.DataFrame(geometry_rows).to_csv(directory/'geometry.csv.gz',index=False)
        pd.DataFrame(samples).to_csv(directory/'samples.csv',index=False)
        finish_run(directory,dict(case=case,signature=signature,draws=draws,train_models=len(train),train_patients=len(np.unique(patients[train])),evaluation_models=len(ids),universe_gene_n=int(universe.sum()),parameters=config,original_bootstrap_replay_verified=case=='TCGA'))
    frames=[pd.read_csv(case_path(out,c)/'draw_metrics.csv.gz') for c in cases]
    draws_frame=pd.concat(frames,ignore_index=True)
    keys=['case','variant','object_id','patient_id','split','k']
    numeric=['topk_overlap','prediction_rmse','ndcg_at_10','true_top10_overlap','selective_precision_at_10']
    summary=draws_frame.groupby(keys,dropna=False)[numeric].mean().reset_index()
    bases=pd.concat([pd.read_csv(case_path(out,c)/'baseline.csv.gz') for c in cases],ignore_index=True)
    summary=summary.merge(bases,on=['case','object_id','patient_id','split','k'],suffixes=('','_baseline'),validate='many_to_one')
    summary.to_csv(out/'object_summary.csv.gz',index=False)
    summary.groupby(['case','variant','split','k'])[numeric].mean().to_csv(out/'cohort_summary.csv')
    assoc=[]; contrasts=[]
    for (variant,k),group in summary[summary.case!='TCGA'].groupby(['variant','k']):
        patient=group.groupby(['case','patient_id'])[['topk_overlap','ndcg_at_10_baseline','boundary_gap']].mean().reset_index()
        within=[]; differences=[]
        for lineage,g in patient.groupby('case'):
            corr=spearmanr(g.topk_overlap,g.ndcg_at_10_baseline).statistic
            within.append(corr)
            ranks=g.topk_overlap.rank(method='average',pct=True)
            low=g[ranks<=1/3]; high=g[ranks>2/3]
            diff=high.ndcg_at_10_baseline.mean()-low.ndcg_at_10_baseline.mean()
            differences.append(diff)
            contrasts.append(dict(variant=variant,k=k,case=lineage,high_n=len(high),low_n=len(low),high_minus_low_ndcg=diff,within_lineage_spearman=corr))
        valid=np.asarray(differences); valid=valid[np.isfinite(valid)]
        generator=torch.Generator(device='cuda').manual_seed(20261012)
        values=tensor(valid); indices=torch.randint(len(valid),(2000,len(valid)),device='cuda',generator=generator)
        interval=torch.quantile(values[indices].mean(axis=1),tensor([.025,.975])).cpu().numpy()
        assoc.append(dict(variant=variant,k=k,patient_n=len(patient),overall_spearman=spearmanr(patient.topk_overlap,patient.ndcg_at_10_baseline).statistic,mean_within_lineage_spearman=np.nanmean(within),lineage_equal_high_minus_low_ndcg=np.mean(valid),ci_low=interval[0],ci_high=interval[1],valid_lineages=len(valid),gap_stability_spearman=spearmanr(patient.boundary_gap,patient.topk_overlap).statistic))
    pd.DataFrame(assoc).to_csv(out/'functional_associations.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'lineage_contrasts.csv',index=False)
    finish_run(out,dict(signature=signature,cases={c:sha256(case_path(out,c)/'run.json') for c in cases},functional_seed_rule='20261011 + zero-based sorted lineage index',elapsed_seconds=time.monotonic()-started,locked_test_expression_access=False,candidate_reranking=False))


if __name__=='__main__': main()
