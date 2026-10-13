"""Validate paired audit artifacts and portable snapshots without re-fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from pcr_transfer_methods import sha256, verify_manifest, cluster_bootstrap
import run_depmap_baseline as baseline
from pcr_instability_methods import case_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots-only',action='store_true')
    parser.add_argument('--report',type=Path,help='Optional new JSON record; refuse overwrite')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    checked=0
    for name in ['pcr_instability_audit_v1','pcr_instability_synthesis_v1']:
        directory=root/'results/historical'/name
        snap=json.loads((directory/'snapshot.json').read_text())
        assert snap['upstream_run_sha256']==sha256(directory/'run.json')
        for filename,expected in snap['included_sha256'].items():
            assert sha256(directory/filename)==expected
            checked+=1
    if args.snapshots_only:
        print(json.dumps(dict(status='passed',snapshot_files=checked))); return
    source=root/'outputs/pcr_instability_audit_v1'
    record=verify_manifest(source)
    synthesis=verify_manifest(root/'outputs/pcr_instability_synthesis_v1')
    assert synthesis['upstream_run_sha256']==sha256(source/'run.json')
    for filename,expected in record['signature'].items():
        if filename=='matrices.npz': path=root/'data/processed/depmap_baseline_24q4_v1'/filename
        elif filename=='patient_run.json': path=root/'outputs/pcr_patient_extension_v1/run.json'
        elif filename=='bridge_run.json': path=root/'data/processed/tcga_kirc_expression_bridge_v1/run.json'
        elif filename.endswith('.json'): path=root/'configs'/filename
        else: path=root/'scripts'/filename
        assert sha256(path)==expected,filename
    models,genes,matrices,essential=baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    labels=models.OncotreeLineage.to_numpy(); pids=models.PatientID.to_numpy()
    summaries=pd.read_csv(source/'object_summary.csv.gz')
    historical=json.loads((root/'results/historical/selective_dependency_benchmark_v1/run.json').read_text())
    cases=['TCGA']+sorted(k for k,v in pd.Series(labels).value_counts().items() if v>=20)
    assert len(cases)==20 and set(record['cases'])==set(cases)
    max_error=0.; total_draws=0
    for number,case in enumerate(cases):
        directory=case_path(source,case)
        assert sha256(directory/'run.json')==record['cases'][case]
        run=verify_manifest(directory)
        assert run['signature']==record['signature']
        lineage='Kidney' if case=='TCGA' else case
        train=np.flatnonzero((labels!=lineage)&~np.isin(pids,pids[labels==lineage]))
        assert len(train)==run['train_models']
        assert run['evaluation_models']==(209 if case=='TCGA' else int((labels==lineage).sum()))
        assert not set(pids[train])&set(pids[labels==lineage])
        assert run['parameters']==historical['selected_parameters'][lineage]['expression_pcr_ridge']
        rng=np.random.default_rng(20261010 if case=='TCGA' else 20261011+number-1)
        samples=pd.read_csv(directory/'samples.csv')
        with np.load(root/'data/processed/tcga_kirc_expression_bridge_v1/expression_inputs.npz') as a:
            train_patients=np.flatnonzero(a['split']=='train')
        for draw,row in samples.iterrows():
            indices=cluster_bootstrap(models,train,rng)
            assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==row.sample_sha256
            if case=='TCGA': rng.choice(train_patients,size=len(train_patients),replace=True)
        total_draws+=len(samples)
        components=pd.read_csv(directory/'components.csv.gz')
        for draw,g in components.groupby('draw'):
            err=abs(g.cross_product.sum()-g.total_mse.iloc[0]); max_error=max(max_error,err)
            assert err<1e-10 and g.reconstruction_error.max()<1e-10
        draws=pd.read_csv(directory/'draw_metrics.csv.gz')
        assert set(draws.variant)==set(summaries.variant)
        assert draws.draw.nunique()==run['draws']
        assert len(draws)==run['draws']*6*3*run['evaluation_models']
        assert not draws.duplicated(['draw','variant','k','object_id']).any()
        assert draws.topk_overlap.between(0,1).all()
        expected=draws.groupby(['variant','object_id','k']).topk_overlap.mean().sort_index()
        actual=summaries[summaries.case==case].set_index(['variant','object_id','k']).topk_overlap.sort_index()
        np.testing.assert_allclose(expected,actual,atol=1e-12)
    old=pd.read_csv(root/'outputs/pcr_patient_extension_v1/bootstrap_patient_overlap.csv.gz')
    old=old[old.kind=='source_patient_bootstrap'].groupby('patient_id').top10_overlap.mean().sort_index()
    new=summaries[(summaries.case=='TCGA')&(summaries.variant=='full_refit_local_reference')&(summaries.k==10)].set_index('patient_id').topk_overlap.sort_index()
    np.testing.assert_allclose(old,new,atol=1e-12)
    assert total_draws==430 and not record['locked_test_expression_access'] and not record['candidate_reranking']
    functional=summaries[(summaries.case!='TCGA')&(summaries.k==10)&(summaries.variant=='full_refit_common_reference')]
    assert len(functional)==805 and functional.patient_id.nunique()==780
    result=dict(status='passed',snapshot_files=checked,cases=20,paired_source_draws=total_draws,maximum_cross_product_identity_error=max_error,original_bootstrap_replay=True,patient_purge=True,locked_test_expression_access=False,candidate_reranking=False)
    if args.report:
        if args.report.exists(): raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
