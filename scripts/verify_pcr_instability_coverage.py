"""Explain historical multi-method versus PCR-only coverage without selecting a new universe."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import run_depmap_baseline as baseline
import run_selective_dependency_benchmark as benchmark
from pcr_transfer_methods import fit_pcr, predict_pcr
from pcr_instability_methods import fast_functional_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Optional new JSON record; refuse overwrite')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    baseline.configure_device('cuda'); torch.set_num_threads(4)
    models,genes,matrices,essential=baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    historical=json.loads((root/'results/historical/selective_dependency_benchmark_v1/run.json').read_text())
    comparisons=pd.read_csv(root/'outputs/pcr_instability_synthesis_v1/historical_coverage_comparison.csv.gz')
    exceptions=comparisons[comparisons.ndcg_difference.abs()>1e-8]
    records=[]
    for _,row in exceptions.iterrows():
        labels=models.OncotreeLineage.to_numpy(); patients=models.PatientID.to_numpy()
        held=np.flatnonzero(models.index.to_numpy()==row.object_id)
        train=np.flatnonzero((labels!=row['case'])&~np.isin(patients,patients[labels==row['case']]))
        config=historical['selected_parameters'][row['case']]
        y=matrices['dependency'].astype(float)
        mean=baseline.observed_mean(y[train],axis=0)
        model=fit_pcr(matrices['expression'][train],y[train],*config['expression_pcr_ridge'])
        pred=predict_pcr(model,matrices['expression'][held])
        kernels=benchmark.build_kernels(models,genes,matrices,train,held)
        rank,alpha=config['expression_pcr_ridge']
        independent=benchmark.pcr_ridge_predictions(*kernels['expression'],y[train]-mean,[rank],[alpha])[(rank,alpha)]
        mask=~essential & (np.isfinite(y[train]).sum(axis=0)>=20) & np.isfinite(y[held][0])
        prediction_error=float(np.max(np.abs(pred[0,mask]-independent[0,mask])))
        assert prediction_error<1e-10
        k=config['expression_knn'][0]
        knn=benchmark.knn_predictions(*kernels['expression'],kernels['train_norm'],kernels['held_norm'],y[train]-mean,[k])[k]
        historical_mask=mask&np.isfinite(knn[0])
        assert int(historical_mask.sum())==row.gene_n
        metrics=fast_functional_metrics(y[held]-mean,pred,historical_mask[None,:])
        assert abs(metrics['ndcg_at_10'][0]-row.ndcg_at_10_historical)<1e-10
        records.append(dict(case=row['case'],ModelID=row.object_id,added_gene_n=int(mask.sum()-historical_mask.sum()),
                            cause='Historical multi-method common mask excluded genes missing all labels among expression-kNN neighbors; audit PCR-only common mask retains them',
                            maximum_independent_prediction_error=prediction_error,historical_mask_ndcg=float(metrics['ndcg_at_10'][0]),audit_mask_ndcg=row.ndcg_at_10_baseline))
    result=json.dumps(dict(status='passed',exception_n=len(records),records=records),indent=2)+'\n'
    if args.report:
        if args.report.exists(): raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(result)
    print(result)


if __name__=='__main__': main()
