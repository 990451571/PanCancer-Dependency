#!/usr/bin/env python3
"""Certified nested-CV upper-grid sensitivity; no patient or external test access."""
from pathlib import Path
import argparse
import json
import shutil
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
import run_selective_dependency as selective
import run_advanced_model_benchmark as advanced
from advanced_model_methods import standardize_expression
from certified_elastic_net import certified_path


def summary(records,lineage,scope,fold):
    f=pd.DataFrame(records);rows=[]
    for alpha,g in f.groupby('alpha'):
        rows.append({'heldout_lineage':lineage,'scope':scope,'inner_fold':fold,'alpha':alpha,
          'target_n':len(g),'maximum_kkt':float(g.kkt.max()),'maximum_relative_bound':float(g.relative_bound.max()),
          'maximum_iterations':int(g.iterations.max()),'maximum_working_set_passes':int(g.working_set_passes.max()),
          'maximum_active_feature_n':int(g.max_active_feature_n.max())})
    return rows


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=root/'outputs/elastic_net_upper_grid_v1')
    args=parser.parse_args();out=args.output_dir
    if out.exists():raise FileExistsError(f'拒绝覆盖：{out}')
    baseline.configure_device('cuda');started=time.monotonic()
    protocol_path=root/'configs/elastic_net_upper_grid_protocol_20260920.json'
    protocol=json.loads(protocol_path.read_text());alphas=tuple(protocol['alphas'])
    internal=root/'results/historical/advanced_model_benchmark_v1'
    historical=json.loads((internal/'run.json').read_text())
    print('【阶段 1/4】校验固定网格、历史划分及输入',flush=True)
    input_dir=root/'data/processed/advanced_model_benchmark_v1'
    _,arrays=advanced.load_inputs(input_dir)
    if advanced.sha256(input_dir/'benchmark_inputs.npz')!=historical['input_matrix_sha256']:
        raise ValueError('输入与历史比较不一致')
    for name,digest in historical['output_sha256'].items():
        if advanced.sha256(internal/name)!=digest:raise ValueError(f'历史文件变化：{name}')
    models,_,_,_=baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    if models.index.tolist()!=arrays['model_ids'].tolist():raise ValueError('模型顺序变化')
    split_path=root/protocol['outer_splits']
    splits=advanced.exact_outer_splits(split_path,arrays['model_ids'],None)
    signature={str(p.relative_to(root)):advanced.sha256(p) for p in [protocol_path,Path(__file__),
        root/'scripts/certified_elastic_net.py',root/'scripts/advanced_model_methods.py',
        root/'scripts/run_advanced_model_benchmark.py',root/'scripts/run_depmap_baseline.py',
        root/'scripts/run_selective_dependency.py',split_path,internal/'run.json',input_dir/'benchmark_inputs.npz']}
    checkpoint=out.parent/('.'+out.name+'_checkpoint');checkpoint.mkdir(parents=True,exist_ok=True)
    state=checkpoint/'signature.json'
    if state.exists() and json.loads(state.read_text())!=signature:
        raise ValueError('断点输入或代码变化；拒绝混合运行')
    state.write_text(json.dumps(signature,indent=2)+'\n')
    labels=models.OncotreeLineage.to_numpy();patients=models.PatientID.to_numpy()
    y=arrays['dependency'].astype(float);expression=arrays['expression']
    primary=~arrays['target_common_essential']
    print(f'【固定规则】19癌系×5内层折｜α={alphas}｜911主评价基因｜不读取患者Test',flush=True)
    print('【阶段 2/4】GPU数值认证与训练内选参',flush=True)
    completed=[]
    for number,(lineage,(train,held)) in enumerate(splits.items(),1):
        path=checkpoint/f'fold_{number:02d}.json'
        if path.exists():
            result=json.loads(path.read_text());completed.append(result)
            print(f'  【断点复用】{number}/19 {lineage}',flush=True);continue
        tick=time.monotonic();certificates=[];inner={a:[] for a in alphas}
        folds=baseline.inner_folds(train,labels,patients,protocol['inner_folds'],protocol['seed']+number-1)
        for fold_number,validation in enumerate(folds,1):
            fit=np.setdiff1d(train,validation)
            mean=baseline.observed_mean(y[fit],axis=0)
            x,hx,_=standardize_expression(expression[fit],expression[validation],torch.float32)
            predictions,records=certified_path(x,y[fit]-mean,hx,alphas,protocol)
            certificates.extend(summary(records,lineage,'inner',fold_number))
            for alpha,prediction in predictions.items():
                scored,_=selective.score_models(y[validation]-mean,{'m':prediction},arrays['model_ids'][validation],
                    labels[validation],arrays['target_genes'],primary,10,-.5)
                inner[alpha].append(pd.DataFrame(scored))
            print(f'  【癌系 {number}/19】{lineage}｜内层 {fold_number}/5｜四个α全部通过证书',flush=True)
        scores={a:advanced.tune_score(frames) for a,frames in inner.items()}
        selected=sorted(alphas,key=lambda a:(-scores[a],-a))[0]
        mean=baseline.observed_mean(y[train],axis=0)
        x,hx,_=standardize_expression(expression[train],expression[held],torch.float32)
        predictions,records=certified_path(x,y[train]-mean,hx,tuple(sorted({.1,selected})),protocol)
        certificates.extend(summary(records,lineage,'outer',0))
        scored,_=selective.score_models(y[held]-mean,{'certified_fixed_alpha':predictions[.1],
            'certified_expanded_grid':predictions[selected]},arrays['model_ids'][held],labels[held],
            arrays['target_genes'],primary,10,-.5)
        result={'heldout_lineage':lineage,'selected_alpha':selected,'elapsed_seconds':time.monotonic()-tick,
            'metrics':scored,'certificates':certificates,
            'tuning':[{'heldout_lineage':lineage,'alpha':a,'inner_lineage_equal_ndcg':scores[a],
                       'selected':a==selected} for a in alphas]}
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False)+'\n');temporary.rename(path)
        completed.append(result)
        print(f'  【选参完成】{lineage}｜α={selected}｜耗时 {time.monotonic()-tick:.1f}秒',flush=True)
    print('【阶段 3/4】固定外层结果与GPU配对bootstrap',flush=True)
    historical_metrics=pd.read_csv(internal/'per_model_metrics.csv.gz')
    metrics=pd.concat([pd.DataFrame([row for f in completed for row in f['metrics']]),
        historical_metrics.loc[historical_metrics.method.eq('expression_pcr_ridge')]],ignore_index=True)
    overall,lineage_summary=advanced.summarize_metrics(metrics)
    deltas=advanced.paired_deltas(metrics)
    change=metrics.loc[~metrics.method.eq('expression_pcr_ridge')].copy()
    change['method']=change.method.replace({'certified_fixed_alpha':'expression_pcr_ridge'})
    change=advanced.paired_deltas(change);change['comparison']='certified_expanded_grid_vs_certified_fixed_alpha'
    deltas=pd.concat([deltas,change],ignore_index=True)
    bootstrap=advanced.bootstrap_deltas(deltas,models,protocol['bootstrap_draws'],20260920)
    gate=bootstrap.loc[bootstrap.comparison.eq('certified_expanded_grid_vs_certified_fixed_alpha') &
        bootstrap.metric.eq('ndcg_at_10') & bootstrap.estimand.eq('lineage_equal')].iloc[0]
    print('【阶段 4/4】保存结果、证书汇总和选参审计',flush=True)
    staging=out.parent/('.'+out.name+'_final')
    if staging.exists():shutil.rmtree(staging)
    staging.mkdir()
    tables={'per_model_metrics.csv.gz':metrics,'overall_summary.csv':overall,'lineage_summary.csv':lineage_summary,
        'paired_deltas.csv.gz':deltas,'bootstrap.csv':bootstrap,
        'tuning.csv':pd.DataFrame([row for f in completed for row in f['tuning']]),
        'certificates.csv':pd.DataFrame([row for f in completed for row in f['certificates']]),
        'folds.csv':pd.DataFrame([{k:f[k] for k in ('heldout_lineage','selected_alpha','elapsed_seconds')} for f in completed])}
    for name,frame in tables.items():frame.to_csv(staging/name,index=False)
    run={'status':'certified_upper_grid_sensitivity_complete','created_utc':datetime.now(timezone.utc).isoformat(),
        'invocation_elapsed_seconds':time.monotonic()-started,'completed_fold_seconds':sum(f['elapsed_seconds'] for f in completed),
        'device':torch.cuda.get_device_name(0),'signature_sha256':signature,
        'external_gate_passed':bool(gate.ci_low>0),'selected_alphas':{f['heldout_lineage']:f['selected_alpha'] for f in completed},
        'limitations':['Only alpha>=0.1; smaller historical alpha paths not recertified.',
            'Post-hoc development sensitivity on reused outer holdouts, not new independent confirmation.',
            'No TCGA patient Test access; no Sanger evaluation in this entry.'],
        'output_sha256':{name:advanced.sha256(staging/name) for name in tables}}
    (staging/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n');staging.rename(out)
    shutil.rmtree(checkpoint)
    names={'expression_pcr_ridge':'PCR-ridge','certified_fixed_alpha':'认证固定α','certified_expanded_grid':'认证扩展网格'}
    for row in overall.loc[overall.estimand.eq('lineage_equal')].itertuples():
        print(f'【结果】{names[row.method]}｜NDCG {row.ndcg_at_10:.6f}｜命中率 {row.selective_precision_at_10:.4f}',flush=True)
    print(f'【上界增益】ΔNDCG {gate["mean"]:+.6f}｜95%区间 [{gate.ci_low:+.6f}, {gate.ci_high:+.6f}]',flush=True)
    print(f'【完成】结果 {out}｜外部重算条件 {"满足" if gate.ci_low>0 else "不满足"}',flush=True)


if __name__=='__main__':main()
