#!/usr/bin/env python3
"""Manual GPU entry for recovered-expression PCR and observed-pair DeepDEP."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
import run_selective_dependency as selective
import run_selective_dependency_benchmark as classic
import run_advanced_model_benchmark as advanced
from advanced_model_methods import ExpressionAutoencoder,expression_kernels
from deepdep_pair_training import atomic_json,benchmark,fit

NAMES={'expression_pcr_ridge':'PCR-ridge','exp_deepdep_residual_primary':'DeepDEP残差主分析',
       'exp_deepdep_absolute_secondary':'DeepDEP绝对依赖次分析'}
VARIANTS=list(NAMES)[1:]


def cache_write(path,payload):
    encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True)
    atomic_json(path,{'payload':payload,'sha256':hashlib.sha256(encoded.encode()).hexdigest()})


def cache_read(path):
    record=json.loads(path.read_text());payload=record['payload']
    if hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=record['sha256']:
        raise ValueError(f'断点校验失败：{path}')
    return payload


def load_context(root):
    protocol_path=root/'configs/deepdep_recovered_comparison_protocol_20260921.json'
    p=json.loads(protocol_path.read_text())
    if p['deep_training']['maximum_epochs']!=100 or p['inner_selection']['folds']!=5 or not p['deep_training']['batch'].startswith('500 observed'):
        raise ValueError('冻结训练设置改变，需要显式更新实现与协议版本')
    input_dir=root/p['inputs']['directory']
    if advanced.sha256(input_dir/'audit.json')!=p['inputs']['audit_sha256']:
        raise ValueError('输入审计变化')
    audit,arrays=advanced.load_inputs(input_dir)
    if advanced.sha256(input_dir/'benchmark_inputs.npz')!=p['inputs']['matrix_sha256']:
        raise ValueError('输入矩阵变化')
    models,*_=baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    if models.index.tolist()!=arrays['model_ids'].tolist():raise ValueError('模型顺序变化')
    splits=advanced.exact_outer_splits(root/p['outer_split'],arrays['model_ids'],None)
    if len(splits)!=19 or len(arrays['target_genes'])!=1204 or int((~arrays['target_common_essential']).sum())!=911:
        raise ValueError('冻结比较范围变化')
    encoder_path=root/'data/processed/tcga_pancan_deepdep_pretrain_v1/expression_encoder.pt'
    if advanced.sha256(encoder_path)!=p['deep_training']['encoder_sha256']:raise ValueError('编码器变化')
    encoder=ExpressionAutoencoder().encoder
    encoder.load_state_dict(torch.load(encoder_path,map_location='cpu',weights_only=True))
    files=[protocol_path,input_dir/'audit.json',input_dir/'benchmark_inputs.npz',encoder_path,
        root/p['outer_split'],root/'data/processed/depmap_baseline_24q4_v1/models.csv']
    files += [root/'scripts'/name for name in ['run_deepdep_recovered_comparison.py','deepdep_pair_training.py',
        'advanced_model_methods.py','run_advanced_model_benchmark.py','run_selective_dependency.py',
        'run_selective_dependency_benchmark.py','run_depmap_baseline.py']]
    signature={'files':{str(f.relative_to(root)):advanced.sha256(f) for f in files},
        'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,
                       'pandas':pd.__version__,'device':torch.cuda.get_device_name(0),
                       'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}}
    return p,arrays,models,splits,encoder,signature


def inner_folds(models,train,number):
    return baseline.inner_folds(train,models.OncotreeLineage.to_numpy(),models.PatientID.to_numpy(),5,20260917+number)


def runtime_audit(root,directory,context):
    p,arrays,models,splits,encoder,signature=context
    directory.mkdir(parents=True,exist_ok=True);path=directory/'run.json'
    if path.exists():
        report=cache_read(path)
        if report['signature']!=signature:raise ValueError('已有计时代码或输入不同；请使用新的 --runtime-dir')
    else:
        lineage,(outer,_) = next(iter(splits.items()))
        validation=inner_folds(models,outer,0)[0];train=np.setdiff1d(outer,validation)
        mean=baseline.observed_mean(arrays['dependency'][train].astype(float),axis=0)
        print('【计时】仅使用首个内层训练子集｜20次预热＋200次更新｜不评价留出集',flush=True)
        report=benchmark(encoder,arrays['expression'][train],arrays['dependency'][train]-mean,
                         arrays['target_fingerprints'],p['deep_training']['seeds'][0])
        report.update({'status':'training_only_runtime_audit','created_utc':datetime.now(timezone.utc).isoformat(),
                       'signature':signature,'outer_lineage':lineage,'heldout_predictions_computed':False})
        report['one_variant_update_only_hours']=report['seconds_per_update']*p['budget']['per_variant_maximum_optimizer_steps']/3600
        report['both_variants_update_only_hours']=report['seconds_per_update']*p['budget']['both_deep_variants_maximum_optimizer_steps']/3600
        cache_write(path,report)
    print(f'【成本估计】每更新 {report["seconds_per_update"]*1000:.2f}毫秒｜单版本满预算 {report["one_variant_update_only_hours"]:.1f}小时｜两个版本 {report["both_variants_update_only_hours"]:.1f}小时',flush=True)
    print('【估计边界】按全部拟合100 epochs外推；未计验证、PCR和磁盘开销，实际受所选epoch及设备负载影响。',flush=True)
    return report


def score(arrays,models,held,truth,predictions):
    if any(not np.isfinite(pred).all() for pred in predictions.values()):
        raise FloatingPointError('预测存在非有限值')
    rows,_=selective.score_models(truth,predictions,arrays['model_ids'][held],
        models.OncotreeLineage.to_numpy()[held],arrays['target_genes'],~arrays['target_common_essential'],10,-.5)
    if len(rows)!=len(held)*len(predictions):raise ValueError('评分模型覆盖不完整')
    return pd.DataFrame(rows)


def fit_seeds(p,arrays,encoder,train,held,variant,epochs,checkpoints,directory,label):
    mean=baseline.observed_mean(arrays['dependency'][train].astype(float),axis=0)
    outcomes=arrays['dependency'][train].astype(float)
    if variant=='exp_deepdep_residual_primary':outcomes=outcomes-mean
    ensemble={};histories=[]
    for seed in p['deep_training']['seeds']:
        seed_dir=directory/f'seed_{seed}'
        predictions=fit(encoder,arrays['expression'][train],outcomes,arrays['expression'][held],
            arrays['target_fingerprints'],seed,epochs,checkpoints,seed_dir,label)
        for epoch,prediction in predictions.items():
            ensemble.setdefault(epoch,[]).append(prediction)
        history=pd.read_csv(seed_dir/'training.csv');history['seed']=seed;history['variant']=variant
        histories.extend(history.to_dict('records'))
    ensemble={epoch:np.mean(values,axis=0) for epoch,values in ensemble.items()}
    if variant=='exp_deepdep_absolute_secondary':ensemble={e:v-mean for e,v in ensemble.items()}
    return ensemble,histories


def train(root,out,context,runtime):
    p,arrays,models,splits,encoder,signature=context
    if out.exists():raise FileExistsError(f'正式结果已存在，拒绝覆盖：{out}')
    work=out.parent/('.'+out.name+'_work');work.mkdir(parents=True,exist_ok=True)
    with (work/'run.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('同一输出已有训练进程运行')
        manifest=work/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=signature:
            raise ValueError('断点的代码、输入或环境变化；拒绝混合运行')
        atomic_json(manifest,signature)
        completed=[];grids=p['models']['pcr_ridge'];checkpoints=p['deep_training']['selection_checkpoints']
        for number,(lineage,(outer,held)) in enumerate(splits.items()):
            print(f'【癌系 {number+1}/19】{lineage}',flush=True)
            fold_dir=work/f'outer_{number+1:02d}';fold_dir.mkdir(exist_ok=True)
            finished=fold_dir/'complete.json'
            if finished.exists():completed.append(cache_read(finished));print('  【续跑】已完成，校验后复用',flush=True);continue
            inner=[]
            for fold_number,validation in enumerate(inner_folds(models,outer,number),1):
                cached=fold_dir/f'inner_{fold_number}.json'
                if cached.exists():inner.append(cache_read(cached));continue
                print(f'  【内层 {fold_number}/5】只用训练折选择参数和epoch',flush=True)
                fit_rows=np.setdiff1d(outer,validation)
                mean=baseline.observed_mean(arrays['dependency'][fit_rows].astype(float),axis=0)
                residual=arrays['dependency'][fit_rows]-mean;truth=arrays['dependency'][validation]-mean
                kernel,hk,*_=expression_kernels(arrays['expression'][fit_rows],arrays['expression'][validation])
                pcr=classic.pcr_ridge_predictions(kernel,hk,residual,grids['ranks'],grids['alphas'])
                tuning=[];history=[]
                for (rank,alpha),prediction in pcr.items():
                    frame=score(arrays,models,validation,truth,{'m':prediction})
                    for held_lineage,value in frame.groupby('heldout_lineage').ndcg_at_10.mean().items():
                        tuning.append({'method':'expression_pcr_ridge','rank':rank,'alpha':alpha,'epoch':0,
                                       'inner_lineage':held_lineage,'ndcg':value})
                del kernel,hk,pcr
                temp=fold_dir/f'inner_{fold_number}_fits'
                for variant in VARIANTS:
                    predictions,rows=fit_seeds(p,arrays,encoder,fit_rows,validation,variant,
                        p['deep_training']['maximum_epochs'],checkpoints,temp/variant,f'{lineage} 内层{fold_number} {NAMES[variant]}')
                    history.extend(rows)
                    for epoch,prediction in predictions.items():
                        frame=score(arrays,models,validation,truth,{'m':prediction})
                        for held_lineage,value in frame.groupby('heldout_lineage').ndcg_at_10.mean().items():
                            tuning.append({'method':variant,'rank':0,'alpha':0,'epoch':epoch,'inner_lineage':held_lineage,'ndcg':value})
                for row in history:row['scope']='inner';row['inner_fold']=fold_number
                result={'tuning':tuning,'history':history};cache_write(cached,result);inner.append(result)
                shutil.rmtree(temp)
                print(f'  【内层完成】{fold_number}/5｜三方法共享评价口径',flush=True)
            tuning=pd.DataFrame([r for f in inner for r in f['tuning']])
            selected={}
            values=tuning.loc[tuning.method.eq('expression_pcr_ridge')].groupby(['rank','alpha']).ndcg.mean()
            selected['expression_pcr_ridge']=sorted(values.index,key=lambda c:(-values[c],c[0],-c[1]))[0]
            for variant in VARIANTS:
                values=tuning.loc[tuning.method.eq(variant)].groupby('epoch').ndcg.mean()
                selected[variant]=int(sorted(values.index,key=lambda e:(-values[e],e))[0])
            selected={k:([int(v[0]),float(v[1])] if k=='expression_pcr_ridge' else v) for k,v in selected.items()}
            print(f'  【内层选择】PCR {selected["expression_pcr_ridge"]}｜残差epoch {selected[VARIANTS[0]]}｜绝对依赖epoch {selected[VARIANTS[1]]}',flush=True)
            mean=baseline.observed_mean(arrays['dependency'][outer].astype(float),axis=0)
            kernel,hk,*_=expression_kernels(arrays['expression'][outer],arrays['expression'][held])
            rank,alpha=selected['expression_pcr_ridge']
            predictions={'expression_pcr_ridge':classic.pcr_ridge_predictions(kernel,hk,arrays['dependency'][outer]-mean,[rank],[alpha])[(rank,alpha)]}
            history=[r for f in inner for r in f['history']]
            refit=fold_dir/'refit'
            for variant in VARIANTS:
                epoch=selected[variant]
                prediction,rows=fit_seeds(p,arrays,encoder,outer,held,variant,epoch,[epoch],refit/variant,
                    f'{lineage} 外层重拟合 {NAMES[variant]}')
                predictions[variant]=prediction[epoch]
                for row in rows:row['scope']='outer';row['inner_fold']=0
                history.extend(rows)
            metrics=score(arrays,models,held,arrays['dependency'][held]-mean,predictions)
            result={'lineage':lineage,'selected':selected,'metrics':metrics.to_dict('records'),
                    'tuning':tuning.to_dict('records'),'history':history}
            cache_write(finished,result);completed.append(result);shutil.rmtree(refit)
            for variant,value in metrics.groupby('method').ndcg_at_10.mean().items():
                print(f'  【外层结果】{NAMES[variant]}｜NDCG {value:.4f}',flush=True)
        print('【最终汇总】计算癌系等权指标与GPU配对区间',flush=True)
        metrics=pd.DataFrame([r for f in completed for r in f['metrics']])
        overall,lineage_summary=advanced.summarize_metrics(metrics)
        deltas=advanced.paired_deltas(metrics)
        bootstrap=advanced.bootstrap_deltas(deltas,models,20000,20260921)
        staging=work/'final';staging.mkdir(exist_ok=True)
        tables={'per_model_metrics.csv.gz':metrics,'overall_summary.csv':overall,'lineage_summary.csv':lineage_summary,
            'paired_deltas.csv.gz':deltas,'bootstrap.csv':bootstrap,
            'tuning.csv.gz':pd.DataFrame([{'outer_lineage':f['lineage'],**r} for f in completed for r in f['tuning']]),
            'training.csv.gz':pd.DataFrame([{'outer_lineage':f['lineage'],**r} for f in completed for r in f['history']])}
        for name,frame in tables.items():frame.to_csv(staging/name,index=False)
        report={'status':'recovered_expression_comparison_complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'signature':signature,'runtime_audit':runtime,'selected_parameters':{f['lineage']:f['selected'] for f in completed},
                'model_n':metrics.ModelID.nunique(),'lineage_n':len(splits),'tcga_kirc_locked_test_read':False,
                'limitations':p['limitations'],'output_sha256':{name:advanced.sha256(staging/name) for name in tables}}
        atomic_json(staging/'run.json',report);staging.rename(out)
    shutil.rmtree(work)
    for row in overall.loc[overall.estimand.eq('lineage_equal')].itertuples():
        print(f'【最终结果】{NAMES[row.method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}｜Spearman {row.spearman:.4f}',flush=True)
    for row in bootstrap.loc[bootstrap.estimand.eq('lineage_equal') & bootstrap.metric.eq('ndcg_at_10')].itertuples():
        method=row.comparison.removesuffix('_vs_expression_pcr_ridge')
        print(f'【相对PCR】{NAMES[method]}｜ΔNDCG {row.mean:+.4f}｜95%区间 [{row.ci_low:+.4f}, {row.ci_high:+.4f}]',flush=True)
    print(f'【完成】结果 {out}｜原论文精确复现与患者功能验证均未建立',flush=True)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--benchmark',action='store_true')
    mode.add_argument('--train',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=root/'outputs/deepdep_recovered_comparison_v1')
    parser.add_argument('--runtime-dir',type=Path,default=root/'outputs/deepdep_recovered_runtime_v1')
    args=parser.parse_args()
    baseline.configure_device('cuda')
    print('【输入检查】恢复表达5384/6016｜主评价911靶点｜不访问患者Test',flush=True)
    context=load_context(root)
    if args.dry_run:
        print('【检查通过】19癌系、五折内层、三种子；尚未训练或生成留出预测。',flush=True);return
    runtime=runtime_audit(root,args.runtime_dir,context)
    if args.benchmark:return
    print('【正式启动】两个DeepDEP目标版本＋PCR｜每批500对｜可用同一命令断点续跑',flush=True)
    train(root,args.output_dir,context,runtime)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n【已中断】保留最近完整epoch断点；再次执行同一 --train 命令续跑。',flush=True)
        raise SystemExit(130)
