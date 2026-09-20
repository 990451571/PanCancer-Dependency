#!/usr/bin/env python3
"""CUDA optimality audit of frozen shared-alpha real-data outer refits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import math
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
import run_selective_dependency as selective
import run_advanced_model_benchmark as advanced
from advanced_model_methods import standardize_expression, elastic_net_path


def certificate(x, y, w, alpha):
    l1 = l2 = alpha * .5
    residual = x @ w - y
    gradient = x.T @ residual / len(x) + l2 * w
    subgradient = torch.where(w != 0, gradient + l1 * torch.sign(w),
                             torch.sign(gradient) * torch.clamp(gradient.abs() - l1, min=0.))
    objective = residual.square().sum(0)/(2*len(x)) + l1*w.abs().sum(0) + .5*l2*w.square().sum(0)
    bound = subgradient.square().sum(0)/(2*l2)
    return objective, subgradient.abs().amax(0), bound, bound/torch.clamp(objective, min=1e-12)


def refine(x, y, initial, alpha, protocol):
    x, y, w = x.double(), y.double(), initial.double().clone()
    lipschitz = torch.linalg.eigvalsh(x @ x.T)[-1].item()/len(x) + alpha*.5
    z, momentum = w.clone(), 1.
    for iteration in range(protocol['maximum_refinement_iterations'] + 1):
        if iteration % 25 == 0:
            cert = certificate(x, y, w, alpha)
            if (cert[1] <= protocol['kkt_absolute_tolerance']).all() and (cert[3] <= protocol['relative_suboptimality_bound_tolerance']).all():
                return w, cert, iteration
        if iteration == protocol['maximum_refinement_iterations']:
            break
        gradient = x.T @ (x @ z - y)/len(x) + alpha*.5*z
        step = z - gradient/lipschitz
        updated = step.sign()*torch.clamp(step.abs()-alpha*.5/lipschitz, min=0.)
        next_momentum = (1.+math.sqrt(1.+4.*momentum*momentum))/2.
        if torch.sum((z-updated)*(updated-w)) > 0:
            z, next_momentum = updated.clone(), 1.
        else:
            z = updated + (momentum-1.)/next_momentum*(updated-w)
        w, momentum = updated, next_momentum
    raise RuntimeError(f'精化未达到逐靶点证书：KKT={cert[1].max().item():.3g}｜相对上界={cert[3].max().item():.3g}')


def closed_form_check(protocol):
    generator = torch.Generator(device='cuda').manual_seed(20260920)
    x = torch.randn((80, 12), generator=generator, dtype=torch.float64, device='cuda')
    x -= x.mean(0)
    x = torch.linalg.qr(x, mode='reduced')[0]*math.sqrt(len(x))
    y = torch.randn((80, 5), generator=generator, dtype=torch.float64, device='cuda')
    y -= y.mean(0)
    alpha = .1
    correlation = x.T @ y/len(x)
    exact = correlation.sign()*torch.clamp(correlation.abs() - alpha*.5,min=0)/(1+alpha*.5)
    fitted, cert, _ = refine(x,y,torch.zeros_like(exact),alpha,protocol)
    error = float((fitted-exact).abs().max())
    if error > 1e-10:
        raise ValueError(f'闭式解核验失败：{error}')
    return error


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=root/'outputs/elastic_net_numerical_audit_v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'拒绝覆盖：{args.output_dir}')
    start = time.monotonic()
    baseline.configure_device('cuda')
    protocol_path = root/'configs/elastic_net_numerical_audit_protocol_20260920.json'
    protocol = json.loads(protocol_path.read_text())
    print('【阶段 1/4】校验历史结果、真实训练划分与GPU闭式解',flush=True)
    check_error = closed_form_check(protocol)
    directory = root/'results/historical/advanced_model_benchmark_v1'
    historical = json.loads((directory/'run.json').read_text())
    for name,digest in historical['output_sha256'].items():
        if advanced.sha256(directory/name) != digest:
            raise ValueError(f'历史结果哈希错误：{name}')
    _, arrays = advanced.load_inputs(root/'data/processed/advanced_model_benchmark_v1')
    models, _, _, _ = baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    if models.index.tolist() != arrays['model_ids'].tolist():
        raise ValueError('模型顺序错误')
    splits = advanced.exact_outer_splits(root/'results/historical/selective_dependency_benchmark_v1/splits.csv.gz',arrays['model_ids'],None)
    old_metrics = pd.read_csv(directory/'per_model_metrics.csv.gz')
    y = arrays['dependency'].astype(float)
    rows, numerical, folds = [], [], []
    print(f'【固定范围】癌系 {len(splits)}｜全部1204靶点｜只审计已选共享α｜闭式解误差 {check_error:.2e}',flush=True)
    print('【阶段 2/4】复算历史解并以逐靶点最优性条件精化',flush=True)
    for number,(lineage,(train,held)) in enumerate(splits.items(),1):
        tick = time.monotonic()
        alpha = float(historical['selected_parameters'][lineage]['per_target_elastic_net_shared'][0])
        if alpha != .1:
            raise ValueError('此审计只支持历史暖启动路径的首个α；其他α需重放完整路径')
        mean = baseline.observed_mean(y[train],axis=0)
        residual, truth = y[train]-mean, y[held]-mean
        x,hx,_ = standardize_expression(arrays['expression'][train],arrays['expression'][held],torch.float32)
        refined_prediction = np.empty((len(held),y.shape[1]),dtype=float)
        fold_records = []
        def audit_callback(a,columns,gx,outcomes,weight,ghx,intercept):
            before = certificate(gx.double(),outcomes.double(),weight.double(),a)
            refined,after,iterations = refine(gx,outcomes,weight,a,protocol)
            if (after[0] > before[0]+1e-10).any():
                raise ValueError('精化后目标函数增大')
            refined_prediction[:,columns] = (ghx.double() @ refined).cpu().numpy()+intercept
            values = [value.detach().cpu().numpy() for value in (*before,*after)]
            for j,column in enumerate(columns):
                fold_records.append({'heldout_lineage':lineage,'Gene':arrays['target_genes'][column],
                    'common_essential':bool(arrays['target_common_essential'][column]),'alpha':a,
                    'observed_train_n':len(gx),'refinement_iterations':iterations,
                    **{name:float(value[j]) for name,value in zip(
                        ('old_objective','old_kkt','old_suboptimality_bound','old_relative_bound',
                         'refined_objective','refined_kkt','refined_suboptimality_bound','refined_relative_bound'),values)}})
        prediction,diagnostics = elastic_net_path(x,residual,hx,(alpha,),.5,5000,1e-4,audit_callback=audit_callback)
        if not diagnostics[alpha]['converged']:
            raise ValueError('历史停止条件无法复现')
        scored,_ = selective.score_models(truth,{'historical_elastic_net':prediction[alpha],
            'refined_elastic_net':refined_prediction},arrays['model_ids'][held],models.OncotreeLineage.to_numpy()[held],
            arrays['target_genes'],~arrays['target_common_essential'],10,-.5)
        frame = pd.DataFrame(scored)
        original = old_metrics.loc[old_metrics.heldout_lineage.eq(lineage)&old_metrics.method.eq('per_target_elastic_net_shared')].set_index('ModelID')
        reproduced = frame.loc[frame.method.eq('historical_elastic_net')].set_index('ModelID')
        error = float((original[list(selective.METRICS)]-reproduced[list(selective.METRICS)]).abs().max().max())
        if not np.isfinite(error) or error > 1e-6:
            raise ValueError(f'{lineage}历史指标复算不一致：{error}')
        rows.extend(scored); numerical.extend(fold_records)
        numerical_frame = pd.DataFrame(fold_records)
        old_failure = int(((numerical_frame.old_kkt>protocol['kkt_absolute_tolerance']) |
                          (numerical_frame.old_relative_bound>protocol['relative_suboptimality_bound_tolerance'])).sum())
        ndcg = frame.groupby('method').ndcg_at_10.mean()
        folds.append({'heldout_lineage':lineage,'old_certificate_fail_n':old_failure,
            'historical_metric_max_error':error,'prediction_max_abs_change':float(np.max(np.abs(refined_prediction-prediction[alpha]))),
            'ndcg_change':float(ndcg['refined_elastic_net']-ndcg['historical_elastic_net'])})
        print(f'  {number}/{len(splits)} {lineage}｜原解未达严格证书 {old_failure}/1204｜精化全部通过｜ΔNDCG {folds[-1]["ndcg_change"]:+.5f}｜{time.monotonic()-tick:.1f}秒',flush=True)
    print('【阶段 3/4】计算精化前后及相对PCR的配对区间',flush=True)
    metrics = pd.concat([pd.DataFrame(rows),old_metrics.loc[old_metrics.method.eq('expression_pcr_ridge')]],ignore_index=True)
    overall,lineage_summary = advanced.summarize_metrics(metrics)
    deltas = advanced.paired_deltas(metrics)
    changes = metrics.loc[~metrics.method.eq('expression_pcr_ridge')].copy()
    changes['method'] = changes.method.replace({'historical_elastic_net':'expression_pcr_ridge'})
    changes = advanced.paired_deltas(changes)
    changes['comparison'] = 'refined_elastic_net_vs_historical_elastic_net'
    deltas = pd.concat([deltas,changes],ignore_index=True)
    bootstrap = advanced.bootstrap_deltas(deltas,models,20000,20260920)
    print('【阶段 4/4】保存数值证书与敏感性结果',flush=True)
    out=args.output_dir;out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.'+out.name,dir=out.parent))
    try:
        tables={'per_target_certificate.csv.gz':pd.DataFrame(numerical),'folds.csv':pd.DataFrame(folds),
                'per_model_metrics.csv.gz':metrics,'overall_summary.csv':overall,'lineage_summary.csv':lineage_summary,
                'paired_deltas.csv.gz':deltas,'bootstrap.csv':bootstrap}
        for name,frame in tables.items():frame.to_csv(temp/name,index=False)
        record={'status':'frozen_shared_alpha_outer_refit_numerical_audit_complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':time.monotonic()-start,'device':torch.cuda.get_device_name(0),
                'protocol_sha256':advanced.sha256(protocol_path),'historical_run_sha256':advanced.sha256(directory/'run.json'),
                'script_sha256':advanced.sha256(Path(__file__)),'methods_sha256':advanced.sha256(root/'scripts/advanced_model_methods.py'),
                'closed_form_max_error':check_error,'target_fold_certificate_n':len(numerical),
                'scope':'Only frozen shared-alpha outer refits; inner alpha selection and secondary targetwise fits remain unaudited.',
                'output_sha256':{name:advanced.sha256(temp/name) for name in tables}}
        (temp/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');temp.rename(out)
    except BaseException:
        shutil.rmtree(temp);raise
    for row in overall.loc[overall.estimand.eq('lineage_equal')].itertuples():
        print(f'【结果】{row.method}｜NDCG {row.ndcg_at_10:.6f}｜Spearman {row.spearman:.6f}',flush=True)
    print(f'【完成】耗时 {time.monotonic()-start:.1f}秒｜结果 {out}',flush=True)
    print('【边界】固定α的数值敏感性检查；不能替代内层选参审计或新的独立验证。',flush=True)


if __name__=='__main__':main()
