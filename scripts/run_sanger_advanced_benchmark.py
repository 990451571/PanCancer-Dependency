#!/usr/bin/env python3
"""External evaluation of the primary advanced methods with DepMap-fixed parameters."""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import run_depmap_baseline as baseline
import run_selective_dependency as selective
import run_selective_dependency_benchmark as classic
import run_advanced_model_benchmark as advanced
from advanced_model_methods import (ExpressionAutoencoder, expression_kernels,
    elastic_net_path, reduced_rank_ridge_predictions, train_exp_deepdep)
from prepare_sanger_external_controls import read_label_subset
from run_sanger_baseline_benchmark import label_header
from run_sanger_selective_dependency import paired_standardization, zscore


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root/'outputs/sanger_advanced_benchmark_v1')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'拒绝覆盖：{args.output_dir}')
    start = time.monotonic()
    baseline.configure_device('cuda')
    print('【阶段 1/4】校验共同输入、内部参数及冻结外部队列', flush=True)
    input_dir = root/'data/processed/advanced_model_benchmark_v1'
    internal = root/'outputs/advanced_model_benchmark_v1'
    run = json.loads((internal/'run.json').read_text())
    for name, digest in run['output_sha256'].items():
        if advanced.sha256(internal/name) != digest:
            raise ValueError(f'内部结果哈希不一致：{name}')
    _, arrays = advanced.load_inputs(input_dir)
    if advanced.sha256(input_dir/'benchmark_inputs.npz') != run['input_matrix_sha256']:
        raise ValueError('内部训练输入变化')
    models, genes, matrices, _ = baseline.load_data(root/'data/processed/depmap_baseline_24q4_v1')
    if models.index.tolist() != arrays['model_ids'].tolist():
        raise ValueError('模型顺序变化')
    frozen_dir = root/'outputs/sanger_external_controls_frozen_v1'
    controls = pd.read_csv(frozen_dir/'frozen_models.csv')
    with np.load(frozen_dir/'matrices.npz', allow_pickle=False) as archive:
        frozen = {key: archive[key] for key in archive.files}
    mapping = pd.read_csv(input_dir/'expression_feature_mapping.csv.gz')
    gene_index = {g: i for i, g in enumerate(genes)}
    target_index = np.array([gene_index[g] for g in arrays['target_genes']])
    expression = np.broadcast_to(mapping.official_ccle_imputation_mean.to_numpy(),
                                  (len(controls), len(mapping))).copy()
    for j, row in enumerate(mapping.itertuples()):
        if row.depmap_observed:
            expression[:, j] = frozen['direct_expression'][:, gene_index[row.canonical_symbol]]
    if not np.isfinite(expression).all():
        raise ValueError('外部表达存在非有限值')
    primary = frozen['primary_gene_mask'][target_index].astype(bool) & ~arrays['target_common_essential']
    lineages = sorted(set(controls.BroadLineage) & set(run['selected_parameters']))
    excluded = controls.loc[~controls.BroadLineage.isin(lineages)]
    deep_epochs = pd.read_csv(internal/'deep_training.csv')
    encoder_path = root/'data/processed/tcga_pancan_deepdep_pretrain_v1/expression_encoder.pt'
    if advanced.sha256(encoder_path) != run['pretrained_encoder_sha256']:
        raise ValueError('预训练编码器变化')
    encoder = ExpressionAutoencoder().encoder
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    external = root/'data/raw/external_sanger_20260914'
    label_path = external/'project_score_release1_scaled_bf.tsv.gz'
    header = set(label_header(label_path))
    annotation = pd.read_csv(external/'model_list_20260814.csv')
    names = annotation.groupby('model_id').model_name.agg(lambda x: list(dict.fromkeys(x.dropna().astype(str))))
    calibration_rows, calibration_names = [], []
    for i, row in enumerate(models.itertuples()):
        if pd.notna(row.SangerModelID) and row.SangerModelID in names:
            matches = [n for n in names[row.SangerModelID] if n in header]
            if len(matches) == 1:
                calibration_rows.append(i)
                calibration_names.append(matches[0])
    calibration_rows = np.asarray(calibration_rows)
    bf, label_audit = read_label_subset(label_path, calibration_names, arrays['target_genes'], binary=False)
    y = arrays['dependency'].astype(float)
    metric_rows, fold_rows, convergence = [], [], []
    print(f'【固定口径】模型 {len(controls)-len(excluded)}｜癌系 {len(lineages)}｜主评价基因 {primary.sum()}｜不调参', flush=True)
    print('【阶段 2/4】GPU重拟合主比较方法并进行外部校准', flush=True)
    for number, lineage in enumerate(lineages, 1):
        tick = time.monotonic()
        held = np.flatnonzero(controls.BroadLineage.to_numpy() == lineage)
        target_patients = set(controls.iloc[held].BroadPatientID)
        train = np.flatnonzero((models.OncotreeLineage.to_numpy() != lineage)
                              & ~models.PatientID.isin(target_patients).to_numpy())
        keep = np.isin(calibration_rows, train)
        bmean, bsd, smean, ssd, prior, _ = paired_standardization(y[calibration_rows[keep]], -bf[keep])
        truth = zscore(-frozen['sanger_scaled_bf'][held][:, target_index], smean, ssd)
        mean = baseline.observed_mean(y[train], axis=0)
        residual = y[train] - mean
        kernel, hk, x, hx, _ = expression_kernels(arrays['expression'][train], expression[held])
        selected = run['selected_parameters'][lineage]
        rank, alpha = selected['expression_pcr_ridge']
        predictions = {'training_selectivity_prior': np.broadcast_to(prior, truth.shape)}
        pred = classic.pcr_ridge_predictions(kernel, hk, residual, (int(rank),), (float(alpha),))[(rank, alpha)]
        predictions['expression_pcr_ridge'] = zscore(pred + mean, bmean, bsd)
        rank, alpha = selected['multitask_reduced_rank_ridge']
        pred = reduced_rank_ridge_predictions(kernel, hk, residual, (int(rank),), (float(alpha),))[0][(rank, alpha)]
        predictions['multitask_reduced_rank_ridge'] = zscore(pred + mean, bmean, bsd)
        alpha = float(selected['per_target_elastic_net_shared'][0])
        pred, diagnostics = elastic_net_path(x.float(), residual, hx.float(), (alpha,), .5, 5000, 1e-4)
        if not diagnostics[alpha]['converged']:
            raise ValueError(f'{lineage} Elastic Net未收敛')
        convergence.append({'heldout_lineage': lineage, 'alpha': alpha, **diagnostics[alpha]})
        predictions['per_target_elastic_net_shared'] = zscore(pred[alpha] + mean, bmean, bsd)
        deep = []
        for row in deep_epochs.loc[deep_epochs.heldout_lineage.eq(lineage)].itertuples():
            fitted = train_exp_deepdep(arrays['expression'][train], residual, expression[held],
                arrays['target_fingerprints'], encoder, int(row.seed), int(row.selected_epoch),
                None, None, 3, 32)
            deep.append(fitted.prediction)
        if len(deep) != 3:
            raise ValueError('DeepDEP种子数变化')
        predictions['exp_deepdep_adapted'] = zscore(np.mean(deep, axis=0) + mean, bmean, bsd)
        rows, _ = selective.score_models(truth, predictions, controls.matched_broad_id.to_numpy()[held],
            controls.BroadLineage.to_numpy()[held], arrays['target_genes'], primary, 10, -1.)
        metric_rows.extend(rows)
        fold_rows.append({'heldout_lineage': lineage, 'external_n': len(held), 'train_n': len(train),
                          'calibration_n': int(keep.sum()), 'patient_overlap_n': len(set(models.iloc[train].PatientID)&target_patients)})
        print(f'  {number}/{len(lineages)} {lineage}｜模型 {len(held)}｜{time.monotonic()-tick:.1f}秒', flush=True)
    print('【阶段 3/4】GPU配对bootstrap与癌系等权汇总', flush=True)
    metrics = pd.DataFrame(metric_rows)
    overall, lineage_summary = advanced.summarize_metrics(metrics)
    deltas = advanced.paired_deltas(metrics)
    units = controls.set_index('matched_broad_id')[['BroadPatientID']].rename(columns={'BroadPatientID':'PatientID'})
    bootstrap = advanced.bootstrap_deltas(deltas, units, 20000, 20260917)
    print('【阶段 4/4】保存外部结果与审计', flush=True)
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.'+out.name, dir=out.parent))
    try:
        tables = {'per_model_metrics.csv.gz':metrics, 'overall_summary.csv':overall,
                  'lineage_summary.csv':lineage_summary, 'paired_deltas.csv.gz':deltas,
                  'bootstrap.csv':bootstrap, 'folds.csv':pd.DataFrame(fold_rows),
                  'elastic_convergence.csv':pd.DataFrame(convergence), 'excluded_models.csv':excluded}
        for name, frame in tables.items():
            frame.to_csv(temp/name, index=False)
        audit = {'status':'sanger_advanced_primary_comparison_complete',
                 'created_utc':datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds':time.monotonic()-start, 'device':torch.cuda.get_device_name(0),
                 'internal_run_sha256':advanced.sha256(internal/'run.json'),
                 'script_sha256':advanced.sha256(Path(__file__)),
                 'methods_script_sha256':advanced.sha256(root/'scripts/advanced_model_methods.py'),
                 'frozen_matrices_sha256':advanced.sha256(frozen_dir/'matrices.npz'),
                 'primary_target_n':int(primary.sum()), 'model_n':metrics.ModelID.nunique(),
                 'lineage_n':len(lineages), 'sanger_labels_used_for_tuning':False,
                 'label_audit':label_audit,
                 'limitations':['Primary shared-alpha Elastic Net only; secondary targetwise-MSE variant not externally refitted.',
                    'Previously examined Sanger cohort; not a new untouched test set.',
                    'Shared 911-target universe may shrink due to Sanger coverage.',
                    'Calibration uses non-target paired Broad/Sanger labels; cross-platform scores are descriptive.',
                    'No patient functional validation.'],
                 'output_sha256':{n:advanced.sha256(temp/n) for n in tables}}
        (temp/'run.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp)
        raise
    print('【外部结果｜癌系等权】', flush=True)
    for row in overall.loc[overall.estimand.eq('lineage_equal')].itertuples():
        print(f'  {advanced.METHOD_LABELS[row.method]}｜NDCG {row.ndcg_at_10:.4f}｜命中率 {row.selective_precision_at_10:.4f}', flush=True)
    print(f'【完成】耗时 {time.monotonic()-start:.1f}秒｜结果 {out}', flush=True)
    print('【结论边界】既有外部队列的方法比较，不能证明患者功能依赖或原论文模型优劣。', flush=True)


if __name__ == '__main__':
    main()
