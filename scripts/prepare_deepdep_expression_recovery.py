#!/usr/bin/env python3
"""Restore official DeepDEP features from the raw expression-only DepMap matrix."""
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch

from build_context_module_stage0 import GeneCanonicalizer
from build_depmap_bridge import gene_symbol
from run_advanced_model_benchmark import load_inputs,sha256
from runtime_paths import source_project_root


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=source_project_root(root))
    parser.add_argument('--input-dir',type=Path,default=root/'data/processed/advanced_model_benchmark_v1')
    parser.add_argument('--output-dir',type=Path,default=root/'data/processed/advanced_model_benchmark_expression_recovered_v1')
    parser.add_argument('--summary-dir',type=Path,default=root/'outputs/deepdep_expression_recovery_v1')
    args=parser.parse_args();start=time.monotonic()
    for path in (args.output_dir,args.summary_dir):
        if path.exists():raise FileExistsError(f'拒绝覆盖：{path}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA不可用；数值一致性检查不退回CPU')
    print('【阶段 1/4】校验原始表达、HGNC和历史输入哈希',flush=True)
    old_audit,arrays=load_inputs(args.input_dir)
    baseline_audit=json.loads((root/'data/processed/depmap_baseline_24q4_v1/audit.json').read_text())
    source=args.source_root/'data/raw/depmap_24q4/OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    hgnc=args.source_root/'outputs/reassessment_20260911/hgnc_complete_set.tsv'
    for path,expected in ((source,baseline_audit['raw_files'][source.name]['sha256']),
                          (hgnc,baseline_audit['hgnc']['sha256'])):
        if sha256(path)!=expected:raise ValueError(f'来源哈希不一致：{path.name}')
    mapping=pd.read_csv(args.input_dir/'expression_feature_mapping.csv.gz')
    canonicalizer=GeneCanonicalizer(hgnc)
    header=pd.read_csv(source,nrows=0).columns.tolist()
    raw_map={}
    for column in header[1:]:
        symbol=canonicalizer.map(gene_symbol(column))
        if symbol in canonicalizer.approved:raw_map.setdefault(symbol,[]).append(column)
    print('【阶段 2/4】逐特征追溯缺失原因并恢复实测值',flush=True)
    official=set(mapping.canonical_symbol.dropna())
    columns=[c for g,cs in raw_map.items() if g in official for c in cs]
    frame=pd.read_csv(source,usecols=[header[0],*columns],dtype={c:np.float32 for c in columns})
    if frame[header[0]].isna().any() or frame[header[0]].duplicated().any():raise ValueError('原始模型ID缺失或重复')
    frame=frame.set_index(header[0]).loc[arrays['model_ids']]
    frame.columns=[canonicalizer.map(gene_symbol(c)) for c in frame.columns]
    frame=frame.T.groupby(level=0).mean().T
    recovered=arrays['expression'].copy()
    hgnc_frame=pd.read_csv(hgnc,sep='\t',dtype=str).set_index('symbol')
    rows=[]
    for j,row in enumerate(mapping.itertuples()):
        canonical=row.canonical_symbol
        available=canonical in frame.columns
        if available:
            values=frame[canonical].to_numpy(np.float32)
            if not np.isfinite(values).all():raise ValueError(f'实测特征存在缺失值：{canonical}')
            recovered[:,j]=values
            status='retained_observed' if row.depmap_observed else 'recovered_from_expression_only'
        elif row.deepdep_feature in canonicalizer.ambiguous and row.deepdep_feature not in canonicalizer.approved:
            status='ambiguous_hgnc_alias'
        elif canonical not in canonicalizer.approved:
            status='unresolved_hgnc_symbol'
        else:
            status='approved_absent_from_raw_expression'
        locus=hgnc_frame.loc[canonical,'locus_group'] if canonical in hgnc_frame.index and 'locus_group' in hgnc_frame else ''
        rows.append({'feature_index':j,'deepdep_feature':row.deepdep_feature,'canonical_symbol':canonical,
            'previously_observed':bool(row.depmap_observed),'now_observed':available,'status':status,
            'hgnc_locus_group':locus,'raw_column_n':len(raw_map.get(canonical,[])),
            'raw_columns':'|'.join(raw_map.get(canonical,[])),
            'official_imputation_mean':row.official_ccle_imputation_mean})
    audit_table=pd.DataFrame(rows)
    print('【阶段 3/4】GPU检查历史特征、模型和依赖标签一致性',flush=True)
    original=torch.as_tensor(arrays['expression'],device='cuda')
    updated=torch.as_tensor(recovered,device='cuda')
    old_mask=torch.as_tensor(mapping.depmap_observed.to_numpy(bool),device='cuda')
    retained_error=float((original[:,old_mask]-updated[:,old_mask]).abs().max())
    if retained_error!=0.:raise ValueError(f'历史实测表达发生变化：{retained_error}')
    if not torch.isfinite(updated).all():raise ValueError('新表达矩阵存在非有限值')
    missing=~audit_table.now_observed.to_numpy(bool)
    if not np.array_equal(recovered[:,missing],arrays['expression'][:,missing]):raise ValueError('未恢复位置的填补值变化')
    new_arrays=dict(arrays);new_arrays['expression']=recovered
    for key in arrays:
        if key == 'expression':
            continue
        options = {'equal_nan': True} if arrays[key].dtype.kind in 'fc' else {}
        if not np.array_equal(new_arrays[key], arrays[key], **options):
            raise ValueError(f'非表达输入变化：{key}')
    counts=audit_table.status.value_counts().to_dict()
    print(f'【表达覆盖】原实测 {int(mapping.depmap_observed.sum())}｜恢复 {counts.get("recovered_from_expression_only",0)}｜现实测 {int(audit_table.now_observed.sum())}｜仍填补 {int(missing.sum())}',flush=True)
    print('【阶段 4/4】保存新输入及逐特征审计；本步不训练模型',flush=True)
    args.output_dir.parent.mkdir(parents=True,exist_ok=True);args.summary_dir.parent.mkdir(parents=True,exist_ok=True)
    data_temp=Path(tempfile.mkdtemp(prefix='.'+args.output_dir.name,dir=args.output_dir.parent))
    summary_temp=Path(tempfile.mkdtemp(prefix='.'+args.summary_dir.name,dir=args.summary_dir.parent))
    try:
        np.savez_compressed(data_temp/'benchmark_inputs.npz',**new_arrays)
        audit_table.to_csv(summary_temp/'feature_recovery.csv.gz',index=False)
        pd.DataFrame([{'status':k,'feature_n':v} for k,v in counts.items()]).to_csv(summary_temp/'coverage_summary.csv',index=False)
        report={'status':'expression_feature_recovery_complete_no_model_fitted','created_utc':datetime.now(timezone.utc).isoformat(),
          'elapsed_seconds':time.monotonic()-start,'device_for_numerical_checks':torch.cuda.get_device_name(0),
          'model_n':len(arrays['model_ids']),'expression_feature_n':len(mapping),'previous_observed_n':int(mapping.depmap_observed.sum()),
          'recovered_feature_n':counts.get('recovered_from_expression_only',0),'observed_feature_n':int(audit_table.now_observed.sum()),
          'imputed_feature_n':int(missing.sum()),'feature_status_counts':counts,
          'retained_expression_max_abs_difference':retained_error,'non_expression_arrays_unchanged':True,
          'tcga_kirc_test_read':False,'target_n':len(arrays['target_genes']),
          'primary_target_n':int((~arrays['target_common_essential']).sum()),
          'sources':{'raw_expression_name':source.name,'raw_expression_sha256':sha256(source),
             'hgnc_sha256':sha256(hgnc),'historical_input_sha256':sha256(args.input_dir/'benchmark_inputs.npz'),
             'historical_audit_sha256':sha256(args.input_dir/'audit.json'),'script_sha256':sha256(Path(__file__))},
          'recovered_input_sha256':sha256(data_temp/'benchmark_inputs.npz'),
          'rules':['Original 873-model order and 1204 targets retained; no target-based feature filtering.',
             'Same HGNC canonicalization and duplicate-column mean as the baseline builder.',
             'Unresolved or ambiguous genes are not guessed; still-missing values retain official CCLE means.',
             'Raw log2(TPM+1) values used unchanged; no TCGA file read.'],
          'limitations':['More complete does not mean all official features are observed.',
             'Recovered features were excluded by the old shared-modality gene intersection, not absent from raw expression.',
             'This input audit gives no model-performance or patient-function evidence.']}
        data_audit=dict(report);data_audit['output_sha256']={'benchmark_inputs.npz':report['recovered_input_sha256']}
        (data_temp/'audit.json').write_text(json.dumps(data_audit,ensure_ascii=False,indent=2)+'\n')
        report['output_sha256']={p.name:sha256(p) for p in summary_temp.iterdir()}
        (summary_temp/'run.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        data_temp.rename(args.output_dir);summary_temp.rename(args.summary_dir)
    except BaseException:
        shutil.rmtree(data_temp,ignore_errors=True);shutil.rmtree(summary_temp,ignore_errors=True);raise
    print(f'【完成】耗时 {time.monotonic()-start:.1f}秒｜审计 {args.summary_dir}',flush=True)


if __name__=='__main__':main()
