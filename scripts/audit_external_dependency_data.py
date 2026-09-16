"""Audit pinned Sanger metadata against the current Broad training cohort.

No functional outcomes are evaluated. Availability is not proof that a particular
download contains a model; source-specific matrix coverage remains to be checked.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from runtime_paths import source_project_root


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def main():
    root = Path(__file__).resolve().parents[1]
    source = source_project_root(root)
    parser = argparse.ArgumentParser(description='外部功能数据覆盖审计（仅元数据，不训练）')
    parser.add_argument('--metadata-dir', type=Path,
                        default=Path('data/raw/external_sanger_20260914'))
    parser.add_argument('--baseline-dir', type=Path,
                        default=Path('data/processed/depmap_baseline_24q4_v1'))
    parser.add_argument('--broad-dir', type=Path,
                        default=source / 'data/raw/depmap_24q4')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('outputs/external_dependency_audit_v1'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f'拒绝覆盖已有结果：{args.output_dir}')
    print('【阶段 1/3】核对模型身份、患者重叠及来源标注', flush=True)
    annotation_path = args.metadata_dir / 'model_list_20260814.csv'
    availability_path = args.metadata_dir / 'model_dataset_availability_20260727.csv'
    broad_path = args.broad_dir / 'Model.csv'
    cohort_path = args.baseline_dir / 'models.csv'
    annotation = pd.read_csv(annotation_path)
    availability = pd.read_csv(availability_path)
    broad = pd.read_csv(broad_path, low_memory=False).set_index('ModelID', verify_integrity=True)
    cohort = pd.read_csv(cohort_path)
    merged = annotation.merge(availability, on='model_id', how='left',
                              suffixes=('', '_availability'), validate='one_to_one', indicator=True)
    merged['availability_known'] = merged['_merge'].eq('both')
    sanger_to_broad = broad.reset_index().dropna(subset=['SangerModelID']).groupby('SangerModelID')['ModelID'].agg(list).to_dict()
    cohort_models = set(cohort.ModelID)
    cohort_patients = set(cohort.PatientID.dropna())
    rows = []
    for _, row in merged.iterrows():
        direct = str(row.BROAD_ID) if pd.notna(row.BROAD_ID) else ''
        reverse = sanger_to_broad.get(row.model_id, [])
        candidates = set(reverse) | ({direct} if direct in broad.index else set())
        conflict = len(candidates) > 1 or bool(direct and reverse and direct not in reverse)
        match = next(iter(candidates)) if len(candidates) == 1 and not conflict else ''
        patient = broad.at[match, 'PatientID'] if match else None
        record = row.drop(labels=['_merge']).to_dict()
        record.update(matched_broad_id=match, identity_conflict=conflict,
                      candidate_broad_ids=';'.join(sorted(candidates)),
                      identity_resolved=bool(match),
                      in_current_cohort=match in cohort_models if match else None,
                      shares_current_patient=patient in cohort_patients if pd.notna(patient) else None,
                      broad_patient_id=patient,
                      broad_subtype=broad.at[match, 'OncotreeSubtype'] if match else None)
        rows.append(record)
    audit = pd.DataFrame(rows)
    print('【阶段 2/3】读取 Broad 各矩阵模型索引，不评价功能标签', flush=True)
    files = {'crispr': 'CRISPRGeneEffect.csv',
             'expression': 'OmicsExpressionProteinCodingGenesTPMLogp1.csv',
             'copy_number': 'OmicsAbsoluteCNGene.csv',
             'mutation': 'OmicsSomaticMutationsMatrixDamaging.csv'}
    index_sources = {}
    for modality, name in files.items():
        path = args.broad_dir / name
        ids = set(pd.read_csv(path, usecols=[0]).iloc[:, 0].dropna().astype(str))
        audit[f'broad_{modality}_row_available'] = audit.matched_broad_id.map(
            lambda value: value in ids if value else None)
        index_sources[modality] = {
            'path': str(path.resolve()), 'bytes': path.stat().st_size, 'unique_ids': len(ids),
            'sorted_ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'hash_scope': 'model IDs only; matrix values not hashed or evaluated'}
    cell = audit[audit['CRISPR Sanger Cell Lines'].eq(True)]
    renal = cell[cell.tissue.eq('Kidney')].copy()
    renal['strict_ccrcc'] = renal.cancer_type_detail.eq('Clear Cell Renal Cell Carcinoma')
    organoid = audit[audit['CRISPR Sanger Organoid'].eq(True)]
    counts = {
        'annotated_models': len(audit), 'availability_unknown': int((~audit.availability_known).sum()),
        'availability_ids_without_annotation': len(set(availability.model_id) - set(annotation.model_id)),
        'sanger_cell_lines': len(cell), 'sanger_identity_conflicts': int(cell.identity_conflict.sum()),
        'sanger_identity_unresolved': int((~cell.identity_resolved).sum()),
        'sanger_in_current_cohort': int(cell.in_current_cohort.eq(True).sum()),
        'sanger_resolved_outside_current_cohort': int(cell.in_current_cohort.eq(False).sum()),
        'sanger_sharing_current_patient': int(cell.shares_current_patient.eq(True).sum()),
        'sanger_with_broad_crispr': int(cell.broad_crispr_row_available.eq(True).sum()),
        'sanger_kidney': len(renal), 'sanger_strict_ccrcc': int(renal.strict_ccrcc.sum()),
        'sanger_organoids': len(organoid), 'sanger_kidney_organoids': int(organoid.tissue.eq('Kidney').sum())}
    run = {'created_utc': datetime.now(timezone.utc).isoformat(), 'counts': counts,
           'sources': [fingerprint(p) for p in [annotation_path, availability_path, broad_path, cohort_path]],
           'official_urls': [
               'https://cog.sanger.ac.uk/cmp/download/model_list_20260814.csv',
               'https://cog.sanger.ac.uk/cmp/download/model_dataset_availability_20260727.csv',
               'https://depmap.sanger.ac.uk/documentation/datasets/wg-crispr-knockout/'],
           'raw_matrix_index_sources': index_sources, 'script': fingerprint(Path(__file__)),
           'limitations': [
               'Source-specific availability does not establish membership in a particular archived matrix.',
               'Unresolved identities remain unknown; aliases are not automatically matched.',
               'Current merged scores are not an independently processed Sanger validation endpoint.',
               'Three kidney models are absent from the current cohort but present in Broad CRISPR.',
               'Two explicitly annotated ccRCC models cannot establish general ccRCC specificity.',
               'This audit covers the pinned Sanger release, not every public CRISPR study.',
               'No model training, outcome evaluation, or TCGA access was performed.']}
    args.output_dir.mkdir(parents=True)
    audit.to_csv(args.output_dir / 'model_coverage.csv.gz', index=False, compression='gzip')
    renal.to_csv(args.output_dir / 'renal_candidates.csv', index=False)
    (args.output_dir / 'run.json').write_text(json.dumps(run, indent=2, ensure_ascii=False) + '\n')
    print(f'【阶段 3/3】审计完成：Sanger 细胞系 {len(cell)}；肾癌 {len(renal)}；明确 ccRCC {int(renal.strict_ccrcc.sum())}', flush=True)
    print(f'【身份核对】当前队列重叠 {counts["sanger_in_current_cohort"]}；已匹配但不在队列 {counts["sanger_resolved_outside_current_cohort"]}；身份未解决 {counts["sanger_identity_unresolved"]}', flush=True)
    print(f'【结论边界】未评价依赖预测；未证明患者域适配。结果：{args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
