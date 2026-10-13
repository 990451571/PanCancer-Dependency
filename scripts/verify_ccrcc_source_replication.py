"""Verify source provenance, donor pooling, frozen candidates and replication snapshots."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from pcr_transfer_methods import sha256,verify_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots-only',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]; checked=0
    for name in ['ccrcc_source_replication_v1','ccrcc_source_replication_synthesis_v1','ccrcc_source_replication_coverage_v1']:
        directory=root/'results/historical'/name; snap=json.loads((directory/'snapshot.json').read_text())
        assert sha256(directory/'run.json')==snap['upstream_run_sha256']
        for filename,expected in snap['included_sha256'].items(): assert sha256(directory/filename)==expected; checked+=1
    result=dict(status='passed',snapshot_file_n=checked)
    if not args.snapshots_only:
        source=root/'outputs/ccrcc_source_replication_v1'; synthesis=root/'outputs/ccrcc_source_replication_synthesis_v1'
        run=verify_manifest(source); summary=verify_manifest(synthesis)
        assert summary['upstream_run_sha256']==sha256(source/'run.json')
        assert run['input_sha256']['pcr_run']==sha256(root/'outputs/pcr_patient_extension_v1/run.json')
        assert run['input_sha256']['old_singlecell_run']==sha256(root/'outputs/ccrcc_singlecell_extension_v1/run.json')
        assert run['protocol_sha256']==sha256(root/'configs/ccrcc_source_replication_protocol_20261010.json')
        for name,expected in run['script_sha256'].items(): assert sha256(root/'scripts'/name)==expected
        assert summary['script_sha256']==sha256(root/'scripts/summarize_ccrcc_source_replication.py')
        raw=root/'data/raw/gse178481'; manifest=json.loads((raw/'source_manifest.json').read_text())
        assert run['input_sha256']['source_manifest']==sha256(raw/'source_manifest.json')
        for name,record in manifest['files'].items(): assert sha256(raw/name)==record['sha256']
        samples=pd.read_csv(source/'sample_inventory.csv'); audit=pd.read_csv(source/'sample_source_audit.csv')
        assert len(audit)==21 and len(audit[audit.scope=='primary'])==16
        assert audit.geo_author_concordant.all() and (audit.umi_error==0).all() and (audit.detected_error==0).all() and (audit.mt_fraction_error<1e-8).all()
        assert not set(audit.donor)&{'PR1','PR9'}
        assert set(samples[samples.scope=='excluded'].donor)=={'PR1','PR9'}
        primary=audit[audit.scope=='primary']; assert primary.donor.nunique()==7
        assert primary[primary.condition=='Tumor'].donor.nunique()==6 and primary[primary.condition=='Normal'].donor.nunique()==6
        cells=pd.read_csv(source/'cell_metadata_qc.csv.gz',index_col=0)
        assert not cells.index.has_duplicates and len(cells)==audit.annotated_cells.sum()
        assert not cells.loc[cells.condition=='Normal','ano.l1'].eq('Tumor').any()
        meta=pd.read_csv(source/'pseudobulk_groups.csv')
        with np.load(source/'pseudobulk_counts.npz') as a: genes=a['genes']; counts=a['counts']; detected=a['detection_count']; cpm=a['cpm']
        assert counts.shape==(len(meta),len(genes)) and len(set(genes))==len(genes)
        assert (detected>=0).all() and (detected<=meta.cell_n.to_numpy()[:,None]).all()
        np.testing.assert_allclose(cpm,counts/counts.sum(axis=1,keepdims=True)*1e6,atol=1e-8)
        max_error=0.
        for i,r in meta[meta.group=='all_cells'].iterrows():
            selected=cells[(cells.donor==r.donor)&(cells.condition==r.condition)&(cells.scope==r.scope)]
            if r.qc=='extra_qc': selected=selected[selected.extra_qc]
            assert len(selected)==r.cell_n
            error=abs(counts[i].sum()-selected.raw_umi.sum()); assert error==0; max_error=max(max_error,error)
        candidates=pd.read_csv(source/'candidate_sets.csv'); original=pd.read_csv(root/'outputs/pcr_patient_extension_v1/pcr_discovery_top20.csv')
        current=candidates[candidates.pcr_discovery_rank.notna()].sort_values('pcr_discovery_rank')
        assert current.Gene.tolist()==original.Gene.tolist()
        old=pd.read_csv(root/'results/historical/final_evidence_synthesis_v2/candidate_evidence_matrix.csv')
        assert set(candidates[candidates.historical_frozen_candidate].Gene)==set(old.Gene)
        assert not run['locked_test_access'] and not run['model_refitting'] and not run['candidate_reranking']
        pairs=pd.read_csv(source/'paired_tumor_proximal_expression.csv')
        np.testing.assert_allclose(pairs.log2_cpm_ratio,np.log2((pairs.tumor_cpm+1)/(pairs.normal_cpm+1)),atol=1e-12)
        assert pairs[pairs.qc=='extra_qc'].donor.nunique()==run['paired_donor_n']
        coverage_dir=root/'outputs/ccrcc_source_replication_coverage_v1'
        coverage_run=verify_manifest(coverage_dir)
        assert coverage_run['upstream_run_sha256']==sha256(source/'run.json')
        assert coverage_run['script_sha256']==sha256(root/'scripts/audit_ccrcc_replication_coverage.py')
        coverage=pd.read_csv(coverage_dir/'candidate_gene_coverage.csv')
        assert set(coverage.loc[~coverage.gene_symbol_measured,'Gene'])=={'GCN1','MRPL57'}
        corrected=pd.read_csv(coverage_dir/'paired_expression_summary_with_coverage.csv')
        assert (corrected.loc[~corrected.gene_symbol_measured,'valid_paired_donor_n']==0).all()
        assert corrected.loc[~corrected.gene_symbol_measured,'median_log2_cpm_ratio'].isna().all()
        for _,row in corrected.iterrows():
            finite=pairs[(pairs.qc==row.qc)&(pairs.Gene==row.Gene)&pairs.log2_cpm_ratio.notna()]
            assert row.valid_paired_donor_n==finite.donor.nunique()
        result.update(missing_symbols_separated_from_zero=True,missing_candidate_symbols=['GCN1','MRPL57'])
        result.update(sample_n=21,primary_donor_n=7,primary_tumor_donor_n=6,primary_normal_donor_n=6,paired_donor_n=run['paired_donor_n'],raw_count_and_donor_pooling_verified=True,maximum_umi_pooling_error=max_error,frozen_candidates_preserved=True,locked_test_access=False)
    if args.report:
        if args.report.exists(): raise FileExistsError(args.report)
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
