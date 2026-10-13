"""Separate structural donor matches, measured pairs, and absent symbols explicitly."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from pcr_transfer_methods import sha256,verify_manifest,finish_run


def main():
    root=Path(__file__).resolve().parents[1]; source=root/'outputs/ccrcc_source_replication_v1'
    verify_manifest(source); out=root/'outputs/ccrcc_source_replication_coverage_v1'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    evidence=pd.read_csv(source/'candidate_donor_cell_sources.csv.gz'); flags=pd.read_csv(source/'candidate_sets.csv')
    coverage=evidence.groupby('Gene').gene_present.all().rename('gene_symbol_measured').reset_index()
    coverage=flags.merge(coverage,on='Gene',validate='one_to_one')
    coverage['interpretation']=np.where(coverage.gene_symbol_measured,'Measured; zero values are distinguishable from absence','Symbol absent; no alias mapping; expression and paired effect are unavailable')
    coverage.to_csv(out/'candidate_gene_coverage.csv',index=False)
    pairs=pd.read_csv(source/'paired_tumor_proximal_expression.csv')
    summaries=[]
    for (qc,gene),g in pairs.groupby(['qc','Gene']):
        finite=g[np.isfinite(g.log2_cpm_ratio)]
        summaries.append(dict(qc=qc,Gene=gene,matched_donor_record_n=g.donor.nunique(),valid_paired_donor_n=finite.donor.nunique(),
                              both_zero_donor_n=int(((finite.tumor_cpm==0)&(finite.normal_cpm==0)).sum()),
                              median_log2_cpm_ratio=finite.log2_cpm_ratio.median() if len(finite) else np.nan,
                              positive_donor_n=int((finite.log2_cpm_ratio>0).sum()),negative_donor_n=int((finite.log2_cpm_ratio<0).sum())))
    summaries=pd.DataFrame(summaries).merge(coverage,on='Gene',validate='many_to_one')
    assert (summaries.loc[~summaries.gene_symbol_measured,'valid_paired_donor_n']==0).all()
    summaries.to_csv(out/'paired_expression_summary_with_coverage.csv',index=False)
    old=pd.read_csv(source/'cross_study_candidate_expression.csv')
    available=old.dropna(); missing=coverage.loc[~coverage.gene_symbol_measured,'Gene'].tolist()
    finish_run(out,dict(status='passed_coverage_audit',upstream_run_sha256=sha256(source/'run.json'),script_sha256=sha256(Path(__file__)),
                        missing_candidate_symbols=missing,cross_study_valid_candidate_n=len(available),
                        note='Original paired_expression_summary.csv paired_donor_n counts structurally matched donor records including missing symbols. This table adds valid measured pair counts; absent symbols have no effect estimate. No model or candidate changes.'))
    target=root/'results/historical'/out.name
    if target.exists(): raise FileExistsError(target)
    target.mkdir()
    selected=[p for p in out.iterdir() if p.is_file()]
    for p in selected: shutil.copy2(p,target/p.name)
    (target/'snapshot.json').write_text(json.dumps(dict(type='compact_result_snapshot',upstream_output_dir=f'outputs/{out.name}',upstream_run_sha256=sha256(out/'run.json'),included_sha256={p.name:sha256(target/p.name) for p in selected}),indent=2)+'\n')
    print(json.dumps(dict(missing_symbols=missing,valid_cross_study_candidates=len(available)),indent=2))


if __name__=='__main__': main()
