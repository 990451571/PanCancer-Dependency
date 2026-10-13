"""Lock GSE178481 author metadata and untreated ccRCC count inputs."""
from pathlib import Path
import gzip
import json
import re
from prepare_ccrcc_singlecell import download, soft_samples
from pcr_transfer_methods import sha256


def main():
    root=Path(__file__).resolve().parents[1]
    raw=root/'data/raw/gse178481'; raw.mkdir(parents=True,exist_ok=True)
    manifest=raw/'source_manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        for n,v in m['files'].items(): assert sha256(raw/n)==v['sha256']
        print('Existing inputs verified'); return
    urls={'cell.ano.csv':'https://dl.dropboxusercontent.com/s/i2cjyqghs6jwaiu/cell.ano.csv',
          'GSE178481_family.soft.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE178nnn/GSE178481/soft/GSE178481_family.soft.gz',
          'author_README.md':'https://raw.githubusercontent.com/shenglinmei/ccRCC.analysis/main/README.md'}
    for n,u in urls.items(): download(u,raw/n)
    samples=soft_samples(raw/'GSE178481_family.soft.gz')
    samples['donor']=samples.sample_id.str.extract(r'RCC-([^-]+)-')[0]
    samples['scope']=samples.donor.map(lambda x:'excluded' if x in ['PR1','PR9'] else ('metastatic_status' if x.startswith('BM') else 'primary'))
    samples['exclusion_reason']=samples.donor.map({'PR1':'pRCC','PR9':'chemotherapy/thrombus donor'})
    samples['filename']=''
    links={}
    for line in gzip.open(raw/'GSE178481_family.soft.gz','rt'):
        if line.startswith('!Sample_supplementary_file_1 = '):
            u=line.strip().split(' = ',1)[1].replace('ftp://','https://')
            accession=re.search(r'/(GSM\d+)_',u)[1]; links[accession]=u
    for i,row in samples.iterrows():
        if row.scope=='excluded': continue
        assert row.Stage=='ccRCC' and row.treatment=='No'
        assert ('Normal' in row.sample_id)==(row.condition=='Normal')
        u=links[row.accession]; n=u.rsplit('/',1)[1]; urls[n]=u; samples.loc[i,'filename']=n
        print('Download '+n,flush=True); download(u,raw/n)
    samples.to_csv(raw/'samples.csv',index=False)
    record={'dataset':'GSE178481','files':{n:{'url':u,'sha256':sha256(raw/n),'bytes':(raw/n).stat().st_size} for n,u in urls.items()},
            'samples_sha256':sha256(raw/'samples.csv'),'protocol_sha256':sha256(root/'configs/ccrcc_source_replication_protocol_20261010.json')}
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('Inputs locked',flush=True)


if __name__=='__main__': main()
