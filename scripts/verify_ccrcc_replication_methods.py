"""Check streamed CUDA counts and overlapping groups against direct NumPy sums."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from run_ccrcc_source_replication import read_sample


def main():
    genes=[f'MT-X{i}' if i<5 else f'GENE{i}' for i in range(205)]
    counts=np.ones((205,3),dtype=int); counts[:5,1]=100; counts[2:,2]=0
    barcodes=['s_A','s_B','s_C']
    totals=counts.sum(axis=0); detected=(counts>0).sum(axis=0); mt=counts[:5].sum(axis=0)/totals
    annotation=pd.DataFrame({'ano.l1':['Tumor','Proximal T','Macro'],'TotalUMI':totals,'gene.num':detected,'MT.ratio':mt},index=barcodes)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'counts.csv.gz'
        pd.DataFrame(counts,index=genes,columns=barcodes).to_csv(path)
        names,sums,detects,meta,cells,errors=read_sample(path,annotation)
    assert cells.extra_qc.tolist()==[True,False,False]
    np.testing.assert_array_equal(names,genes)
    for i,row in meta.iterrows():
        mask=np.ones(3,bool) if row.qc=='author_only' else cells.extra_qc.to_numpy()
        if row.group=='nonmalignant': mask &= ~cells['ano.l1'].eq('Tumor').to_numpy()
        elif row.group!='all_cells': mask &= cells.group.eq(row.group).to_numpy()
        np.testing.assert_array_equal(sums[i],counts[:,mask].sum(axis=1))
        np.testing.assert_array_equal(detects[i],(counts[:,mask]>0).sum(axis=1))
    assert errors=={'umi_error':0.,'detected_error':0.,'mt_fraction_error':0.}
    print('PASS: streamed raw-count audit, overlapping GPU sums, detection counts, QC masks')


if __name__=='__main__': main()
