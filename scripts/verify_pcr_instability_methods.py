"""Independent numerical and ranking checks for the instability audit."""
import numpy as np
import torch
from pathlib import Path
from pcr_transfer_methods import fit_pcr, predict_pcr, tensor
from pcr_instability_methods import *


def main():
    torch.set_num_threads(4)
    assert case_path(Path('outputs'), 'Bladder/Urinary Tract').parent==Path('outputs')
    assert case_path(Path('outputs'), 'Head and Neck').name=='Head%20and%20Neck'
    rng=np.random.default_rng(41)
    x=rng.normal(size=(35,70)); hx=rng.normal(size=(4,70))
    x[:,0]=2; hx[:,0]=2
    y=rng.normal(size=(35,65)); y[:3,4]=np.nan
    base=fit_pcr(x,y,12,5.)
    bp=predict_pcr(base,hx)
    z=standardized(base,x); hz=standardized(base,hx)
    preds,spaces,_=paired_predictions(base,base,bp,z,hz,np.arange(35),x,hx,y)
    for name,pred in preds.items():
        np.testing.assert_allclose(pred,bp,atol=1e-10,rtol=1e-10,err_msg=name)
    # Independent NumPy SVD + augmented ridge regression, with unpenalized intercept.
    zn=z.cpu().numpy(); hzn=hz.cpu().numpy()
    _,_,vt=np.linalg.svd(zn,full_matrices=False)
    sx=zn@vt[:12].T; sh=hzn@vt[:12].T
    reference=base['gene_mean'].cpu().numpy()
    expected=np.empty_like(bp)
    for j in range(y.shape[1]):
        mask=np.isfinite(y[:,j]); design=np.column_stack([np.ones(mask.sum()),sx[mask]])
        penalty=np.diag([0.]+[5.]*12)
        coefficients=np.linalg.solve(design.T@design+penalty,design.T@(y[mask,j]-reference[j]))
        expected[:,j]=np.column_stack([np.ones(len(hx)),sh])@coefficients
    np.testing.assert_allclose(bp,expected,atol=1e-10,rtol=1e-10)
    q,_=np.linalg.qr(rng.normal(size=(12,12)))
    diagnostics=subspace_diagnostics(base['loadings'],base['loadings']@tensor(q))
    assert abs(diagnostics['mean_squared_canonical_correlation']-1)<1e-12
    sampled=rng.choice(35,35,replace=True)
    boot=fit_pcr(x[sampled],y[sampled],12,5.)
    preds,_,shift=paired_predictions(base,boot,bp,z,hz,sampled,x[sampled],hx,y[sampled])
    np.testing.assert_allclose(preds[VARIANTS[1]]-preds[VARIANTS[0]],np.broadcast_to(shift,bp.shape),atol=1e-14)
    masks=np.ones(bp.shape,dtype=bool)
    cross,error,total=prediction_components(preds,bp,masks)
    np.testing.assert_allclose(cross.sum(),total,atol=1e-12)
    truth=rng.normal(size=bp.shape)
    _,top=masked_ranks(bp,masks)
    metrics=fast_functional_metrics(truth,bp,masks,top)
    for row in range(len(bp)):
        oracle=np.argsort(truth[row],kind='stable')[:10]
        selected=np.argsort(bp[row],kind='stable')[:10]
        discounts=1/np.log2(np.arange(2,12))
        expected=(np.maximum(0,-truth[row,selected])*discounts).sum()/(np.maximum(0,-truth[row,oracle])*discounts).sum()
        np.testing.assert_allclose(metrics['ndcg_at_10'][row],expected)
        assert metrics['true_top10_overlap'][row]==len(set(oracle)&set(selected))/10
    # Constant outcome shifts change neither common-reference raw predictions nor geometry.
    offset=np.linspace(-.2,.2,65)
    shifted=fit_pcr(x,y+offset,12,5.)
    np.testing.assert_allclose(predict_pcr(shifted,hx),bp,atol=1e-10)
    print('PASS: independent SVD, six identity controls, rotation invariance, reference alignment, cross-products, functional scoring',flush=True)


if __name__=='__main__': main()
