"""Small independent numerical and identity checks for the new extension."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch

from pcr_transfer_methods import cluster_bootstrap, fit_pcr, load_model, predict_pcr, save_model


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(32, 12))
    y = x @ rng.normal(size=(12, 4)) + rng.normal(size=(32, 4)) * .02
    y[:7, 1] = np.nan
    y[:, 3] = np.nan
    x[:, -1] = 5
    hx = rng.normal(size=(7, 12))
    hx[0, 2] = np.nan
    rank, alpha = 6, 8.0
    model = fit_pcr(x, y, rank, alpha)
    actual = predict_pcr(model, hx)
    # Independent NumPy SVD and primal closed-form ridge, including missing labels.
    mean, sd = x.mean(axis=0), x.std(axis=0)
    keep = sd > 1e-8
    z = (x[:, keep] - mean[keep]) / sd[keep]
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = u[:, :rank] * s[:rank]
    h = np.where(np.isfinite(hx), hx, mean)
    hscores = ((h[:, keep] - mean[keep]) / sd[keep]) @ vt[:rank].T
    expected = np.full_like(actual, np.nan)
    for gene in range(3):
        observed = np.isfinite(y[:, gene])
        ym = y[observed, gene].mean()
        xm = scores[observed].mean(axis=0)
        design = scores[observed] - xm
        beta = np.linalg.solve(design.T @ design + alpha*np.eye(rank), design.T @ (y[observed, gene]-ym))
        expected[:, gene] = (hscores-xm) @ beta
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-10, rtol=1e-10)
    assert np.isnan(actual[:, 3]).all()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.npz"
        save_model(model, path, np.asarray([f"g{i}" for i in range(12)]), np.asarray([f"m{i}" for i in range(32)]))
        loaded, genes = load_model(path)
        np.testing.assert_allclose(predict_pcr(loaded, hx), actual, atol=0, rtol=0, equal_nan=True)
        assert len(genes) == 12
    metadata = pd.DataFrame({"PatientID": ["a", "a", "b", "c", "d", "d"],
                             "OncotreeLineage": ["L1", "L1", "L1", "L2", "L2", "L2"]}, index=list("ABCDEF"))
    for _ in range(30):
        draw = cluster_bootstrap(metadata, np.arange(6), rng)
        counts = np.bincount(draw, minlength=6)
        assert counts[0] == counts[1] and counts[4] == counts[5]
    from run_ccrcc_singlecell_extension import gpu_group_sum
    counts = sparse.csr_matrix([[1, 0, 3], [0, 2, 4], [1, 0, 0], [0, 5, 1]])
    codes = np.array([0, 1, 0, 1])
    total, detection = gpu_group_sum(counts, codes, 2)
    np.testing.assert_array_equal(total, [[2, 0, 3], [0, 7, 5]])
    np.testing.assert_array_equal(detection, [[2, 0, 1], [0, 2, 2]])
    print("【验证通过】独立SVD/缺失标签/零方差特征/模型保存读取/患者分组重采样/稀疏GPU计数与检出率")


if __name__ == "__main__":
    main()
