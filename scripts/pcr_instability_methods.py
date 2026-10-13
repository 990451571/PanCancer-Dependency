"""GPU conditional PCR refits and reference-aligned ranking diagnostics.

The frozen transfer module is not modified: v1 artifacts remain verifiable.
"""
from __future__ import annotations

import numpy as np
import torch
from urllib.parse import quote

from pcr_transfer_methods import tensor


VARIANTS = ("full_refit_local_reference", "full_refit_common_reference", "reference_shift_only",
            "fixed_geometry_beta_refit", "fixed_scaler_pca_beta_refit", "refit_scaler_fixed_pca_beta_refit")
COMPONENTS = ("regression_given_geometry", "pca_given_fixed_scaler", "scaler_propagated_to_pca", "target_reference")


def case_path(directory, case):
    return directory / quote(case, safe="")


def standardized(model, expression):
    x = tensor(expression)
    z = (torch.where(torch.isfinite(x), x, model["mean"]) - model["mean"]) / model["sd"]
    z[:, ~model["feature_keep"]] = 0
    return z - model["feature_center"]


def refit_scores(x, hx, dependency, reference_mean, rank, alpha):
    y = tensor(dependency) - reference_mean
    observed = np.isfinite(dependency)
    _, groups = np.unique(np.packbits(observed.T, axis=1), axis=0, return_inverse=True)
    prediction = torch.full((len(hx), dependency.shape[1]), torch.nan, dtype=torch.float64, device="cuda")
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed[:, columns[0]])
        if len(rows) < 2:
            continue
        r = min(rank, x.shape[1], len(rows)-1)
        design = x[rows, :r]
        center = design.mean(dim=0)
        design = design - center
        values = y[rows][:, columns]
        mean = values.mean(dim=0)
        beta = torch.linalg.solve(design.T @ design + alpha*torch.eye(r, dtype=torch.float64, device="cuda"),
                                   design.T @ (values-mean))
        prediction[:, columns] = (hx[:, :r]-center) @ beta + mean
    return prediction.cpu().numpy()


def refit_fixed_scaler_pca(base, train_z, held_z, sampled, dependency):
    x = train_z[sampled]
    center = x.mean(dim=0)
    x, hx = x-center, held_z-center
    eig, vectors = torch.linalg.eigh(x @ x.T)
    order = torch.argsort(eig, descending=True)
    r = min(base["requested_rank"], int((eig > 1e-8).sum()))
    if r < 1:
        raise ValueError("重采样没有有效PCA分量")
    eig, vectors = eig[order][:r], vectors[:, order][:, :r]
    loadings = (x.T @ vectors) / torch.sqrt(eig)[None, :]
    scores = vectors*torch.sqrt(eig)[None, :]
    held_scores = hx @ loadings
    prediction = refit_scores(scores, held_scores, dependency, base["gene_mean"], base["requested_rank"], base["alpha"])
    return prediction, loadings


def subspace_diagnostics(left, right):
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(singular))
    return {"left_rank": left.shape[1], "right_rank": right.shape[1],
            "mean_squared_canonical_correlation": float((singular*singular).sum()/max(left.shape[1], right.shape[1])),
            "mean_principal_angle_degrees": float(angles.mean()),
            "largest_principal_angle_degrees": float(angles.max())}


def paired_predictions(base, boot, base_prediction, train_z, held_z, sampled, sampled_expression, held_expression, dependency):
    full_local = (standardized(boot, held_expression) @ boot["loadings"] @ boot["beta"] + boot["bias"]).cpu().numpy()
    shift = (boot["gene_mean"]-base["gene_mean"]).cpu().numpy()
    fixed_scores, fixed_held_scores = train_z @ base["loadings"], held_z @ base["loadings"]
    fixed_prediction = refit_scores(fixed_scores[sampled], fixed_held_scores, dependency,
                                    base["gene_mean"], base["requested_rank"], base["alpha"])
    fixed_scaler_prediction, pca = refit_fixed_scaler_pca(base, train_z, held_z, sampled, dependency)
    new_scores = standardized(boot, sampled_expression) @ base["loadings"]
    new_held_scores = standardized(boot, held_expression) @ base["loadings"]
    scaler_only = refit_scores(new_scores, new_held_scores, dependency, base["gene_mean"],
                               base["requested_rank"], base["alpha"])
    predictions = dict(zip(VARIANTS, [full_local, full_local+shift, base_prediction-shift,
                                      fixed_prediction, fixed_scaler_prediction, scaler_only]))
    subspaces = {"fixed_scaler_refit_pca": subspace_diagnostics(base["loadings"], pca),
                 "full_refit_pca_standardized_coordinates": subspace_diagnostics(base["loadings"], boot["loadings"])}
    original_keep = base["feature_keep"] & boot["feature_keep"]
    subspaces["scaler"] = {"standardized_mean_drift_rmse": float(torch.sqrt((((boot["mean"]-base["mean"])/base["sd"])[original_keep]**2).mean())),
                            "log_sd_ratio_rmse": float(torch.sqrt((torch.log(boot["sd"][original_keep]/base["sd"][original_keep])**2).mean()))}
    return predictions, subspaces, shift


def masked_ranks(values, masks, max_k=50):
    if np.any(masks & ~np.isfinite(values)):
        raise ValueError("共同排序空间出现非有限预测")
    if np.any(masks.sum(axis=1) <= max_k):
        raise ValueError("共同排序空间过小")
    work = np.where(masks, values, np.inf)
    order = np.argsort(work, axis=1, kind="stable")
    top = order[:, :max_k]
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, values.shape[1]+1)[None, :], axis=1)
    return ranks, top


def overlap_at_k(left, right, k):
    return np.array([len(set(a[:k]) & set(b[:k]))/k for a,b in zip(left,right)])


def boundary_diagnostics(values, masks, top, ks=(10,20,50)):
    columns = np.argsort(np.where(masks, values, np.inf), axis=1, kind="stable")[:, :max(ks)+1]
    rows = np.arange(len(values))
    result = {}
    for k in ks:
        cutoff = values[rows, columns[:, k-1]]
        gap = values[rows, columns[:, k]]-cutoff
        near = (masks & (np.abs(values-cutoff[:, None]) <= .01)).sum(axis=1)
        result[k] = {"boundary_gap": gap, "near_boundary_gene_n": near, "exact_boundary_tie": gap == 0}
    return result


def fast_functional_metrics(truth, prediction, masks, top=None):
    if top is None:
        _, top = masked_ranks(prediction, masks)
    rows = np.arange(len(truth))[:, None]
    selected = truth[rows, top[:, :10]]
    oracle = np.argsort(np.where(masks, truth, np.inf), axis=1, kind="stable")[:, :10]
    best = truth[rows, oracle]
    discounts = 1/np.log2(np.arange(2,12))
    ideal = (np.maximum(0,-best)*discounts).sum(axis=1)
    ndcg = np.divide((np.maximum(0,-selected)*discounts).sum(axis=1), ideal,
                      out=np.full(len(truth),np.nan), where=ideal>0)
    return {"ndcg_at_10": ndcg, "true_top10_overlap": overlap_at_k(top,oracle,10),
            "selective_precision_at_10": (selected <= -.5).mean(axis=1)}


def prediction_components(predictions, base_prediction, masks):
    a = predictions["fixed_geometry_beta_refit"]-base_prediction
    b = predictions["fixed_scaler_pca_beta_refit"]-predictions["fixed_geometry_beta_refit"]
    c = predictions["full_refit_common_reference"]-predictions["fixed_scaler_pca_beta_refit"]
    d = predictions["full_refit_local_reference"]-predictions["full_refit_common_reference"]
    components = np.stack([a[masks],b[masks],c[masks],d[masks]])
    total = (predictions["full_refit_local_reference"]-base_prediction)[masks]
    error = float(np.max(np.abs(components.sum(axis=0)-total)))
    if error > 1e-10:
        raise ValueError(f"条件对照预测重构失败: {error}")
    gpu = tensor(components)
    cross = (gpu @ gpu.T / gpu.shape[1]).cpu().numpy()
    return cross, error, float(np.mean(total*total))
