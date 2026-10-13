"""Reusable GPU PCR, patient ranking and provenance for development transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_depmap_baseline import observed_mean


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory, filename="run.json"):
    record = json.loads((Path(directory) / filename).read_text())
    for name, expected in record["output_sha256"].items():
        if sha256(Path(directory) / name) != expected:
            raise ValueError(f"上游哈希校验失败: {directory}/{name}")
    return record


def finish_run(directory, record):
    record["output_sha256"] = {p.name: sha256(p) for p in sorted(Path(directory).iterdir())
                               if p.is_file() and p.name != "run.json"}
    (Path(directory) / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def tensor(values):
    return torch.as_tensor(values, dtype=torch.float64, device="cuda")


def fit_pcr(expression, dependency, rank, alpha):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动退回CPU")
    x = tensor(expression)
    mean = torch.nan_to_num(torch.nanmean(x, dim=0), nan=0.0)
    x = torch.where(torch.isfinite(x), x, mean) - mean
    sd = torch.sqrt((x * x).mean(dim=0))
    keep = sd > 1e-8
    safe_sd = torch.where(keep, sd, torch.ones_like(sd))
    x = x / safe_sd
    x[:, ~keep] = 0
    # Center again to match historical kernel PCR, including roundoff.
    feature_center = x.mean(dim=0)
    x = x - feature_center
    kernel = x @ x.T
    eig, vectors = torch.linalg.eigh(kernel)
    order = torch.argsort(eig, descending=True)
    eig, vectors = eig[order], vectors[:, order]
    actual_rank = min(rank, int((eig > 1e-8).sum()))
    if actual_rank < 1:
        raise ValueError("无可用主成分")
    eig, vectors = eig[:actual_rank], vectors[:, :actual_rank]
    scores = vectors * torch.sqrt(eig)[None, :]
    loadings = (x.T @ vectors) / torch.sqrt(eig)[None, :]
    outcome = tensor(dependency)
    # Preserve the historical float32-input target centering convention exactly.
    gene_mean = tensor(observed_mean(dependency, axis=0))
    residual = outcome - gene_mean
    beta = torch.full((actual_rank, dependency.shape[1]), torch.nan, dtype=torch.float64, device="cuda")
    bias = torch.full((dependency.shape[1],), torch.nan, dtype=torch.float64, device="cuda")
    observed = np.isfinite(dependency)
    _, groups = np.unique(np.packbits(observed.T, axis=1), axis=0, return_inverse=True)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed[:, columns[0]])
        if len(rows) < 2:
            continue
        used_rank = min(actual_rank, len(rows) - 1)
        design = scores[rows, :used_rank]
        design_mean = design.mean(dim=0)
        centered = design - design_mean
        y = residual[rows][:, columns]
        y_mean = y.mean(dim=0)
        b = torch.linalg.solve(centered.T @ centered + alpha * torch.eye(used_rank, device="cuda", dtype=torch.float64),
                               centered.T @ (y - y_mean))
        beta[:, columns] = 0
        beta[:used_rank, columns] = b
        bias[columns] = y_mean - design_mean @ b
    represented = (scores * scores).sum(dim=1)
    reconstruction = torch.sqrt(torch.clamp((x * x).sum(dim=1) - represented, min=0) / keep.sum())
    return {"mean": mean, "sd": safe_sd, "feature_keep": keep, "feature_center": feature_center,
            "loadings": loadings, "beta": beta, "bias": bias, "gene_mean": gene_mean,
            "low": torch.quantile(x, .01, dim=0), "high": torch.quantile(x, .99, dim=0),
            "training_reconstruction_rmse": reconstruction,
            "rank": actual_rank, "requested_rank": rank, "alpha": alpha}


def predict_pcr(model, expression, diagnostics=False):
    values = tensor(expression)
    finite = torch.isfinite(values)
    z = (torch.where(finite, values, model["mean"]) - model["mean"]) / model["sd"]
    z[:, ~model["feature_keep"]] = 0
    z = z - model["feature_center"]
    scores = z @ model["loadings"]
    prediction = (scores @ model["beta"] + model["bias"]).cpu().numpy()
    if not diagnostics:
        return prediction
    keep = model["feature_keep"]
    available = finite[:, keep]
    outside = ((z[:, keep] < model["low"][keep]) | (z[:, keep] > model["high"][keep])) & available
    represented = (scores * scores).sum(dim=1)
    reconstruction = torch.sqrt(torch.clamp((z * z).sum(dim=1) - represented, min=0) / keep.sum())
    training = model["training_reconstruction_rmse"]
    return prediction, {
        "out_of_reference_fraction": (outside.sum(dim=1) / available.sum(dim=1).clamp_min(1)).cpu().numpy(),
        "missing_feature_fraction": (~finite[:, keep]).double().mean(dim=1).cpu().numpy(),
        "pca_reconstruction_rmse": reconstruction.cpu().numpy(),
        "training_reconstruction_percentile": ((training[None, :] <= reconstruction[:, None]).double().mean(dim=1)).cpu().numpy(),
    }


def save_model(model, path, genes, train_ids):
    values = {k: v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k, v in model.items()}
    np.savez_compressed(path, genes=genes, train_ids=train_ids, **values)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        genes = archive["genes"].copy()
        result = {}
        for name in archive.files:
            if name in {"genes", "train_ids"}:
                continue
            value = archive[name]
            if name in {"rank", "requested_rank"}:
                result[name] = int(value)
            elif name == "alpha":
                result[name] = float(value)
            elif name == "feature_keep":
                result[name] = torch.as_tensor(value, device="cuda", dtype=torch.bool)
            else:
                result[name] = tensor(value)
    return result, genes


def ranks_and_top(predictions, universe, top_k=10):
    columns = np.flatnonzero(universe)
    selected_values = predictions[:, columns]
    if not np.isfinite(selected_values).all():
        raise ValueError("候选空间中存在非有限预测")
    # Unique ordinal ranks, ties resolved by stable original gene order.
    order = np.argsort(selected_values, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, len(columns) + 1)[None, :], axis=1)
    return ranks, columns[order[:, :top_k]]


def candidate_tables(predictions, patient_ids, splits, genes, universe, method):
    ranks, top = ranks_and_top(predictions, universe)
    columns = np.flatnonzero(universe)
    selected = ranks <= 10
    rows = []
    for split in ["train", "validation"]:
        use = splits == split
        for j, column in enumerate(columns):
            rows.append({"method": method, "split": split, "Gene": genes[column], "patient_n": int(use.sum()),
                         "top10_frequency": float(selected[use, j].mean()),
                         "mean_predicted_residual": float(predictions[use, column].mean()),
                         "mean_ordinal_rank": float(ranks[use, j].mean())})
    ranking = pd.DataFrame([{"method": method, "patient_id": patient_ids[i], "split": splits[i],
                             "rank": j + 1, "Gene": genes[c], "predicted_residual": predictions[i, c]}
                            for i in range(len(top)) for j, c in enumerate(top[i])])
    return pd.DataFrame(rows), ranking, ranks, top


def overlap(left, right):
    return np.array([len(set(a) & set(b)) / len(a) for a, b in zip(left, right)])


def cluster_bootstrap(models, train, rng):
    draws = []
    source = models.iloc[train]
    # This estimator keeps the original lineage mixture and samples patients within lineage.
    if source.groupby("PatientID").OncotreeLineage.nunique().max() != 1:
        raise ValueError("训练患者跨癌系，不能进行按癌系分层患者bootstrap")
    for _, group in source.groupby("OncotreeLineage", sort=True):
        patients = group.PatientID.unique()
        for patient in rng.choice(patients, size=len(patients), replace=True):
            draws.extend(models.index.get_indexer(group.index[group.PatientID.eq(patient)]))
    return np.asarray(draws, dtype=int)
