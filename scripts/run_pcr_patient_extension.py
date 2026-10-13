"""Fit historical-config PCR and quantify TCGA development transfer sensitivity."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import run_depmap_baseline as baseline
import run_selective_dependency_benchmark as benchmark
from pcr_transfer_methods import (candidate_tables, cluster_bootstrap, finish_run, fit_pcr,
                                  overlap, predict_pcr, ranks_and_top, save_model, sha256, verify_manifest)
from runtime_paths import source_project_root


METHODS = {"nonkidney_mean_shift_clip0": "nonkidney_mean_shift_expression",
           "kidney_mean_shift_clip0": "kidney_mean_shift_expression",
           "raw_xena_log_expression": "raw_expression"}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=Path, default=source_project_root(root) / "data/processed/context_module_stage0/patients.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/pcr_patient_extension_v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not args.train and not args.dry_run:
        parser.error("请选择 --dry-run（输入检查）或 --train（正式开发分析）")
    started = time.monotonic()
    protocol_path = root / "configs/pcr_transfer_singlecell_protocol_20261010.json"
    protocol = json.loads(protocol_path.read_text())
    directory = root / "data/processed/depmap_baseline_24q4_v1"
    bridge_dir = root / "data/processed/tcga_kirc_expression_bridge_v1"
    prior_dir = root / "results/historical/selective_dependency_benchmark_v1"
    baseline.configure_device("cuda")
    torch.set_num_threads(4)
    models, genes, matrices, essential = baseline.load_data(directory)
    bridge_run = verify_manifest(bridge_dir)
    historical = json.loads((prior_dir / "run.json").read_text())
    if historical["input_sha256"]["matrices.npz"] != sha256(directory / "matrices.npz"):
        raise ValueError("历史选参输入与当前矩阵不同")
    config = historical["selected_parameters"]["Kidney"]["expression_pcr_ridge"]
    tuning = pd.read_csv(prior_dir / "tuning.csv")
    chosen = tuning[(tuning.heldout_lineage == "Kidney") & (tuning.method == "expression_pcr_ridge") & tuning.selected]
    if len(chosen) != 1 or chosen.iloc[0]["config"] != f"{config[0]}|{float(config[1])}":
        raise ValueError("历史PCR选参表与运行记录不一致")
    if config != [protocol["pcr"]["rank"], protocol["pcr"]["alpha"]]:
        raise ValueError("新协议与历史选参不一致")
    labels, patients = models.OncotreeLineage.to_numpy(), models.PatientID.to_numpy()
    kidney_patients = patients[labels == "Kidney"]
    train = np.flatnonzero((labels != "Kidney") & ~np.isin(patients, kidney_patients))
    historical_splits = pd.read_csv(prior_dir / "splits.csv.gz")
    prior_train = historical_splits[(historical_splits.heldout_lineage == "Kidney") & (historical_splits.role == "train")]
    if set(prior_train.ModelID) != set(models.index[train]):
        raise ValueError("历史Kidney外层训练模型与新PCR训练集合不同")
    with np.load(bridge_dir / "expression_inputs.npz", allow_pickle=False) as a:
        bridge = {k: a[k] for k in a.files}
    if not np.array_equal(bridge["genes"], genes):
        raise ValueError("桥接基因顺序不匹配")
    ids, splits = bridge["patient_ids"].astype(str), bridge["split"].astype(str)
    metadata = pd.read_csv(args.patients, index_col=0)
    if sha256(args.patients) != bridge_run["source_sha256"]["patients"]:
        raise ValueError("历史患者分组哈希发生变化")
    locked = set(metadata.index[metadata.split.eq("test")])
    if set(ids) & locked or set(splits) != {"train", "validation"} or not np.array_equal(metadata.loc[ids, "split"], splits):
        raise ValueError("Locked Test隔离或患者分组校验失败")
    covered = pd.read_csv(bridge_dir / "gene_mapping.csv.gz").tcga_covered.to_numpy(dtype=bool)
    observed = np.isfinite(matrices["dependency"][train]).sum(axis=0) >= 2
    universe = covered & ~essential & observed
    print(f"【固定输入】训练{len(train)}模型｜{models.iloc[train].PatientID.nunique()}患者｜TCGA {len(ids)}患者｜候选{universe.sum()}基因｜Test 0", flush=True)
    print(f"【固定参数】PCR rank={config[0]}｜alpha={config[1]}｜沿用历史训练内选参", flush=True)
    if args.dry_run:
        print("【检查通过】未训练、未预测、未写结果", flush=True)
        return
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖结果或未完成分析目录: {output}")
    output.mkdir(parents=True)
    print("【阶段1/4】GPU拟合可复用PCR模型", flush=True)
    model = fit_pcr(matrices["expression"][train], matrices["dependency"][train], *config)
    # Independently verify against the existing kernel-PCR implementation on actual inputs.
    check_rows = np.arange(min(12, len(ids)))
    reference_expression = bridge["nonkidney_mean_shift_expression"][check_rows]
    x = torch.as_tensor(matrices["expression"][train], dtype=torch.float64, device="cuda")
    h = torch.as_tensor(reference_expression, dtype=torch.float64, device="cuda")
    mean = torch.nan_to_num(torch.nanmean(x, dim=0), nan=0.0)
    x, h = torch.where(torch.isfinite(x), x, mean) - mean, torch.where(torch.isfinite(h), h, mean) - mean
    sd = torch.sqrt((x * x).mean(dim=0))
    keep = sd > 1e-8
    x, h = x[:, keep] / sd[keep], h[:, keep] / sd[keep]
    y = matrices["dependency"][train]
    ymean = baseline.observed_mean(y, axis=0)
    reference = benchmark.pcr_ridge_predictions(x @ x.T, h @ x.T, y - ymean, [config[0]], [config[1]])[tuple(config)]
    delta = np.nanmax(np.abs(reference - predict_pcr(model, reference_expression)))
    if delta > 1e-7:
        raise ValueError(f"独立核PCR等价检查失败: {delta}")
    print(f"【等价核验】与历史实现最大预测差 {delta:.3g}", flush=True)
    del x, h, reference
    save_model(model, output / "pcr_model.npz", genes, models.index[train].to_numpy(dtype=str))
    np.savez_compressed(output / "ranking_universe.npz", genes=genes, universe=universe)
    models.iloc[train].to_csv(output / "training_models.csv")
    primary = protocol["mapping"]["primary"]
    predictions, tables, rankings, tops, diagnostics, sensitivity = {}, [], [], {}, [], []
    for method, key in METHODS.items():
        prediction, diagnostic = predict_pcr(model, bridge[key], diagnostics=True)
        predictions[method] = prediction
        table, ranking, _, top = candidate_tables(prediction, ids, splits, genes, universe, method)
        tables.append(table)
        rankings.append(ranking)
        tops[method] = top
        diagnostics.append(pd.DataFrame({"patient_id": ids, "split": splits, "method": method, **diagnostic}))
    candidates = pd.concat(tables, ignore_index=True)
    candidates.to_csv(output / "candidate_summary.csv.gz", index=False)
    pd.concat(rankings).to_csv(output / "patient_top10.csv.gz", index=False)
    pd.concat(diagnostics).to_csv(output / "domain_diagnostics.csv", index=False)
    discovery = candidates[(candidates.method == primary) & (candidates.split == "train")].sort_values(
        ["top10_frequency", "mean_predicted_residual", "Gene"], ascending=[False, True, True]).head(20).copy()
    discovery.insert(0, "pcr_discovery_rank", np.arange(1, 21))
    discovery.to_csv(output / "pcr_discovery_top20.csv", index=False)
    target_columns = [int(np.flatnonzero(genes == gene)[0]) for gene in discovery.Gene]
    feature_weights = model["loadings"] @ model["beta"][:, target_columns]
    patient_x = torch.as_tensor(bridge[METHODS[primary]], dtype=torch.float64, device="cuda")
    patient_x = (torch.where(torch.isfinite(patient_x), patient_x, model["mean"]) - model["mean"]) / model["sd"]
    patient_x[:, ~model["feature_keep"]] = 0
    patient_x = patient_x - model["feature_center"]
    reconstructed = patient_x @ feature_weights + model["bias"][target_columns]
    attribution_error = float(torch.max(torch.abs(reconstructed-torch.as_tensor(predictions[primary][:, target_columns], device="cuda"))))
    if attribution_error > 1e-9:
        raise ValueError(f"PCR输入贡献重构失败: {attribution_error}")
    mean_abs = patient_x[splits == "train"].abs().mean(dim=0)[:, None] * feature_weights.abs()
    mean_signed = patient_x[splits == "train"].mean(dim=0)[:, None] * feature_weights
    feature_rows = []
    for j, target in enumerate(discovery.Gene):
        selected = torch.argsort(mean_abs[:, j], descending=True, stable=True)[:20].cpu().numpy()
        for importance_rank, column in enumerate(selected, 1):
            feature_rows.append({"target_gene": target, "feature_rank": importance_rank, "feature_gene": genes[column],
                                 "standardized_expression_coefficient": float(feature_weights[column, j]),
                                 "train_mean_absolute_prediction_contribution": float(mean_abs[column, j]),
                                 "train_mean_signed_prediction_contribution": float(mean_signed[column, j]),
                                 "interpretation": "linear_model_contribution_not_causal_mechanism"})
    pd.DataFrame(feature_rows).to_csv(output / "pcr_candidate_input_features.csv", index=False)
    stability = []
    for method in METHODS:
        table = candidates[candidates.method == method].pivot(index="Gene", columns="split", values="top10_frequency")
        active = (table.train > 0) | (table.validation > 0)
        train_top = set(table.train[table.train > 0].sort_values(ascending=False, kind="stable").head(100).index)
        validation_top = set(table.validation[table.validation > 0].sort_values(ascending=False, kind="stable").head(100).index)
        stability.append({"method": method, "active_gene_n": int(active.sum()),
                          "active_frequency_spearman": spearmanr(table.loc[active, "train"], table.loc[active, "validation"]).statistic,
                          "positive_top100_train_n": len(train_top), "positive_top100_validation_n": len(validation_top),
                          "positive_top100_shared_n": len(train_top & validation_top),
                          "positive_top100_jaccard": len(train_top & validation_top) / len(train_top | validation_top),
                          "top100_frequency_overlap": len(train_top & validation_top) / 100 if min(len(train_top), len(validation_top)) == 100 else np.nan})
        if method != primary:
            agreement = overlap(tops[primary], tops[method])
            sensitivity.append(pd.DataFrame({"patient_id": ids, "split": splits, "comparison": f"{method}_vs_{primary}",
                                             "top10_overlap": agreement}))
    pd.DataFrame(stability).to_csv(output / "cohort_stability.csv", index=False)
    pd.concat(sensitivity).to_csv(output / "mapping_sensitivity.csv", index=False)
    np.savez_compressed(output / "predicted_residuals.npz", genes=genes, patient_ids=ids, split=splits,
                        **{k: v.astype(np.float32) for k, v in predictions.items()})
    print("【阶段2/4】三种映射预测完成，Train前20已固定", flush=True)
    draws, seed = protocol["uncertainty"]["draws"], protocol["uncertainty"]["seed"]
    rng = np.random.default_rng(seed)
    columns = np.flatnonzero(universe)
    base_ranks, _ = ranks_and_top(predictions[primary], universe)
    accumulators = {kind: {"selection": np.zeros((len(ids), len(columns)), dtype=np.uint16), "ranks": []}
                    for kind in ["source_patient_bootstrap", "tcga_train_mapping_bootstrap"]}
    boot_overlap, cohort = [], []
    tcga_train = np.flatnonzero(splits == "train")
    reference_mean = pd.read_csv(bridge_dir / "gene_mapping.csv.gz").depmap_nonkidney_mean.to_numpy()
    discovery_columns = [np.flatnonzero(genes[columns] == g)[0] for g in discovery.Gene]
    for draw in range(draws):
        sampled = cluster_bootstrap(models, train, rng)
        boot_model = fit_pcr(matrices["expression"][sampled], matrices["dependency"][sampled], *config)
        source_pred = predict_pcr(boot_model, bridge[METHODS[primary]])
        del boot_model
        tcga_sampled = rng.choice(tcga_train, size=len(tcga_train), replace=True)
        shift = reference_mean - baseline.observed_mean(bridge["raw_expression"][tcga_sampled], axis=0)
        mapped = np.maximum(bridge["raw_expression"] + shift, 0)
        mapping_pred = predict_pcr(model, mapped)
        for kind, prediction in [("source_patient_bootstrap", source_pred), ("tcga_train_mapping_bootstrap", mapping_pred)]:
            rank, top = ranks_and_top(prediction, universe)
            accumulators[kind]["selection"] += (rank <= 10)
            accumulators[kind]["ranks"].append(rank[:, discovery_columns].astype(np.uint16))
            agreement = overlap(tops[primary], top)
            boot_overlap.append(pd.DataFrame({"draw": draw, "kind": kind, "patient_id": ids, "split": splits,
                                               "top10_overlap": agreement}))
            freq = (rank[splits == "train"] <= 10).mean(axis=0)
            for j in discovery_columns:
                cohort.append({"draw": draw, "kind": kind, "Gene": genes[columns[j]], "train_top10_frequency": freq[j]})
        print(f"【阶段3/4】患者重采样 {draw+1}/{draws}｜累计{time.monotonic()-started:.1f}秒", flush=True)
    uncertainty = []
    for kind, accumulator in accumulators.items():
        frequency = accumulator["selection"] / draws
        np.savez_compressed(output / f"{kind}_inclusion.npz", patient_ids=ids, split=splits,
                            genes=genes[columns], selection_count=accumulator["selection"], draws=draws)
        rank_draws = np.stack(accumulator["ranks"])
        for i, patient in enumerate(ids):
            for j, gene in enumerate(discovery.Gene):
                local = discovery_columns[j]
                rank_values = rank_draws[:, i, j]
                uncertainty.append({"kind": kind, "patient_id": patient, "split": splits[i], "Gene": gene,
                                    "base_ordinal_rank": int(base_ranks[i, local]),
                                    "top10_inclusion_frequency": frequency[i, local],
                                    "rank_p025": np.quantile(rank_values, .025), "rank_median": np.median(rank_values),
                                    "rank_p975": np.quantile(rank_values, .975), "draws": draws})
    pd.DataFrame(uncertainty).to_csv(output / "candidate_patient_uncertainty.csv.gz", index=False)
    pd.concat(boot_overlap).to_csv(output / "bootstrap_patient_overlap.csv.gz", index=False)
    pd.DataFrame(cohort).to_csv(output / "bootstrap_cohort_frequency.csv.gz", index=False)
    source_paths = [protocol_path, args.patients, directory / "matrices.npz", directory / "models.csv",
                    directory / "gene_coverage.csv", bridge_dir / "expression_inputs.npz", prior_dir / "run.json",
                    prior_dir / "tuning.csv", prior_dir / "splits.csv.gz", Path(__file__), root / "scripts/pcr_transfer_methods.py"]
    finish_run(output, {"status": "complete_development_analysis", "protocol_sha256": sha256(protocol_path),
                        "input_sha256": {str(p): sha256(p) for p in source_paths},
                        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "numpy": np.__version__,
                        "dtype": "float64_fit_float32_prediction_storage", "elapsed_seconds": time.monotonic()-started,
                        "training_model_n": len(train), "training_patient_n": models.iloc[train].PatientID.nunique(),
                        "patient_n": len(ids), "locked_test_overlap": 0, "ranking_gene_n": int(universe.sum()),
                        "pcr_rank": model["rank"], "alpha": model["alpha"], "bootstrap_draws": draws,
                        "historical_kernel_prediction_max_difference": float(delta),
                        "feature_contribution_reconstruction_max_difference": attribution_error,
                        "candidate_policy": "Train-only PCR development top20, separate from historical frozen top20",
                        "limitations": protocol["limitations"]})
    print(f"【阶段4/4 完成】结果{output}｜耗时{time.monotonic()-started:.1f}秒", flush=True)


if __name__ == "__main__":
    main()
