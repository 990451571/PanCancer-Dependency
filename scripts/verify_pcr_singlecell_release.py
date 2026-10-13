"""Verify the computational extension, compact snapshots and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcr_transfer_methods import load_model, predict_pcr, ranks_and_top, sha256, verify_manifest


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-only", action="store_true", help="Verify versioned files without large local inputs or GPU")
    parser.add_argument("--report", type=Path, help="Write a new JSON verification record; refuse overwrite")
    args = parser.parse_args()
    names = ["pcr_patient_extension_v1", "ccrcc_singlecell_extension_v1", "pcr_singlecell_synthesis_v1"]
    snapshot_files = 0
    for name in names:
        path = root / "results/historical" / name
        manifest = json.loads((path / "snapshot.json").read_text())
        assert manifest["upstream_run_sha256"] == sha256(path / "run.json")
        for filename, expected in manifest["included_sha256"].items():
            assert sha256(path / filename) == expected, filename
            if not args.snapshots_only:
                assert sha256(root / "outputs" / name / filename) == expected
            snapshot_files += 1
    if args.snapshots_only:
        print(json.dumps({"status": "passed", "snapshot_file_n": snapshot_files,
                          "scope": "versioned compact snapshots only; no GPU, model fit or large input read"}, indent=2))
        return
    records = {name: verify_manifest(root / "outputs" / name) for name in names}
    pcr_dir = root / "outputs/pcr_patient_extension_v1"
    sc_dir = root / "outputs/ccrcc_singlecell_extension_v1"
    pcr_run = records[names[0]]
    for path, expected in pcr_run["input_sha256"].items():
        assert sha256(Path(path)) == expected, path
    sc_run = records[names[1]]
    for name, expected in sc_run["script_sha256"].items():
        assert sha256(root / "scripts" / name) == expected
    assert sc_run["input_sha256"]["pcr_run"] == sha256(pcr_dir / "run.json")
    assert pcr_run["protocol_sha256"] == sc_run["protocol_sha256"] == sha256(root / "configs/pcr_transfer_singlecell_protocol_20261010.json")
    model, genes = load_model(pcr_dir / "pcr_model.npz")
    with np.load(root / "data/processed/tcga_kirc_expression_bridge_v1/expression_inputs.npz", allow_pickle=False) as a:
        bridge = {k: a[k] for k in a.files}
    with np.load(pcr_dir / "ranking_universe.npz", allow_pickle=False) as a:
        universe = a["universe"].copy()
    with np.load(pcr_dir / "predicted_residuals.npz", allow_pickle=False) as a:
        prediction_archive = {k: a[k] for k in a.files}
    assert np.array_equal(genes, bridge["genes"])
    assert np.array_equal(bridge["patient_ids"], prediction_archive["patient_ids"])
    patient_path = next(Path(p) for p in pcr_run["input_sha256"] if p.endswith("/patients.csv"))
    patients = pd.read_csv(patient_path, index_col=0)
    ids, splits = bridge["patient_ids"], bridge["split"]
    assert len(set(ids)) == 209 and set(splits) == {"train", "validation"}
    assert not set(ids) & set(patients.index[patients.split.eq("test")])
    training = pd.read_csv(pcr_dir / "training_models.csv", index_col=0)
    assert not training.OncotreeLineage.eq("Kidney").any()
    candidate_rows = pd.read_csv(pcr_dir / "patient_top10.csv.gz")
    method_inputs = {"nonkidney_mean_shift_clip0": "nonkidney_mean_shift_expression",
                     "kidney_mean_shift_clip0": "kidney_mean_shift_expression", "raw_xena_log_expression": "raw_expression"}
    model_error = 0
    for method, key in method_inputs.items():
        prediction = predict_pcr(model, bridge[key])
        model_error = max(model_error, float(np.nanmax(np.abs(prediction-prediction_archive[method]))))
        assert model_error < 1e-6
        _, top = ranks_and_top(prediction, universe)
        expected = pd.DataFrame([{"patient_id": patient, "rank": k+1, "Gene": genes[column]}
                                 for i, patient in enumerate(ids) for k, column in enumerate(top[i])])
        actual = candidate_rows[candidate_rows.method.eq(method)][["patient_id", "rank", "Gene"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    candidates = pd.read_csv(pcr_dir / "candidate_summary.csv.gz")
    calculated = candidates[(candidates.method == "nonkidney_mean_shift_clip0") & (candidates.split == "train")].sort_values(
        ["top10_frequency", "mean_predicted_residual", "Gene"], ascending=[False, True, True]).head(20)
    discovery = pd.read_csv(pcr_dir / "pcr_discovery_top20.csv")
    assert calculated.Gene.tolist() == discovery.Gene.tolist()
    evidence = pd.read_csv(root / "outputs/pcr_singlecell_synthesis_v1/pcr_candidate_evidence_table.csv")
    assert evidence.Gene.tolist() == discovery.Gene.tolist()
    bootstrap = pd.read_csv(pcr_dir / "bootstrap_patient_overlap.csv.gz")
    assert len(bootstrap) == 2 * 50 * 209
    assert bootstrap.groupby(["kind", "draw"]).patient_id.nunique().eq(209).all()
    assert bootstrap.groupby(["kind", "patient_id"]).draw.nunique().eq(50).all()
    assert bootstrap.top10_overlap.between(0, 1).all()
    sources = pd.read_csv(sc_dir / "sample_source_audit.csv")
    assert sources.raw_umi_max_difference.eq(0).all()
    assert (~sources.geo_author_concordant).sum() == 8
    assert sources.donor.nunique() == 8
    groups = pd.read_csv(sc_dir / "pseudobulk_groups.csv")
    with np.load(sc_dir / "pseudobulk_counts.npz", allow_pickle=False) as a:
        counts, detects, cpm = a["counts"], a["detection_count"], a["cpm"]
    assert len(groups) == len(counts)
    np.testing.assert_allclose(cpm.sum(axis=1), 1e6, atol=1e-7, rtol=1e-12)
    assert np.all(detects >= 0) and np.all(detects <= groups.cell_n.to_numpy()[:, None])
    for _, group in groups.groupby(["donor", "author_cohort", "geo_author_concordant"]):
        all_index = group.index[group.group.eq("all_cells")]
        assert len(all_index) == 1
        broad = group.index[~group.group.str.startswith("label:") & ~group.group.isin(["all_cells", "nonmalignant"])]
        np.testing.assert_allclose(counts[all_index[0]], counts[broad].sum(axis=0), atol=0, rtol=0)
    mixture = pd.read_csv(sc_dir / "mixture_sensitivity.csv")
    assert len(mixture) == 7 * 4 and mixture.groupby("scenario").donor.nunique().eq(7).all()
    assert mixture.top10_overlap.between(0, 1).all()
    for suffix in ["png", "pdf"]:
        figure_manifest = root / "results/historical/mechdec_figure_redraw_20261013/render_manifest.json"
        if figure_manifest.exists():
            stems = [r["stem"] for r in json.loads(figure_manifest.read_text())["figures"] if r["group"] == "pcr_singlecell_synthesis_v1"]
        else:
            stems = ["figure1_pcr_transfer_stability", "figure2_candidate_cell_sources", "figure3_mixture_and_source_audit"]
        for stem in stems:
            file = root / f"outputs/pcr_singlecell_synthesis_v1/{stem}.{suffix}"
            signature = file.read_bytes()[:8]
            assert signature.startswith(b"\x89PNG") if suffix == "png" else signature.startswith(b"%PDF")
    record = {"status": "passed", "snapshot_file_n": snapshot_files, "model_reload_prediction_storage_max_error": model_error,
              "test_patient_overlap": 0, "checks": ["source/output/script hashes", "snapshot integrity", "GPU model reload",
              "all patient Top10 recomputation", "Train-only candidate ordering", "50 patient-cluster draws per uncertainty type",
              "barcode/UMI provenance", "pseudobulk conservation/detection/CPM", "mixture donor identity", "PNG/PDF signatures"]}
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"拒绝覆盖验证记录: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
