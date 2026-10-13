"""Generate planning artifacts; contains no wet-lab observations or guide sequences."""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from statsmodels.stats.power import TTestPower

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/historical/functional_validation_design_v1")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    config_path = ROOT / "configs/functional_validation_design_20261010.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rng = random.Random(cfg["seed"])
    genes = cfg["core_genes"] + cfg["exploratory_genes"]
    conditions = [(gene, f"{gene}_g{i}_UNVALIDATED") for gene in genes for i in range(1, 4)]
    conditions += [("non_targeting", f"NT{i}_UNVALIDATED") for i in range(1, 4)]
    conditions += [("mock", "mock"), ("positive_control", "LAB_VALIDATED_CONTROL_REQUIRED")]
    inner = [f"{row}{col:02}" for row in "BCDEFG" for col in range(2, 12)]
    all_wells = [f"{row}{col:02}" for row in "ABCDEFGH" for col in range(1, 13)]
    layout, blind = [], []
    for donor_i in range(1, cfg["pilot_donor_pairs"] + 1):
        donor = f"PLACEHOLDER_D{donor_i:03}"
        for run in range(1, cfg["independent_runs"] + 1):
            for rep in range(1, 4):
                plate = f"PILOT_D{donor_i:03}_R{run}_TECH{rep}"
                wells = inner.copy()
                rng.shuffle(wells)
                assigned = {}
                for tissue in ("tumor", "normal"):
                    for gene, guide in conditions:
                        well = wells.pop()
                        code = f"BLIND_{len(blind) + 1:05}"
                        assigned[well] = dict(stage="pilot", plate_id=plate, well=well,
                            donor_id=donor, tissue=tissue, independent_run=run,
                            technical_replicate=rep, blinded_condition_id=code,
                            well_role="assay", model_id="", medium_id="")
                        blind.append(dict(blinded_condition_id=code, gene=gene, guide_id=guide,
                            panel="core" if gene in cfg["core_genes"] else "exploratory" if gene in genes else "control",
                            suppression_level="calibrate_before_use"))
                for well in all_wells:
                    layout.append(assigned.get(well, dict(stage="pilot", plate_id=plate,
                        well=well, donor_id=donor, tissue="", independent_run=run,
                        technical_replicate=rep, blinded_condition_id="",
                        well_role="unused_interior" if well in inner else "edge_buffer",
                        model_id="", medium_id="")))
    assay = [row for row in layout if row["well_role"] == "assay"]
    assert len(layout) == 18 * 96 and len(assay) == 18 * 52
    for plate in {row["plate_id"] for row in assay}:
        rows = [row for row in assay if row["plate_id"] == plate]
        assert len(rows) == 52 and len({row["well"] for row in rows}) == 52
        assert all(row["well"] in inner for row in rows)
        assert [sum(row["tissue"] == tissue for row in rows) for tissue in ("tumor", "normal")] == [26, 26]
    power_rows = []
    solver = TTestPower()
    # Four genes x three co-primary bounds: conservative simultaneous inference.
    alpha = cfg["family_alpha"] / (len(cfg["core_genes"]) * 3)
    for endpoint, margin, means, deviations in (
        ("paired_selectivity", 0.5, [0.75, 1.0], [0.25, 0.5, 0.75]),
        ("tumor_loss", 1.0, [1.25, 1.5], [0.25, 0.5, 0.75]),
        ("normal_preservation", cfg["normal_loss_margin_log2"], [0.0, 0.1, 0.2], [0.2, 0.4]),
    ):
        for mean in means:
            for sd in deviations:
                effect = abs(mean - margin) / sd
                # Union bound: allocate per-gene failure probability across three bounds.
                for desired_joint in (0.8, 0.9):
                    target_power = 1 - (1 - desired_joint) / 3
                    n = math.ceil(float(solver.solve_power(effect_size=effect, alpha=alpha,
                        power=target_power, alternative="larger")))
                    achieved = float(solver.power(effect_size=effect, nobs=n,
                        alpha=alpha, alternative="larger"))
                    assert achieved >= target_power - 1e-7
                    power_rows.append(dict(endpoint=endpoint, null_margin_log2=margin,
                        assumed_mean_log2=mean, assumed_donor_sd_log2=sd,
                        standardized_distance_from_margin=effect, one_sided_alpha=alpha,
                        desired_per_gene_joint_power_lower_bound=desired_joint,
                        target_per_bound_power=target_power, complete_donor_pairs=n,
                        recruitment_with_assumed_20pct_pair_failure=math.ceil(n / 0.8),
                        achieved_per_bound_power=achieved, status="assumed_not_observed"))
    out.mkdir(parents=True)
    write_csv(out / "pilot_plate_layout.csv", layout)
    write_csv(out / "allocation_key.csv", blind)
    observations = []
    for row in assay:
        for day in cfg["readout_days_after_pre_induction_baseline"]:
            observations.append({**row, "day": day, "viable_cell_count": "",
                "apoptotic_fraction": "", "target_rna_remaining_fraction": "",
                "target_protein_remaining_fraction": "", "qc_status": "", "exclusion_reason": ""})
    write_csv(out / "observations_blank_template.csv", observations)
    write_csv(out / "power_sensitivity.csv", power_rows)
    inventory = [dict(donor_id=f"PLACEHOLDER_D{i:03}", tissue=tissue,
        model_id="", availability="unknown", histology="", authentication="",
        mycoplasma_status="", passage="", proximal_identity="", tumor_identity="",
        medium_id="", baseline_growth="", consent_approval_reference="")
        for i in range(1, 4) for tissue in ("tumor", "normal")]
    write_csv(out / "model_inventory_blank_template.csv", inventory)
    write_csv(out / "engagement_calibration_blank_template.csv", [dict(gene=gene,
        tissue=tissue, intended_suppression_fraction=level, guide_id="", model_id="",
        induction_setting="", measured_rna_suppression="", measured_protein_suppression="",
        assay_time="", qc_status="") for gene in genes for tissue in ("tumor", "normal")
        for level in cfg["engagement_calibration_targets_fraction"]])
    sources = [ROOT / "results/historical/final_evidence_synthesis_v2/candidate_evidence_matrix.csv",
        ROOT / "results/historical/ccrcc_source_replication_coverage_v1/paired_expression_summary_with_coverage.csv",
        config_path, Path(__file__).resolve()]
    files = sources + sorted(out.glob("*.csv"))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(status=cfg["status"], seed=cfg["seed"], measurements_present=False,
        pilot_donor_pairs=3, independent_runs=2, plates=18, assay_wells=936,
        edge_buffer_wells=648, unused_interior_wells=144,
        observation_template_rows=len(observations), power_scenarios=len(power_rows),
        numerical_method="statsmodels TTestPower noncentral-t; CPU closed-form planning",
        additional_resources_not_in_plate_budget=["engagement calibration", "molecular QC",
            "destructive readouts", "orthogonal perturbation", "rescue", "confirmation"],
        checks=["unique interior allocation", "paired tissues on each plate", "balanced conditions",
            "empty outcome fields", "power target achieved after rounding"], sha256=manifest)
    (out / "design_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
