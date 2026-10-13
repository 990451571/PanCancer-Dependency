"""Read-only validation of planning artifacts, independent of the generator."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/historical/functional_validation_design_v1"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    report = json.loads((OUT / "design_validation.json").read_text(encoding="utf-8"))
    for name, digest in report["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    key = {r["blinded_condition_id"]: r for r in read("allocation_key.csv")}
    plates = defaultdict(list)
    for row in read("pilot_plate_layout.csv"):
        plates[row["plate_id"]].append(row)
    assert len(plates) == 18
    for rows in plates.values():
        assert len(rows) == len({r["well"] for r in rows}) == 96
        assays = [r for r in rows if r["well_role"] == "assay"]
        assert len(assays) == 52
        counts = Counter((r["tissue"], key[r["blinded_condition_id"]]["guide_id"]) for r in assays)
        assert len(counts) == 52 and set(counts.values()) == {1}
        assert {g for t, g in counts if t == "tumor"} == {g for t, g in counts if t == "normal"}
        assert all(r["well"][0] in "BCDEFG" and 2 <= int(r["well"][1:]) <= 11 for r in assays)
    observations = read("observations_blank_template.csv")
    assert len(observations) == 2808
    for row in observations:
        assert all(row[field] == "" for field in ("viable_cell_count", "apoptotic_fraction",
            "target_rna_remaining_fraction", "target_protein_remaining_fraction"))
    # Anchor against a known published/standard paired-t planning example.
    from statsmodels.stats.power import TTestPower
    solver = TTestPower()
    assert abs(solver.solve_power(effect_size=0.5, alpha=0.05, power=0.8) - 33.36713) < 0.001
    for row in read("power_sensitivity.csv"):
        effect = abs(float(row["assumed_mean_log2"]) - float(row["null_margin_log2"])) / float(row["assumed_donor_sd_log2"])
        n = int(row["complete_donor_pairs"])
        kwargs = dict(effect_size=effect, alpha=float(row["one_sided_alpha"]), alternative="larger")
        target = float(row["target_per_bound_power"])
        assert solver.power(nobs=n, **kwargs) >= target
        assert n <= 2 or solver.power(nobs=n - 1, **kwargs) < target
    print("设计核验通过：18块板、936检测孔、2808条空白观测记录、36组样本量情景；未执行湿实验。")


if __name__ == "__main__":
    main()
