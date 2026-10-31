#!/usr/bin/env python3
"""Export descriptive, per-dataset Table 3 results from the frozen public records.

Standard library only. No training, inference, networking, or repository writes.
The input boundary is the four historical result JSONL files, two configuration
files, and two public reference tables explicitly named below.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path


DATASETS = ("criteo", "avazu", "kkbox", "kuairec")
SEED_RANGE = range(2024, 2032)
EXPECTED_RECORDS = 798
CORE_CSV = "table3_per_dataset.csv"
AGGREGATE_CSV = "table3_four_dataset_aggregates.csv"
VALIDATION_JSON = "table3_export_validation.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(dataset, arm, seed):
    return f"fair_{dataset}_{arm}_s{seed}"


def shared_seeds(records, arm, reference):
    """Match reproduce/build_tables.py: intersection over all four datasets."""
    return [
        seed for seed in SEED_RANGE
        if all(run_name(ds, a, seed) in records
               for ds in DATASETS for a in (arm, reference))
    ]


def matrix_flops(record):
    # Every selected Table 3 arm has model='pml_mv'. For that model the public
    # reproduce/matrix_flops.py returns this logged matrix-product count as-is.
    # Fail closed rather than silently applying this rule to other model families.
    if record["model"] != "pml_mv":
        raise ValueError("Unexpected model family in the Table 3 subset")
    value = record["serving_linear_flops_2x"]
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Invalid logged matrix-product FLOP count")
    return value


def mean_sd(records, metric):
    values = [r[metric] for r in records]
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("At least two finite seed results are required")
    return statistics.mean(values), statistics.stdev(values)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_cell(value):
    return json.dumps(value, separators=(",", ":"))


def build_dataset_rows(records, conditions, reference):
    rows = []
    for number, condition in enumerate(conditions, 1):
        arm = condition["arm"]
        seeds = shared_seeds(records, arm, reference)
        if len(seeds) not in (3, 8):
            raise ValueError(f"Unexpected shared cohort for {arm}: {seeds}")
        for ds in DATASETS:
            names = [run_name(ds, arm, seed) for seed in seeds]
            reference_names = [run_name(ds, reference, seed) for seed in seeds]
            group = [records[name] for name in names]
            comparison = [records[name] for name in reference_names]
            auc, auc_sd = mean_sd(group, "test_auc")
            ll, ll_sd = mean_sd(group, "test_logloss")
            ref_auc, ref_auc_sd = mean_sd(comparison, "test_auc")
            ref_ll, ref_ll_sd = mean_sd(comparison, "test_logloss")
            cost = {matrix_flops(r) for r in group}
            ref_cost = {matrix_flops(r) for r in comparison}
            widths = {tuple(r["dims"]) for r in group}
            ref_widths = {tuple(r["dims"]) for r in comparison}
            if any(len(values) != 1 for values in (cost, ref_cost, widths, ref_widths)):
                raise ValueError("Architecture or FLOP count varies within a seed cohort")
            cost, ref_cost = cost.pop(), ref_cost.pop()
            rows.append({
                "condition_order": number,
                "dataset": ds,
                "comparison": condition["comparison"],
                "core_arm": condition["core_arm"],
                "arm": arm,
                "reference_arm": reference,
                "seeds": json_cell(seeds),
                "n": len(group),
                "reference_n": len(comparison),
                "run_ids": json_cell(names),
                "reference_run_ids": json_cell(reference_names),
                "auc_mean": auc,
                "auc_sample_sd": auc_sd,
                "logloss_mean": ll,
                "logloss_sample_sd": ll_sd,
                "reference_auc_mean": ref_auc,
                "reference_auc_sample_sd": ref_auc_sd,
                "reference_logloss_mean": ref_ll,
                "reference_logloss_sample_sd": ref_ll_sd,
                "auc_absolute_change": auc - ref_auc,
                "auc_relative_change_percent": 100 * (auc / ref_auc - 1),
                "logloss_absolute_change": ll - ref_ll,
                "logloss_relative_change_percent": 100 * (ll / ref_ll - 1),
                "matrix_flops": cost,
                "reference_matrix_flops": ref_cost,
                "matrix_flops_ratio": cost / ref_cost,
                "widths": json_cell(list(widths.pop())),
                "reference_widths": json_cell(list(ref_widths.pop())),
            })
    return rows


def aggregate_rows(dataset_rows, conditions):
    aggregated = []
    for condition in conditions:
        rows = [r for r in dataset_rows if r["arm"] == condition["arm"]]
        if len(rows) != 4:
            raise ValueError("Each condition must have exactly four dataset rows")
        aggregated.append({
            "core_arm": condition["core_arm"],
            "arm": condition["arm"],
            "comparison": condition["comparison"],
            "datasets": json_cell(list(DATASETS)),
            "seeds": rows[0]["seeds"],
            "n_per_dataset": rows[0]["n"],
            "mean_auc_absolute_change": statistics.mean(r["auc_absolute_change"] for r in rows),
            "mean_auc_relative_change_percent": statistics.mean(r["auc_relative_change_percent"] for r in rows),
            "mean_logloss_absolute_change": statistics.mean(r["logloss_absolute_change"] for r in rows),
            "mean_logloss_relative_change_percent": statistics.mean(r["logloss_relative_change_percent"] for r in rows),
            "mean_matrix_flops_ratio": statistics.mean(r["matrix_flops_ratio"] for r in rows),
        })
    return aggregated


def decimal_mean(values):
    return sum(values, Decimal(0)) / Decimal(len(values))


def decimal_mean_sd(records, metric):
    values = [Decimal(str(record[metric])) for record in records]
    mean = decimal_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / Decimal(len(values) - 1)
    return mean, variance.sqrt()


def rounded_percent(value):
    return Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def numeric_tex_cell(value):
    value = value.strip().replace(r"\\", "").replace(r"\%", "")
    value = value.replace("$-$", "-").replace("$", "").strip()
    return None if value in ("", "--") else Decimal(value)


def reference_table_rows(path, expected):
    rows = [line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("%") and len(line.split("&")) == 5]
    if len(rows) != expected:
        raise ValueError(f"Unexpected number of public reference rows in {path.name}")
    return [[numeric_tex_cell(c) for c in row.split("&")[-3:]] for row in rows]


def validate(repo, out, records, conditions, specs, core_arms, input_paths):
    """Independent Decimal recomputation; compare with frozen public TeX cells."""
    with (out / CORE_CSV).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (out / AGGREGATE_CSV).open(newline="", encoding="utf-8") as handle:
        combined = {r["arm"]: r for r in csv.DictReader(handle)}
    maximum_error = Decimal(0)
    numeric_checks = 0
    with localcontext() as ctx:
        ctx.prec = 45
        for row in rows:
            seeds = json.loads(row["seeds"])
            arm, ds, ref = row["arm"], row["dataset"], row["reference_arm"]
            # Derive the cohort independently using set intersections.
            eligible = []
            for dataset in DATASETS:
                for candidate in (arm, ref):
                    eligible.append({r["seed"] for r in records.values()
                                     if r["name"].startswith(f"fair_{dataset}_{candidate}_s")
                                     and r["seed"] in SEED_RANGE})
            expected_seeds = sorted(set.intersection(*eligible))
            if seeds != expected_seeds or int(row["n"]) != len(seeds) or int(row["reference_n"]) != len(seeds):
                raise ValueError("Independent seed-cohort check failed")
            group = [records[run_name(ds, arm, seed)] for seed in seeds]
            reference = [records[run_name(ds, ref, seed)] for seed in seeds]
            expected = {}
            for label, metric in (("auc", "test_auc"), ("logloss", "test_logloss")):
                mean, sd = decimal_mean_sd(group, metric)
                ref_mean, ref_sd = decimal_mean_sd(reference, metric)
                expected.update({
                    f"{label}_mean": mean,
                    f"{label}_sample_sd": sd,
                    f"reference_{label}_mean": ref_mean,
                    f"reference_{label}_sample_sd": ref_sd,
                    f"{label}_absolute_change": mean - ref_mean,
                    f"{label}_relative_change_percent": 100 * (mean / ref_mean - 1),
                })
            for field, value in expected.items():
                error = abs(Decimal(row[field]) - value)
                maximum_error = max(maximum_error, error)
                numeric_checks += 1
                if error > Decimal("1e-11"):
                    raise ValueError(f"Independent descriptive statistic check failed: {field}")
            if json.loads(row["run_ids"]) != [r["name"] for r in group]:
                raise ValueError("Run identifiers do not match the selected cohort")
            if json.loads(row["reference_run_ids"]) != [r["name"] for r in reference]:
                raise ValueError("Reference run identifiers do not match the selected cohort")
            if any(int(row["matrix_flops"]) != matrix_flops(r) for r in group):
                raise ValueError("Selected-run FLOP count mismatch")
            if any(int(row["reference_matrix_flops"]) != matrix_flops(r) for r in reference):
                raise ValueError("Reference FLOP count mismatch")
            if abs(Decimal(row["matrix_flops_ratio"]) - Decimal(row["matrix_flops"]) / Decimal(row["reference_matrix_flops"])) > Decimal("1e-14"):
                raise ValueError("FLOP ratio mismatch")

        # Re-normalize each exported dataset mean separately, then average over
        # the four fixed datasets; do not normalize a pooled cross-dataset mean.
        for condition in conditions:
            group = [r for r in rows if r["arm"] == condition["arm"]]
            expected_delta = 100 * decimal_mean([
                Decimal(r["auc_mean"]) / Decimal(r["reference_auc_mean"]) - 1 for r in group
            ])
            expected_cost = decimal_mean([
                Decimal(r["matrix_flops"]) / Decimal(r["reference_matrix_flops"]) for r in group
            ])
            actual = combined[condition["arm"]]
            for field, expected in (("mean_auc_relative_change_percent", expected_delta),
                                    ("mean_matrix_flops_ratio", expected_cost)):
                error = abs(Decimal(actual[field]) - expected)
                maximum_error = max(maximum_error, error)
                numeric_checks += 1
                if error > Decimal("1e-11"):
                    raise ValueError("Independent four-dataset aggregate check failed")

    core_reference = reference_table_rows(
        repo / "results/reference/tables/table3_core_gpt_rows.tex", len(core_arms))
    full_reference = reference_table_rows(
        repo / "results/reference/tables/table3_full_gpt_rows.tex", len(specs))
    full_by_arm = {spec["arm"]: cells for spec, cells in zip(specs, full_reference)}
    spec_by_arm = {spec["arm"]: spec for spec in specs}
    reference_checks = []
    for arm, core_cells in zip(core_arms, core_reference):
        same = combined[arm]
        expected = [Decimal(f"{round(float(same['mean_matrix_flops_ratio']), 2):.2f}"),
                    None if arm == spec_by_arm[arm]["reference"] else
                    rounded_percent(same["mean_auc_relative_change_percent"]), None]
        matched = spec_by_arm[arm]["matched_arm"]
        if matched:
            expected[2] = rounded_percent(combined[matched]["mean_auc_relative_change_percent"])
        if core_cells != expected or full_by_arm[arm] != expected:
            raise ValueError(f"Public core/full reference numeric cells disagree for {arm}")
        reference_checks.append({"core_arm": arm, "core_cells_match": True,
                                 "full_cells_match": True})

    matched_rows = [r for r in rows if r["comparison"] == "near_budget"]
    ratios = [float(r["matrix_flops_ratio"]) for r in matched_rows]
    selected = {name for row in rows for name in json.loads(row["run_ids"])}
    n_counts = {str(n): sum(int(r["n"]) == n for r in rows) for n in (3, 8)}
    return {
        "validation_passed": True,
        "historical_records_loaded": len(records),
        "dataset_rows": len(rows),
        "conditions": len(conditions),
        "core_conditions": len(core_arms),
        "near_budget_conditions": sum(c["comparison"] == "near_budget" for c in conditions),
        "unique_selected_run_ids": len(selected),
        "row_counts_by_seed_cohort_size": n_counts,
        "cohort_rule": "Intersection of arm and reference availability over all four datasets, within seeds 2024-2031.",
        "cohorts": [{"arm": condition["arm"],
                     "seeds": shared_seeds(records, condition["arm"], specs[0]["reference"])}
                    for condition in conditions],
        "independent_validation": {
            "method": "45-digit Decimal recomputation from scalar source records; independent set-intersection cohort selection; exported-CSV normalization; frozen public core/full table checks.",
            "numeric_values_checked": numeric_checks,
            "maximum_absolute_numeric_difference": float(maximum_error),
            "tolerance": 1e-11,
            "public_table_cells_checked": len(core_arms) * 3 * 2,
            "public_table_numeric_cells_checked":
                2 * sum(cell is not None for cells in core_reference for cell in cells),
            "core_and_full_reference_checks": reference_checks,
        },
        "near_budget_scope": {
            "dataset_condition_rows": len(matched_rows),
            "minimum_cost_ratio": min(ratios),
            "maximum_cost_ratio": max(ratios),
            "minimum_budget_percent_rounded_one_decimal": round(100 * min(ratios), 1),
            "maximum_budget_percent_rounded_one_decimal": round(100 * max(ratios), 1),
            "description": "Discrete widths use 97.9%-99.8% of reference matrix FLOPs, rather than exactly equal costs.",
            "per_dataset": [{"core_arm": r["core_arm"], "arm": r["arm"],
                             "dataset": r["dataset"], "matrix_flops": int(r["matrix_flops"]),
                             "reference_matrix_flops": int(r["reference_matrix_flops"]),
                             "ratio": float(r["matrix_flops_ratio"])} for r in matched_rows],
        },
        "input_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in input_paths},
        "output_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                          for name in (CORE_CSV, AGGREGATE_CSV)},
        "privacy_boundary": {
            "input_scope": "Only the four named historical scalar JSONL files, two public configuration files, and two public reference tables.",
            "excluded": "No training examples, checkpoints, new training outputs, external services, author identities, private review text, or review scores. Internal per-run provenance and dense-parameter counts are not exported.",
            "paths": "All recorded input paths are repository-relative; historical run names are reproducibility identifiers.",
        },
        "analysis_scope": "Descriptive seed summaries only; no tests, significance labels, new predictive training, or new manuscript claims.",
    }


README = """# Per-dataset Table 3 descriptive results

This package expands the existing, frozen Table 3 core comparisons into one row
per dataset. It adds no predictive training or inferential tests. It is an
auxiliary CSV export, not an additional manuscript table.

## Reproduce

Python 3.11 or later is sufficient; no third-party packages are required.

```sh
python export_table3_by_dataset.py --repo-dir /path/to/repository --out-dir /path/to/output
```

`--repo-dir` must point to the repository containing `configs`, `results`, and
the frozen public reference tables. The exporter reads only the four named
historical JSONL files, two configuration files, and two reference TeX tables.
It does not import the training code or modify the input repository. No network
access is used. It fails if the frozen input set does not contain 798 records,
if selected model families or cohorts differ, or if the public table checks fail.

## Files

- `table3_per_dataset.csv`: 56 rows: 11 core conditions and 3 reported
  near-budget variants, each on four datasets.
- `table3_four_dataset_aggregates.csv`: the 14 condition-level descriptive
  aggregates used to check the existing Table 3 numbers.
- `table3_export_validation.json`: input hashes, cohort membership, numeric
  checks, comparisons with both frozen public reference tables, and cost scope.
- `export_table3_by_dataset.py`: the portable exporter and independent checker.

## Cohorts and interpretation

The exporter follows `reproduce/build_tables.py`: for each condition, it takes
the seed intersection shared by that arm and the reference over **all four
datasets**, restricted to 2024-2031. Each arm uses three or eight seeds. Every
comparison uses its own matching reference cohort. In particular, three-seed
arms are compared with three-seed references, not the eight-seed reference
summary shown in the reference arm's own row. Thus reference means and sample
standard deviations can differ between CSV rows by design.

Means and sample standard deviations describe seed variability on the fixed
dataset splits. Sample SD uses denominator `n - 1`; it is not a standard error
or a confidence interval. The fixed datasets are not treated as independent
seed replications. No missing results are imputed, and no seeds are selected by
outcome value.

`auc_absolute_change` is AUC minus its matching reference mean.
`auc_relative_change_percent` is `100 * (AUC / reference_AUC - 1)`.
The corresponding LogLoss changes use the same formulas; positive AUC changes
and negative LogLoss changes favor the variant. These relative percentages are
not percentage-point differences. The aggregate AUC change is the arithmetic
mean of the four **within-dataset relative changes**, as in the existing table;
it is not the relative change of a pooled cross-dataset mean. Widths, run names,
seeds, sample sizes, and matching reference identifiers are supplied for each
condition. Full numeric precision is retained in the CSV.

## Computation scope

All selected historical arms have model family `pml_mv`, for which the public
matrix-FLOP helper returns the logged `serving_linear_flops_2x` value unchanged.
The count covers matrix products in the common input projection, backbone, and
head; it excludes embedding lookups, normalization, and elementwise operations.
It is not latency or an energy estimate. Architecture and FLOP counts are checked
to be constant across the seeds within each condition. Historical dense-parameter
counts are deliberately neither used nor exported.

The three variants labeled as matched FLOPs in the existing table actually use
**97.9%-99.8%** of reference matrix FLOPs across their 12 dataset-condition
combinations because widths are discrete. They are labeled `near_budget` in
this export; exact per-dataset costs and ratios are included. This terminology
does not change the existing experimental configurations or results.

## Validation and disclosure boundary

The exporter independently recomputes means and sample SDs with 45-digit Decimal
arithmetic and independently derives cohorts by set intersection. It then reads
back the CSV, checks the four-dataset normalization, and verifies all reported
numeric cells against both frozen public core and full Table 3 files.

Only historical scalar experiment summaries and public configuration/table
metadata are processed. No dataset examples, checkpoints, new predictive runs,
external services, author identities, private review text, or review scores are
included. The run names are technical reproducibility identifiers. All recorded
paths are relative to the input repository. The source JSONL records contain
additional fields, but internal provenance, timing traces, and parameter-count
fields are not copied into this export. No manuscript claims are changed by
generating these descriptive artifacts.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    repo, out = args.repo_dir.expanduser().resolve(), args.out_dir.expanduser().resolve()
    if out == repo or out.is_relative_to(repo):
        raise ValueError("Choose an output directory outside the read-only input repository")
    index_path = repo / "configs/experiment_index.json"
    specification_path = repo / "configs/table3_specification.json"
    index, specs = read_json(index_path), read_json(specification_path)
    if tuple(index["datasets"]) != DATASETS:
        raise ValueError("The historical dataset list differs")
    reference = index["reference_arm"]
    core_arms = index["core_table3_arms"]
    if len(core_arms) != 11 or len(set(core_arms)) != 11:
        raise ValueError("The frozen Table 3 core must contain 11 unique conditions")
    spec_by_arm = {spec["arm"]: spec for spec in specs}
    if len(spec_by_arm) != len(specs):
        raise ValueError("Duplicate arm specification")
    records = {}
    result_paths = [repo / f"results/{ds}.jsonl" for ds in DATASETS]
    for ds, path in zip(DATASETS, result_paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            name = record["name"]
            if name in records:
                raise ValueError("Duplicate historical run identifier")
            records[name] = record
    if len(records) != EXPECTED_RECORDS or index["selected_run_count"] != EXPECTED_RECORDS:
        raise ValueError("The input is not the frozen 798-record historical release")
    conditions = []
    for arm in core_arms:
        spec = spec_by_arm[arm]
        if spec["reference"] != reference:
            raise ValueError("Unexpected Table 3 reference arm")
        conditions.append({"core_arm": arm, "arm": arm,
                           "comparison": "reference" if arm == reference else "same_widths"})
    for arm in core_arms:
        matched = spec_by_arm[arm]["matched_arm"]
        if matched:
            conditions.append({"core_arm": arm, "arm": matched, "comparison": "near_budget"})
    if len(conditions) != 14 or len({c["arm"] for c in conditions}) != 14:
        raise ValueError("Expected exactly three distinct reported near-budget variants")
    rows = build_dataset_rows(records, conditions, reference)
    combined = aggregate_rows(rows, conditions)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / CORE_CSV, rows)
    write_csv(out / AGGREGATE_CSV, combined)
    input_paths = result_paths + [index_path, specification_path,
        repo / "results/reference/tables/table3_core_gpt_rows.tex",
        repo / "results/reference/tables/table3_full_gpt_rows.tex"]
    validation = validate(repo, out, records, conditions, specs, core_arms, input_paths)
    (out / VALIDATION_JSON).write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    print(json.dumps({key: validation[key] for key in
                     ("validation_passed", "historical_records_loaded", "conditions",
                      "dataset_rows", "unique_selected_run_ids", "row_counts_by_seed_cohort_size")},
                     indent=2))


if __name__ == "__main__":
    main()
