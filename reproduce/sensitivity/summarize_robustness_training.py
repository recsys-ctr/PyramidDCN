#!/usr/bin/env python3
"""Aggregate the initial frozen sensitivity plan only after both queues (18 fits) complete.

This script uses Python's standard library and never imports the training harness.
An unfinished queue is refused before any metric-bearing file is opened. A finished
queue with failures may produce a status-only report with the explicit CLI flag.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from inspection_gate import require_all_queues


DATASETS = ("criteo", "avazu")
SEEDS = (2032, 2033, 2034)
ARMS = ("selection", "learned_budget")
ROWS = {"criteo": (36672493, 4584062, 4584062),
        "avazu": (32343172, 4042897, 4042898)}
TOLERANCE = 1e-6
PLAN_HASHES = {
    "PROTOCOL.md": "43e921803857713fa69f748644bebf9bf8f4b05839a596fc0606a864baa22f8b",
    "criteo.json": "64752de46c5238723cbbb9ed3962959c5a865feff7637bf7c957327386b3dea9",
    "avazu.json": "6bb7f92cad042db9d4781d1f3eccce6bd5267b14835e67aaac45595c8c136b95",
    "manifest.json": "983590030437b051ff6192dfa40ba110a2e8266c3d7fcbaf062ed9ee7a9b1b76"
}
LIMITATIONS = [
    "One prespecified alternative protocol: fixed Adam LR 3e-4, two full epochs, validation AUC checkpoint selection.",
    "Datasets are analyzed separately; no pooled primary estimate, significance test, p-value, superiority, equivalence, or noninferiority claim.",
    "Three paired seeds per fixed split/configuration; sample SD describes only this seed variation.",
    "Existing benchmark test splits are reused; new seeds are not new evaluation data.",
    "Near-matched matrix-FLOP recipe comparison with different state widths; not a pure input-map causal effect.",
    "LR and checkpoint/epoch effects are not separately identified; no convergence or native-input SOTA conclusion.",
    "No historical 798-run pooling, historical metric replacement, result-dependent tuning, or new practical-equivalence margin.",
    "Restoration of checkpoint weights relies on the released runner; this analysis verifies recorded best_epoch and evaluation metadata, not saved weights.",
    "Original rounded history times/losses retain their recorded precision; full-precision evaluation returns are used for metric analysis.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Invalid finite numeric field: " + label)
    return value


def integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool), "Invalid integer field: " + label)
    return value


def expected_name(dataset, arm, seed):
    return f"robustness_r2_{dataset}_{arm}_lr3e4_ep2_s{seed}"


def mean_sd(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_status_report(status, expected):
    completed = {item["run"]: item for item in status["completed_runs"]}
    failed = {item["run"]: item for item in status["failed_runs"]}
    rows = []
    for name in expected:
        item = completed.get(name, failed.get(name, {}))
        # Deliberately whitelist fields: arbitrary error messages can contain private paths.
        row = {"run": name, "state": item.get("state", "missing"),
               "retry_attempts": item.get("retry_attempts", 0)}
        if name in failed:
            row["failure_class"] = (item.get("failure_class") if item.get("failure_class")
                                    in ("numerical", "infrastructure_or_code") else "unspecified")
        rows.append(row)
    return {"state": status["state"], "metrics_accessed": False,
            "planned_runs": 12, "completed_runs": len(completed), "failed_runs": len(failed),
            "runs": rows, "interpretation": "Incomplete fixed plan; no metric analysis or positive conclusion.",
            "failure_details": "Failure classes are reported here; detailed reasons remain in source status files."}


def aggregate(run_dir, out, allow_incomplete):
    # This must be the first input read. Do not move any metric-bearing reads above it.
    queue = read_json(run_dir / "status.json")
    require_all_queues(run_dir)
    names = [expected_name(d, a, s) for d in DATASETS for s in SEEDS for a in ARMS]
    require(queue.get("planned_runs") == 12, "Queue does not declare the frozen 12-run plan")
    complete = queue.get("completed_runs", [])
    failures = queue.get("failed_runs", [])
    if queue.get("state") == "completed_with_failures" and allow_incomplete:
        require(failures and len(complete) + len(failures) == 12,
                "Finished-failure queue does not account for all 12 runs")
        seen = [item.get("run") for item in complete + failures]
        require(len(set(seen)) == 12 and set(seen) == set(names), "Queue run roster mismatch")
        report = safe_status_report(queue, names)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "incomplete_status_only.json", report)
        return "Wrote status-only report; no metrics were accessed."
    require(queue.get("state") == "completed" and len(complete) == 12 and not failures,
            "Metric access refused: require state=completed, 12 completed runs, and zero failures")
    require({item.get("run") for item in complete} == set(names), "Completed run roster mismatch")
    require(all(item.get("state") == "completed" for item in complete), "Noncompleted queue entry")

    plan = run_dir / "frozen_plan"
    require(read_json(plan / "SHA256SUMS.json") == PLAN_HASHES, "Frozen hash manifest mismatch")
    require(all(sha256(plan / name) == value for name, value in PLAN_HASHES.items()),
            "Frozen plan content mismatch")
    manifest = read_json(plan / "manifest.json")
    require(manifest["seeds"] == list(SEEDS) and manifest["training_runs"] == 12,
            "Frozen manifest protocol mismatch")
    frozen_configs = {config["name"]: config for d in DATASETS
                      for config in read_json(plan / (d + ".json"))}
    require(set(frozen_configs) == set(names), "Frozen configuration roster mismatch")
    statuses = {}
    for name in names:
        state = read_json(run_dir / "runs" / name / "status.json")
        require(state.get("run") == name and state.get("state") == "completed",
                "Per-run completion guard failed")
        require(read_json(run_dir / "runs" / name / "config.json") == frozen_configs[name],
                "Executed configuration differs from frozen plan")
        statuses[name] = state

    # All completion guards passed. Only here may metric-bearing files be opened.
    rounded = [json.loads(line) for line in (run_dir / "results_rounded.jsonl").read_text().splitlines()
               if line.strip()]
    require(len(rounded) == 12 and {item.get("name") for item in rounded} == set(names),
            "Rounded results roster mismatch")
    rounded = {item["name"]: item for item in rounded}
    runs, sources = [], {"status.json": sha256(run_dir / "status.json"),
                         "results_rounded.jsonl": sha256(run_dir / "results_rounded.jsonl")}
    for dataset in DATASETS:
        train_rows, valid_rows, test_rows = ROWS[dataset]
        for seed in SEEDS:
            for arm in ARMS:
                name = expected_name(dataset, arm, seed)
                directory = run_dir / "runs" / name
                evaluation = read_json(directory / "evaluations_full_precision.json")
                profile = read_json(directory / "model_profile.json")
                state, result = statuses[name], rounded[name]
                require(len(evaluation) == 3 and [e["phase"] for e in evaluation]
                        == ["validation", "validation", "test"], "Evaluation sequence mismatch")
                require([e["epoch"] for e in evaluation] == [1, 2, None], "Evaluation epoch mismatch")
                require([e["rows"] for e in evaluation] == [valid_rows, valid_rows, test_rows],
                        "Full-split evaluation row mismatch")
                for item in evaluation:
                    for key in ("auc", "logloss", "elapsed_seconds"):
                        number(item[key], key)
                    require(0 <= item["auc"] <= 1 and item["logloss"] >= 0
                            and item["elapsed_seconds"] >= 0, "Evaluation value outside allowed range")
                best = 0
                for idx in range(1, 2):
                    if evaluation[idx]["auc"] > evaluation[best]["auc"] + TOLERANCE:
                        best = idx
                require(integer(result["best_epoch"], "best_epoch") == best + 1,
                        "Original best_epoch violates inherited >1e-6 update rule; no strict-max substitution")
                require(len(result["history"]) == 2, "Two epoch histories required")
                for idx, history in enumerate(result["history"]):
                    require(history["epoch"] == idx + 1 and history["lr"] == 0.0003,
                            "History epoch or effective LR mismatch")
                    for raw, recorded in (("auc", "valid_auc"), ("logloss", "valid_logloss")):
                        require(abs(evaluation[idx][raw] - number(history[recorded], recorded)) <= 0.500001e-6,
                                "Rounded validation history disagrees with full precision")
                for value, key in ((evaluation[best]["auc"], "valid_auc"),
                                   (evaluation[best]["logloss"], "valid_logloss"),
                                   (evaluation[2]["auc"], "test_auc"),
                                   (evaluation[2]["logloss"], "test_logloss")):
                    require(abs(value - number(result[key], key)) <= 0.500001e-6,
                            "Rounded result disagrees with full precision")
                steps = math.ceil(train_rows / 4096)
                for key, expected in {"actual_training_epochs": 2, "actual_training_rows": 2 * train_rows,
                                      "actual_training_steps": 2 * steps, "validation_evaluations": 2,
                                      "selected_checkpoint_test_evaluations": 1}.items():
                    require(state.get(key) == expected, "Full execution count mismatch: " + key)
                require(profile["total_params"] == profile["embedding_params"] + profile["nonembedding_params"],
                        "Parameter count decomposition mismatch")
                flops = 2 * sum(shape["in"] * shape["out"] for shape in profile["linear_shapes"])
                require(flops == profile["matrix_flops_per_example"] == result["serving_linear_flops_2x"],
                        "Matrix-FLOP count mismatch")
                require(profile["parameter_dtypes"] == ["torch.float32"]
                        and all(device.startswith("mps") for device in profile["parameter_devices"])
                        and profile["parameter_devices"], "Device or precision mismatch")
                record = {"run": name, "dataset": dataset, "arm": arm, "seed": seed,
                          "best_epoch": best + 1, "checkpoint_rule_verified": True,
                          "valid_auc_epoch1": evaluation[0]["auc"], "valid_logloss_epoch1": evaluation[0]["logloss"],
                          "valid_auc_epoch2": evaluation[1]["auc"], "valid_logloss_epoch2": evaluation[1]["logloss"],
                          "selected_valid_auc": evaluation[best]["auc"],
                          "selected_valid_logloss": evaluation[best]["logloss"],
                          "test_auc": evaluation[2]["auc"], "test_logloss": evaluation[2]["logloss"],
                          "validation_auc_change_epoch2_minus_epoch1": evaluation[1]["auc"] - evaluation[0]["auc"],
                          "training_epochs": 2, "train_rows_per_epoch": train_rows,
                          "train_steps_per_epoch": steps, "actual_training_rows": state["actual_training_rows"],
                          "actual_training_steps": state["actual_training_steps"], "validation_rows": valid_rows,
                          "test_rows": test_rows, "run_elapsed_seconds": number(state["elapsed_seconds"], "run_elapsed_seconds"),
                          "harness_total_seconds_rounded": number(result["total_s"], "total_s"),
                          "epoch1_seconds_rounded": number(result["history"][0]["epoch_s"], "epoch_s"),
                          "epoch2_seconds_rounded": number(result["history"][1]["epoch_s"], "epoch_s"),
                          "validation1_seconds": evaluation[0]["elapsed_seconds"],
                          "validation2_seconds": evaluation[1]["elapsed_seconds"],
                          "test_seconds": evaluation[2]["elapsed_seconds"],
                          "epoch1_lr": result["history"][0]["lr"], "epoch2_lr": result["history"][1]["lr"],
                          "total_params": profile["total_params"], "embedding_params": profile["embedding_params"],
                          "nonembedding_params": profile["nonembedding_params"], "matrix_flops_per_example": flops,
                          "retry_attempts": state.get("retry_attempts", 0), "state": "completed"}
                runs.append(record)
                for filename in ("status.json", "config.json", "evaluations_full_precision.json", "model_profile.json"):
                    relative = f"runs/{name}/{filename}"
                    sources[relative] = sha256(run_dir / relative)
    paired, arm_statistics, paired_statistics = [], [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            pair = {r["arm"]: r for r in runs if r["dataset"] == dataset and r["seed"] == seed}
            a, b = pair["selection"], pair["learned_budget"]
            paired.append({"dataset": dataset, "seed": seed,
                           "delta_auc": a["test_auc"] - b["test_auc"],
                           "delta_logloss": a["test_logloss"] - b["test_logloss"],
                           "selection_matrix_flops": a["matrix_flops_per_example"],
                           "learned_budget_matrix_flops": b["matrix_flops_per_example"],
                           "selection_over_learned_matrix_flops": a["matrix_flops_per_example"] / b["matrix_flops_per_example"]})
        for arm in ARMS:
            group = [r for r in runs if r["dataset"] == dataset and r["arm"] == arm]
            item = {"dataset": dataset, "arm": arm}
            for metric in ("test_auc", "test_logloss", "validation_auc_change_epoch2_minus_epoch1"):
                stats = mean_sd([r[metric] for r in group])
                item.update({metric + "_" + key: value for key, value in stats.items()})
            item["epoch2_selected_count"] = sum(r["best_epoch"] == 2 for r in group)
            arm_statistics.append(item)
        item = {"dataset": dataset}
        for metric in ("delta_auc", "delta_logloss"):
            stats = mean_sd([r[metric] for r in paired if r["dataset"] == dataset])
            item.update({metric + "_" + key: value for key, value in stats.items()})
        item["metric_tradeoff"] = item["delta_auc_mean"] * item["delta_logloss_mean"] > 0
        item["mixed_auc_seed_signs"] = (any(r["delta_auc"] > 0 for r in paired if r["dataset"] == dataset)
                                         and any(r["delta_auc"] < 0 for r in paired if r["dataset"] == dataset))
        paired_statistics.append(item)
    interpretation = []
    if all(item["delta_auc_mean"] > 0 for item in paired_statistics):
        interpretation.append("Mean AUC ordering is preserved on both datasets under this one fixed alternative protocol.")
    else:
        interpretation.append("Positive mean AUC ordering is not preserved on both datasets; report each dataset and the protocol dependence within this scope.")
    if all(item["delta_auc"] > 0 for item in paired):
        interpretation.append("All six AUC differences are positive, so every new seed has the same observed favorable AUC direction.")
    for item in paired_statistics:
        if item["mixed_auc_seed_signs"]:
            interpretation.append(item["dataset"] + ": AUC difference signs vary across seeds.")
        if item["metric_tradeoff"]:
            interpretation.append(item["dataset"] + ": mean AUC and LogLoss favor different arms, showing a metric tradeoff.")
    interpretation.extend([
        "Interpret difference sizes together with sample SD; mean ordering does not establish general superiority or statistical significance.",
        "Report validation AUC changes and epoch-2 selection counts by arm. They may be compatible with different early optimization speed but do not separate LR/epoch causal effects or establish shrinkage of an old gap.",
        "Results are conditional on a two-epoch cap. Report epoch-2 validation change without claiming convergence.",
    ])
    report = {"state": "completed", "metrics_accessed": True, "planned_runs": 12, "completed_runs": 12,
              "failed_runs": 0, "retry_attempts": sum(r["retry_attempts"] for r in runs),
              "checkpoint_rule": "Sequential AUC update only when auc > best_auc + 1e-6; otherwise retain earlier epoch.",
              "delta_direction": "selection minus learned_budget; positive AUC and negative LogLoss favor selection",
              "matrix_flops_convention": "2*Linear MACs per example; these pml_mv arms have no non-Linear matrix products",
              "runs": runs, "paired_deltas": paired, "per_dataset_arm_statistics": arm_statistics,
              "per_dataset_paired_statistics": paired_statistics, "interpretation_en": interpretation,
              "limitations": LIMITATIONS, "frozen_plan_sha256": PLAN_HASHES, "source_sha256": sources}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", report)
    for filename, rows in (("runs.csv", runs), ("paired_deltas.csv", paired),
                           ("per_dataset_arm_statistics.csv", arm_statistics),
                           ("per_dataset_paired_statistics.csv", paired_statistics)):
        write_csv(out / filename, rows)
    lines = ["# Fixed alternative-protocol sensitivity analysis", "", "Both queues passed the eighteen-fit guard; initial twelve fits completed without failure. Analysis uses full-precision full-test evaluation returns.", "",
             "Delta is selection minus learned_budget; positive AUC and negative LogLoss favor selection.", "",
             "| Dataset | Seed | Delta AUC | Delta LogLoss | Selection/learned FLOPs |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset']} | {r['seed']} | {r['delta_auc']:.12g} | {r['delta_logloss']:.12g} | {r['selection_over_learned_matrix_flops']:.12g} |" for r in paired]
    lines += ["", "| Dataset | Mean delta AUC | Sample SD | Mean delta LogLoss | Sample SD |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['dataset']} | {r['delta_auc_mean']:.12g} | {r['delta_auc_sample_sd']:.12g} | {r['delta_logloss_mean']:.12g} | {r['delta_logloss_sample_sd']:.12g} |" for r in paired_statistics]
    lines += ["", *interpretation, "", "All twelve runs, per-epoch validation, selected epochs, rows/steps, runtime, parameters and FLOPs are retained in runs.csv and summary.json.", "", *["- " + item for item in LIMITATIONS]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return "Wrote complete 12-run analysis; checkpoint tolerance and full execution counts verified."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-incomplete-status-only", action="store_true",
                        help="For completed_with_failures only, emit status without opening metric files")
    args = parser.parse_args()
    try:
        print(aggregate(args.run_dir, args.out, args.allow_incomplete_status_only))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        # Avoid leaking supplied paths, arbitrary run messages, or host identifiers.
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else "Input schema/read failure"
        print("Analysis refused: " + message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
