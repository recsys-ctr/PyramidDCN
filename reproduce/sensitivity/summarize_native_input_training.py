#!/usr/bin/env python3
"""Analyze both frozen controls only after all eighteen fits complete."""
import argparse
import json
import math
from pathlib import Path
import sys
import summarize_robustness_training as core
from inspection_gate import require_all_queues


NATIVE_HASHES = {
    "PROTOCOL.md": "a142f8c7f8a8aa9f6debf067d9ee3f00ad3e60c17d8aec6009703c001e87bf3d",
    "criteo.json": "d8dea7d88054cc0b10b9072e822a04ced313d9c9e06c6f310935d7023888c908",
    "avazu.json": "cfc93a0c0a2887c13d4b3c95370b56345baea9f103e54444064081fd37353689",
    "synthetic_shape_audit.json": "25b4572ebf5d0934fdaf0b0ddf6d92c1deb75570eb38f75a6d10062afda45cdc",
    "manifest.json": "c60d588f8a4f1c9d05896b5e1ab1c9c1d5dbbeda5da33a49851756425d71ea90"
}


def aggregate(original, native, out):
    require_all_queues(original, native)
    # Status-only guards precede every metric-bearing read in both analyses.
    for directory, expected in ((original, 12), (native, 6)):
        queue = core.read_json(directory / "status.json")
        core.require(queue.get("state") == "completed" and queue.get("planned_runs") == expected
                     and len(queue.get("completed_runs", [])) == expected and not queue.get("failed_runs"),
                     "All original twelve plus native six must complete before metric access")
    plan = native / "frozen_plan"
    core.require(core.read_json(plan / "SHA256SUMS.json") == NATIVE_HASHES
                 and all(core.sha256(plan / name) == h for name, h in NATIVE_HASHES.items()),
                 "Native six frozen plan hash mismatch")
    manifest = core.read_json(plan / "manifest.json")
    configs = {item["name"]: item for dataset in core.DATASETS
               for item in core.read_json(plan / f"{dataset}.json")}
    names = [f"native_field_{d}_fibinetpp_lr3e4_ep2_s{s}" for d in core.DATASETS for s in core.SEEDS]
    core.require(set(configs) == set(names) and manifest["training_runs"] == 6, "Native roster mismatch")
    original_provenance = core.read_json(original / "provenance.json")
    native_provenance = core.read_json(native / "provenance.json")
    core.require(original_provenance["source_sha256"] == native_provenance["source_sha256"],
                 "Training source changed between frozen queues")
    for dataset in core.DATASETS:
        before = original_provenance["datasets"][dataset]
        after = native_provenance["datasets"][dataset]
        core.require(before["metadata_sha256"] == after["metadata_sha256"]
                     and before["cache_files"] == after["cache_files"],
                     "Dataset metadata or full cache differs between queues")
    # Reuse the independent twelve-run validator rather than silently reimplementing its checks.
    core.aggregate(original, out / "learned_map_control", False)
    original_report = core.read_json(out / "learned_map_control" / "summary.json")
    pyramids = {(r["dataset"], r["seed"]): r for r in original_report["runs"] if r["arm"] == "selection"}
    rounded = [json.loads(line) for line in (native / "results_rounded.jsonl").read_text().splitlines() if line.strip()]
    core.require(len(rounded) == 6 and {r["name"] for r in rounded} == set(names), "Native rounded roster mismatch")
    rounded = {r["name"]: r for r in rounded}
    audit = core.read_json(plan / "synthetic_shape_audit.json")
    runs, sources = [], {"status.json": core.sha256(native / "status.json"),
                        "results_rounded.jsonl": core.sha256(native / "results_rounded.jsonl")}
    for dataset in core.DATASETS:
        train_rows, valid_rows, test_rows = core.ROWS[dataset]
        for seed in core.SEEDS:
            name = f"native_field_{dataset}_fibinetpp_lr3e4_ep2_s{seed}"
            directory = native / "runs" / name
            state = core.read_json(directory / "status.json")
            core.require(state["state"] == "completed" and state["run"] == name, "Native completion mismatch")
            core.require(core.read_json(directory / "config.json") == configs[name], "Native config differs from frozen plan")
            evaluation = core.read_json(directory / "evaluations_full_precision.json")
            profile = core.read_json(directory / "model_profile.json")
            rec = rounded[name]
            core.require(len(evaluation) == 3 and [v["phase"] for v in evaluation] == ["validation", "validation", "test"]
                         and [v["epoch"] for v in evaluation] == [1, 2, None]
                         and [v["rows"] for v in evaluation] == [valid_rows, valid_rows, test_rows],
                         "Native full evaluation sequence mismatch")
            for value in evaluation:
                core.number(value["auc"], "auc")
                core.number(value["logloss"], "logloss")
                core.require(0 <= value["auc"] <= 1 and value["logloss"] >= 0, "Native metric out of range")
            best = 1 if evaluation[1]["auc"] > evaluation[0]["auc"] + 1e-6 else 0
            core.require(rec["best_epoch"] == best + 1 and len(rec["history"]) == 2,
                         "Native inherited checkpoint selection mismatch")
            for idx, history in enumerate(rec["history"]):
                core.require(history["epoch"] == idx + 1 and history["lr"] == 0.0003, "Native epoch/LR mismatch")
                for raw, recorded in (("auc", "valid_auc"), ("logloss", "valid_logloss")):
                    core.require(abs(evaluation[idx][raw] - history[recorded]) <= 0.500001e-6,
                                 "Native rounded validation mismatch")
            for value, key in ((evaluation[best]["auc"], "valid_auc"),
                               (evaluation[best]["logloss"], "valid_logloss"),
                               (evaluation[2]["auc"], "test_auc"), (evaluation[2]["logloss"], "test_logloss")):
                core.require(abs(value - rec[key]) <= 0.500001e-6, "Native rounded result mismatch")
            steps = math.ceil(train_rows / 4096)
            for key, expected in {"actual_training_epochs": 2, "actual_training_rows": 2 * train_rows,
                "actual_training_steps": 2 * steps, "validation_evaluations": 2,
                "selected_checkpoint_test_evaluations": 1}.items():
                core.require(state[key] == expected, "Native full execution count mismatch")
            fields, dim = audit[dataset]["native_fields"], 16
            linear = 2 * sum(v["in"] * v["out"] for v in profile["linear_shapes"])
            matrix = linear + 2 * (fields * dim * dim + fields * fields * dim)
            core.require(profile["native_fields"] == fields and profile["embedding_dim"] == 16
                         and profile["bilinear_W_shape"] == [16, 16]
                         and profile["bilinear_pairs"] == fields * (fields - 1) // 2,
                         "Actual trained native grouping mismatch")
            core.require(profile["parameter_dtypes"] == ["torch.float32"] and profile["parameter_devices"]
                         and all(d.startswith("mps") for d in profile["parameter_devices"]), "Native device/precision mismatch")
            core.require(profile["total_params"] == profile["embedding_params"] + profile["nonembedding_params"]
                         and profile["total_params"] == audit[dataset]["full_vocabulary_total_params_analytic"],
                         "Native full parameter count mismatch")
            core.require(linear == rec["serving_linear_flops_2x"] == profile["linear_flops_per_example"]
                         and matrix == profile["matrix_flops_per_example"] == audit[dataset]["complete_matrix_flops_per_example"],
                         "Native complete matrix FLOPs mismatch")
            runs.append({"run": name, "dataset": dataset, "arm": "native_field_fibinetpp", "seed": seed,
                "best_epoch": best + 1, "valid_auc_epoch1": evaluation[0]["auc"], "valid_auc_epoch2": evaluation[1]["auc"],
                "valid_logloss_epoch1": evaluation[0]["logloss"], "valid_logloss_epoch2": evaluation[1]["logloss"],
                "test_auc": evaluation[2]["auc"], "test_logloss": evaluation[2]["logloss"],
                "actual_training_epochs": 2, "actual_training_rows": 2 * train_rows,
                "actual_training_steps": 2 * steps, "validation_rows": valid_rows, "test_rows": test_rows,
                "run_elapsed_seconds": state["elapsed_seconds"], "epoch1_seconds": rec["history"][0]["epoch_s"],
                "epoch2_seconds": rec["history"][1]["epoch_s"], "total_params": profile["total_params"],
                "nonembedding_params": profile["nonembedding_params"], "matrix_flops_per_example": matrix,
                "native_fields": fields, "embedding_dim": 16, "retry_attempts": state.get("retry_attempts", 0)})
            for file in ("status.json", "config.json", "evaluations_full_precision.json", "model_profile.json"):
                relative = f"runs/{name}/{file}"
                sources[relative] = core.sha256(native / relative)
    paired, arm_stats, paired_stats = [], [], []
    for dataset in core.DATASETS:
        for seed in core.SEEDS:
            p = pyramids[(dataset, seed)]
            n = next(v for v in runs if v["dataset"] == dataset and v["seed"] == seed)
            paired.append({"dataset": dataset, "seed": seed, "pyramid_auc": p["test_auc"], "native_fibinetpp_auc": n["test_auc"],
                "delta_auc": p["test_auc"] - n["test_auc"], "pyramid_logloss": p["test_logloss"],
                "native_fibinetpp_logloss": n["test_logloss"], "delta_logloss": p["test_logloss"] - n["test_logloss"],
                "pyramid_best_epoch": p["best_epoch"], "native_fibinetpp_best_epoch": n["best_epoch"],
                "pyramid_matrix_flops": p["matrix_flops_per_example"], "native_fibinetpp_matrix_flops": n["matrix_flops_per_example"]})
        for arm in ("pyramid_selection", "native_field_fibinetpp"):
            values = ([v for (d, s), v in pyramids.items() if d == dataset] if arm == "pyramid_selection"
                      else [v for v in runs if v["dataset"] == dataset])
            stats = {"dataset": dataset, "arm": arm}
            for metric in ("test_auc", "test_logloss"):
                stats.update({metric + "_" + k: v for k, v in core.mean_sd([r[metric] for r in values]).items()})
            arm_stats.append(stats)
        stats = {"dataset": dataset}
        for metric in ("delta_auc", "delta_logloss"):
            stats.update({metric + "_" + k: v for k, v in core.mean_sd([r[metric] for r in paired if r["dataset"] == dataset]).items()})
        paired_stats.append(stats)
    limits = ["Native-field FiBi++ comparison under the added protocol, not a pure projection-removal causal ablation.",
        "No projected-FiBi++ run under the added protocol; old Table1 differs in LR, epoch, device, and seeds.",
        "Exact Table1 hand-port DNN overrides retained; not an independently tuned official reproduction or native-input SOTA test.",
        "Different input grouping and parameter/FLOP budgets; descriptive paired AUC/LogLoss only, no p-values or general superiority.",
        "The same six new PyramidDCN results are reused across the two separate comparisons; they are not independent replications.",
        "Three seeds per fixed benchmark split; two-epoch cap, reused test splits, and no convergence claim."]
    report = {"state": "completed", "all18_completed_before_metric_access": True, "native_runs": runs,
        "paired_deltas": paired, "per_dataset_arm_statistics": arm_stats, "per_dataset_paired_statistics": paired_stats,
        "delta_direction": "new PyramidDCN minus native-field FiBi++; positive AUC and negative LogLoss favor PyramidDCN",
        "frozen_plan_sha256": NATIVE_HASHES, "source_sha256": sources, "limitations": limits}
    destination = out / "native_field_control"
    destination.mkdir(parents=True, exist_ok=True)
    core.write_json(destination / "summary.json", report)
    for name, values in (("native_runs.csv", runs), ("paired_deltas.csv", paired),
                         ("per_dataset_arm_statistics.csv", arm_stats), ("per_dataset_paired_statistics.csv", paired_stats)):
        core.write_csv(destination / name, values)
    lines = ["# Native-field FiBiNet++ comparison under the added protocol", "", "Both separately frozen queues, all eighteen fits, completed before analysis.", "",
        "| Dataset | Seed | Delta AUC (Pyramid - native FiBiNet++) | Delta LogLoss |", "|---|---:|---:|---:|"]
    lines += [f"| {v['dataset']} | {v['seed']} | {v['delta_auc']:.12g} | {v['delta_logloss']:.12g} |" for v in paired]
    lines += ["", "| Dataset | Mean delta AUC | Sample SD | Mean delta LogLoss | Sample SD |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {v['dataset']} | {v['delta_auc_mean']:.12g} | {v['delta_auc_sample_sd']:.12g} | {v['delta_logloss_mean']:.12g} | {v['delta_logloss_sample_sd']:.12g} |" for v in paired_stats]
    lines += ["", "Report all seed and metric directions. This comparison does not identify a pure projection-removal effect or establish a native-input SOTA reproduction.", "", *["- " + v for v in limits]]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return "Both frozen analyses written after all eighteen-fit guards passed."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run-dir", required=True, type=Path)
    p.add_argument("--native-run-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    try:
        print(aggregate(a.original_run_dir, a.native_run_dir, a.out))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        message = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else "Input schema/read failure"
        print("Analysis refused: " + message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
