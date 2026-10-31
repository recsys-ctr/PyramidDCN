#!/usr/bin/env python3
"""Execute an immutable two-dataset sensitivity plan using the released harness.

No configuration search, partial datasets, altered selection, or metric-based retry.
Full-precision evaluation returns are recorded without changing harness semantics.
Console/status output excludes test metrics; the global eighteen-fit guard controls analysis.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
import warnings


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--criteo-data-dir", required=True, type=Path)
    parser.add_argument("--avazu-data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise RuntimeError("Output already has status; use a new directory, do not silently repeat runs")
    plan = args.plan_dir.resolve()
    code = args.code_dir.resolve()
    datasets = {"criteo": args.criteo_data_dir.resolve(), "avazu": args.avazu_data_dir.resolve()}
    substitutions = {str(code): "<code-dir>", str(plan): "<plan-dir>",
                     str(output): "<output-dir>", str(Path.cwd()): "<workspace>"}
    substitutions.update({str(path): f"<{name}-data-dir>" for name, path in datasets.items()})

    def sanitize(value):
        value = str(value)
        for private, replacement in sorted(substitutions.items(), key=lambda x: -len(x[0])):
            value = value.replace(private, replacement)
        return re.sub(r"/Users/[^/\s'\"]+", "<user-home>", value)

    status = {"state": "verifying", "started_utc": utc(), "completed_runs": [], "failed_runs": [],
              "planned_runs": 12, "test_metrics_human_inspection": "deferred_until_both_original12_plus_native6_finish"}

    def event(kind, **fields):
        value = {"utc": utc(), "event": kind, **fields}
        print(json.dumps(value), flush=True)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        status["elapsed_seconds"] = time.perf_counter() - started
        status["last_event"] = value
        atomic_json(output / "status.json", status)

    frozen_hashes = json.loads((plan / "SHA256SUMS.json").read_text())
    if not all(digest(plan / name) == expected for name, expected in frozen_hashes.items()):
        raise RuntimeError("Frozen plan hash mismatch")
    manifest = json.loads((plan / "manifest.json").read_text())
    for relative, expected in manifest["source_files"].items():
        source_relative = relative.removeprefix("exports/PyramidDCN_recsys/")
        if digest(code / source_relative) != expected:
            raise RuntimeError("Frozen source hash mismatch: " + source_relative)
    arm_lists = {dataset: json.loads((plan / f"{dataset}.json").read_text()) for dataset in datasets}
    for dataset, arms in arm_lists.items():
        if [arm["name"] for arm in arms] != manifest["arm_names"][dataset]:
            raise RuntimeError("Frozen execution order mismatch")
        for arm in arms:
            if (arm["max_epochs"] != 2 or arm["lr"] != 0.0003 or arm["batch"] != 4096
                    or arm.get("train_frac", 1) != 1 or arm.get("sched") is not None
                    or any(arm.get(flag) for flag in ("erank", "funnel", "agop"))):
                raise RuntimeError("Unexpected arm protocol")
    snapshot = output / "frozen_plan"
    snapshot.mkdir()
    for name in [*frozen_hashes, "SHA256SUMS.json"]:
        shutil.copyfile(plan / name, snapshot / name)
    event("frozen_plan_verified", arm_count=sum(map(len, arm_lists.values())))

    os.environ.pop("CTR_DATA", None)
    os.environ.pop("CTR_EMB", None)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    if not torch.backends.mps.is_available():
        status["state"] = "failed_device_unavailable"
        event("device_unavailable")
        raise RuntimeError("MPS unavailable; no CPU fallback or changed protocol")
    sys.path.insert(0, str(code))
    from experiments import x4_runner as xr
    expected_metadata = json.loads((code / "configs" / "dataset_metadata.json").read_text())
    provenance = {
        "started_utc": status["started_utc"], "plan_sha256": frozen_hashes,
        "launcher_sha256": digest(Path(__file__)),
        "source_sha256": {str(path.relative_to(code)): digest(path) for path in sorted(code.rglob("*.py"))},
        "environment": {"system": platform.system(), "system_release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "device": "mps", "precision": "float32",
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "mps_available": True, "mps_cpu_fallback": False,
            "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "mps_prefer_metal": os.environ.get("PYTORCH_MPS_PREFER_METAL", "unset")},
        "datasets": {}, "effective_training_lr": 0.0003,
        "selection": "Released run_arm: full-validation AUC > previous best + 1e-6, otherwise earliest",
        "test_policy": "Once after selected checkpoint restoration; no human test inspection before both queues, all 18 runs, finish",
        "code_changes": "None to released harness; evaluate return instrumentation and console redaction only",
    }
    for dataset, directory in datasets.items():
        metadata = json.loads((directory / "meta.json").read_text())
        expected = expected_metadata[dataset]
        for key in ("vocab_size", "fields"):
            if metadata[key] != expected[key]:
                raise RuntimeError(f"Dataset metadata mismatch: {dataset}/{key}")
        information = {"metadata": metadata, "cache_files": {}}
        for split in ("train", "valid", "test"):
            expected_rows = expected[split]["rows"]
            if metadata[split]["rows"] != expected_rows:
                raise RuntimeError(f"Metadata row count mismatch: {dataset}/{split}")
            for suffix in ("cats", "y"):
                name = f"{split}_{suffix}.npy"
                path = directory / name
                mapped = np.load(path, mmap_mode="r")
                expected_shape = (expected_rows, len(metadata["fields"])) if suffix == "cats" else (expected_rows,)
                if mapped.shape != expected_shape:
                    raise RuntimeError(f"Cache shape mismatch: {dataset}/{name}")
                information["cache_files"][name] = {"shape": list(mapped.shape), "dtype": str(mapped.dtype),
                    "file_size_bytes": path.stat().st_size, "sha256": digest(path)}
                del mapped
                event("cache_file_verified", dataset=dataset, file=name, rows=expected_rows)
        information["metadata_sha256"] = digest(directory / "meta.json")
        provenance["datasets"][dataset] = information
    atomic_json(output / "provenance.json", provenance)

    class HarnessLog:
        def __init__(self, stream, name):
            self.stream, self.name, self.pending = stream, name, ""
        def write(self, text):
            text = sanitize(text)
            self.stream.write(text)
            self.stream.flush()
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                prefix = f"[{self.name}] "
                if line.startswith(prefix + "{"):
                    record = json.loads(line[len(prefix):])
                    event("epoch_complete", run=self.name, epoch=record["epoch"],
                          epoch_seconds=record["epoch_s"], lr=record["lr"])
            return len(text)
        def flush(self):
            self.stream.flush()

    original_evaluate = xr.evaluate
    warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: event(
        "warning", category=category.__name__, message=sanitize(message))
    status["state"] = "running"
    for dataset, directory in datasets.items():
        xr.configure_data(directory)
        data = (xr.load("train"), xr.load("valid"))
        for arm in arm_lists[dataset]:
            name = arm["name"]
            status["active_run"] = name
            run_dir = output / "runs" / name
            run_dir.mkdir(parents=True)
            atomic_json(run_dir / "config.json", arm)
            raw_evaluations = []
            model_profile = {}
            run_start = time.perf_counter()
            actual_counts = {"train_rows_per_epoch": len(data[0][1]),
                "train_steps_per_epoch": (len(data[0][1]) + arm["batch"] - 1) // arm["batch"],
                "validation_rows_per_epoch": len(data[1][1]), "full_training_epochs_planned": 2}
            event("run_started", run=name, dataset=dataset, **actual_counts)

            def instrumented_evaluate(model, cats, labels, batch, device):
                if not model_profile:
                    model_profile.update({"total_params": sum(p.numel() for p in model.parameters()),
                        "embedding_params": model.embedding.weight.numel(),
                        "nonembedding_params": sum(p.numel() for p in model.parameters()) - model.embedding.weight.numel(),
                        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                        "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                        "linear_shapes": [{"name": name, "in": module.in_features, "out": module.out_features}
                            for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]})
                phase = "validation" if len(raw_evaluations) < 2 else "test"
                if len(raw_evaluations) >= 3:
                    raise RuntimeError("More evaluations than frozen plan permits")
                t0 = time.perf_counter()
                auc, logloss = original_evaluate(model, cats, labels, batch, device)
                raw_evaluations.append({"phase": phase,
                    "epoch": len(raw_evaluations) + 1 if phase == "validation" else None,
                    "rows": len(labels), "auc": float(auc), "logloss": float(logloss),
                    "elapsed_seconds": time.perf_counter() - t0})
                atomic_json(run_dir / "evaluations_full_precision.json", raw_evaluations)
                return auc, logloss

            xr.evaluate = instrumented_evaluate
            try:
                with (run_dir / "harness.log").open("w") as stream:
                    redactor = HarnessLog(stream, name)
                    with contextlib.redirect_stdout(redactor), contextlib.redirect_stderr(redactor):
                        xr.run_arm(arm, data, output / "results_rounded.jsonl", True, "mps")
                if [value["phase"] for value in raw_evaluations] != ["validation", "validation", "test"]:
                    raise RuntimeError("Incomplete validation/test evaluation sequence")
                rounded = json.loads((output / "results_rounded.jsonl").read_text().splitlines()[-1])
                if rounded["name"] != name:
                    raise RuntimeError("Harness result/name mismatch")
                # These two pml_mv arms use only nn.Linear matrix products.
                model_profile["matrix_flops_per_example"] = rounded["serving_linear_flops_2x"]
                model_profile["matrix_flops_convention"] = "2*Linear MACs; pml_mv has no non-Linear matrix products"
                atomic_json(run_dir / "model_profile.json", model_profile)
                completed = {"run": name, "state": "completed", "elapsed_seconds": time.perf_counter() - run_start,
                             "actual_training_epochs": 2, "actual_training_rows": 2 * len(data[0][1]),
                             "actual_training_steps": 2 * actual_counts["train_steps_per_epoch"],
                             "validation_evaluations": 2, "selected_checkpoint_test_evaluations": 1}
                atomic_json(run_dir / "status.json", completed)
                status["completed_runs"].append(completed)
                event("run_completed", **completed)
            except Exception as exc:
                message = sanitize(exc)
                numerical = any(word in message.lower() for word in ("nan", "infinity", "non-finite", "not finite"))
                failed = {"run": name, "state": "failed", "exception": type(exc).__name__, "message": message,
                          "failure_class": "numerical" if numerical else "infrastructure_or_code",
                          "elapsed_seconds": time.perf_counter() - run_start,
                          "completed_validation_evaluations": sum(x["phase"] == "validation" for x in raw_evaluations),
                          "test_evaluations": sum(x["phase"] == "test" for x in raw_evaluations), "retry_attempts": 0}
                atomic_json(run_dir / "status.json", failed)
                status["failed_runs"].append(failed)
                event("run_failed", **failed)
            finally:
                xr.evaluate = original_evaluate
                gc.collect()
                torch.mps.empty_cache()
        del data
        gc.collect()
    status["active_run"] = None
    status["state"] = "completed" if not status["failed_runs"] else "completed_with_failures"
    status["finished_utc"] = utc()
    status["test_metrics_human_inspection"] = "original12_finished_global18_inspection_gate_still_required"
    event("queue_finished", completed_count=len(status["completed_runs"]), failed_count=len(status["failed_runs"]))


if __name__ == "__main__":
    main()
