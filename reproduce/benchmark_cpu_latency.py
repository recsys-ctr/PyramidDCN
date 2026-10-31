#!/usr/bin/env python3
"""Bounded CPU graph diagnostic on released Criteo configs and real held-out IDs.

Weights are randomly initialized: these measurements are not checkpoint,
production, GPU, or accelerator latency results. No training is performed.
Run from the research workspace, or pass --code-dir for an exported release.
Only selected rows of test_cats.npy are copied from its read-only memory map.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_code_dir(explicit):
    if explicit is not None:
        return explicit.resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "exports" / "PyramidDCN_recsys"
        if (candidate / "experiments" / "x4_runner.py").is_file():
            return candidate
    raise ValueError("Pass --code-dir pointing to the portable release root")


def cpu_brand():
    # Capture only the CPU description, never hostnames, serials, or account IDs.
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0:
            # Select only the chip name; never persist other hardware fields.
            for record in json.loads(result.stdout).get("SPHardwareDataType", []):
                if record.get("chip_type"):
                    return record["chip_type"]
    return platform.processor() or "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-timed-seconds", type=float, default=120.0)
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    if min(args.batches + [args.sample_rows, args.iterations, args.repetitions]) < 1:
        parser.error("Batch sizes, sample rows, iterations, and repetitions must be positive")
    if args.warmup < 0 or max(args.batches) > args.sample_rows:
        parser.error("Warmup must be nonnegative and every batch must fit the sample")

    # These settings are local to this process. Do not pick up an unrelated dataset
    # or embedding-width override from the caller's shell environment.
    os.environ.pop("CTR_DATA", None)
    os.environ.pop("CTR_EMB", None)
    import numpy as np
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    code_dir = find_code_dir(args.code_dir)
    sys.path.insert(0, str(code_dir))
    xr = importlib.import_module("experiments.x4_runner")
    flops = importlib.import_module("experiments.flops")

    metadata = json.loads((args.data_dir / "meta.json").read_text())
    xr.configure_metadata(metadata, dataset="criteo_x4_001_cpu_diagnostic")
    if metadata["vocab_size"] != 910708 or len(metadata["fields"]) != 39 or xr.EMB != 16:
        raise ValueError("This diagnostic requires released Criteo dimensions: vocab 910708, F=39, E=16")
    mapped = np.load(args.data_dir / "test_cats.npy", mmap_mode="r")
    if mapped.ndim != 2 or mapped.shape[1] != len(metadata["fields"]):
        raise ValueError("Held-out input shape does not match metadata")
    rng = np.random.default_rng(args.seed)
    row_count = min(args.sample_rows, mapped.shape[0])
    selected = rng.choice(mapped.shape[0], size=row_count, replace=False)
    copied = np.array(mapped[selected], dtype=np.int64, copy=True)
    if copied.min() < 0 or copied.max() >= metadata["vocab_size"]:
        raise ValueError("Global categorical IDs exceed the vocabulary range")
    cats = torch.from_numpy(copied)
    # Views and input tuples are created before any timed forward.
    input_batches = {
        batch: [(None, cats[i:i + batch]) for i in range(0, row_count - batch + 1, batch)]
        for batch in args.batches
    }
    del mapped

    table1 = json.loads((code_dir / "configs" / "table1" / "criteo.json").read_text())
    table4 = json.loads((code_dir / "configs" / "table4" / "criteo.json").read_text())
    requested = [
        ("PyramidDCN_512_256_128", table1, "fair_criteo_v3n_stagger_s2024"),
        ("Constant_512_512_512", table4, "fair_criteo_v3n_r1_s2024"),
        ("BudgetConstant_344_344_344", table4, "fair_criteo_v3n_o1_bal_s2024"),
        ("ECN_Table1", table1, "v3_criteo_ecn_s2024"),
        ("FiBiNetPP_Table1", table1, "v3_criteo_fibinetpp_s2024"),
        ("DCNv2_rank_W_div_4_Table1", table1, "fair_criteo_v3f_r4_s2024"),
    ]
    models, model_info = {}, []
    build_start = time.perf_counter()
    with torch.inference_mode():
        for label, source, arm_name in requested:
            arm = next(record for record in source if record["name"] == arm_name)
            torch.manual_seed(args.seed)
            model = xr.CTRModel(arm).cpu().float().eval()
            embedding_params = model.embedding.weight.numel()
            total_params = sum(parameter.numel() for parameter in model.parameters())
            linear_flops = 2 * flops.serving_linear_macs(model, torch.zeros(1, 0), cats[:1])
            with FlopCounterMode(display=False) as counter:
                prediction = model(None, cats[:1])
            if prediction.shape != (1,) or not torch.isfinite(prediction).all():
                raise ValueError(f"Invalid diagnostic forward: {label}")
            models[label] = model
            model_info.append({
                "model": label, "released_arm": arm, "total_params": total_params,
                "embedding_params": embedding_params,
                "nonembedding_params": total_params - embedding_params,
                "linear_flops_per_example": linear_flops,
                "matrix_flops_per_example": counter.get_total_flops(),
                "training": False, "weight_source": "random_initialization",
            })
    initialization_seconds = time.perf_counter() - build_start

    cases = [(label, batch) for label in models for batch in args.batches]
    pilot = []
    with torch.inference_mode():
        for label, batch in cases:
            # The pilot is recorded separately and never pooled with measurements.
            model = models[label]
            inputs = input_batches[batch]
            for i in range(args.warmup):
                model(*inputs[i % len(inputs)])
            measurements = []
            for i in range(3):
                call_input = inputs[(i + args.warmup) % len(inputs)]
                start = time.perf_counter_ns()
                prediction = model(*call_input)
                measurements.append((time.perf_counter_ns() - start) / 1e6)
            pilot.append({"model": label, "batch": batch,
                          "median_ms": statistics.median(measurements), "calls": 3})
    projected_seconds = sum(item["median_ms"] for item in pilot) / 1000 * (
        args.warmup + args.iterations) * args.repetitions
    timed_iterations = args.iterations
    if projected_seconds > args.max_timed_seconds:
        timed_iterations = max(5, int(
            args.max_timed_seconds * 1000 /
            (sum(item["median_ms"] for item in pilot) * args.repetitions)
        ) - args.warmup)
        timed_iterations = min(args.iterations, timed_iterations)
    print(json.dumps({"event": "pilot", "projected_seconds": projected_seconds,
                      "actual_iterations_per_repetition": timed_iterations}), flush=True)

    repetitions, execution_order = [], []
    order_rng = random.Random(args.seed)
    measure_start = time.perf_counter()
    with torch.inference_mode():
        for rep in range(args.repetitions):
            order = cases.copy()
            order_rng.shuffle(order)
            execution_order.append([{"model": label, "batch": batch} for label, batch in order])
            for label, batch in order:
                model, inputs = models[label], input_batches[batch]
                offset = (rep * (args.warmup + timed_iterations)) % len(inputs)
                for i in range(args.warmup):
                    model(*inputs[(offset + i) % len(inputs)])
                measurements = []
                for i in range(timed_iterations):
                    call_input = inputs[(offset + args.warmup + i) % len(inputs)]
                    start = time.perf_counter_ns()
                    prediction = model(*call_input)
                    measurements.append((time.perf_counter_ns() - start) / 1e6)
                repetitions.append({
                    "model": label, "batch": batch, "repetition": rep + 1,
                    "calls": timed_iterations, "latencies_ms": measurements,
                    "median_ms": statistics.median(measurements),
                    "p95_ms": float(np.percentile(measurements, 95)),
                })
            print(json.dumps({"event": "repetition_complete", "repetition": rep + 1}), flush=True)
    measurement_wall_seconds = time.perf_counter() - measure_start
    summaries = []
    info_by_label = {item["model"]: item for item in model_info}
    for label, batch in cases:
        reps = [item for item in repetitions if item["model"] == label and item["batch"] == batch]
        pooled = [value for item in reps for value in item["latencies_ms"]]
        rep_medians = [item["median_ms"] for item in reps]
        rep_mean = statistics.mean(rep_medians)
        rep_sd = statistics.stdev(rep_medians) if len(rep_medians) > 1 else 0.0
        info = info_by_label[label]
        summaries.append({
            "model": label, "batch": batch,
            "total_params": info["total_params"], "embedding_params": info["embedding_params"],
            "nonembedding_params": info["nonembedding_params"],
            "matrix_flops_per_example": info["matrix_flops_per_example"],
            "timed_calls": len(pooled), "median_ms": statistics.median(pooled),
            "p95_ms": float(np.percentile(pooled, 95)),
            "repetition_medians_ms": rep_medians,
            "repetition_median_mean_ms": rep_mean,
            "repetition_median_sd_ms": rep_sd,
            "repetition_median_cv_pct": 100 * rep_sd / rep_mean,
            "repetition_median_min_ms": min(rep_medians),
            "repetition_median_max_ms": max(rep_medians),
        })
    source_files = [
        "experiments/x4_runner.py", "experiments/cond_pyramid_screen.py",
        "experiments/pyramid_baselines.py", "experiments/x4_baselines.py",
        "experiments/extra_baselines.py", "experiments/flops.py",
        "configs/table1/criteo.json", "configs/table4/criteo.json",
    ]
    result = {
        "started_utc": started_utc,
        "scope": "CPU graph diagnostic, untrained random weights; no training or checkpoint claim",
        "environment": {
            "system": platform.system(), "system_release": platform.release(),
            "machine": platform.machine(), "cpu_brand": cpu_brand(),
            "python_version": platform.python_version(), "torch_version": torch.__version__,
            "numpy_version": np.__version__, "device": "cpu", "dtype": "float32",
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "execution": "eager eval + torch.inference_mode; no compilation",
        },
        "input": {
            "dataset": "Criteo_x4_001", "split": "test", "available_rows": int(metadata["test"]["rows"]),
            "selected_rows": row_count, "selection": "numpy default_rng(seed).choice without replacement",
            "seed": args.seed, "vocab_size": metadata["vocab_size"], "num_fields": len(metadata["fields"]),
            "embedding_dim": xr.EMB, "shared_input_width": 512,
            "input_id_min": int(copied.min()), "input_id_max": int(copied.max()),
            "distinct_input_ids": int(np.unique(copied).size),
            "batches_preloaded_per_size": {str(batch): len(values) for batch, values in input_batches.items()},
            "timed_io": False, "full_array_loaded": False,
        },
        "protocol": {
            "warmup_per_case_per_repetition": args.warmup,
            "requested_iterations_per_repetition": args.iterations,
            "actual_iterations_per_repetition": timed_iterations,
            "repetitions": args.repetitions, "weight_seed": args.seed,
            "pilot_calls_per_case": 3, "pilot": pilot,
            "pilot_projected_seconds": projected_seconds,
            "max_timed_seconds": args.max_timed_seconds,
            "initialization_seconds_excluded": initialization_seconds,
            "measurement_wall_seconds_including_warmup": measurement_wall_seconds,
            "timer": "time.perf_counter_ns around one full model(None, cats) call",
            "includes": ["embedding lookup", "shared input projection", "full backbone", "logit head"],
            "excludes": ["initialization", "input I/O and selection", "input batch view creation", "training"],
            "case_order": "seeded shuffle of all model/batch cases separately in each repetition",
            "execution_order": execution_order,
            "aggregation": "pooled per-call median and NumPy linear-interpolation P95; SD/CV of repetition medians",
        },
        "source_sha256": {relative: sha256(code_dir / relative) for relative in source_files},
        "benchmark_script_sha256": sha256(Path(__file__)),
        "models": model_info, "summaries": summaries, "repetitions": repetitions,
        "limitations": [
            "Random weights; trained checkpoint latency and quality are not measured.",
            "Single CPU process, one machine, eager PyTorch; no isolated serving or production load.",
            "Resident 4096-row input working set with warmup; not a cold or full-test embedding-cache measurement.",
            "All six full-vocabulary models stay resident during timing; no peak-memory measurement.",
            "P95 describes per-call samples in this diagnostic, not a service-level tail-latency guarantee.",
            "CPU affinity, clock frequency, thermals, and competing OS work are not controlled.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cpu_latency_results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "cpu_latency_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for item in summaries:
        print(f"{item['model']:28s} B={item['batch']:3d} median={item['median_ms']:.6f} ms "
              f"p95={item['p95_ms']:.6f} ms CV={item['repetition_median_cv_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
