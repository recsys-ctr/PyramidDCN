# CPU graph latency diagnostic

This is a supplementary diagnostic of **randomly initialized, untrained models** on one Apple M2 Max CPU. It measures the released execution graphs, including full-vocabulary embedding lookup, the shared input projection, the backbone, and the logit head. It does not measure trained checkpoints, GPU performance, production throughput, or service latency.

## Recorded protocol

- PyTorch 2.11.0, NumPy 2.4.4, Python 3.14.4; CPU float32; one intra-op and one inter-op thread; eager `eval()` with `torch.inference_mode()`.
- Criteo_x4_001: vocabulary 910,708, 39 categorical fields, embedding dimension 16, shared projection 624→512. The same 4,096 sampled held-out rows are used for every model. The full test array is opened with a read-only memory map; only the selected rows are copied. No raw examples or selected row indices are released.
- Batch sizes 1 and 128. Inputs are preloaded and reused. Six full-vocabulary models remain resident during measurement.
- Each model/batch case receives 10 warmup calls followed by 50 timed calls, repeated three times in seeded shuffled case order. The three repetitions are timing repetitions, not independent training seeds. A separate three-call pilot is excluded from the reported timing samples.
- A timed sample covers one complete `model(None, cats)` call with `time.perf_counter_ns()`. Initialization, data loading, row selection, and batch creation are excluded.
- Median and P95 pool all 150 calls per case; P95 uses NumPy linear interpolation. SD and CV summarize the three repetition medians. The JSON preserves every timed call, case order, complete architecture configurations, code/config hashes, parameter counts, and environment details.

The recorded data are the **final chronological run**, after a preliminary run and a rerun to add safe CPU-description collection. The final run was not selected for favorable timing. The recorded UTC start is `2026-10-31T06:41:36.981724+00:00`.

## Reproduce

From the repository root, install `requirements.txt` and prepare the matching Criteo cache as described in [data/README.md](../../data/README.md). Its directory must contain `meta.json` and `test_cats.npy`.

```sh
python reproduce/benchmark_cpu_latency.py \
  --code-dir . \
  --data-dir /path/to/prepared/criteo \
  --out outputs/cpu_latency \
  --batches 1 128 --sample-rows 4096 \
  --warmup 10 --iterations 50 --repetitions 3 --seed 2024
```

The script records its actual iterations and may reduce them if its pilot projects a timing phase longer than 120 seconds. No reduction occurred in the archived run. Timing numbers depend on hardware, software, and system load; this command reproduces the procedure rather than the exact durations.

## Results and limits

| Graph | B=1 median (ms) | B=128 median (ms) |
| --- | ---: | ---: |
| PyramidDCN [512,256,128] | 0.096542 | 0.450313 |
| Constant [512,512,512] | 0.100000 | 0.596709 |
| Budget constant [344,344,344] | 0.094959 | 0.515626 |
| ECN | 0.147375 | 0.664896 |
| FiBiNet++ | 0.152355 | 2.030396 |
| DCNv2 (rank W/4) | 0.162312 | 1.124958 |

At batch 128, PyramidDCN has a lower observed median than the two constant-width controls; at batch 1, the budget-matched constant-width graph has a slightly lower median. Small single-request differences do not establish a reliable advantage. DCNv2 at batch 128 has 34.33% CV across repetition medians, so precise speedup or tail claims are especially weak there. FLOPs are not a latency guarantee.

Warmup and the resident 4,096-row working set do not represent cold-cache or full-test-vocabulary access. CPU affinity, clock frequency, thermals, and competing OS work were not controlled. These short single-process measurements provide neither confidence intervals nor service-level tail guarantees. Matrix FLOPs count matrix products rather than every operation. Non-embedding parameter counts are recomputed from each instantiated graph and include the shared input projection; older log fields named `dense_params` have differing counting scopes.

## Files

- [cpu_latency_results.json](cpu_latency_results.json): raw timings, summaries, protocol, configurations, and source hashes.
- [cpu_latency_results.csv](cpu_latency_results.csv): the same per-case summaries in CSV form.
- [cpu_latency_validation.json](cpu_latency_validation.json): checks of arm identity, recorded matrix/linear FLOPs, parameter decomposition, timing aggregation, source hashes, and absence of targeted personal paths or account names. This is a bounded validation record, not a universal privacy guarantee.
- [benchmark_cpu_latency.py](../../reproduce/benchmark_cpu_latency.py): the exact script used for the archived run. Its content hash matches the JSON record.

The data files contain no raw categorical inputs, checkpoints, private filesystem paths, hostnames, hardware serials, personal account names, or credentials.
