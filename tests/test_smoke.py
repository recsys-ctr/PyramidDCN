"""CPU smoke checks; no benchmark downloads or original research checkout required.

    python tests/test_smoke.py
    python tests/test_smoke.py --full-catalog

The optional catalog pass preserves every recorded width/operator setting while
replacing only the embedding vocabulary with 128 synthetic IDs. It does not run
benchmark training or validate the published benchmark metrics.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# A smoke test must never load an unrelated dataset selected in the shell.
_CONFIGURED_DATA = os.environ.pop("CTR_DATA", None)
from experiments import x4_runner as xr

FULL_CATALOG = "--full-catalog" in sys.argv
if FULL_CATALOG:
    sys.argv.remove("--full-catalog")
CATALOG_SUMMARY = {}


def small_metadata():
    return {"fields": ["a", "b", "c", "d"], "vocab_size": 32, "emb_dim": 8}


def pyramid_arm():
    return dict(model="pml_mv", views=1, anchor="stagger", dims=[32, 16, 8],
                rows=[0, 0, 0], t=1.0, input_proj=32)


def distinct_catalog():
    """Deduplicate only names, seeds, and post-training diagnostic flags."""
    metadata = json.loads((ROOT / "configs/dataset_metadata.json").read_text())
    excluded = {"name", "seed", "erank", "funnel", "agop"}
    records = []
    for path in sorted((ROOT / "configs/all").glob("*.json")):
        seen = set()
        for arm in json.loads(path.read_text()):
            key = json.dumps({k: v for k, v in arm.items() if k not in excluded}, sort_keys=True)
            if key not in seen:
                seen.add(key)
                records.append((path.stem, arm, metadata[path.stem]))
    return records


class ExperimentSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(2024)
        xr.configure_metadata(small_metadata())

    def assert_backward(self, model, cats, labels):
        model.eval()
        with torch.no_grad():
            logits = model(None, cats)
        self.assertEqual(tuple(logits.shape), (len(labels),))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        model.train()
        loss = model.loss(cats, labels)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_pyramid_formula_and_input_slices(self):
        # Direct head input is already the projected representation.
        # Three-layer halving reads all coordinates, then odd positions,
        # then positions 2,6,10,14 (one-based) from the ORIGINAL input.
        x = torch.randn(4, 32, requires_grad=True)
        head = xr.PyramidMultiViewHead([32, 16, 8], [0, 0, 0], 1.0, anchor="stagger")
        expected_indices = [list(range(32)), list(range(0, 32, 2)), list(range(1, 32, 4))]
        expected_states = []
        state = x
        for indices, layer, norm in zip(expected_indices, head.layers, head.norms):
            selected = x[:, indices]
            torch.testing.assert_close(layer.proj_x0(x), selected, rtol=0, atol=0)
            cross = F.linear(state, layer.cross[0].weight, layer.cross[0].bias)
            state = F.layer_norm(selected * cross + selected, norm.normalized_shape,
                                 norm.weight, norm.bias, norm.eps)
            expected_states.append(state)
        for actual, expected in zip(head.states([x]), expected_states):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        head([x]).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_baselines_forward_loss_backward(self):
        width = 32
        cases = {
            "pyramid": pyramid_arm(),
            "dcnv2": dict(model="pyramid:o256_dcnv2", input_proj=width,
                          preset=dict(project=False, rank=8, dnn_widths=[width] * 5),
                          kw=dict(net_dropout=0.0)),
            "mldcn": dict(model="mldcn", input_proj=width, rank=width, layers=3, t=0.5),
            "masknet": dict(model="baseline:masknet", input_proj=width,
                            overrides=dict(parallel_block_dim=width, reduction_ratio=2,
                                           dnn_hidden_units=[width] * 5, net_dropout=0.0)),
            "qnn": dict(model="pyramid:p_qnn", input_proj=width,
                        preset=dict(widths=[width] * 4, num_rows=[5] * 4), kw=dict(net_dropout=0.0)),
            "ecn": dict(model="pyramid:p_ecn", input_proj=width,
                        preset=dict(widths=[width] * 4), kw=dict(net_dropout=0.0)),
            "fcn": dict(model="baseline:fcn", input_proj=width, overrides=dict(net_dropout=0.0)),
            "gdcn": dict(model="pyramid:p_gdcn", input_proj=width,
                         preset=dict(d=width, in_proj=False, cross_widths=[width] * 3,
                                     dnn_hidden_units=[width] * 3), kw=dict(net_dropout=0.0)),
            "finalmlp": dict(model="baseline:finalmlp", input_proj=width,
                             overrides=dict(mlp1_hidden_units=[width] * 2, mlp2_hidden_units=[width],
                                            num_heads=8, mlp1_dropout=0.0, mlp2_dropout=0.0)),
            "fibinetpp": dict(model="baseline:fibinetpp", input_proj=width,
                              overrides=dict(dnn_hidden_units=[width] * 3)),
        }
        cats = torch.arange(16).reshape(4, 4)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        for name, arm in cases.items():
            with self.subTest(model=name):
                torch.manual_seed(2024)
                self.assert_backward(xr.CTRModel(arm), cats, labels)

    def test_one_epoch_training_validation_test_cli(self):
        with tempfile.TemporaryDirectory(prefix="pyramiddcn_smoke_") as directory:
            path = Path(directory)
            (path / "meta.json").write_text(json.dumps(small_metadata()))
            for split in ("train", "valid", "test"):
                np.save(path / f"{split}_cats.npy", np.arange(32, dtype=np.int32).reshape(8, 4))
                np.save(path / f"{split}_y.npy", np.array([0, 1] * 4, dtype=np.float32))
            arm = dict(pyramid_arm(), name="synthetic_s2024", seed=2024, max_epochs=1,
                       batch=4, emb_reg=0.0, clip=10.0, emb_init_std=1e-4, protocol="fair1ep")
            (path / "arms.json").write_text(json.dumps([arm]))
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            environment.pop("CTR_DATA", None)
            # Also supports development environments with dependencies on sys.path.
            environment["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
            result = subprocess.run(
                [sys.executable, "-m", "experiments.x4_runner", "--data-dir", str(path),
                 "--arms", str(path / "arms.json"), "--device", "cpu", "--threads", "1",
                 "--eval_test", "--out", str(path / "results.jsonl")],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["best_epoch"], 1)
            self.assertEqual(len(record["history"]), 1)
            for key in ("valid_auc", "valid_logloss", "test_auc", "test_logloss"):
                self.assertTrue(np.isfinite(record[key]), key)
            self.assertEqual(record["serving_linear_flops_2x"], 5392)
            self.assertTrue(0 <= record["test_auc"] <= 1)

    @unittest.skipUnless(FULL_CATALOG, "enable with --full-catalog")
    def test_all_distinct_recorded_configurations(self):
        counts = {}
        for dataset, arm, metadata in distinct_catalog():
            with self.subTest(dataset=dataset, arm=arm["name"]):
                torch.manual_seed(2024)
                xr.configure_metadata(dict(metadata, vocab_size=128), dataset=f"synthetic_{dataset}")
                cats = torch.arange(4 * len(metadata["fields"])).reshape(4, -1) % 128
                labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
                model = xr.CTRModel(arm)
                self.assert_backward(model, cats, labels)
                del model
                gc.collect()
                counts[dataset] = counts.get(dataset, 0) + 1
        CATALOG_SUMMARY.update(counts)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    if FULL_CATALOG:
        print("CATALOG_SMOKE_COUNTS=" + json.dumps(CATALOG_SUMMARY, sort_keys=True))
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
