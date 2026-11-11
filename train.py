"""Train and evaluate the paper's configurations (Tables 1-4) with the one-epoch protocol.

    python -m train --arms configs/table1/criteo.json --data-dir /path/to/prepared/criteo --eval_test

Data: a prepared dataset directory (data/README.md) containing meta.json and {train,valid,test}_{cats,y}.npy.
Protocol (identical for every model): every model sees each training example once (one epoch), Adam with learning
rate 1e-3, batch size 4096, gradient-norm clipping at 10, no regularization and no learning-rate schedule.  The
embedding table is re-initialized with N(0, 1e-4^2) for every model.  The epoch-end model is evaluated on the
validation split and, with --eval_test, on the test split.

A configuration ("arm") names a class of models/ in its field "model" and gives the keyword arguments of that class
in "args".  Every model reads the same shared projected input (ProjEmbedding, width "input_proj").
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from models import MODELS, BaseModel

ROOT = Path(__file__).resolve().parent
LR = 1e-3

# The models can be imported without downloading datasets. Training requires
# an explicit prepared directory, selected with --data-dir or CTR_DATA.
DATA = None
DATASET = None
META = None
N_FIELDS, EMB = 39, 16


def resolve_data_dir(value=None):
    selected = value or os.environ.get("CTR_DATA")
    if not selected:
        raise ValueError("Pass --data-dir or set CTR_DATA to a prepared dataset directory")
    path = Path(selected).expanduser().resolve()
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Prepared metadata not found: {path / 'meta.json'}")
    return path


def configure_metadata(metadata, dataset="synthetic"):
    """Configure model shapes from prepared metadata (also used for smoke tests)."""
    global META, DATASET, N_FIELDS, EMB
    if "fields" not in metadata or "vocab_size" not in metadata:
        raise ValueError("metadata needs fields and vocab_size")
    META = dict(metadata)
    DATASET = dataset
    N_FIELDS = len(META["fields"])
    EMB = int(os.environ.get("CTR_EMB", META.get("emb_dim", 16)))


def configure_data(data_dir=None):
    """Select the prepared dataset."""
    global DATA
    path = resolve_data_dir(data_dir)
    with (path / "meta.json").open() as stream:
        metadata = json.load(stream)
    DATA = str(path)
    dataset = f"{path.parent.name}_{path.name.replace('fuxi_', '')}"
    configure_metadata(metadata, dataset)


if os.environ.get("CTR_DATA"):
    configure_data()


def load(split):
    if DATA is None:
        configure_data()
    cats = torch.from_numpy(np.asarray(np.load(f"{DATA}/{split}_cats.npy", mmap_mode="r")))
    y = torch.from_numpy(np.asarray(np.load(f"{DATA}/{split}_y.npy", mmap_mode="r")))
    return cats, y


@torch.no_grad()
def linear_macs(model: nn.Module, dense: torch.Tensor, cats: torch.Tensor) -> int:
    """Per-example multiply-adds (MACs) of the nn.Linear calls in the eval graph; FLOPs = 2 x MACs.

    Counts every nn.Linear call via forward hooks (so reused layers count per call);
    embedding lookups, activations, LayerNorm, Hadamard products and reductions are excluded.
    Direct matrix products outside nn.Linear are added in reproduce/matrix_flops.py.
    """
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        rows = inputs[0].numel() // module.in_features
        total += rows * module.in_features * module.out_features

    was_training = model.training
    model.eval()
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    try:
        model(dense[:1], cats[:1])
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total


class ProjEmbedding(nn.Module):
    """Shared input stage of every model: one EMB-wide table, flattened (d0), then a bias-free Linear d0 -> W
    (xavier normal, as FuxiCTR initializes Linear layers).  Every backbone sees the same W-wide input, viewed as
    W/8 groups of width 8 (models that only flatten their input are unaffected by the grouping)."""

    def __init__(self, vocab_size, num_fields, emb_dim, width, pe=8):
        super().__init__()
        assert width % pe == 0, width
        self.table = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.table.weight, std=1e-4)
        self.proj = nn.Linear(num_fields * emb_dim, width, bias=False)
        nn.init.xavier_normal_(self.proj.weight)
        self.pf, self.pe = width // pe, pe

    @property
    def weight(self):  # the runner treats model.embedding.weight as the embedding table
        return self.table.weight

    def forward(self, cats):
        return self.proj(self.table(cats.long()).flatten(1)).view(-1, self.pf, self.pe)


class CTRModel(nn.Module):
    """The model named by arm["model"] (models/) on the shared projected input of width arm["input_proj"]."""

    def __init__(self, arm):
        super().__init__()
        if META is None:
            configure_data()
        if arm["model"] not in MODELS:
            raise ValueError(f"unknown model {arm['model']!r}; choose from {tuple(MODELS)}")
        cls, args, width = MODELS[arm["model"]], arm.get("args", {}), arm["input_proj"]
        if issubclass(cls, BaseModel):  # built on the W-wide projected input, viewed as W/8 groups of width 8
            self.base = cls(META["vocab_size"], width // 8, 8, **args)
        else:
            self.base = None
            # Placeholder table, replaced by the shared projected input below. It is still created and initialized
            # here so that the random-number stream, and therefore every released run, is reproduced exactly.
            self.embedding = nn.Embedding(META["vocab_size"], 8)
            nn.init.normal_(self.embedding.weight, std=1e-4)
            self.net = cls(width, **args)
        pe = ProjEmbedding(META["vocab_size"], N_FIELDS, EMB, width)
        if self.base is not None:
            self.base.embedding = pe
        self.embedding = pe

    def forward(self, dense, cats):
        if self.base is not None:
            return self.base(dense, cats)
        return self.net(self.embedding(cats).flatten(1))

    def loss(self, cats, y):
        if self.base is not None:
            return self.base.loss(cats, y)
        return F.binary_cross_entropy_with_logits(self(None, cats), y)


@torch.no_grad()
def evaluate(model, cats, y, batch, device):
    model.eval()
    logits = [model(None, cats[i : i + batch].to(device)).float().cpu() for i in range(0, len(y), batch)]
    p = torch.sigmoid(torch.cat(logits)).double().numpy()
    return roc_auc_score(y.numpy(), p), log_loss(y.numpy(), np.clip(p, 1e-7, 1 - 1e-7))


def run_arm(arm, data, out, eval_test, device):
    """Train one configuration for one epoch and append its result record to `out`."""
    if arm["max_epochs"] != 1 or arm["emb_reg"]:
        raise ValueError("the released protocol trains for one epoch without regularization")
    (tr_cats, tr_y), (va_cats, va_y) = data
    seed = arm["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CTRModel(arm)
    nn.init.normal_(model.embedding.weight, 0.0, arm["emb_init_std"])  # one embedding initialization for every model
    macs = linear_macs(model, torch.zeros(2, 0), tr_cats[:2])
    model = model.to(device)
    emb_params = [model.embedding.weight]
    dense_params = [p for p in model.parameters() if p is not model.embedding.weight]
    opt = torch.optim.Adam([{"params": emb_params, "lr": LR}, {"params": dense_params, "lr": LR}])
    batch, t_start = arm["batch"], time.time()
    model.train()
    perm = torch.randperm(len(tr_y), generator=torch.Generator().manual_seed(seed * 1000 + 1))
    losses = []
    for i in range(0, len(tr_y), batch):
        idx = perm[i : i + batch]
        loss = model.loss(tr_cats[idx].to(device), tr_y[idx].to(device))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), arm["clip"])
        opt.step()
        losses.append(loss.item())
    auc, ll = evaluate(model, va_cats, va_y, batch, device)
    history = [dict(epoch=1, train_loss=round(float(np.mean(losses)), 6), valid_auc=round(auc, 6),
                    valid_logloss=round(ll, 6), lr=opt.param_groups[0]["lr"], epoch_s=round(time.time() - t_start))]
    print(f"[{arm['name']}] " + json.dumps(history[-1]), flush=True)
    # params counts every parameter; dense_params excludes the embedding table (the count of Section 3.3)
    rec = dict(arm, dataset=DATASET, best_epoch=1, valid_auc=round(auc, 6), valid_logloss=round(ll, 6),
               linear_macs=macs, linear_flops=2 * macs,
               params=sum(p.numel() for p in emb_params + dense_params),
               dense_params=sum(p.numel() for p in dense_params),
               total_s=round(time.time() - t_start), history=history)
    if eval_test:
        rec["test_auc"], rec["test_logloss"] = (round(v, 6) for v in evaluate(model, *load("test"), batch, device))
    print("RESULT " + json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True, help="JSON list of configurations, or path to a JSON file")
    p.add_argument("--data-dir", help="Prepared dataset directory (defaults to CTR_DATA)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--out", default=str(ROOT / "outputs" / "results.jsonl"))
    a = p.parse_args()
    configure_data(a.data_dir)
    torch.set_num_threads(a.threads)
    data = (load("train"), load("valid"))
    arm_path = Path(a.arms) if not a.arms.lstrip().startswith("[") else None
    arms = json.loads(arm_path.read_text() if arm_path is not None else a.arms)
    failures = []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        try:
            run_arm(arm, data, a.out, a.eval_test, a.device)
        except Exception as exc:  # keep the queue alive; the failure is logged for diagnosis
            print(f"ARM FAILED [{arm['name']}]: {type(exc).__name__}: {exc}", flush=True)
            failures.append(arm["name"])
    if failures:
        raise SystemExit(f"{len(failures)} arm(s) failed: {failures}")
