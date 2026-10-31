"""Gradient outer-product spectrum in a trained model.
For every nn.Linear inside the interaction network (the shared input projection excluded), C = E[g g^T] with
g = d logit / d (layer output) over validation inputs, and
  width   the output width,
  erank   effective rank of C (Roy & Vetterli: exp of the entropy of its normalized eigenvalues),
  frac    erank / width,
  rank99  eigenvalues needed for 99% of trace(C),
ordered by the first call in the forward pass (depth).  Per-example gradients come from one backward of the summed
logits in eval mode, where examples do not interact."""
import copy
import torch
import torch.nn as nn


def _spectrum(C):
    lam = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)
    tot = lam.sum()
    if tot <= 0:
        return 0.0, 0
    p = lam / tot
    q = p[p > 0]
    erank = float(torch.exp(-(q * q.log()).sum()))
    rank99 = int((torch.cumsum(p, 0) < 0.99).sum()) + 1
    return erank, rank99


def agop_check(model, cats, n=8192, chunk=1024, max_width=4096):
    m = copy.deepcopy(model).cpu().float().eval()
    skip = {id(mod) for mod in m.embedding.modules()} if getattr(m, "embedding", None) is not None else set()
    lin = [(name, mod) for name, mod in m.named_modules() if isinstance(mod, nn.Linear) and id(mod) not in skip]
    store, acc, width, order = [], {}, {}, {}

    def hook(name):
        def f(mod, inp, out):
            if out.requires_grad:
                out.retain_grad()
                store.append((name, out))
        return f

    hs = [mod.register_forward_hook(hook(name)) for name, mod in lin]
    n = min(n, cats.shape[0])
    try:
        for s in range(0, n, chunk):
            store.clear()
            y = m(None, cats[s:s + chunk].long())
            y.sum().backward()
            calls = {}
            for name, out in store:
                k = calls.get(name, 0)
                calls[name] = k + 1
                key = f"{name}#{k}" if k else name
                if out.grad is None:
                    continue
                g = out.grad.reshape(out.shape[0], -1).double()
                if g.shape[1] > max_width:
                    continue
                acc[key] = acc.get(key, 0) + g.T @ g
                width[key] = g.shape[1]
                order.setdefault(key, len(order))
            m.zero_grad(set_to_none=True)
    finally:
        for h in hs:
            h.remove()
    out = []
    for key in sorted(acc, key=order.get):
        er, r99 = _spectrum(acc[key] / n)
        out.append(dict(name=key, width=width[key], erank=round(er, 2), frac=round(er / width[key], 4), rank99=r99))
    return out
