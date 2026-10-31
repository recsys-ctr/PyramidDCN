"""Effective rank (Roy & Vetterli 2007: exp of the entropy of the normalized singular values) inside a trained
PyramidDCN (pml_mv head on the shared projection), per layer:
  rep   effective rank of the centred outputs x_i (n x d_i) on validation inputs, layer 0 = the input x
  rep_frac  rep / d_i
  w     effective rank of the crossing weight A_i (d_i x d_{i-1}), w_frac = w / min(d_i, d_{i-1})"""
import copy
import torch


def erank(M):
    s = torch.linalg.svdvals(M.double())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def erank_check(model, cats, n=20000):
    m = copy.deepcopy(model).cpu().float().eval()
    net = m.net
    views = m.embed_views(cats[:n].long())
    xs = [views[0]] + net.states(views)
    x = torch.cat(xs[1:], dim=-1) if getattr(net, "readout", "last") == "concat" else xs[-1]  # multi-layer readout
    assert torch.allclose(net.fc(x).squeeze(-1), m(None, cats[:n].long()), atol=1e-5)
    rep = [erank(z - z.mean(0)) for z in xs]
    ws = [layer.cross[0].weight for layer in net.layers if len(layer.cross)]  # the "mix" cross has no single A_i
    w = [erank(a) for a in ws]
    return dict(width=[z.shape[1] for z in xs], rep=[round(v, 2) for v in rep],
                rep_frac=[round(v / z.shape[1], 4) for v, z in zip(rep, xs)],
                w=[round(v, 2) for v in w], w_frac=[round(v / min(a.shape), 4) for v, a in zip(w, ws)])
