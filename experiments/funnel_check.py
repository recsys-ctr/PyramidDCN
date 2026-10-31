"""Funnel check (Thm. 2.1/2.2) on a trained protocol-v3 low-rank DCNv2 (pyramid:o256_dcnv2, project=False).

The cross stream s_{l+1} = s_l + x0 * (U_l V_l s_l + b_l) is a read-write network with reads R_l = V_l (r x d) and
readout R_L = the cross half of fc (the head also reads the parallel DNN of x0, which the theorem allows).  In float64
on validation inputs this records, per depth l = 0..L:
  m_theory  min{d, (L-l) r + 1}                    (Thm. 2.2, generic weights)
  m         dim S_l, S_l = row(R_l) + ... + row(R_L)  (numerical rank of the stacked reads)
  maxdiff   max |logit change| when s_l is replaced by its projection onto S_l and the rest is rerun  (Thm. 2.1a)
  grad_rank rank of the per-instance gradients d logit / d s_l, and sv_gap = sigma_{m+1} / sigma_1  (tightness)
  dead      share of the energy sum ||s_l||^2 orthogonal to S_l"""
import copy
import torch


def _rank(M, tol=1e-10):
    s = torch.linalg.svdvals(M)
    k = int((s > s[0] * tol).sum())
    return k, s


@torch.no_grad()
def _stream(m, x0):
    S = [x0]
    for V, U in zip(m.V, m.U):
        S.append(S[-1] + x0 * U(V(S[-1])))
    return S


def funnel_check(model, cats, n_proj=20000, n_grad=3000):
    b = model.base
    assert b.cross_widths is None and not isinstance(b.V[0], torch.nn.Identity), "needs a low-rank DCNv2 cross stack"
    m = copy.deepcopy(b).cpu().double().eval()
    emb = copy.deepcopy(model.embedding).cpu().double().eval()
    with torch.no_grad():
        x0 = emb(cats[:n_proj].long()).flatten(1)
        dnn = m.parallel_dnn(x0)
    d, L, r = x0.shape[1], len(m.V), m.V[0].weight.shape[0]
    reads = [V.weight.detach() for V in m.V] + [m.fc.weight.detach()[:, :d]]
    Q = []
    for l in range(L + 1):
        k, _ = _rank(torch.cat(reads[l:], 0))
        Q.append(torch.linalg.svd(torch.cat(reads[l:], 0), full_matrices=False)[2][:k])  # k x d, orthonormal rows

    def rest(l, s, x0, dnn):  # run layers l..L-1 from stream s, then the head
        for V, U in list(zip(m.V, m.U))[l:]:
            s = s + x0 * U(V(s))
        return m.fc(torch.cat([s, dnn], -1)).squeeze(-1)

    S = _stream(m, x0)
    with torch.no_grad():
        y = rest(L, S[L], x0, dnn)
    out = dict(d=d, r=r, m_theory=[min(d, (L - l) * r + 1) for l in range(L + 1)], m=[q.shape[0] for q in Q],
               maxdiff=[], grad_rank=[], sv_gap=[], dead=[])
    for l in range(L + 1):
        P = Q[l].T @ Q[l]
        with torch.no_grad():
            sp = S[l] @ P
            out["maxdiff"].append(float((rest(l, sp, x0, dnn) - y).abs().max()))
            out["dead"].append(round(float(((S[l] - sp) ** 2).sum() / (S[l] ** 2).sum()), 4))
        s = S[l][:n_grad].clone().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(rest(l, s, x0[:n_grad], dnn[:n_grad]).sum(), s)  # row i = d logit_i / d s_l
        k, sv = _rank(g)
        out["grad_rank"].append(k)
        out["sv_gap"].append(float(sv[k] / sv[0]) if k < len(sv) else 0.0)
    return out
