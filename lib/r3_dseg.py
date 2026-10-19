"""S619: which ordinal does each later token belong to, by the model's own Gated DeltaNet memory?
The contribution of a source position s to the output at t in one head is w(t, s) v_s with
    w(t, s) = kappa_t . q_t,   kappa_s = beta_s k_s,   kappa_t = A_t kappa_{t-1},   A_t = exp(g_t)(I - beta_t k_t k_t^T)
(q, k l2-normalised, q scaled by 1/sqrt(d); the S610b effective weight written forward from s). Over the selected heads,
C(t, s) = sum |w(t, s)| ||v_s|| / ||o_t||; each later token is owned by the detected ordinal with the largest C.

install(v)            record q/k/v/g/beta/output of every DeltaNet layer when started
start(layers)/stop()  -> {layer: (q, k, vnorm, g, beta, onorm)} each [T, H, ...] (float32, on GPU)
contrib(keep, heads, s, t_hi)   C(t, s) for t in [s, t_hi)
select(records, k=8)  heads with the largest mean C(cue, ordinal) on canonical sentences
"""
from __future__ import annotations

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M

DS = dict(on=False, layers=None, cur=None, keep={})
_orig = M.torch_chunk_gated_delta_rule


def _l2(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _rec(query, key, value, g, beta, **kw):
    out, st = _orig(query, key, value, g=g, beta=beta, **kw)
    if DS["on"] and DS["cur"] is not None and (DS["layers"] is None or DS["cur"] in DS["layers"]):
        DS["keep"][DS["cur"]] = (_l2(query[0].float()) / (query.shape[-1] ** .5), _l2(key[0].float()), value[0].float().norm(dim=-1),
                                 g[0].float(), beta[0].float(), out[0].float().norm(dim=-1))
    return out, st


def install(v):
    if M.torch_chunk_gated_delta_rule is not _rec:
        prev = M.torch_chunk_gated_delta_rule
        global _orig
        _orig = prev
        M.torch_chunk_gated_delta_rule = _rec
    for i, L in enumerate(v.layers):
        if getattr(L, "linear_attn", None) is not None:
            L.linear_attn.register_forward_pre_hook(lambda mod, args, i=i: DS.__setitem__("cur", i))


def start(layers=None):
    DS.update(on=True, layers=None if layers is None else set(layers), keep={})


def stop():
    DS["on"] = False
    keep = DS["keep"]; DS["keep"] = {}
    return keep


@torch.no_grad()
def contrib(keep, heads, s, t_hi):
    """C(t, s) summed over heads [(layer, head)], for t = s .. t_hi-1 (vectorised over the heads of a layer)."""
    by = {}
    for l, h in heads:
        by.setdefault(l, []).append(h)
    C = torch.zeros(max(0, t_hi - s))
    if t_hi <= s:                                                                  # nothing after the source inside the range
        return C
    for l, hs in by.items():
        q, k, vn, gg, bb, on = keep[l]
        hs = torch.tensor(hs, device=q.device)
        kap = bb[s, hs][:, None] * k[s, hs]                                        # [h, d]
        out = []
        for t in range(s, t_hi):
            if t > s:
                kt = k[t, hs]
                kap = torch.exp(gg[t, hs])[:, None] * (kap - bb[t, hs][:, None] * kt * (kt * kap).sum(-1, keepdim=True))
            out.append(((kap * q[t, hs]).sum(-1).abs() * vn[s, hs] / (on[t, hs] + 1e-8)).sum())
        C += torch.stack(out).cpu()
    return C


@torch.no_grad()
def per_head(keep, s, c):
    """C(c, s) for every head of every recorded layer -> {layer: tensor[H]} (head selection)."""
    res = {}
    for l, (q, k, vn, gg, bb, on) in keep.items():
        kap = bb[s][:, None] * k[s]
        for t in range(s + 1, c + 1):
            kap = torch.exp(gg[t])[:, None] * (kap - bb[t][:, None] * k[t] * (k[t] * kap).sum(-1, keepdim=True))
        res[l] = ((kap * q[c]).sum(-1).abs() * vn[s] / (on[c] + 1e-8)).cpu()
    return res


def owners(keep, heads, srcs, t_hi):
    """for each position t in [min(srcs), t_hi): the source with the largest contribution (sources before t only)."""
    srcs = sorted(srcs)
    best = {}
    for s in srcs:
        if s >= t_hi:                                                              # a later source owns nothing before t_hi
            continue
        c = contrib(keep, heads, s, t_hi)
        for i, t in enumerate(range(s, t_hi)):
            if t not in best or float(c[i]) > best[t][1]:
                best[t] = (s, float(c[i]))
    return {t: s for t, (s, _) in best.items()}


def select(records, k=8):
    """records: list of {layer: tensor[H]} C(cue, ordinal) per head; top-k (layer, head) by mean."""
    layers = sorted(records[0])
    Mx = torch.stack([torch.stack([r[l] for l in layers]) for r in records]).mean(0)
    flat = torch.argsort(Mx.flatten(), descending=True)[:k]
    H = Mx.shape[1]
    return [(layers[int(i) // H], int(i) % H) for i in flat], float(Mx.flatten()[flat].mean())
