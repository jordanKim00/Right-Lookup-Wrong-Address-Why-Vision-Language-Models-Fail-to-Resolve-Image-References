"""S618: the model's own "from the end" code in a Gated DeltaNet head (S617). During the plan pass one layer's q/k/v/g/beta
for one head are kept; afterwards, for each source position c, W_b = sum_{r in b} |w(c, r)| ||v_r|| (exact backward sweep);
ranking blocks by W (descending) is the position counted from the end.
"""
from __future__ import annotations

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M

NB = dict(on=False, layer=None, head=None, keep=None, cur=None)
_orig = M.torch_chunk_gated_delta_rule


def _l2(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _rec(query, key, value, g, beta, **kw):
    out, st = _orig(query, key, value, g=g, beta=beta, **kw)
    if NB["on"] and NB["cur"] == NB["layer"]:
        h = NB["head"]
        NB["keep"] = (_l2(query[0, :, h].float()) / (query.shape[-1] ** .5), _l2(key[0, :, h].float()), value[0, :, h].float().norm(dim=-1),
                      g[0, :, h].float(), beta[0, :, h].float())
    return out, st


def install(v, layer):
    M.torch_chunk_gated_delta_rule = _rec
    NB["layer"] = layer
    for i, L in enumerate(v.layers):
        if getattr(L, "linear_attn", None) is not None:
            L.linear_attn.register_forward_pre_hook(lambda mod, args, i=i: NB.__setitem__("cur", i))


def start(head):
    NB.update(on=True, head=head, keep=None)


def stop():
    NB["on"] = False
    keep = NB["keep"]; NB["keep"] = None
    return keep


@torch.no_grad()
def ranks(keep, blocks, c):
    """block contributions to position c's output (exact), one value per block."""
    q, k, vn, gg, bb = keep
    lo = blocks[0][0]
    bid = torch.full((q.shape[0],), -1, dtype=torch.long)
    for j, (a, z) in enumerate(blocks):
        bid[a:z + 1] = j
    u = q[c].clone(); W = torch.zeros(len(blocks), device=q.device)
    for r in range(c, lo - 1, -1):
        w = bb[r] * (k[r] @ u)
        if bid[r] >= 0:
            W[bid[r]] += w.abs() * vn[r]
        u = torch.exp(gg[r]) * (u - bb[r] * k[r] * (k[r] @ u))
    return W.cpu()
