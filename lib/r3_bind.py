"""S609: the model's own binding of a frame cue to an ordinal, read from its attention (S607 / S607b).

install(v)            switch the text attention to a pass-through sdpa wrapper that can also return attention rows
start(rows, layers)   capture, on the next forward, attention probabilities (softmax over all keys) of the given query rows
                      restricted to the same rows as keys, at the given layers -> CAP["A"][layer] = [H, S, S]
select_heads(...)     the top-8 heads of a binding measure on canonical calibration sentences (S607 / S607b rules)
prior(...)            the per-token binding prior p(q) of a source: tokens after it by A(q -> s) over the post-nominal heads,
                      tokens before it by A(s+1 -> q) over the fronted heads, each side scaled by its calibration level, cut at 1
"""
from __future__ import annotations

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAP = dict(on=False, rows=None, layers=None, A={})


def _cap_attn(module, query, key, value, attention_mask, **kw):
    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    li = getattr(module, "layer_idx", None)
    if CAP["on"] and li is not None and (CAP["layers"] is None or li in CAP["layers"]):
        rows = CAP["rows"].to(query.device)
        q = query[0][:, rows].float()                                              # [H, S, D]
        k = key[0].float()
        k = k.repeat_interleave(q.shape[0] // k.shape[0], 0)                      # grouped-query heads
        sc = (q @ k.transpose(-1, -2)) * (kw.get("scaling") or module.scaling)
        L = k.shape[1]
        sc = sc.masked_fill(torch.arange(L, device=sc.device)[None, None, :] > rows[None, :, None], float("-inf"))
        CAP["A"][li] = sc.softmax(-1)[:, :, rows].cpu()                           # [H, S, S]
    return out


AttentionInterface.register("r3cap", _cap_attn)


def install(v):
    v.model.config.get_text_config()._attn_implementation = "r3cap"


def start(rows, layers=None):
    CAP.update(on=True, rows=rows, layers=None if layers is None else set(layers), A={})


def stop():
    CAP["on"] = False
    A = CAP["A"]; CAP["A"] = {}
    return A


def head_mean(A, heads, qi, ki):
    """mean over the given (layer, head) pairs of A[layer][head, qi, ki]."""
    return float(sum(float(A[l][h, qi, ki]) for l, h in heads) / len(heads))


def select_heads(records, k=8):
    """records: list of dicts {layer: [H] binding of the cue}; top-k (layer, head) by mean binding."""
    layers = sorted(records[0])
    M = torch.stack([torch.stack([r[l] for l in layers]) for r in records]).mean(0)   # [L, H]
    flat = torch.argsort(M.flatten(), descending=True)[:k]
    H = M.shape[1]
    return [(layers[int(i) // H], int(i) % H) for i in flat], float(M.flatten()[flat].mean())


_MCACHE = {}


def _mean_maps(A, bind):
    """[S, S] mean attention maps of the post-nominal and fronted binding heads, cached for the current capture only
    (held by reference: an id() key can be reused by a later capture of another size)."""
    if _MCACHE.get("A") is not A or _MCACHE.get("bind") is not bind:
        _MCACHE.update(A=A, bind=bind, maps=(torch.stack([A[l][h] for l, h in bind["heads_after"]]).mean(0),
                                            torch.stack([A[l][h] for l, h in bind["heads_front"]]).mean(0)))
    return _MCACHE["maps"]


def prior(A, s, n_span, bind, capL=32, capR=32, srcs=None):
    """window positions (relative to the span) around source s and their binding prior.
    S609b (bind["compete"]): a cue's binding is shared among the detected ordinals that can bind it -- the share of s."""
    MA, MF = _mean_maps(A, bind)
    left = list(range(max(0, s - capL), s)); right = list(range(s + 1, min(n_span, s + 1 + capR)))
    noun = lambda s_: min(s_ + 1, n_span - 1)
    comp = bool(bind.get("compete")) and srcs is not None
    p = []
    for q in left:
        b = float(MF[noun(s), q]) if noun(s) > q else 0.0
        if comp and b > 0:
            b_all = sum(float(MF[noun(s_), q]) for s_ in srcs if s_ > q and noun(s_) > q)
            p.append(min(1.0, b / bind["c_front"]) * b / max(b_all, 1e-8))
        else:
            p.append(min(1.0, b / bind["c_front"]))
    for q in right:
        b = float(MA[q, s])
        if comp:
            b_all = sum(float(MA[q, s_]) for s_ in srcs if s_ < q)
            p.append(min(1.0, b / bind["c_after"]) * b / max(b_all, 1e-8))
        else:
            p.append(min(1.0, b / bind["c_after"]))
    return left + right, torch.tensor(p)


def window_hard(A, s, n_span, bind, srcs, capL=32, capR=32):
    """S609c: the tokens the model binds to source s -- among the detected sources a cue binds to s most (argmax),
    and its binding reaches the calibrated threshold of its side. Returns (positions, bindings)."""
    MA, MF = _mean_maps(A, bind)
    noun = lambda s_: min(s_ + 1, n_span - 1)
    srcs = list(srcs) if srcs else [s]
    win, bv = [], []
    for q in range(max(0, s - capL), s):
        cand = [s_ for s_ in srcs if s_ > q and noun(s_) > q]
        if not cand or noun(s) <= q:
            continue
        b = {s_: float(MF[noun(s_), q]) for s_ in cand}
        if max(b, key=b.get) == s and b[s] >= bind["th_front"]:
            win.append(q); bv.append(b[s])
    for q in range(s + 1, min(n_span, s + 1 + capR)):
        b = {s_: float(MA[q, s_]) for s_ in srcs if s_ < q}
        if max(b, key=b.get) == s and b[s] >= bind["th_after"]:
            win.append(q); bv.append(b[s])
    return win, bv
