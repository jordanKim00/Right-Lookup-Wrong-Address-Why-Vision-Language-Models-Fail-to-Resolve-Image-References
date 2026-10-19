"""S701 tool: make chosen image blocks invisible to the question's text rows through one module type of Qwen3.5.

HideDN (Gated DeltaNet layers): the text rows' core outputs are recomputed with the hidden positions' beta = 0 and g = 0, so the
recurrent state passes those positions unchanged (no write, no decay) and the text rows read the memory of the visible images only
-- with a matched prefix this is exactly the memory a run without the hidden blocks would have.  The short convolution is closed too:
the first three text rows' 4-tap windows take the pre-convolution inputs of the three positions before the hidden span instead of the
hidden span's last three.  Image rows keep their original outputs (the hidden images stay intact for every other route).
Full attention: use c_tools.FullAttn with ctx.rows = the text rows and edit {"ko": hidden positions} (keys masked for those rows).
NOOP: the same recomputation without hiding (the convolution rows recomputed from unmodified inputs); conv_err records the largest
absolute difference between recomputed and original convolution outputs of those rows.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as M


class HideDN:
    def __init__(self, v):
        self.v = v; self.orig = M.torch_chunk_gated_delta_rule; M.torch_chunk_gated_delta_rule = self._rule
        self.cfg = None; self.cur = None; self.x = None; self.hooks = []; self.conv_err = 0.0
        self.mods = {}
        for i, L in enumerate(v.layers):
            la = getattr(L, "linear_attn", None)
            if la is None:
                continue
            self.mods[i] = la
            self.hooks.append(la.register_forward_pre_hook(lambda mod, args, i=i: setattr(self, "cur", i)))
            self.hooks.append(la.in_proj_qkv.register_forward_hook(lambda mod, inp, out, i=i: self._grab(i, out)))

    def set(self, cfg):
        """cfg = dict(layers=set, H=list, rows=list, src3=list, noop=bool) or None."""
        self.cfg = cfg; self.conv_err = 0.0

    def _grab(self, i, out):
        if self.cfg is not None and i in self.cfg["layers"]:
            self.x = out[0].detach()
        return None

    def _conv_rows(self, la, x, ps, sub):
        """silu(4-tap causal conv) at positions ps, with x rows replaced per sub {dst: src}; returns [len(ps), conv_dim] float32."""
        w = la.conv1d.weight[:, 0, :].float()                                         # [C, 4]
        K = w.shape[1]; outs = []
        for p in ps:
            win = [sub.get(s, s) for s in range(p - K + 1, p + 1)]
            xw = x[torch.tensor(win, device=x.device)].float()                         # [4, C]
            outs.append(F.silu((w.T * xw).sum(0)))
        return torch.stack(outs)

    def _rule(self, query, key, value, g, beta, **kw):
        out = self.orig(query, key, value, g=g, beta=beta, **kw)
        c = self.cfg; li = self.cur
        if c is None or li not in c["layers"]:
            return out
        la = self.mods[li]
        H, rows, src3 = c["H"], c["rows"], c["src3"]
        q2, k2, v2, g2, b2 = query.clone(), key.clone(), value.clone(), g.clone(), beta.clone()
        p0 = H[-1] + 1
        ps = [p for p in (p0, p0 + 1, p0 + 2) if p in set(rows)]
        sub = {} if c.get("noop") else dict(zip(H[-3:], src3))
        y = self._conv_rows(la, self.x, ps, sub)
        kd, vd = la.key_dim, la.value_dim
        rep = la.num_v_heads // la.num_k_heads
        qn = y[:, :kd].view(len(ps), la.num_k_heads, la.head_k_dim).repeat_interleave(rep, dim=1)
        kn = y[:, kd:2 * kd].view(len(ps), la.num_k_heads, la.head_k_dim).repeat_interleave(rep, dim=1)
        vn = y[:, 2 * kd:].view(len(ps), la.num_v_heads, la.head_v_dim)
        P = torch.tensor(ps, device=query.device)
        if c.get("noop"):
            self.conv_err = max(self.conv_err, float((qn.to(query.dtype) - query[0, P]).abs().max()),
                                float((kn.to(key.dtype) - key[0, P]).abs().max()), float((vn.to(value.dtype) - value[0, P]).abs().max()))
        q2[0, P] = qn.to(q2.dtype); k2[0, P] = kn.to(k2.dtype); v2[0, P] = vn.to(v2.dtype)
        if not c.get("noop"):
            Ht = torch.tensor(H, device=g.device)
            g2[0, Ht] = 0.; b2[0, Ht] = 0.
        out2 = self.orig(q2, k2, v2, g=g2, beta=b2, **kw)
        o0 = out[0].clone()
        R = torch.tensor(rows, device=o0.device)
        o0[0, R] = out2[0][0, R]
        return (o0,) + tuple(out[1:])

    def remove(self):
        M.torch_chunk_gated_delta_rule = self.orig
        for h in self.hooks:
            h.remove()
