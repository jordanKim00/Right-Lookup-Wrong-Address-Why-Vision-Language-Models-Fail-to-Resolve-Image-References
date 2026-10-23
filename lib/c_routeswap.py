"""S921 tool: position-restricted swaps of a full-attention layer's read-row output (the late-read route split; advisor review 3).
At a full-attention layer the gated output of a sample's read rows decomposes over key positions: o_h = g_h * sum_t A_h,t V_h,t.  For a set of
key positions P (a route: picture tokens, menu-name tokens, the rest) the route part of head h is g_h * sum_{t in P} A_h,t V_h,t.  A route
swap replaces this sample's route part by the source sample's (its own queries, keys, values and gate, same pass), every head:
  out_rows += sum_h W_O[:, h] (g_src,h * R_src,h(P) - g_this,h * R_this,h(P)).
The three routes add up to the whole gated output, so swapping every route equals S917's full swap.  cfgs[b] = None or dict(rows=[read-row
positions], src=batch index, pos=[key positions]); the source must carry the same rows (use a cfg without 'pos' as a passive source).
Float32 recompute of q / k / v / gate at the rows (as c_headpatch); no pins at these layers."""
from __future__ import annotations

import threading
import torch
from c_tools import rotate_half


class FARouteSwap:
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def __init__(self, v, layers):
        self._tl = threading.local(); self.v = v; self.cfgs = None; self.hooks = []
        for l in layers:
            self.hooks.append(v.layers[l].self_attn.register_forward_hook(self._post(l), with_kwargs=True))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    @staticmethod
    def _route(mod, kwargs, b, rows, pos):
        """per-head gated route part [R, H, hd] and the gate, for sample b's rows over key positions pos"""
        hs = kwargs["hidden_states"][b].float()
        cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
        S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]; R = len(rows)
        qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = torch.sigmoid(qg[..., hd:])
        k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
        vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
        kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
        K = k[:, kh]; V = vv[:, kh]
        qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
        ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
        lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
        mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(rows, device=hs.device)[:, None]
        att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)               # [R, H, S]
        p_ = torch.tensor(pos, device=hs.device)
        return gate * torch.einsum("rhs,shd->rhd", att[:, :, p_], V[p_])

    def _post(self, l):
        def h(mod, args, kwargs, out):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and c.get("pos") is not None]
            if not todo:
                return None
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            nh, hd = mod.config.num_attention_heads, mod.head_dim
            Wo = mod.o_proj.weight.float().view(-1, nh, hd)
            for b, c in todo:
                rows, src, pos = c["rows"], c["src"], c["pos"]
                assert cfgs[src] is not None and cfgs[src]["rows"] == rows, "the source must carry the same rows"
                d = self._route(mod, kwargs, src, rows, pos) - self._route(mod, kwargs, b, rows, pos)
                t2[b, rows] = (t[b, rows].float() + torch.einsum("rhd,ohd->ro", d, Wo)).to(t2.dtype)
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h
