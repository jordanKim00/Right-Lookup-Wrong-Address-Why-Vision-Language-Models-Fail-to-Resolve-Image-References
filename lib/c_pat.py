"""S704 tool: grab / transplant the attention pattern of chosen rows over chosen source positions at full-attention layers.

The rows are recomputed in float32 exactly as c_tools.FullAttn does (q_norm, partial RoPE, output gate, o_proj).
grab: store, per layer, the post-softmax weights of the rows over `pos` ([R, H, P]).
edit: replace the rows' weights over `pos` with a donor's ([R, H, P]); the remaining weights are rescaled so that every row
still sums to one (the recipient keeps its own reading of every other position).  cond=True: keep the recipient's total mass on `pos` and
transplant only the donor's distribution within `pos`.  heads={layer: [h, ...]}: only those heads take the edit.  With identical keys and values at `pos` in both
runs (image tokens precede the question), this transplants only which images are read.
NOOP: recompute without change.
"""
from __future__ import annotations

import torch
from c_tools import rotate_half


class PatAttn:
    def __init__(self, v, layers):
        self.hooks = []; self.cfg = None; self.store = {}
        for li in layers:
            self.hooks.append(v.layers[li].self_attn.register_forward_hook(self._hook(li), with_kwargs=True))

    def set(self, cfg):
        """cfg = dict(rows=list, pos=list, grab=set(layers), edit={layer: donor [R, H, P]}, noop=set(layers)) or None"""
        self.cfg = cfg
        if cfg is not None and cfg.get("grab"):
            self.store = {}

    def _hook(self, li):
        def h(mod, args, kwargs, out):
            c = self.cfg
            if c is None or not (li in c.get("grab", ()) or li in c.get("edit", {}) or li in c.get("noop", ())):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads
            qg = torch.nn.functional.linear(hs, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(S, nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            v = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
            rows = c["rows"]; rd = cos.shape[-1]
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            K = k[:, kh]; V = v[:, kh]
            qr = q[rows]
            qe = torch.cat([qr[..., :rd] * cos[rows][:, None, :] + rotate_half(qr[..., :rd]) * sin[rows][:, None, :], qr[..., rd:]], -1)
            ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
            lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
            mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
            att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)               # [R, H, S]
            P = torch.tensor(c["pos"], device=q.device)
            if li in c.get("grab", ()):
                self.store[li] = att[:, :, P].detach().cpu()
            if li in c.get("edit", {}):
                don = c["edit"][li].to(q.device)
                a_in = att[:, :, P].sum(-1, keepdim=True); d_in = don.sum(-1, keepdim=True)
                if c.get("cond"):                                                       # keep the recipient's image mass, move only which images
                    new_img = a_in * don / d_in.clamp_min(1e-12); new = att.clone()
                else:                                                                   # donor's image weights; text rescaled to sum to one
                    new_img = don; new = att * ((1 - d_in) / (1 - a_in).clamp_min(1e-6))
                new[:, :, P] = new_img
                hs_ = c.get("heads", {}).get(li)
                if hs_ is not None:                                                     # only these heads take the edit
                    Hm = torch.zeros(att.shape[1], dtype=torch.bool, device=q.device); Hm[torch.tensor(hs_, device=q.device)] = True
                    new = torch.where(Hm[None, :, None], new, att)
                att = new
            if li in c.get("grab", ()) and li not in c.get("edit", {}) and li not in c.get("noop", ()):
                return out
            o = torch.sigmoid(gate[rows]) * torch.einsum("ths,shd->thd", att, V)
            new = torch.nn.functional.linear(o.reshape(len(rows), -1), mod.o_proj.weight.float(),
                                             None if mod.o_proj.bias is None else mod.o_proj.bias.float())
            t0 = out[0] if isinstance(out, tuple) else out
            t0 = t0.clone(); t0[0, rows] = new.to(t0.dtype)
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()
