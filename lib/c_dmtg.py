"""S770 tool: split the output of a full-attention layer at chosen rows into four parts and recompose it with any subset taken from a donor.

For row r and head h the layer output (before o_proj) is  o = G * (M * D + (1 - M) * T)  with
  M  the attention mass on image positions,
  D  the image read  sum_img w v / M   (which images are read; image keys and values are the same in donor and recipient when the images
     precede every edited token),
  T  the text read   sum_txt w v / (1 - M)   (where in the text it reads and what those text positions carry),
  G  the output gate sigmoid(gate).
Rows are recomputed in float32 exactly as c_tools.FullAttn / c_pat.PatAttn do (q_norm, partial RoPE, output gate, o_proj).
cfg = dict(rows=list, img=bool list over positions,
           grab=set(layers)       store m, D, T, G, the within-text weights Tw, the query q (after q_norm, before RoPE), the module output rows
                                  and (layers in keep_att) the full weights of rows keep_rows as float16;
           give={layer: (donor store, comps)}  comps among D, M, T, Tw (donor within-text weights read on the recipient's values), G;
           qsrc={layer: (donor q, comps)}      weights from the donor query on the recipient's keys (c_tools.FullAttn "q"); comps among D, M, T;
           out={layer: donor rows}             replace the module output rows by the donor's (instrument comparison);
           noop=set(layers))      recompute without change.
"""
from __future__ import annotations

import torch
from c_tools import rotate_half


class DMTG:
    def __init__(self, v, layers):
        self.hooks = []; self.cfg = None; self.store = {}
        for li in layers:
            self.hooks.append(v.layers[li].self_attn.register_forward_hook(self._hook(li), with_kwargs=True))

    def set(self, cfg):
        self.cfg = cfg
        if cfg is not None and cfg.get("grab"):
            self.store = {}

    def _hook(self, li):
        def h(mod, args, kwargs, out):
            c = self.cfg
            if c is None:
                return out
            give, qsrc, outd = c.get("give", {}), c.get("qsrc", {}), c.get("out", {})
            grab = li in c.get("grab", ())
            edit = li in give or li in qsrc or li in c.get("noop", ())
            t0 = out[0] if isinstance(out, tuple) else out
            rows = c["rows"]
            if li in outd:
                t0 = t0.clone(); t0[0, rows] = outd[li].to(t0.device, t0.dtype)
                return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
            if not grab and not edit:
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            qg = torch.nn.functional.linear(hs, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(S, nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            K = k[:, kh]; V = vv[:, kh]
            ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
            mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
            IMG = torch.tensor(c["img"], device=q.device, dtype=torch.bool)

            def weights(qr):
                qe = torch.cat([qr[..., :rd] * cos[rows][:, None, :] + rotate_half(qr[..., :rd]) * sin[rows][:, None, :], qr[..., rd:]], -1)
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                return torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)              # [R, H, S]

            def parts(att):
                ai = att.masked_fill(~IMG[None, None, :], 0.); at = att.masked_fill(IMG[None, None, :], 0.)
                m = ai.sum(-1)
                Di = torch.einsum("ths,shd->thd", ai, V) / m[..., None].clamp_min(1e-12)
                tw = at / (1 - m)[..., None].clamp_min(1e-12)
                Tt = torch.einsum("ths,shd->thd", tw, V)
                return m, Di, Tt, tw

            att = weights(q[rows])
            m, Di, Tt, tw = parts(att)
            g = torch.sigmoid(gate[rows])
            if grab:
                st = dict(m=m.cpu(), D=Di.cpu(), T=Tt.cpu(), G=g.cpu(), Tw=tw.cpu(), q=q[rows].cpu(), out=t0[0, rows].detach().cpu())
                if li in c.get("keep_att", ()):
                    st["att"] = att[c["keep_rows"]].to(torch.float16).cpu()
                self.store[li] = st
            if not edit:
                return out
            if li in qsrc:
                qd, comps = qsrc[li]
                m2, D2, T2, _ = parts(weights(qd.to(q.device).float()))
                if "M" in comps:
                    m = m2
                if "D" in comps:
                    Di = D2
                if "T" in comps:
                    Tt = T2
            if li in give:
                dn, comps = give[li]
                if "M" in comps:
                    m = dn["m"].to(q.device)
                if "D" in comps:
                    Di = dn["D"].to(q.device)
                if "T" in comps:
                    Tt = dn["T"].to(q.device)
                if "Tw" in comps:
                    Tt = torch.einsum("ths,shd->thd", dn["Tw"].to(q.device), V)
                if "G" in comps:
                    g = dn["G"].to(q.device)
            o = g * (m[..., None] * Di + (1 - m[..., None]) * Tt)
            new = torch.nn.functional.linear(o.reshape(len(rows), -1), mod.o_proj.weight.float(),
                                             None if mod.o_proj.bias is None else mod.o_proj.bias.float())
            t0 = t0.clone(); t0[0, rows] = new.to(t0.dtype)
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()
