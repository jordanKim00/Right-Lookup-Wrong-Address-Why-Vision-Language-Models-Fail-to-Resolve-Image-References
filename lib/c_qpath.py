"""S771 tool: query-path patching at full-attention reader layers.

At a reader layer the attention of the chosen rows is recomputed in float32 (q_norm, partial RoPE, output gate, o_proj, as c_tools.FullAttn) with a
replaced query and everything else of the running pass kept (keys, values, the output gate, every other row):
  qd     the donor's query (after q_norm, before RoPE) - the whole query of another run (S767);
  delta  a donor-minus-recipient residual difference taken at a boundary (the input of an earlier decoder layer), added to the recipient's residual
         at this layer's input before its input RMSNorm and q_proj: the reader sees the donor's message as of the boundary and none of the later
         layers' processing of it (a path patch from "everything up to the boundary" to the reader queries);  rmask = rows that take it;
         gate_too = the output gate is also computed from the patched input (query + gate path);
  ko     (row indices into rows, key positions): those rows may not attend to those positions (attention knockout, S778).
Residuals at decoder-layer inputs are captured by forward pre-hooks (grab at the boundaries; the running one at reader layers).
cfg = dict(rows=list, grab=set(layers) residual capture, grabq=set(readers) query capture, edit={reader: dict(qd=... | delta=..., rmask=...)},
           noop=set(readers), cap=dict(rows=[row indices into rows], owner=[block of every position or -1], nb=number of blocks)): with cap, every reader
           layer stores the fixed-bank block scores (logsumexp of the scaled q.k over each image block, S748) of the query it finally uses, [len, H, nb].
  qsub   (S808) dict(owner, nb, part='U' | 'perp', donors=[(weight, donor query after q_norm, before RoPE)], gain=None | g): in the post-RoPE
         query, per head, replace the part inside U = span of the centred block-mean image keys (part='U') by that of the weighted donor
         mixture (gain g rescales it to g times the norm of the recipient's own part in U), or keep the recipient's part in U and take the
         donor's complement (part='perp').  A single donor equal to the recipient reproduces the original query; self=True uses the
         running query itself as the donor (an exactness check of the edit).
         uref (S811f) = a reference query (after q_norm, before RoPE): with part='perp' the part inside U comes from uref instead of the running
         query (the recipient's natural U kept fixed while the complement is swapped).
  grabqe (S811f) = set of readers: store the query actually used at the last row (after RoPE and any edit), [H, hd];  grabkb = set of
  readers: store the post-RoPE block-mean keys per query head, [nb, H, hd] (needs cap for the owners).
"""
from __future__ import annotations

import torch
from c_tools import rotate_half


class QPath:
    def __init__(self, v, readers, bounds):
        self.v = v; self.cfg = None; self.cur = {}; self.store = {}; self.hooks = []; self.readers = set(readers)
        for li in sorted(set(readers) | set(bounds)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._pre(li), with_kwargs=True))
        for li in readers:
            self.hooks.append(v.layers[li].self_attn.register_forward_hook(self._hook(li), with_kwargs=True))

    def set(self, cfg):
        self.cfg = cfg; self.cur = {}
        if cfg is not None:
            self.store = {}

    def _pre(self, li):
        def h(mod, args, kwargs):
            c = self.cfg
            if c is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            r = hs[0, c["rows"]].detach().float()
            if li in c.get("grab", ()):
                self.store[("res", li)] = r.cpu()
            if li in self.readers:
                self.cur[li] = r
            return None
        return h

    def _hook(self, li):
        def h(mod, args, kwargs, out):
            c = self.cfg
            if c is None:
                return out
            ed = c.get("edit", {}).get(li); grabq = li in c.get("grabq", ()); noop = li in c.get("noop", ()); cap = c.get("cap")
            if ed is None and not grabq and not noop and cap is None:
                return out
            hs = kwargs["hidden_states"][0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            rows = c["rows"]; R = len(rows)
            qproj = lambda x: torch.nn.functional.linear(x, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
            qg = qproj(hs[rows]).view(R, nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
            if grabq:
                self.store[("q", li)] = q.cpu()
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            K = k[:, kh]; V = vv[:, kh]
            qr = q
            if ed is not None and "qd" in ed:
                qr = ed["qd"].to(q.device).float()
            if ed is not None and "delta" in ed:
                ln = self.v.layers[li].input_layernorm
                res = self.cur[li].to(q.device) + ed["delta"].to(q.device).float()
                hn = ln(res.to(ln.weight.dtype)).float()
                qg2 = qproj(hn).view(R, nh, 2 * hd)
                q2 = mod.q_norm(qg2[..., :hd].to(mod.q_norm.weight.dtype)).float()
                rm = ed.get("rmask")
                rmt = None if rm is None else torch.tensor(rm, device=q.device)[:, None, None]
                qr = q2 if rm is None else torch.where(rmt, q2, q)
                if ed.get("gate_too"):                                     # the output gate also from the patched input
                    gate = qg2[..., hd:] if rm is None else torch.where(rmt, qg2[..., hd:], gate)
            qe = torch.cat([qr[..., :rd] * cos[rows][:, None, :] + rotate_half(qr[..., :rd]) * sin[rows][:, None, :], qr[..., rd:]], -1)
            ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
            if ed is not None and "qsub" in ed:
                qe = self._qsub(qe, ke, ed["qsub"], cos[rows], sin[rows], rd)
            if li in c.get("grabqe", ()):
                self.store[("qe", li)] = qe[-1].detach().cpu()
            if li in c.get("grabkb", ()) and cap is not None:
                own_ = torch.tensor(cap["owner"], device=q.device)
                self.store[("kb", li)] = torch.stack([ke[own_ == b].mean(0) for b in range(cap["nb"])]).detach().cpu()
            lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
            if cap is not None:
                own = torch.tensor(cap["owner"], device=q.device); cr = cap["rows"]
                self.store[("cap", li)] = torch.stack([torch.logsumexp(lg[cr][:, :, own == b], -1) for b in range(cap["nb"])], -1).cpu()
                if cap.get("opt") is not None:                              # weights of the cap rows on the menu words and on all image tokens
                    rr = torch.tensor([rows[i] for i in cr], device=q.device)
                    mk = torch.arange(S, device=q.device)[None, :] > rr[:, None]
                    ac = torch.softmax(lg[cr].masked_fill(mk[:, None, :], float("-inf")), -1)
                    self.store[("opt", li)] = ac[:, :, torch.tensor(cap["opt"], device=q.device)].cpu()
                    self.store[("img", li)] = ac[:, :, own >= 0].sum(-1).cpu()
            if ed is None and not noop:
                return out
            mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
            if ed is not None and "ko" in ed:                                  # attention knockout: rows in ko_rows may not read positions ko
                kr, kp = ed["ko"]
                blk = torch.zeros(R, S, dtype=torch.bool, device=q.device)
                blk[torch.tensor(kr, device=q.device)[:, None], torch.tensor(kp, device=q.device)[None, :]] = True
                mask = mask | blk
            att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
            o = torch.sigmoid(gate) * torch.einsum("ths,shd->thd", att, V)
            new = torch.nn.functional.linear(o.reshape(R, -1), mod.o_proj.weight.float(), None if mod.o_proj.bias is None else mod.o_proj.bias.float())
            t0 = out[0] if isinstance(out, tuple) else out
            t0 = t0.clone(); t0[0, rows] = new.to(t0.dtype)
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h

    @staticmethod
    def _rope(x, cr, sr, rd):
        return torch.cat([x[..., :rd] * cr[:, None, :] + rotate_half(x[..., :rd]) * sr[:, None, :], x[..., rd:]], -1)

    def _qsub(self, qe, ke, spec, cr, sr, rd):
        if spec.get("part") == "local":                                                  # S827: one part of a natural change (other parts untouched)
            return self._qlocal(qe, ke, spec, cr, sr, rd)
        if spec.get("part") == "proj":                                                   # S827: remove alpha x the change along u_k
            return self._qproj(qe, ke, spec, cr, sr, rd)
        if spec.get("part") == "all":                                                    # S829: the whole query <- the donor's
            return sum(w * self._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["donors"]).clone()
        if spec.get("part") == "mass":                                                   # S829: the picture-vs-text offset <- the donor's
            return self._qmass(qe, ke, spec, cr, sr, rd)
        own = torch.tensor(spec["owner"], device=qe.device)
        Kb = torch.stack([ke[own == b].mean(0) for b in range(spec["nb"])])            # [nb, H, hd], post-RoPE keys per query head
        C = Kb - Kb.mean(0, keepdim=True)
        don = qe if spec.get("self") else sum(w * self._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["donors"])
        out = qe.clone()
        for h in range(qe.shape[1]):
            U, s, _ = torch.linalg.svd(C[:, h].T, full_matrices=False)
            U = U[:, s > 1e-6 * s.max()]
            a_rec = (qe[:, h] @ U) @ U.T; a_don = (don[:, h] @ U) @ U.T
            if spec.get("uref") is not None and spec["part"] != "U":                   # S811f: the recipient's natural U part, kept fixed
                ur = self._rope(spec["uref"].to(qe.device).float(), cr, sr, rd)
                a_rec = (ur[:, h] @ U) @ U.T
            if spec["part"] == "U":
                if spec.get("gain") is not None:
                    a_don = a_don * (spec["gain"] * a_rec.norm(dim=-1, keepdim=True) / (a_don.norm(dim=-1, keepdim=True) + 1e-12))
                out[:, h] = qe[:, h] - a_rec + a_don
            else:
                out[:, h] = a_rec + (don[:, h] - (don[:, h] @ U) @ U.T)
        return out

    @staticmethod
    def _qlocal(qe, ke, spec, cr, sr, rd):
        """S827.  delta = [(w, q)]: a natural change of the query (e.g. END - START, pre-RoPE, rotated here like donors).  Per head, its centred
        block pattern c = C d (C = centred block-mean post-RoPE picture keys, [nb, hd]); local coefficients a_i = c_i - mean of c over the blocks
        outside loc; the query component x = pinv(C) (sum over i in sel of a_i e_i) (e_i = centred indicator of block i) - it changes, in the
        block-mean linear scores, only block i's score relative to the others - is added with coef (-1 removes that part of the change)."""
        own = torch.tensor(spec["owner"], device=qe.device); nb = spec["nb"]
        Kb = torch.stack([ke[own == b].mean(0) for b in range(nb)])
        C = (Kb - Kb.mean(0, keepdim=True)).permute(1, 0, 2)                              # [H, nb, hd]
        d = sum(w * QPath._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["delta"])   # [R, H, hd]
        far = [i for i in range(nb) if i not in spec["loc"]]; sel = list(spec["sel"])
        c = torch.einsum("rhd,hbd->rhb", d, C)
        a = c[..., sel] - c[..., far].mean(-1, keepdim=True)                              # [R, H, |sel|]
        E = torch.full((len(sel), nb), -1.0 / nb, device=qe.device)
        E[torch.arange(len(sel)), torch.tensor(sel)] += 1.0
        x = torch.einsum("hdb,rhb->rhd", torch.linalg.pinv(C, rtol=1e-6), torch.einsum("rhs,sb->rhb", a, E))
        return qe + spec["coef"] * x

    @staticmethod
    def _qproj(qe, ke, spec, cr, sr, rd):
        """S827 (the advisor's definition, 20:1x): u_k = Kbar_k - mean of Kbar_b over b != k (block-mean post-RoPE keys, per head); the part of a
        natural change d (delta, as in _qlocal) along u_k, coef x (d . u) u / |u|^2, is added (coef = -alpha removes alpha of it)."""
        own = torch.tensor(spec["owner"], device=qe.device); nb = spec["nb"]; k0 = spec["block"]
        Kb = torch.stack([ke[own == b].mean(0) for b in range(nb)])                         # [nb, H, hd]
        u = Kb[k0] - torch.cat([Kb[:k0], Kb[k0 + 1:]]).mean(0)                              # [H, hd]
        d = sum(w * QPath._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["delta"])
        x = (d * u[None]).sum(-1, keepdim=True) / (u * u).sum(-1)[None, :, None] * u[None]
        return qe + spec["coef"] * x

    @staticmethod
    def _qmass(qe, ke, spec, cr, sr, rd):
        """S829.  Per head, the picture-vs-text direction m = mean key over picture tokens - mean key over the text tokens before spec['text_end']
        (post-RoPE), made orthogonal to U (the centred block-mean picture keys) so that the block address is untouched; the query's component
        along m is set to the donor's (coef 1) - it shifts all picture scores together against the text, i.e. how much the head reads pictures."""
        own = torch.tensor(spec["owner"], device=qe.device); nb = spec["nb"]
        Kb = torch.stack([ke[own == b].mean(0) for b in range(nb)]); C = Kb - Kb.mean(0, keepdim=True)            # [nb, H, hd]
        te = spec["text_end"]; txt = (own < 0) & (torch.arange(len(own), device=qe.device) < te)
        m = ke[own >= 0].mean(0) - ke[txt].mean(0)                                                           # [H, hd]
        don = sum(w * QPath._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["donors"])
        out = qe.clone()
        for h in range(qe.shape[1]):
            U, s_, _ = torch.linalg.svd(C[:, h].T, full_matrices=False); U = U[:, s_ > 1e-6 * s_.max()]
            mh = m[h] - U @ (U.T @ m[h]); mh = mh / (mh.norm() + 1e-12)
            out[:, h] = qe[:, h] + ((don[:, h] - qe[:, h]) @ mh)[:, None] * mh[None, :]
        return out

    def remove(self):
        for h in self.hooks:
            h.remove()
