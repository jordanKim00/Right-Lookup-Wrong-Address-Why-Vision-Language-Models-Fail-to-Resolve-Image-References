"""Batched QPathB + residual path patching into reader queries (S836; the batched form of QPath's 'grab' / 'delta').
Extra per-sample cfg keys (QPathB's keys keep their meaning):
  grabres   set of layers: store the residual at the input of decoder layer l for the sample's rows, ('res', l) -> [R, E] float32 (cpu)
  grabcomp  set of layers: store the token-mixer output (DeltaNet or full attention, as added to the residual) and the MLP output of the
            sample's rows, ('mix', l) / ('mlp', l) -> [R, E] float32 (cpu)
  keepres   set of layers: keep this sample's residual of its rows at those layers' inputs on the device for cross-sample deltas (S836)
  delta     {reader: dict(vec=[R, E] tensor | src=batch index, layer=l; rmask=None | list of bools per row)}: added to the reader layer's input
            residual of the sample's rows; with src/layer the vector is computed in the same forward pass as (the src sample's residual at the
            input of layer l) - (this sample's residual at the input of layer l), both kept with keepres (no batch-composition drift: a delta
            at the reader's own layer makes the query exactly the src sample's); otherwise vec is
            rows before input_layernorm and q_proj; that reader's query (and only it) is recomputed from the patched residual - keys,
            values and the output gate stay those of the running pass; the reader's attention output of the rows is written back
            (float32 recompute, as QPathB's edits).  A path patch of 'whatever the delta carries' into the reader queries only.
  dcheck    set of readers: also store the recomputed query of the rows ('qd', reader) -> [R, H, hd] (the positive control compares it with
            the donor's own query)
Readers must be full-attention layers.  Store layout as QPathB: self.store[b][(key, layer)]."""
from __future__ import annotations

import torch
from c_tools import rotate_half
from c_qpath import QPath
from c_qpath_b import QPathB


class QPathD(QPathB):
    cur = property(lambda self: getattr(self._tl, "cur", {}), lambda self, x: setattr(self._tl, "cur", x))

    def __init__(self, v, readers, res_layers=(), comp_layers=()):
        super().__init__(v, readers)
        self.cur = {}
        for li in sorted(set(readers) | set(res_layers)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._pre(li), with_kwargs=True))
        for li in sorted(set(comp_layers)):
            L = v.layers[li]
            mix = L.linear_attn if getattr(L, "linear_attn", None) is not None else L.self_attn
            self.hooks.append(mix.register_forward_hook(self._comp(li, "mix"), with_kwargs=True))
            self.hooks.append(L.mlp.register_forward_hook(self._comp(li, "mlp"), with_kwargs=True))
        # the delta recompute must run before QPathB's reader hook writes anything: wrap QPathB's reader hooks
        for h in list(self.hooks[:len(readers)]):
            h.remove()
        self.hooks = self.hooks[len(readers):]
        for li in readers:
            self.hooks.append(v.layers[li].self_attn.register_forward_hook(self._hook_d(li), with_kwargs=True))

    def set(self, cfgs):
        super().set(cfgs); self.cur = {}

    def _pre(self, li):
        def h(mod, args, kwargs):
            if self.cfgs is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            for b, c in enumerate(self.cfgs):
                if c is None:
                    continue
                if li in c.get("grabres", ()) or li in c.get("delta", {}) or li in c.get("keepres", ()):
                    r = hs[b, c["rows"]].detach().float()
                    if li in c.get("grabres", ()):
                        self.store[b][("res", li)] = r.cpu()
                    if li in c.get("delta", {}) or li in c.get("keepres", ()):
                        self.cur[(b, li)] = r
            return None
        return h

    def _comp(self, li, kind):
        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            for b, c in enumerate(self.cfgs):
                if c is not None and li in c.get("grabcomp", ()):
                    self.store[b][(kind, li)] = t[b, c["rows"]].detach().float().cpu()
            return None
        return h

    def _hook_d(self, li):
        base = QPathB._hook(self, li)

        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return out
            dsamp = [b for b, c in enumerate(self.cfgs) if c is not None and li in c.get("delta", {})]
            if not dsamp:
                return base(mod, args, kwargs, out)
            # samples with a delta at this reader: recompute their rows' attention with the patched query here; the others go through QPathB
            others = [None if (c is None or b in dsamp) else c for b, c in enumerate(self.cfgs)]
            saved = self.cfgs
            self.cfgs = others
            try:
                out2 = base(mod, args, kwargs, out)
            finally:
                self.cfgs = saved
            t0 = (out2[0] if isinstance(out2, tuple) else out2).clone()
            for b in dsamp:
                c = self.cfgs[b]; d = c["delta"][li]
                hs = kwargs["hidden_states"][b].float()
                cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
                S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
                rows = c["rows"]; R = len(rows)
                qproj = lambda x: torch.nn.functional.linear(x, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
                gated = mod.q_proj.weight.shape[0] == 2 * nh * hd
                qg = qproj(hs[rows]).view(R, nh, 2 * hd if gated else hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:] if gated else None
                ln = self.v.layers[li].input_layernorm
                if "src" in d:
                    vec = self.cur[(d["src"], d["layer"])].to(q.device) - self.cur[(b, d["layer"])].to(q.device)
                else:
                    vec = d["vec"].to(q.device).float()
                res = self.cur[(b, li)].to(q.device) + vec
                hn = ln(res.to(ln.weight.dtype)).float()
                qg2 = qproj(hn).view(R, nh, 2 * hd if gated else hd)
                q2 = mod.q_norm(qg2[..., :hd].to(mod.q_norm.weight.dtype)).float()
                rm = d.get("rmask")
                if rm is not None:
                    rmt = torch.tensor(rm, device=q.device)[:, None, None]
                    q2 = torch.where(rmt, q2, q)
                if li in c.get("dcheck", ()):
                    self.store[b][("qd", li)] = q2.cpu()
                k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
                vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
                kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
                K = k[:, kh]; V = vv[:, kh]
                qe = torch.cat([q2[..., :rd] * cos[rows][:, None, :] + rotate_half(q2[..., :rd]) * sin[rows][:, None, :], q2[..., rd:]], -1)
                ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                cap = c.get("cap")
                if cap is not None:                                             # passive readouts of the patched query, as QPathB
                    own = torch.tensor(cap["owner"], device=q.device); cr = cap["rows"]
                    self.store[b][("cap", li)] = torch.stack([torch.logsumexp(lg[cr][:, :, own == bb], -1) for bb in range(cap["nb"])], -1).cpu()
                    if cap.get("opt") is not None:
                        rr = torch.tensor([rows[i] for i in cr], device=q.device)
                        mk = torch.arange(S, device=q.device)[None, :] > rr[:, None]
                        ac = torch.softmax(lg[cr].masked_fill(mk[:, None, :], float("-inf")), -1)
                        self.store[b][("opt", li)] = ac[:, :, torch.tensor(cap["opt"], device=q.device)].cpu()
                        self.store[b][("img", li)] = ac[:, :, own >= 0].sum(-1).cpu()
                mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
                o = torch.einsum("ths,shd->thd", att, V)
                if gate is not None:
                    o = torch.sigmoid(gate) * o
                new = torch.nn.functional.linear(o.reshape(R, -1), mod.o_proj.weight.float(), None if mod.o_proj.bias is None else mod.o_proj.bias.float())
                t0[b, rows] = new.to(t0.dtype)
            return (t0,) + tuple(out2[1:]) if isinstance(out2, tuple) else t0
        return h
