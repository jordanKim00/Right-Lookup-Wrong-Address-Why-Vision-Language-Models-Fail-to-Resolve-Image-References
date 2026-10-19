"""QPathD + component-level path patching and residual row patching (S837 on; exp/c_qpath_d.py is left as S836 used it).
Extra per-sample cfg keys:
  keepcomp  set of layers: keep this sample's token-mixer output and MLP output of its rows on the device (for same-pass component deltas)
  delta     {reader: dict(src=batch index, comps=[(layer, 'mix' | 'mlp'), ...], scale=1.0)}: in addition to QPathD's vec / src+layer forms,
            the vector is scale * sum over comps of (src sample's component output - this sample's), both kept with keepcomp in the same
            forward pass: the direct path from those components to the reader's query only (scale -1 doubles this sample's own contribution)
  rpatch    {layer: dict(src=batch index, rows=[positions])}: at the input of decoder layer l, this sample's residual at those positions is
            replaced by the src sample's (same forward pass); everything downstream sees it (an ordinary activation patch)
Readers and stores as QPathD."""
from __future__ import annotations

import torch
from c_qpath_d import QPathD


class QPathE(QPathD):
    ccur = property(lambda self: getattr(self._tl, "ccur", {}), lambda self, x: setattr(self._tl, "ccur", x))

    def __init__(self, v, readers, res_layers=(), comp_layers=(), patch_layers=()):
        super().__init__(v, readers, res_layers=res_layers, comp_layers=())
        self.ccur = {}
        for li in sorted(set(comp_layers)):
            L = v.layers[li]
            mix = L.linear_attn if getattr(L, "linear_attn", None) is not None else L.self_attn
            self.hooks.append(mix.register_forward_hook(self._keepc(li, "mix"), with_kwargs=True))
            self.hooks.append(L.mlp.register_forward_hook(self._keepc(li, "mlp"), with_kwargs=True))
        for li in sorted(set(patch_layers)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._rpatch(li), with_kwargs=True, prepend=True))   # before any grab

    def set(self, cfgs):
        super().set(cfgs); self.ccur = {}

    def _keepc(self, li, kind):
        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            for b, c in enumerate(self.cfgs):
                if c is not None and li in c.get("keepcomp", ()):
                    self.ccur[(b, li, kind)] = t[b, c["rows"]].detach().float()
                    if li in c.get("grabcomp", ()):
                        self.store[b][(kind, li)] = self.ccur[(b, li, kind)].cpu()
            return None
        return h

    def _rpatch(self, li):
        def h(mod, args, kwargs):
            if self.cfgs is None:
                return None
            todo = [(b, c["rpatch"][li]) for b, c in enumerate(self.cfgs) if c is not None and li in c.get("rpatch", {})]
            if not todo:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            hs2 = hs.clone()
            for b, rp in todo:
                pos = torch.tensor(rp["rows"], device=hs.device)
                hs2[b, pos] = hs[rp["src"], pos]
            if args:
                return (hs2,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = hs2
            return args, kwargs
        return h

    def _hook_d(self, li):
        inner = QPathD._hook_d(self, li)

        def h(mod, args, kwargs, out):
            if self.cfgs is not None:                                         # turn component deltas into plain vectors for QPathD
                saved = self.cfgs; new = []
                for b, c in enumerate(saved):
                    d = None if c is None else c.get("delta", {}).get(li)
                    if d is not None and "comps" in d:
                        dev = mod.q_proj.weight.device                             # components live on the cards of their layers (device map)
                        vec = sum((self.ccur[(d["src"], l, kd)] - self.ccur[(b, l, kd)]).to(dev) for l, kd in d["comps"]) * float(d.get("scale", 1.0))
                        c = dict(c); c["delta"] = dict(c["delta"]); c["delta"][li] = dict(vec=vec, rmask=d.get("rmask"))
                    new.append(c)
                self.cfgs = new
                try:
                    return inner(mod, args, kwargs, out)
                finally:
                    self.cfgs = saved
            return inner(mod, args, kwargs, out)
        return h
