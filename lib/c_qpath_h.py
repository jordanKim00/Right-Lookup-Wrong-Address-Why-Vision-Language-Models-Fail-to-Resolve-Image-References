"""QPathG + (1) residual row patches at chosen layers (QPathE's 'rpatch', registered here for any layer) and (2) passive direct-path predictions
that add a fixed vector to the measured component changes (S850; exp/c_qpath_g.py and below are left as they were).
Extra per-sample cfg keys:
  rpatch    {layer: dict(src=batch index, rows=[positions])}: as QPathE (the sample's residual at those positions replaced by the src sample's at the
            input of the layer; everything downstream sees it); layers given at construction (rpatch_layers)
  delta[reader] = dict(src=..., comps=[...] (may be empty), extra=[R, E] tensor, passive=True): the reader's query is recomputed from this
            sample's residual + (src's component outputs - this sample's, summed over comps) + extra, without writing back (QPathG passive).
            With comps empty the prediction is 'the added vector persists unchanged to the reader'; with the band's components it also carries
            the writer's measured response."""
from __future__ import annotations

from c_qpath_g import QPathG


class QPathH(QPathG):
    def __init__(self, v, readers, res_layers=(), comp_layers=(), patch_comps=(), add_layers=(), rpatch_layers=()):
        super().__init__(v, readers, res_layers=res_layers, comp_layers=comp_layers, patch_comps=patch_comps, add_layers=add_layers)
        for li in sorted(set(rpatch_layers)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._rpatch(li), with_kwargs=True, prepend=True))

    def _hook_d(self, li):
        inner = QPathG._hook_d(self, li)

        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return inner(mod, args, kwargs, out)
            saved = self.cfgs; new = []; changed = False
            for b, c in enumerate(saved):
                d = None if c is None else (c.get("delta", {}) or {}).get(li)
                if d is not None and "comps" in d and d.get("extra") is not None:
                    dev = mod.q_proj.weight.device
                    vec = d["extra"].to(dev).float()
                    if d["comps"]:
                        vec = vec + sum((self.ccur[(d["src"], l, kd)] - self.ccur[(b, l, kd)]).to(dev) for l, kd in d["comps"]) * float(d.get("scale", 1.0))
                    c = dict(c); c["delta"] = dict(c["delta"]); c["delta"][li] = dict(vec=vec, rmask=d.get("rmask"), passive=d.get("passive"))
                    changed = True
                new.append(c)
            if not changed:
                return inner(mod, args, kwargs, out)
            self.cfgs = new
            try:
                return inner(mod, args, kwargs, out)
            finally:
                self.cfgs = saved
        return h
