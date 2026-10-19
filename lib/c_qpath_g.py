"""QPathF + passive direct-path predictions and residual steering (S848 / S849; exp/c_qpath_f.py and below are left as they were).
Extra per-sample cfg keys:
  delta[reader]['passive'] = True   the reader's query of the rows is recomputed from (this sample's residual + the delta vector) and its readouts
                                    are stored (cap / opt / img with the recomputed query; 'qd' with dcheck) as in QPathD, but the attention
                                    output is NOT written back: the forward pass stays exactly the unpatched one.  With QPathE's component form
                                    (src, comps) this is the weight-only prediction of a reader query from measured component changes along the
                                    direct path (every late reader gets its own prediction from the same unchanged residual).
  radd      {layer: [R, E] tensor}: added to this sample's residual at its rows at the input of decoder layer l; everything downstream sees it
            (steering).  Runs before any grab at that layer, so grabres there sees the steered residual.
Layers that may receive radd are given at construction (add_layers)."""
from __future__ import annotations

import torch
from c_qpath_e import QPathE
from c_qpath_f import QPathF


class QPathG(QPathF):
    def __init__(self, v, readers, res_layers=(), comp_layers=(), patch_comps=(), add_layers=()):
        super().__init__(v, readers, res_layers=res_layers, comp_layers=comp_layers, patch_comps=patch_comps)
        for li in sorted(set(add_layers)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._radd(li), with_kwargs=True, prepend=True))

    def _radd(self, li):
        def h(mod, args, kwargs):
            if self.cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(self.cfgs) if c is not None and li in c.get("radd", {})]
            if not todo:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            hs2 = hs.clone()
            for b, c in todo:
                pos = torch.tensor(c["rows"], device=hs.device)
                hs2[b, pos] = (hs[b, pos].float() + c["radd"][li].to(hs.device).float()).to(hs.dtype)
            if args:
                return (hs2,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = hs2
            return args, kwargs
        return h

    def _hook_d(self, li):
        inner = QPathE._hook_d(self, li)

        def h(mod, args, kwargs, out):
            res = inner(mod, args, kwargs, out)
            if self.cfgs is None:
                return res
            pas = [b for b, c in enumerate(self.cfgs) if c is not None and (c.get("delta", {}).get(li) or {}).get("passive")]
            if not pas:
                return res
            t_new = res[0] if isinstance(res, tuple) else res
            t_old = out[0] if isinstance(out, tuple) else out
            t_new = t_new.clone()
            for b in pas:
                rows = self.cfgs[b]["rows"]
                t_new[b, rows] = t_old[b, rows]                               # passive: keep the unpatched attention output
            return (t_new,) + tuple(res[1:]) if isinstance(res, tuple) else t_new
        return h
