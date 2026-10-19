"""QPathI + linear-combination residual patches (S854-2; exp/c_qpath_i.py and below are left as they were).
Extra per-sample cfg key:
  rpatch_lin  {layer: dict(terms=[(coef, src batch index), ...], rows=[positions])}: at the input of decoder layer l, this sample's residual at
              those positions is set to sum(coef * src's residual at the same positions), summed in float32 in the order given and cast back
              (same forward pass; sources are read from the layer's unpatched inputs).  Terms [(1, a), (1, b), (-1, a)] return b's residual exactly
              for bf16 values (the self-difference arm uses this as a bit-identity check).
Layers are given at construction (rpatch_lin_layers); QPathE's plain 'rpatch' keeps working."""
from __future__ import annotations

import torch
from c_qpath_i import QPathI


class QPathJ(QPathI):
    def __init__(self, v, readers, res_layers=(), comp_layers=(), patch_comps=(), add_layers=(), rpatch_layers=(), rpatch_lin_layers=()):
        super().__init__(v, readers, res_layers=res_layers, comp_layers=comp_layers, patch_comps=patch_comps, add_layers=add_layers,
                         rpatch_layers=rpatch_layers)
        for li in sorted(set(rpatch_lin_layers)):
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._rpatch_lin(li), with_kwargs=True, prepend=True))

    def _rpatch_lin(self, li):
        def h(mod, args, kwargs):
            if self.cfgs is None:
                return None
            todo = [(b, c["rpatch_lin"][li]) for b, c in enumerate(self.cfgs) if c is not None and li in (c.get("rpatch_lin") or {})]
            if not todo:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            hs2 = hs.clone()
            for b, rp in todo:
                pos = torch.tensor(rp["rows"], device=hs.device); acc = None
                for coef, src in rp["terms"]:
                    x = hs[src, pos].float() * float(coef)
                    acc = x if acc is None else acc + x
                hs2[b, pos] = acc.to(hs2.dtype)
            if args:
                return (hs2,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs); kwargs["hidden_states"] = hs2
            return args, kwargs
        return h
