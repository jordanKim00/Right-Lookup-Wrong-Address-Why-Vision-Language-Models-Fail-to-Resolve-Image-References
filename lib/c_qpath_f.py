"""QPathE + component activation patches (S840): a component's output at the sample's rows is replaced by the donor's (same forward pass) and
everything downstream sees it - the full-downstream counterpart of QPathE's direct-path 'comps' deltas.
Extra per-sample cfg keys:
  cpatch  dict(src=batch index, comps=[(layer, 'mix' | 'mlp'), ...]): at those components the output of this sample's rows is set to the src
          sample's output of the same rows (the src sample itself is untouched)
Layers whose components may be patched are given at construction (patch_comps)."""
from __future__ import annotations

from c_qpath_e import QPathE


class QPathF(QPathE):
    def __init__(self, v, readers, res_layers=(), comp_layers=(), patch_comps=()):
        super().__init__(v, readers, res_layers=res_layers, comp_layers=comp_layers)
        for li in sorted(set(patch_comps)):
            L = v.layers[li]
            mix = L.linear_attn if getattr(L, "linear_attn", None) is not None else L.self_attn
            self.hooks.append(mix.register_forward_hook(self._cp(li, "mix"), with_kwargs=True))      # after the reader / keep hooks
            self.hooks.append(L.mlp.register_forward_hook(self._cp(li, "mlp"), with_kwargs=True))

    def _cp(self, li, kind):
        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(self.cfgs) if c is not None and (li, kind) in [tuple(x) for x in c.get("cpatch", {}).get("comps", [])]]
            if not todo:
                return None
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            for b, c in todo:
                rows = c["rows"]; src = c["cpatch"]["src"]
                t2[b, rows] = t[src, rows]
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h
