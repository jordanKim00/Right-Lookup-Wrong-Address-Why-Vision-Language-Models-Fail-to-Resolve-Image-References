"""QPathH + linear-combination component patches (S852; exp/c_qpath_h.py and below are left as they were).
Extra per-sample cfg key:
  cpatch_lin  dict(terms=[(coef, src batch index), ...], comps=[(layer, 'mix' | 'mlp'), ...]): at those components the output of this sample's
              rows is set to sum(coef * src's output of the same rows), summed in float32 in the order given and cast back to the output
              dtype (same forward pass; sources are read from the layer's unpatched outputs).  For bf16 outputs, terms [(1, a), (1, b), (-1, a)]
              return b's output exactly, which the self-difference arm uses as a bit-identity check.
Layers whose components may be patched are given at construction (patch_comps), as for QPathF's 'cpatch' (which keeps working)."""
from __future__ import annotations

from c_qpath_f import QPathF
from c_qpath_h import QPathH


class QPathI(QPathH):
    def _cp(self, li, kind):
        base = QPathF._cp(self, li, kind)

        def h(mod, args, kwargs, out):
            o1 = base(mod, args, kwargs, out)
            if self.cfgs is None:
                return o1
            todo = [(b, c) for b, c in enumerate(self.cfgs)
                    if c is not None and (li, kind) in [tuple(x) for x in (c.get("cpatch_lin") or {}).get("comps", [])]]
            if not todo:
                return o1
            t0 = out[0] if isinstance(out, tuple) else out                    # unpatched outputs of every sample (the sources)
            cur = o1 if o1 is not None else out
            t = cur[0] if isinstance(cur, tuple) else cur
            t2 = t.clone()
            for b, c in todo:
                rows = c["rows"]; acc = None
                for coef, src in c["cpatch_lin"]["terms"]:
                    x = t0[src, rows].float() * float(coef)
                    acc = x if acc is None else acc + x
                t2[b, rows] = acc.to(t2.dtype)
            return (t2,) + tuple(cur[1:]) if isinstance(cur, tuple) else t2
        return h
