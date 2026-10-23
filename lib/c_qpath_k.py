"""QPathJ + component patches at their own positions (S863 BAND_TAIL; exp/c_qpath_j.py and below are left as they were).
Extra per-sample cfg key:
  cpatch_rows  dict(src=batch index, comps=[(layer, 'mix' | 'mlp'), ...], rows=[positions]): at those components this sample's output at the
               given positions is set to src's (same forward pass; sources are the components' unpatched outputs), applied after the parent's
               'cpatch' / 'cpatch_lin'.  A self source returns the output bit for bit.  Components must be among patch_comps."""
from __future__ import annotations

from c_qpath_i import QPathI
from c_qpath_j import QPathJ


class QPathK(QPathJ):
    def _cp(self, li, kind):
        base = QPathI._cp(self, li, kind)

        def h(mod, args, kwargs, out):
            o1 = base(mod, args, kwargs, out)
            if self.cfgs is None:
                return o1
            todo = [(b, c) for b, c in enumerate(self.cfgs)
                    if c is not None and (li, kind) in [tuple(x) for x in (c.get("cpatch_rows") or {}).get("comps", [])]]
            if not todo:
                return o1
            t0 = out[0] if isinstance(out, tuple) else out                    # unpatched outputs of every sample (the sources)
            cur = o1 if o1 is not None else out
            t = cur[0] if isinstance(cur, tuple) else cur
            t2 = t.clone()
            for b, c in todo:
                rows = c["cpatch_rows"]["rows"]; src = c["cpatch_rows"]["src"]
                t2[b, rows] = t0[src, rows]
            return (t2,) + tuple(cur[1:]) if isinstance(cur, tuple) else t2
        return h
