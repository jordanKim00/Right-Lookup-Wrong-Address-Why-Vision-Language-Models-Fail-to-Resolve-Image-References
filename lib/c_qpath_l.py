"""QPathK + component outputs carried between forward passes (S867 cross-layout band transplant; exp/c_qpath_k.py and below are left as they were).
Extra per-sample cfg keys (components must be among patch_comps):
  cgrab  [(layer, 'mix' | 'mlp'), ...]: this sample's component outputs at its rows are stored (self.store[b][('cg', layer, kind)], float32 cpu),
         after every other patch of that component.
  cinj   dict(vals={(layer, kind): tensor [R, d]}): at those components this sample's output at its rows is set to the given values (cast to the
         output dtype), after every other patch of that component.  Values grabbed from the sample's own input in another pass give the cross-pass
         numerical control."""
from __future__ import annotations

from c_qpath_k import QPathK


class QPathL(QPathK):
    def _cp(self, li, kind):
        base = QPathK._cp(self, li, kind)

        def h(mod, args, kwargs, out):
            o1 = base(mod, args, kwargs, out)
            if self.cfgs is None:
                return o1
            cur = o1 if o1 is not None else out
            t = cur[0] if isinstance(cur, tuple) else cur
            t2 = None
            for b, c in enumerate(self.cfgs):
                if c is None:
                    continue
                if (li, kind) in [tuple(x) for x in c.get("cgrab", [])]:
                    self.store[b][("cg", li, kind)] = t[b, c["rows"]].detach().float().cpu()
                vals = (c.get("cinj") or {}).get("vals", {})
                if (li, kind) in vals:
                    if t2 is None:
                        t2 = t.clone()
                    t2[b, c["rows"]] = vals[(li, kind)].to(t2.device, t2.dtype)
            if t2 is None:
                return o1
            return (t2,) + tuple(cur[1:]) if isinstance(cur, tuple) else t2
        return h
