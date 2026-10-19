"""Multi-device-safe subclass of r3_build.Model (S709 / S710): the same capture and write hooks, with every index tensor moved to the
device of the hooked layer's output (27B is split over several cards with device_map).  No numerical change on a single device."""
import torch
import r3_build


class ModelMD(r3_build.Model):
    def _hook(self, l):
        def h(m, i, o):
            ST = self.ST
            if not ST:
                return o
            t = o[0] if isinstance(o, tuple) else o
            dev = t.device
            if ST.get("cap"):
                if l in (self.Lb, self.Lr) or l in self.Lrel:
                    ST.setdefault("H", {})[l] = t[0, ST["span"].to(dev)].detach().float().cpu()
                if l in self.Rlayers and ST.get("B"):
                    f = []
                    for b in ST["B"]:
                        b = b.to(dev)
                        allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                               t[0, int(b[-1])].float()])
                        f.append(torch.cat([allsite[self.SITES.index(s_)] for s_ in self.Rsites]))
                    ST.setdefault("bf", {})[l] = torch.stack(f).cpu()
                return o
            if l == self.Lb and ST.get("writes"):
                t2 = t.clone()
                for p, dlt in ST["writes"]:
                    t2[0, p] = (t[0, p].float() + dlt.to(dev)).to(t.dtype)
                return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
            return o
        return h
