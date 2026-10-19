"""R³ with a learned frame binder and the mechanism-derived Resolve / Rebind (S724 option "v1_gate").
Read: the earlier module's detector, binding window and learned frame gate g (exp/r3_repair.R3Repair, trained on start-relative targets and
frame labels only) decide which ordinal sources are end-bound and where their frame word is.  Resolve: t = N_hat + 1 - k_hat with N_hat from
the image-block codes (max or last block, chosen per model in S724) -- the learned block mixture of the earlier module is not used.
Rebind: for every end-bound source, the frame write found by the earlier module (frame word -> START), and the ordinal write when t != k_hat.
Interface = exp/r3_min_md.R3MinMD's (set / logits / both / both_gen / run / last_detail / last_codes / stats)."""
from __future__ import annotations

import torch

from r3_repair import R3Repair

C = 10


class R3V1C:
    def __init__(self, v, ckpt, reader, n_mode="last", ext_cal=None):
        self.v, self.on, self.n_mode = v, False, n_mode
        self.inner = R3Repair(v, ckpt, reader)
        self.ext = None                                                             # S735: frame words added by calibration (exp/s717_framecal.py)
        if ext_cal:
            fr = torch.load(ext_cal, weights_only=False)["cal"]["frame"]
            self.ext = {f: (fr["mu"][f], fr["thr"][f]) for f in fr["mu"] if f not in ("end", "last", "final", "start", "beginning")}
            self.ext_start = fr["mu_start"]
            self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.inner.tau = 0.0                       # admission is the Read's (g > .5); the earlier pointer's confidence belongs to the replaced Resolve
        self.stats = dict(inputs=0, sources=0, rebound=0, frame=0, frame_only=0)
        self.last_detail, self.last_codes, self.last_logits = [], None, None

    def set(self, on):
        self.on = on

    @property
    def ST(self):                                  # harnesses that drive the writes themselves (exp/s626_patch.py) set rep.ST
        return self.inner.ST

    @ST.setter
    def ST(self, x):
        self.inner.ST = x

    @torch.no_grad()
    def plan(self, text, pics):
        R = self.inner
        inp, writes = R.plan(text, pics)
        codes = R.last_codes or []
        nh = (len(codes) if self.n_mode == "oracle" else int(codes[-1]) if self.n_mode == "last" else int(max(codes))) if codes else 0   # oracle: diagnosis only
        Mbw = R.cal_w["Mb"]
        ord_pos = {int(d["pos"]) for d in R.last_detail}
        frame_w = [(p, dlt) for p, dlt in writes if p not in ord_pos]               # frame writes chosen by the earlier module
        frame_at = {}
        for p, dlt in frame_w:
            frame_at[p] = dlt
        new, self.last_detail = [], []
        for d in R.last_detail:
            g = float(d["map"][1]) if d["map"][0] == "scan" else 0.0
            kh = int(d["kh"]); bound = g > .5
            t_ = nh + 1 - kh
            fpos = d.get("fpos"); ext_hit = None
            if not bound and self.ext:                                              # S735: the gate did not bind -> a calibrated new frame word in the clause
                from r3_build import clause_window
                ids = R.last_ids; turn = max(i for i, x in enumerate(ids) if x == self.im_start)
                Lw_, Rw_ = clause_window(self.v.tok, ids, int(d["pos"]), R.last_s0, turn)
                for q in Lw_ + Rw_:
                    h = R.last_Hb[q - R.last_s0]
                    hit = next((f for f, (mu, th) in self.ext.items() if float((h - mu).norm()) <= th), None)
                    if hit is not None:
                        bound, fpos, ext_hit = True, q, hit
                        frame_at[q] = self.ext_start - self.ext[hit][0]
                        break
            act = bound and 1 <= t_ <= C and t_ != kh
            if act:
                new.append((int(d["pos"]), Mbw[t_ - 1] - Mbw[kh - 1])); self.stats["rebound"] += 1
            if bound and 1 <= t_ <= C and fpos is not None and int(fpos) in frame_at:   # Rebind as R3MinMD: an unresolvable reference gets no write
                new.append((int(fpos), frame_at[int(fpos)])); self.stats["frame"] += 1; self.stats["frame_only"] += int(not act)
            self.last_detail.append(dict(tok=d["tok"], kh=kh, t=t_, N=d.get("N"), N_hat=nh, act=bool(act), fonly=bool(bound and 1 <= t_ <= C and not act),
                                         map=("v1c", round(g, 3)), wmax=1.0, frame=d.get("frame") if bound else None, anchor=None, attn=None,
                                         pos=d["pos"], fpos=fpos))
        dedup = {}
        for p, dlt in new:
            dedup[p] = dlt
        self.stats["sources"] += len(R.last_detail)
        self.last_codes = codes
        return inp, list(dedup.items())

    @torch.no_grad()
    def run(self, inp, writes):
        R = self.inner
        R.ST = dict(writes=writes)
        out = self.v.model(**inp, use_cache=False, return_dict=True, **self.v.forward_extra)
        R.ST = {}
        return out.logits[0, -1].float()

    @torch.no_grad()
    def logits(self, text, pics):
        if self.on and pics:
            inp, writes = self.plan(text, pics); self.stats["inputs"] += 1
        else:
            inp, writes = self.v.encode(text, pics), []
        return self.run(inp, writes)

    @torch.no_grad()
    def generate(self, text, pics, max_new=16):
        R = self.inner
        if self.on and pics:
            inp, writes = self.plan(text, pics); self.stats["inputs"] += 1
        else:
            inp, writes = self.v.encode(text, pics), []
        R.ST = dict(writes=writes, wmax=max([p_ for p_, _ in writes], default=-1))
        out = self.v.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        R.ST = {}
        return self.v.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def both(self, text, pics, cand):
        """base = the unedited forward; repaired = plan (capture) + the forward with the writes, run only when something is written."""
        if not pics:
            a = int(self.run(self.v.encode(text, pics), [])[cand].argmax())
            return a, a
        import time
        tm = getattr(self, "timing", None)                                           # S721 latency, as R3MinMD
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a = int(self.run(self.v.encode(text, pics), [])[cand].argmax())
        torch.cuda.synchronize(); t1 = time.perf_counter()
        inp, writes = self.plan(text, pics)
        self.stats["inputs"] += 1
        b = int(self.run(inp, writes)[cand].argmax()) if writes else a
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if tm is not None:
            tm.append(dict(base=t1 - t0, r3=t2 - t1, wrote=bool(writes), tokens=int(inp["input_ids"].shape[1])))
        return a, b

    def both_gen(self, text, pics, max_new=16):
        self.set(False); a = self.generate(text, pics, max_new)
        self.set(True); b = self.generate(text, pics, max_new)
        self.set(False)
        return a, b
