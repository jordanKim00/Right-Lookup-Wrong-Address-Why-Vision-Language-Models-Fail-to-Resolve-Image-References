"""R³-min for models split over several cards (S712+): the r3_repair.R3Repair min_mode plan with the S709 readers and calibration
(exp/s709_cal.py), every index tensor moved to the hooked layer's device (as exp/r3_build_md.py), and the S711 rule (v2): an ordinal
bound to an END anchor whose resolved target equals the literal ordinal (t = k_hat) still gets the frame write (END -> START).
Interface = R3Repair's (set / logits / generate / both / last_detail / last_codes / stats), so exp/s582_evalrepair.run takes it as `rep`;
both() reads the base answer from the unedited capture forward and runs the repaired forward only when something is written.
Same plan as exp/s709_eval.plan + exp/s711_eval (R3V2) on the same input; no learned relation reader, mixture or fallback."""
from __future__ import annotations

import torch

from r3_build import clause_window
from s588c_reader import probs as reader_probs

C = 10


class R3MinMD:
    def __init__(self, v, calfile, v2=True, det_rule="abs", det_param=None, anchor_scale=1.0, ablate=None, n_mode="max"):
        st = torch.load(calfile, weights_only=False)
        self.v, self.on, self.v2 = v, False, v2
        self.n_mode = n_mode                                                         # "max": largest block code; "last": code of the last block (end anchor)
        self.ablate = ablate                                                         # S721: None | no_frame | no_ord | rand_target | oracle_n
        self.det_rule, self.det_param, self.anchor_scale = det_rule, det_param, anchor_scale   # S716 READ rule (defaults = R³-v2)
        self.cal, self.Lb = st["cal"], st["Lb"]
        self.Lw = st.get("Lw", self.Lb)
        self.cal_w = st.get("cal_w") or dict(Mb=self.cal["Mb"], frame=self.cal["frame"])   # S731: write vectors at L_w (tests stay at L_b)
        R = torch.load(f"/vlm/results/s588c_reader_{st['reader']}.pt", weights_only=False)
        self.RP, self.Rlayers, self.Rsites, self.SITES = R["P"], R["layers"], R["sites"], R["SITES"]
        self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.ST = {}
        self.stats = dict(inputs=0, sources=0, rebound=0, frame=0, frame_only=0)
        self.last_detail, self.last_codes = [], None
        for l in sorted(set([self.Lb, self.Lw] + list(self.Rlayers))):
            v.layers[l].register_forward_hook(self._hook(l))

    def set(self, on):
        self.on = on

    def _hook(self, l):
        def h(m, i, o):
            ST = self.ST
            if not ST:
                return o
            t = o[0] if isinstance(o, tuple) else o
            dev = t.device
            if ST.get("cap"):
                if l == self.Lb:
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
            if l == self.Lw and ST.get("writes") and t.shape[1] > ST.get("wmax", -1):      # prefill only (generation)
                t2 = t.clone()
                for p, dlt in ST["writes"]:
                    t2[:, p] = (t[:, p].float() + dlt.to(dev)).to(t.dtype)
                return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
            return o
        return h

    @torch.no_grad()
    def plan(self, text, pics):
        v, cal = self.v, self.cal
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b) for b in v.image_blocks(inp["input_ids"], len(pics))]
        turn = max(i for i, x in enumerate(ids) if x == self.im_start)
        s0 = int(B[-1][-1]) + 2
        self.ST = dict(cap=True, span=torch.arange(s0, turn), B=B)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        self.last_logits = out.logits[0, -1].float()                               # the capture forward is unedited (native logits)
        H, bf = self.ST["H"], self.ST.get("bf", {})
        self.ST = {}
        Hb = H[self.Lb]
        if cal["det_space"] == "full":
            Dm = torch.cdist(Hb, cal["Mb"]); th = cal["th_full"]
        else:
            Dm = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]); th = cal["th_chart"]
        if self.det_rule in ("margin", "abs_or_margin"):                             # the margin is computed in the ordinal chart (identity lives there)
            Dm = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"])
        d2, k2 = Dm.topk(2, dim=1, largest=False)
        dist, kh0 = d2[:, 0], k2[:, 0]
        X = torch.cat([bf[l] for l in self.Rlayers], 1)
        codes = (reader_probs(self.RP, X).argmax(1) + 1).long(); maxcode = int(codes.max())
        if self.n_mode == "last":                                                    # count from the model's own end anchor: the last block's code
            maxcode = int(codes[-1])
        if self.ablate == "oracle_n":                                                # S721 "without Read": N from the input image count
            maxcode = len(B)
        self.last_codes = codes.tolist()
        fr = cal["frame"]
        writes, self.last_detail = [], []
        if self.det_rule == "abs":                                                   # R³-v2: absolute chart threshold
            det = dist <= (th if self.det_param is None else self.det_param)
        elif self.det_rule == "margin":                                              # nearest prototype much closer than the second
            det = dist / d2[:, 1].clamp_min(1e-6) <= self.det_param
        elif self.det_rule == "abs_or_margin":
            det = (dist <= th) | (dist / d2[:, 1].clamp_min(1e-6) <= self.det_param)
        elif self.det_rule == "rel":                                                 # in-context: relative to the span's median distance
            det = dist / dist.median().clamp_min(1e-6) <= self.det_param
        else:
            raise ValueError(self.det_rule)
        srcs = det.nonzero().flatten().tolist()
        for jx in srcs:
            kh = int(kh0[jx]) + 1
            L, Rr = clause_window(v.tok, ids, s0 + jx, s0, turn)
            win = [q - s0 for q in L + Rr]
            hits = [(qq, a_) for qq in win for a_ in fr["mu"] if float((Hb[qq] - fr["mu"][a_]).norm()) <= self.anchor_scale * fr["thr"][a_]]
            best = hits[0] if hits else None                                        # a token bound to this ordinal is an end anchor
            t_ = (maxcode + 1 - kh) if best is not None else kh                      # N_hat + 1 - k_hat
            act = best is not None and 1 <= t_ <= C and t_ != kh
            fonly = self.v2 and best is not None and not act and t_ == kh           # S711: END-bound, t = k_hat -> frame write only
            fpos = (s0 + best[0]) if best is not None else None
            if act and self.ablate != "no_ord":
                tw = t_
                if self.ablate == "rand_target":                                     # S721: same admission, a wrong valid target
                    import random as _r
                    alts = [x for x in range(1, min(C, maxcode) + 1) if x not in (kh, t_)]
                    if alts:
                        tw = _r.Random(hash((tuple(ids), int(jx))) & 0xffffffff).choice(alts)
                writes.append((s0 + jx, self.cal_w["Mb"][tw - 1] - self.cal_w["Mb"][kh - 1]))
            if (act or fonly) and self.ablate != "no_frame":
                writes.append((fpos, self.cal_w["frame"]["mu_start"] - self.cal_w["frame"]["mu"][best[1]]))
                self.stats["frame"] += 1; self.stats["frame_only"] += int(fonly)
            self.stats["rebound"] += int(act)
            self.last_detail.append(dict(tok=v.tok.decode([ids[s0 + jx]]), kh=kh, t=t_, N=len(B), N_hat=maxcode, act=bool(act),
                                         fonly=bool(fonly), map=("min", int(best is not None)), wmax=1.0,
                                         frame=(v.tok.decode([ids[fpos]]) if (act or fonly) else None),
                                         anchor=(best[1] if best is not None else None), attn=None, pos=s0 + jx, fpos=fpos))
        self.stats["sources"] += len(srcs)
        return inp, writes

    @torch.no_grad()
    def logits(self, text, pics):
        v = self.v
        if self.on and pics:
            inp, writes = self.plan(text, pics)
            self.stats["inputs"] += 1
        else:
            inp, writes = v.encode(text, pics), []
        self.ST = dict(writes=writes)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        self.ST = {}
        return out.logits[0, -1].float()

    @torch.no_grad()
    def generate(self, text, pics, max_new=16):
        v = self.v
        if self.on and pics:
            inp, writes = self.plan(text, pics)
            self.stats["inputs"] += 1
        else:
            inp, writes = v.encode(text, pics), []
        self.ST = dict(writes=writes, wmax=max([p_ for p_, _ in writes], default=-1))
        out = v.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        self.ST = {}
        return v.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def both_gen(self, text, pics, max_new=16):
        self.set(False); a = self.generate(text, pics, max_new)
        self.set(True); b = self.generate(text, pics, max_new)
        self.set(False)
        return a, b

    @torch.no_grad()
    def run(self, inp, writes):
        self.ST = dict(writes=writes)
        out = self.v.model(**inp, use_cache=False, return_dict=True, **self.v.forward_extra)
        self.ST = {}
        return out.logits[0, -1].float()

    def both(self, text, pics, cand):
        """base and repaired answers on the same input: the base logits come from the plan's capture forward (unedited; the hooks only
        read), and the repaired forward runs only when the plan writes something (otherwise repaired = base)."""
        if not pics:
            a = int(self.logits(text, pics)[cand].argmax())
            return a, a
        tm = getattr(self, "timing", None)                                           # S721 latency: plain forward vs R³ (capture + plan + run)
        if tm is not None:
            import time
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                self.v.model(**self.v.encode(text, pics), use_cache=False, return_dict=True, **self.v.forward_extra)
            torch.cuda.synchronize(); t1 = time.perf_counter()
        inp, writes = self.plan(text, pics)
        self.stats["inputs"] += 1
        a = int(self.last_logits[cand].argmax())
        b = int(self.run(inp, writes)[cand].argmax()) if writes else a
        if tm is not None:
            torch.cuda.synchronize(); t2 = time.perf_counter()
            tm.append(dict(base=t1 - t0, r3=t2 - t1, wrote=bool(writes), tokens=int(inp["input_ids"].shape[1])))
        return a, b
