"""R³ repair at inference (checkpoints from r3_build.py): the same interface as nrr_eval.NRRMLRepair
(set / logits / both / plan / last_detail / stats).

Per input: pass 1 reads the question span at L_b (detector, pointer), at the relation layers, and the image blocks at
the end-reader layers; every detected source gets the scanner's distribution over targets from its two-sided clause
window; a confident choice (tau) of a target != k_hat writes h_src += Mb[t] - Mb[k_hat] at L_b, and a confident
end-relative choice (relation gate > .5) also writes the frame carrier: a calibrated anchor in the clause
(mu_start - mu_anchor), else the clause token the relation head attends to most (generic frame direction).
Pass 2 runs with the writes.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/vlm/exp")
from s588b_nrr import Scanner
from s588c_reader import probs as reader_probs
from r3_build import clause_window
import r3_bind

C = 10


class R3Repair:
    def __init__(self, v, ckpt, reader, tau=None):
        st = torch.load(ckpt, weights_only=False, map_location="cpu")
        self.v, self.on = v, False
        self.cal = st["cal"]; self.Lb, self.Lr, self.Lrel = st["Lb"], st["Lr"], list(st["Lrel"])
        self.tau = tau if tau is not None else st.get("tau", .9)
        self.left_scope = st.get("left_scope")                                    # S606 V1 window rule (None = two-sided)
        self.fwd_safe = st.get("fwd_safe", False)                                  # S606 R1: never rebind a forward-gated source
        self.anchor_gate = st.get("anchor_gate", False)                            # S619e
        self.anchor_own = st.get("anchor_own")                                     # S619f: None / "abs" / "rel"
        self.anchor_span = st.get("anchor_span")                                   # S619g: "own" = search the owned span (before rho)
        self.det_rthr = st.get("det_rthr")                                         # S624c: in-context relative source detection
        self.codes_physical = st.get("codes_physical", False)                      # S625: block coordinate = input order
        self.native_back = st.get("native_back")                                   # S618: count from the end with the model's own
        if self.native_back is not None:                                           # DeltaNet decay ranking (one layer, one head)
            import r3_native
            r3_native.install(v, self.native_back["layer"])
        if st.get("bind") is not None and st["bind"].get("mode") == "dseg":          # installed after r3_native (chained)
            import r3_dseg
            r3_dseg.install(v)
            self.dseg_layers = sorted({l for l, h in st["bind"]["dheads"]})
        reader = st.get("reader_tag") or reader                                    # S609 modules carry their own end reader
        self.bind = st.get("bind")                                                 # S609: binding prior from the model's attention
        if self.bind is not None:
            r3_bind.install(v)
            self.att_layers = (sorted({l for l, h in self.bind["heads_after"] + self.bind["heads_front"]})
                               if self.bind.get("mode") != "dseg" else None)
        R = torch.load(f"/vlm/results/s588c_reader_{reader}.pt", weights_only=False)
        self.RP, self.Rlayers, self.Rsites, self.SITES = R["P"], R["layers"], R["sites"], R["SITES"]
        self.dseg = self.bind is not None and self.bind.get("mode") == "dseg"       # S619: windows from the DeltaNet memory
        d_ = st["mod"]["rq.weight"].shape[1]
        self.mod = Scanner(d_, mix=st.get("scan_mix", "prob")).to(v.device).float()
        self.mod.load_state_dict(st["mod"]); self.mod.eval()
        self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.ST = {}
        self.stats = dict(inputs=0, sources=0, rebound=0, frame=0, frame_attn=0)
        self.last_detail = []
        self.Lw = st.get("Lw", self.Lb)                                            # S623: write layer (default = read layer L_b)
        self.cal_w = st.get("cal_w") or dict(Mb=self.cal["Mb"], frame=self.cal["frame"])
        for l in sorted(set([self.Lb, self.Lw] + self.Lrel + list(self.Rlayers))):
            v.layers[l].register_forward_hook(self._hook(l))
        fr = self.cal["frame"]
        self.fgen = fr["mu_start"] - 0.5 * (fr["mu"]["end"] + fr["mu"]["last"])
        frw = self.cal_w["frame"]
        self.fgen_w = frw["mu_start"] - 0.5 * (frw["mu"]["end"] + frw["mu"]["last"])

    def set(self, on):
        self.on = on

    def _hook(self, l):
        def h(m, i, o):
            ST = self.ST
            if not ST:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST.get("cap"):
                if l == self.Lb or l in self.Lrel:
                    ST.setdefault("H", {})[l] = t[0, ST["span"]].detach().float().cpu()
                if l in self.Rlayers and ST.get("B"):
                    f = []
                    for b in ST["B"]:
                        allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                               t[0, int(b[-1])].float()])
                        f.append(torch.cat([allsite[self.SITES.index(s_)] for s_ in self.Rsites]))
                    ST.setdefault("bf", {})[l] = torch.stack(f).cpu()
                return o
            if l == self.Lw and ST.get("writes") and t.shape[1] > ST.get("wmax", -1):      # prefill only (generation)
                t2 = t.clone()
                for p, dlt in ST["writes"]:
                    t2[:, p] = (t[:, p].float() + dlt.to(t.device)).to(t.dtype)                   # every batch row (S626 batched patching)
                return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
            return o
        return h

    @torch.no_grad()
    def plan(self, text, pics):
        v, cal = self.v, self.cal
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b, device=v.device) for b in v.image_blocks(inp["input_ids"], len(pics))]
        turn = max(i for i, x in enumerate(ids) if x == self.im_start)
        s0 = int(B[-1][-1]) + 2
        self.ST = dict(cap=True, span=torch.arange(s0, turn, device=v.device), B=B)
        if self.bind is not None and not self.dseg:
            r3_bind.start(torch.arange(s0, turn), self.att_layers)
        if self.dseg:
            import r3_dseg
            r3_dseg.start(self.dseg_layers)
        if self.native_back is not None:                                           # S618: keep one DeltaNet head's q/k/v/g/beta
            import r3_native
            r3_native.start(self.native_back["head"])
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        A = r3_bind.stop() if (self.bind is not None and not self.dseg) else None
        self.last_A = A                                                            # S624d: binding maps for diagnosis
        KD = r3_dseg.stop() if self.dseg else None
        NBk = r3_native.stop() if self.native_back is not None else None
        nblocks = [(int(b[0]) - 1, int(b[-1]) + 1) for b in B]
        H, bf = self.ST["H"], self.ST.get("bf", {})
        self.ST = {}
        Hb = H[self.Lb]
        self.last_Hb, self.last_s0, self.last_ids = Hb, s0, ids                      # S735: read-layer states for an added frame lexicon (no effect here)
        fr0 = cal["frame"]
        dA = {a: (Hb - fr0["mu"][a]).norm(dim=1) for a in fr0["mu"]}             # S619f: anchor distances, in-context medians
        medA = {a: float(x.median()) for a, x in dA.items()}
        if cal["det_space"] == "full":
            dist, kh0 = torch.cdist(Hb, cal["Mb"]).min(1); th = cal["th_full"]
        else:
            dist, kh0 = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]).min(1); th = cal["th_chart"]
        X = torch.cat([bf[l] for l in self.Rlayers], 1)
        PR = reader_probs(self.RP, X)
        codes = (PR.argmax(1) + 1).long(); maxcode = int(codes.max())
        if self.codes_physical:                                                    # S625: the b-th block of the input is block b
            codes = torch.arange(1, len(B) + 1, device=codes.device).long(); maxcode = len(B)
        self.last_codes = codes.tolist()
        hrel = torch.cat([H[l] for l in self.Lrel], 1)
        fr = cal["frame"]; frw = self.cal_w["frame"]; Mbw = self.cal_w["Mb"]        # tests read at L_b, writes use L_w vectors
        writes, plan = [], []
        self.last_detail = []
        srcs = ((dist / dist.median() <= self.det_rthr) if self.det_rthr else (dist <= th)).nonzero().flatten().tolist()
        own = None; win_own = None; self.last_C = {}
        if self.dseg and srcs:                                                     # S619: one ownership map for all sources
            own = r3_dseg.owners(KD, self.bind["dheads"], [s0 + x for x in srcs], min(turn, s0 + max(srcs) + 1 + self.bind["capR"]))
            self.last_own = own
        for jx in srcs:
            kh = int(kh0[jx]) + 1
            bv = None
            if self.dseg:                                                          # S619: tokens the DeltaNet memory gives this source
                hi = min(turn, s0 + jx + 1 + self.bind["capR"])
                win = [q - s0 for q in range(s0 + jx + 1, hi) if own.get(q) == s0 + jx]
                win_own = list(win)                                                # S621c: the owned span before any threshold
                Cs = r3_dseg.contrib(KD, self.bind["dheads"], s0 + jx, hi)
                self.last_C[s0 + jx] = (Cs.tolist(), hi)
                if self.bind.get("th") is not None:                                # S619b: contribution threshold
                    win = [q for q in win if float(Cs[q - jx]) >= self.bind["th"]]
                if self.bind.get("rho") is not None and win:                       # S619c: relative to the source's own peak
                    peak = max(float(Cs[q - jx]) for q in win)
                    win = [q for q in win if float(Cs[q - jx]) >= self.bind["rho"] * peak]
                bv = [float(Cs[q - jx]) for q in win]; pr = None
                L, Rr = [], [s0 + q for q in win]
            elif self.bind is not None and self.bind.get("mode") == "hard":                # S609c: the model's binding window
                win, bv = r3_bind.window_hard(A, jx, turn - s0, self.bind, srcs, self.bind["capL"], self.bind["capR"])
                pr = None
                L, Rr = [s0 + q for q in win if q < jx], [s0 + q for q in win if q > jx]
            elif self.bind is not None:
                win, pr = r3_bind.prior(A, jx, turn - s0, self.bind, self.bind["capL"], self.bind["capR"], srcs=srcs)
                L, Rr = [s0 + q for q in win if q < jx], [s0 + q for q in win if q > jx]
            else:
                L, Rr = clause_window(v.tok, ids, s0 + jx, s0, turn, left_scope=self.left_scope)
                win, pr = [q - s0 for q in L + Rr], None
            if getattr(self, "min_mode", False):                                   # S644 R3-min: no relation reader, mixture or fallbacks
                hits = [(qq, bq, a) for qq, bq in zip(win, bv or [1.0] * len(win)) for a in fr["mu"]
                        if float((Hb[qq] - fr["mu"][a]).norm()) <= fr["thr"][a]]     # a token bound to this ordinal is an end anchor
                best = max(hits, key=lambda h_: h_[1]) if hits else None
                t_ = (maxcode + 1 - kh) if best is not None else kh                  # the missing step: N_hat + 1 - k_hat
                act = best is not None and 1 <= t_ <= C and t_ != kh
                fpos = None
                if act:
                    writes.append((s0 + jx, Mbw[t_ - 1] - Mbw[kh - 1]))
                    fpos = s0 + best[0]; writes.append((fpos, frw["mu_start"] - frw["mu"][best[2]])); self.stats["frame"] += 1
                plan.append((s0 + jx, kh, t_ if act else kh))
                self.last_detail.append(dict(tok=v.tok.decode([ids[s0 + jx]]), kh=kh, t=t_, N=len(B), act=bool(act), map=("min", int(best is not None)),
                                             wmax=1.0, frame=(v.tok.decode([ids[fpos]]) if fpos is not None else None), attn=None,
                                             pos=s0 + jx, winpos=[s0 + q for q in win], fpos=fpos, winown=[s0 + q for q in win]))
                continue
            hw = hrel[win] if win else hrel[jx:jx + 1]
            pi, g = self.mod(hrel[jx].to(v.device), hw.to(v.device), len(B), kh, prior=(pr if win else None))
            a_ = self.mod.last_a.cpu()
            if NBk is not None and 1 <= kh <= len(B) and float(g) > .5:            # S618: end side = k-th strongest block
                import r3_native
                order = torch.argsort(-r3_native.ranks(NBk, nblocks, s0 + jx))
                onehot = torch.zeros(len(B), device=v.device); onehot[int(order[kh - 1])] = 1.0
                pi = g * onehot + (1 - g) * self.mod.last_pf
            q = torch.zeros(C, device=v.device).index_add(0, (codes - 1).clamp(0, C - 1).to(v.device), pi)
            bi = int(pi.argmax()); conf = float(pi[bi]); t_ = int(codes[bi])
            isa = None
            if self.anchor_own:                                                    # the model's anchor test (S619f(a))
                isa = ((lambda q_: any(float(dA[a][q_]) <= fr["thr"][a] for a in dA)) if self.anchor_own == "abs" else
                       (lambda q_: any(float(dA[a][q_]) / medA[a] <= fr["rthr"][a] for a in dA)))
            aspan = (win_own if (self.anchor_span == "own" and win_own is not None) else win)
            if self.anchor_own and float(g) > .5:                                  # S619f: some token of this window is an anchor
                if not any(isa(qq) for qq in aspan):                               # S619g: ... of the owned span
                    g = torch.zeros_like(g); pi = self.mod.last_pf
                    bi = int(pi.argmax()); conf = float(pi[bi]); t_ = int(codes[bi])
            if self.anchor_gate and float(g) > .5:                                 # S619e: the token the gate read must be a frame anchor
                qa = win[int(a_.argmax())] if (win and len(a_) == len(win)) else None
                if qa is None or not any(float((Hb[qa] - fr["mu"][an]).norm()) <= fr["thr"][an] for an in fr["mu"]):
                    g = torch.zeros_like(g); pi = self.mod.last_pf
                    bi = int(pi.argmax()); conf = float(pi[bi]); t_ = int(codes[bi])
            act = conf >= self.tau and 1 <= t_ <= C and t_ != kh and (not self.fwd_safe or float(g) > .5)
            if act:
                tw = t_
                if getattr(self, "ablate", None) == "rand_target":                  # S641: same admission, random wrong target
                    import random as _r
                    alts = [x for x in range(1, min(C, maxcode) + 1) if x not in (kh, t_)]
                    if alts:
                        tw = _r.Random(hash((tuple(ids), jx)) & 0xffffffff).choice(alts)
                writes.append((s0 + jx, Mbw[tw - 1] - Mbw[kh - 1]))
            fpos = None
            if self.anchor_span == "own" and float(g) > .5 and conf >= self.tau and 1 <= t_ <= min(C, maxcode):
                best = None                                                       # S619g: the strongest anchor of the owned span
                for qq in aspan:
                    bq = float(self.last_C[s0 + jx][0][qq - jx])
                    if isa(qq) and (best is None or bq > best[2]):
                        a_n = min(dA, key=lambda a: float(dA[a][qq]) / (medA[a] if self.anchor_own == "rel" else 1.0))
                        best = (qq, frw["mu_start"] - frw["mu"][a_n], bq)
                if best is not None:
                    fpos = s0 + best[0]; writes.append((fpos, best[1])); self.stats["frame"] += 1
            elif float(g) > .5 and conf >= self.tau and 1 <= t_ <= min(C, maxcode) and bv is not None and win:
                best = None                                                       # S609c: the anchor the model binds most
                same_test = isa is not None and self.bind is not None and self.bind.get("mode") == "hard"   # S624d: decision's test
                for qq, bq in zip(win, bv):
                    for a in fr["mu"]:
                        hit = (isa(qq) and min(dA, key=lambda a_: float(dA[a_][qq]) / (medA[a_] if self.anchor_own == "rel" else 1.0)) == a) if same_test \
                            else float((Hb[qq] - fr["mu"][a]).norm()) <= fr["thr"][a]
                        if hit and (best is None or bq > best[2]):
                            best = (qq, frw["mu_start"] - frw["mu"][a], bq)
                if best is not None:
                    fpos = s0 + best[0]; writes.append((fpos, best[1])); self.stats["frame"] += 1
                elif len(a_) == len(win):
                    fpos = s0 + win[int(a_.argmax())]; writes.append((fpos, self.fgen_w)); self.stats["frame_attn"] += 1
            elif float(g) > .5 and conf >= self.tau and 1 <= t_ <= min(C, maxcode) and self.bind is not None and win:
                qq = win[int(a_.argmax())]                                        # S609: the frame is written where the gate read it
                best = None
                for a in fr["mu"]:
                    if float((Hb[qq] - fr["mu"][a]).norm()) <= fr["thr"][a]:
                        best = (qq, frw["mu_start"] - frw["mu"][a])
                fpos = s0 + qq
                writes.append((fpos, best[1] if best is not None else self.fgen_w))
                self.stats["frame" if best is not None else "frame_attn"] += 1
            elif float(g) > .5 and conf >= self.tau and 1 <= t_ <= min(C, maxcode):
                best = None
                for qq in win:
                    for a in fr["mu"]:
                        dd = float((Hb[qq] - fr["mu"][a]).norm())
                        if dd <= fr["thr"][a] and (best is None or abs(qq - jx) < abs(best[0] - jx)):
                            best = (qq, frw["mu_start"] - frw["mu"][a])
                if best is not None:
                    fpos = s0 + best[0]; writes.append((fpos, best[1])); self.stats["frame"] += 1
                elif win and len(a_) == len(win):
                    fpos = s0 + win[int(a_.argmax())]; writes.append((fpos, self.fgen_w)); self.stats["frame_attn"] += 1
            plan.append((s0 + jx, kh, t_ if act else kh))
            self.last_detail.append(dict(tok=v.tok.decode([ids[s0 + jx]]), kh=kh, t=t_, N=len(B), act=bool(act),
                                         map=("scan", round(float(g), 3)), wmax=conf,
                                         frame=(v.tok.decode([ids[fpos]]) if fpos is not None else None),
                                         win=v.tok.decode([ids[q] for q in L]) + " [" + v.tok.decode([ids[s0 + jx]]) + "] " +
                                         v.tok.decode([ids[q] for q in Rr]),
                                         attn=(v.tok.decode([ids[s0 + win[int(a_.argmax())]]]) if win and len(a_) == len(win) else None),
                                         pos=s0 + jx, winpos=[s0 + q for q in win], fpos=fpos,
                                         winown=[s0 + q for q in (win_own if win_own is not None else win)]))
        self.stats["sources"] += len(srcs); self.stats["rebound"] += sum(1 for p_, k_, t__ in plan if t__ != k_)
        ab = getattr(self, "ablate", None)                                          # S641: remove one write type
        if ab in ("no_frame", "no_ord"):
            ordpos = {s0 + jx for jx in srcs}
            writes = [w for w in writes if (w[0] in ordpos) == (ab == "no_frame")]
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
        """greedy generation; the writes are applied in the prefill (S609 reasoning suites)."""
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

    def both(self, text, pics, cand):
        self.set(False); a = int(self.logits(text, pics)[cand].argmax())
        self.set(True); b = int(self.logits(text, pics)[cand].argmax())
        self.set(False)
        return a, b
