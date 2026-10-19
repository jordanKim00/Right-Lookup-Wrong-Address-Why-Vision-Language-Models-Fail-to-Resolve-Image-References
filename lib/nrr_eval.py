"""Native Reference Rebinding at evaluation time (S588b index variant / S588d circuit variant), with the Repair interface
used by s582_evalrepair.py and s583_signatures.py: set(on), logits(text, pics), both(text, pics, cand).

Per input with images: pass 1 reads the question tokens' states at the source layer and the image blocks' structural
features (the image prefix is complete before the question, so a cached prefix would give the same states in one
pass); sources = question tokens whose state lies near the ordinal prototype chart; pass 2 writes the rebinding at the
source layer. Inputs without images, or without a detected source, run unchanged.
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/vlm/exp")
from clause import clause_len
from s588b_nrr import RelationReader, MicroCircuit
from s588c_reader import probs as reader_probs

C = 10


class NRRRepair:
    def __init__(self, v, ckpt):
        self.v, self.on = v, False
        st = torch.load(ckpt, weights_only=False)
        R = torch.load(f"/vlm/results/s588c_reader_{st['reader']}.pt", weights_only=False)
        self.RP, self.Rlayers, self.Rsites, self.SITES = R["P"], R["layers"], R["sites"], R["SITES"]
        self.M, self.Pb, self.theta = st["M"], st["Pb"], st["theta"]
        self.zmu, self.zsd = st.get("zmu"), st.get("zsd")
        self.Lb, self.w, self.sigma, self.variant = st["Lb"], st["w"], st["sigma"], st.get("variant", "index")
        cfg = v.model.config
        d = cfg.get_text_config().hidden_size if hasattr(cfg, "get_text_config") else cfg.text_config.hidden_size
        self.dev = v.device
        fdim = len(self.Rlayers) * len(self.Rsites) * d
        self.mod = (RelationReader(d) if self.variant == "index" else MicroCircuit(d, fdim, C)).to(self.dev).float()
        self.mod.load_state_dict(st["rel"])
        self.mod.eval()
        self.MP = self.M @ self.Pb
        self.Mdev = self.M.to(self.dev)
        self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.ST = {}
        self.stats = dict(inputs=0, sources=0)
        for l in self.Rlayers:
            v.layers[l].register_forward_hook(self._reader_hook(l))
        v.layers[self.Lb].register_forward_hook(self._bind_hook)

    def set(self, on):
        self.on = on

    def _reader_hook(self, l):
        def h(m, i, o):
            if self.ST.get("cap_blocks"):
                t = o[0] if isinstance(o, tuple) else o
                f = []
                for b in self.ST["B"]:
                    allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                           t[0, int(b[-1])].float()])
                    f.append(torch.cat([allsite[self.SITES.index(s_)] for s_ in self.Rsites]))
                self.ST["bf"][l] = torch.stack(f).cpu()
            return o
        return h

    def _bind_hook(self, m, i, o):
        ST = self.ST
        if ST.get("cap_src"):
            t = o[0] if isinstance(o, tuple) else o
            ST["hsrc"] = t[0, ST["cand_pos"]].detach().float().cpu()
            return o
        if not ST.get("src"):
            return o
        t = o[0] if isinstance(o, tuple) else o
        tt = torch.arange(1, C + 1, device=t.device).float()
        blk, rows = ST["blk"], []
        for p, kh, zc in ST["src"]:
            hw = t[0, p + 1:min(p + 1 + self.w, ST["turn"])].float()
            if self.variant == "index":
                c, g, _ = self.mod(t[0, p].float(), hw)
                ctr = c[0] + c[1] * tt + c[2] * kh
                K = torch.softmax(-(tt[None, :] - ctr[:, None]) ** 2 / (2 * self.sigma ** 2), 1)
                alpha = (blk["e"][:, None] * K).sum(0)
            else:
                pi, g, _ = self.mod(t[0, p].float(), hw, zc.to(t.device), blk["X"])
                alpha = pi @ blk["PR"]
            beta = torch.exp(-(tt - kh) ** 2 / (2 * self.sigma ** 2)); beta = beta / beta.sum()
            rows.append((p, g * ((alpha - beta) @ self.Mdev)))
        t2 = t.clone()
        for p, dlt in rows:
            t2[0, p] = (t[0, p].float() + dlt).to(t.dtype)
        return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2

    @torch.no_grad()
    def logits(self, text, pics):
        v = self.v
        inp = v.encode(text, pics)
        self.ST = {}
        if self.on and pics:
            ids = inp["input_ids"][0].tolist()
            B = [torch.tensor(b, device=self.dev) for b in v.image_blocks(inp["input_ids"], len(pics))]
            turn = max(i for i, x in enumerate(ids) if x == self.im_start)
            s0 = int(B[-1][-1]) + 2
            self.ST = dict(cap_src=True, cand_pos=torch.arange(s0, turn, device=self.dev), cap_blocks=True, B=B, bf={})
            v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
            hsrc, bf = self.ST["hsrc"], self.ST["bf"]
            dd = torch.cdist(hsrc @ self.Pb, self.MP)
            dist, kh = dd.min(1)
            X = torch.cat([bf[l] for l in self.Rlayers], 1)
            PR = reader_probs(self.RP, X)
            E = PR @ torch.arange(1, C + 1).float()
            blk = dict(e=PR[int(E.argmax())].to(self.dev), X=X.to(self.dev), PR=PR.to(self.dev))
            zall = (hsrc @ self.Pb - self.zmu) / self.zsd if self.zmu is not None else hsrc @ self.Pb
            src = [(s0 + jx, float(kh[jx] + 1), zall[jx]) for jx in (dist <= self.theta).nonzero().flatten().tolist()]
            self.stats["inputs"] += 1; self.stats["sources"] += len(src)
            self.ST = dict(src=src, blk=blk, turn=turn)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        self.ST = {}
        return out.logits[0, -1].float()

    def both(self, text, pics, cand):
        self.set(False); a = int(self.logits(text, pics)[cand].argmax())
        self.set(True); b = int(self.logits(text, pics)[cand].argmax())
        self.set(False)
        return a, b


class NRRMLRepair:
    """S588e modules (variant map / retarget, trained by the marginal likelihood of candidate rebindings): hard rebinding of
    every detected source to argmax_t q(t). Same interface as Repair."""
    def __init__(self, v, ckpt, tau=None):
        from s588b_nrr import MapSelector, Retargeter, MAPS
        self.MAPS = MAPS
        self.v, self.on = v, False
        st = torch.load(ckpt, weights_only=False)
        R = torch.load(f"/vlm/results/s588c_reader_{st['reader']}.pt", weights_only=False)
        self.RP, self.Rlayers, self.Rsites, self.SITES = R["P"], R["layers"], R["sites"], R["SITES"]
        self.M, self.Pb, self.theta, self.zmu, self.zsd = st["M"], st["Pb"], st["theta"], st["zmu"], st["zsd"]
        self.Lb, self.w, self.variant = st["Lb"], st["w"], st["variant"]
        self.tau = tau if tau is not None else st.get("tau", 0.9)                  # act only when the chosen map is this confident
        self.gate = st.get("ood_gate")                                              # S588o Mahalanobis gate (None = off)
        self.gate_mode = st.get("ood_mode", "mahal")
        self.Lr = st.get("Lr", self.Lb)                                            # S588g: reads at L_r, write at L_b
        self.Mb = st.get("Mb", self.M)
        self.frame = st.get("frame")                                               # S588v: reflect the frame (anchor) too (None = off)
        self.clause = st.get("clause", False)                                      # S588x: clause-bounded relation window
        self.last_frame = []
        self.detect_at = st.get("detect_at", "read")
        if self.detect_at == "write":
            self.Pbb, self.theta_b = st["Pbb"], st["theta_b"]
            self.MPb = self.Mb @ self.Pbb
        self.det_space = st.get("det_space", "chart")                              # S590g: "full" = full-state nearest prototype
        self.theta_full = st.get("theta_full")
        cfg = v.model.config
        d = cfg.get_text_config().hidden_size if hasattr(cfg, "get_text_config") else cfg.text_config.hidden_size
        self.dev = v.device
        from s588b_nrr import Scanner
        self.Lrel = st.get("Lrel", [self.Lr])
        dr_ = d * len(self.Lrel)
        self.mod = {"map": lambda: MapSelector(dr_), "retarget": lambda: Retargeter(dr_, C), "scan": lambda: Scanner(dr_, mix=st.get("scan_mix", "score"))}[self.variant]().to(self.dev).float()
        self.mod.load_state_dict(st["mod"]); self.mod.eval()
        self.MP = self.M @ self.Pb
        self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.ST = {}
        self.stats = dict(inputs=0, sources=0, rebound=0)
        for l in self.Rlayers:
            v.layers[l].register_forward_hook(self._reader_hook(l))
        v.layers[self.Lb].register_forward_hook(self._bind_hook)
        v.layers[self.Lr].register_forward_hook(self._read_hook)
        for l in self.Lrel:
            v.layers[l].register_forward_hook(self._rel_hook(l))

    def set(self, on):
        self.on = on

    def _rel_hook(self, l):
        def h(m, i, o):
            if self.ST.get("cap_src"):
                t = o[0] if isinstance(o, tuple) else o
                self.ST.setdefault("rel", {})[l] = t[0, self.ST["cand_pos"]].detach().float().cpu()
            return o
        return h

    def _reader_hook(self, l):
        def h(m, i, o):
            if self.ST.get("cap_blocks"):
                t = o[0] if isinstance(o, tuple) else o
                f = []
                for b in self.ST["B"]:
                    allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                           t[0, int(b[-1])].float()])
                    f.append(torch.cat([allsite[self.SITES.index(s_)] for s_ in self.Rsites]))
                self.ST["bf"][l] = torch.stack(f).cpu()
            return o
        return h

    def _read_hook(self, m, i, o):
        if self.ST.get("cap_src"):
            t = o[0] if isinstance(o, tuple) else o
            self.ST["hsrc"] = t[0, self.ST["cand_pos"]].detach().float().cpu()
        return o

    def _bind_hook(self, m, i, o):
        if self.ST.get("cap_src"):
            t = o[0] if isinstance(o, tuple) else o
            self.ST["hsrc_b"] = t[0, self.ST["cand_pos"]].detach().float().cpu()
            return o
        if not self.ST.get("writes"):
            return o
        t = o[0] if isinstance(o, tuple) else o
        t2 = t.clone()
        for p, dlt in self.ST["writes"]:
            t2[0, p] = (t[0, p].float() + dlt.to(t.device)).to(t.dtype)
        return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2

    @torch.no_grad()
    def plan(self, text, pics):
        """Detected sources and their rebinding targets: list of (position, k_hat, t)."""
        v = self.v
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b, device=self.dev) for b in v.image_blocks(inp["input_ids"], len(pics))]
        turn = max(i for i, x in enumerate(ids) if x == self.im_start)
        s0 = int(B[-1][-1]) + 2
        self.ST = dict(cap_src=True, cand_pos=torch.arange(s0, turn, device=self.dev), cap_blocks=True, B=B, bf={})
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        hsrc, bf, hsrc_b = self.ST["hsrc"], self.ST["bf"], self.ST.get("hsrc_b")
        hrel = torch.cat([self.ST["rel"][l] for l in self.Lrel], 1)
        self.ST = {}
        if self.detect_at == "write" and self.det_space == "full":
            dist, kh0 = torch.cdist(hsrc_b, self.Mb).min(1)
            thr = self.theta_full
        elif self.detect_at == "write":
            dist, kh0 = torch.cdist(hsrc_b @ self.Pbb, self.MPb).min(1)
            thr = self.theta_b
        else:
            dist, kh0 = torch.cdist(hsrc @ self.Pb, self.MP).min(1)
            thr = self.theta
        X = torch.cat([bf[l] for l in self.Rlayers], 1)
        PR = reader_probs(self.RP, X)
        E = PR @ torch.arange(1, C + 1).float()
        e = PR[int(E.argmax())]; ehat = int(e.argmax()) + 1
        self.last_ehat = ehat; self.last_codes = (PR.argmax(1) + 1).tolist()
        z = (hsrc @ self.Pb - self.zmu) / self.zsd
        N = len(B); out = []
        self.last_detail = []; self.last_frame = []
        src_list = (dist <= thr).nonzero().flatten().tolist()
        for si, jx in enumerate(src_list):
            kh = int(kh0[jx]) + 1
            wl = clause_len(v.tok, ids, s0 + jx, self.w) if self.clause else self.w  # S588x: window ends at the clause
            hs = hrel[jx].to(self.dev); hw = hrel[jx + 1:jx + 1 + wl].to(self.dev)
            if len(hw) == 0:
                hw = hs[None]
            # targets are CODES present in the scene (not 1..N physical: a counter shift moves codes to 3..N+2 -- §619, s592b)
            present = torch.zeros(C, device=self.dev)
            codes = (PR.argmax(1) + 1).long()
            present[(codes - 1).clamp(0, C - 1).to(self.dev)] = 1.0
            if self.variant == "map":
                wm, _ = self.mod(hs, hw)
                cm = torch.tensor(self.MAPS).float()
                tm = (cm[:, 0] + cm[:, 1] * ehat + cm[:, 2] * kh).round().long()
                ok = (tm >= 1) & (tm <= C)
                q = torch.zeros(C, device=self.dev).index_add(0, (tm[ok] - 1).to(self.dev), wm[ok.to(wm.device)])
            elif self.variant == "scan":
                pi, gsc = self.mod(hs, hw, N, kh)                                  # gsc = relation gate: count from the end
                q = torch.zeros(C, device=self.dev).index_add(0, (codes - 1).clamp(0, C - 1).to(self.dev), pi)
            else:
                pi = self.mod(hs, hw, z[jx].to(self.dev), e.to(self.dev), PR.to(self.dev), float(kh))
                q = torch.zeros(C, device=self.dev).index_add(0, (codes - 1).clamp(0, C - 1).to(self.dev), pi)
            # decision (S590 fix): act only on a confident, non-identity choice whose target lies within the read structure
            maxcode = int(codes.max())
            if self.variant == "map":
                mi = int(wm.argmax()); conf = float(wm[mi])
                t_ = int(round(self.MAPS[mi][0] + self.MAPS[mi][1] * ehat + self.MAPS[mi][2] * kh))
                act = (self.MAPS[mi] != (0, 0, 1)) and conf >= self.tau and 1 <= t_ <= min(C, maxcode) and t_ != kh
                if act and self.gate is not None:                                   # S588o / S588q gates
                    g_ = self.gate.get(mi)
                    r_ = self.mod.last_r.cpu()
                    if g_ is None:
                        act = False
                    elif self.gate_mode == "qda":                                   # log-likelihood ratio vs identity
                        gi = self.gate.get(self.MAPS.index((0, 0, 1)))
                        nll = lambda g: 0.5 * float(((r_ - g["mu"]) @ g["P"] * (r_ - g["mu"])).sum()) + 0.5 * g["logdet"]
                        act = gi is None or (nll(gi) - nll(g_)) > 0
                    else:
                        dd_ = float(((r_ - g_["mu"]) @ g_["P"] * (r_ - g_["mu"])).sum())
                        act = dd_ <= g_["thr"]
                    if not act:
                        self.stats["ood_blocked"] = self.stats.get("ood_blocked", 0) + 1
            else:
                bi = int(pi.argmax()); conf = float(pi[bi]); t_ = int(codes[bi])
                act = conf >= self.tau and 1 <= t_ <= C and t_ != kh
            out.append((s0 + jx, kh, t_ if act else kh))
            fpos = None
            frame_ok = False
            if self.frame is not None and hsrc_b is not None:
                # S588v: an end-relative choice made with confidence reflects the frame as well -- (k, end) -> (E+1-k, start);
                # map: the chosen map reads the end (c_e = 1); scanner (S597): its relation gate chose the count from the end;
                # retargeter: it rebinds. The anchor is the first end-frame anchor token after the source, before the next
                # source, inside the source's clause window.
                if self.variant == "map":
                    mi = int(wm.argmax())
                    frame_ok = self.MAPS[mi][1] == 1 and float(wm[mi]) >= self.tau and 1 <= t_ <= min(C, maxcode)
                    if frame_ok and self.gate is not None:
                        frame_ok = act
                elif self.variant == "scan":
                    frame_ok = float(gsc) > 0.5 and conf >= self.tau and 1 <= t_ <= min(C, maxcode)
                else:
                    frame_ok = act
                if frame_ok:
                    nxt = src_list[si + 1] if si + 1 < len(src_list) else len(hsrc_b)
                    found = False
                    for q in range(jx + 1, min(jx + 1 + wl, nxt, len(hsrc_b))):
                        thr_f = self.frame["thr_margin"] if self.frame.get("use_margin") else self.frame["thr"]   # S599a
                        hit = [a for a, mu_a in self.frame["mu"].items()
                               if float((hsrc_b[q] - mu_a).norm()) <= thr_f[a]]
                        if hit:
                            fpos = s0 + q
                            self.last_frame.append((fpos, self.frame["mu_start"] - self.frame["mu"][hit[0]]))
                            self.stats["frame"] = self.stats.get("frame", 0) + 1
                            found = True
                            break
                    # S600b: no calibrated anchor in the clause -> the token the relation head attends to most (the expression's own
                    # direction carrier) gets the generic frame direction learned from the calibrated anchors
                    if not found and self.frame.get("attn_fallback") and self.variant == "scan":
                        a_ = getattr(self.mod, "last_a", None)
                        if a_ is not None and len(a_) == len(hw) and wl > 0:
                            qa = jx + 1 + int(a_.argmax())
                            if qa < min(nxt, len(hsrc_b)):
                                fpos = s0 + qa
                                fgen = self.frame["mu_start"] - 0.5 * (self.frame["mu"]["end"] + self.frame["mu"]["last"])
                                self.last_frame.append((fpos, fgen))
                                self.stats["frame_attn"] = self.stats.get("frame_attn", 0) + 1
            if self.variant == "map":
                self.last_detail.append(dict(tok=v.tok.decode([ids[s0 + jx]]), kh=kh, t=t_, ehat=ehat, N=N, act=bool(act),
                                             map=tuple(self.MAPS[int(wm.argmax())]), wmax=float(wm.max()),
                                             frame=(v.tok.decode([ids[fpos]]) if fpos is not None else None)))
            else:
                a_ = getattr(self.mod, "last_a", None) if self.variant == "scan" else None
                self.last_detail.append(dict(tok=v.tok.decode([ids[s0 + jx]]), kh=kh, t=t_, ehat=ehat, N=N, act=bool(act),
                                             map=("scan", round(float(gsc), 3)) if self.variant == "scan" else ("retarget",),
                                             wmax=conf, frame=(v.tok.decode([ids[fpos]]) if fpos is not None else None),
                                             pos=s0 + jx, win=[s0 + jx + 1 + i for i in range(len(hw))],
                                             attn=(a_.cpu().tolist() if a_ is not None and len(a_) == len(hw) else None)))
        return inp, out

    @torch.no_grad()
    def logits(self, text, pics):
        v = self.v
        writes = []
        if self.on and pics:
            inp, plan = self.plan(text, pics)
            self.stats["inputs"] += 1; self.stats["sources"] += len(plan)
            for p, kh, t in plan:
                if t != kh:
                    writes.append((p, (self.Mb[t - 1] - self.Mb[kh - 1])))
            self.stats["rebound"] += len(writes)
            if self.frame is not None and self.frame.get("frame_only"):            # check only: frame writes without the index
                writes = []
            writes += self.last_frame                                               # S588v frame writes (same layer L_b)
        else:
            inp = v.encode(text, pics)
        self.ST = dict(writes=writes)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        self.ST = {}
        return out.logits[0, -1].float()

    def both(self, text, pics, cand):
        self.set(False); a = int(self.logits(text, pics)[cand].argmax())
        self.set(True); b = int(self.logits(text, pics)[cand].argmax())
        self.set(False)
        return a, b
