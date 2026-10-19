"""R³ builder on R³-Ref (protocols/r3_ref_dataset_v1.md §6, §9, A1): calibration, interventional discovery with the
deployed actuator, caching of every realization with two-sided clause windows, scanner fit (binary MML + paraphrase
consistency), checkpoint for r3_repair.R3Repair.

Stages (cached on disk, resumable):
  calib   ordinal prototypes at L_r and L_b (forward texts, PROC images), detector (chart / full-state rule, all ordinal
          tokens), frame anchors at L_b (start / end / last / final; margin-midpoint thresholds)
  discover  per program, canonical (Tier A) realization: p_t = P(gold | do(source := t) + frame write at the
          program's frame carrier), Z = {t : p_t >= .5}; context / frame-phrase ordinals -> identity; other options ->
          referent (S588u)
  cache   every realization: per detected source, relation states of the source and of its two-sided clause window,
          pointer, end read, codes; matched to the program's labels by role
  fit     scanner (warm start, probability mixing) on the cache, MML + lambda * symmetric KL between realizations of one
          program-source, group-balanced; model selection on DEV
"""
from __future__ import annotations

import argparse, collections, json, math, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from s588b_nrr import Scanner
from s588c_reader import probs as reader_probs
from r3ref import ORD, load_image
import r3_bind

C = 10
OUT = "/vlm/results/r3"
FRAME_WORDS = {"END": ["end", "last", "final"], "START": ["start", "beginning", "first"]}


def is_boundary(s):
    st = s.strip()
    return "\n" in s or (st != "" and st[0] in ".?!")


LEFT_SCOPE = None                                                                  # S606 V1: "comma" (set by --left_scope)
BIND = False                                                                       # S609: binding prior (set by --bind)
BIND_MODE = "prior"                                                                # S609 "prior" / S609c "hard" (--bind_mode)
BIND_TH = "mid"                                                                    # S609c "mid" / S609d "noncue95" (--bind_th)
DSEG_TH = None                                                                     # S619b "abs" / S619c "rel" (--dseg_th)
FIT_EXTRA = {}


def clause_window(tok, ids, pos, lo, hi, w=8, left_scope="default"):
    """Positions of the source's clause: left back to the previous boundary, right to the next one (at most w each).
    left_scope "comma" (S606 V1): the left side keeps only a fronted phrase -- the tokens before the comma nearest the
    source -- and is empty when there is no comma."""
    left = []
    q = pos - 1
    while q >= lo and len(left) < w and not is_boundary(tok.decode([ids[q]])):
        left.append(q); q -= 1
    if (LEFT_SCOPE if left_scope == "default" else left_scope) == "comma":
        cut = next((i for i, q_ in enumerate(left) if "," in tok.decode([ids[q_]])), None)
        left = left[cut + 1:] if cut is not None else []
    right = []
    q = pos + 1
    while q < hi and len(right) < w and not is_boundary(tok.decode([ids[q]])):
        right.append(q); q += 1
    return left[::-1], right


class Model:
    def __init__(self, v, Lb, Lr, Lrel, reader):
        self.v, self.Lb, self.Lr, self.Lrel = v, Lb, Lr, list(Lrel)
        R = torch.load(f"/vlm/results/s588c_reader_{reader}.pt", weights_only=False)
        self.RP, self.Rlayers, self.Rsites, self.SITES = R["P"], R["layers"], R["sites"], R["SITES"]
        self.THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
        self.im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
        self.ST = {}
        self.bind, self.att_layers = None, None                                    # S609: binding prior from the model's attention
        self.dseg_layers = None                                                    # S619: DeltaNet ownership
        layers = sorted(set([Lb, Lr] + self.Lrel + list(self.Rlayers)))
        for l in layers:
            v.layers[l].register_forward_hook(self._hook(l))

    def _hook(self, l):
        def h(m, i, o):
            ST = self.ST
            if not ST:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST.get("cap"):
                if l in (self.Lb, self.Lr) or l in self.Lrel:
                    ST.setdefault("H", {})[l] = t[0, ST["span"]].detach().float().cpu()
                if l in self.Rlayers and ST.get("B"):
                    f = []
                    for b in ST["B"]:
                        allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                               t[0, int(b[-1])].float()])
                        f.append(torch.cat([allsite[self.SITES.index(s_)] for s_ in self.Rsites]))
                    ST.setdefault("bf", {})[l] = torch.stack(f).cpu()
                return o
            if l == self.Lb and ST.get("writes"):
                t2 = t.clone()
                for p, dlt in ST["writes"]:
                    t2[0, p] = (t[0, p].float() + dlt.to(t.device)).to(t.dtype)
                return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
            return o
        return h

    def encode(self, body, pics):
        v = self.v
        return v.encode(v.chat([("user", f"{v.IMG*len(pics)}\n{body}")], prefix_assistant=self.THINK), pics)

    @torch.no_grad()
    def capture(self, body, pics):
        v = self.v
        inp = self.encode(body, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b, device=v.device) for b in v.image_blocks(inp["input_ids"], len(pics))]
        turn = max(i for i, x in enumerate(ids) if x == self.im_start)
        s0 = int(B[-1][-1]) + 2
        self.ST = dict(cap=True, span=torch.arange(s0, turn, device=v.device), B=B)
        if self.att_layers is not None:
            r3_bind.start(torch.arange(s0, turn), None if self.att_layers == "all" else self.att_layers)
        if self.dseg_layers is not None:                                           # S619: the model's DeltaNet memory
            import r3_dseg
            r3_dseg.start(None if self.dseg_layers == "all" else self.dseg_layers)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        A = r3_bind.stop() if self.att_layers is not None else None
        if self.dseg_layers is not None:
            A = r3_dseg.stop()
        H, bf = self.ST["H"], self.ST.get("bf", {})
        self.ST = {}
        return dict(inp=inp, ids=ids, s0=s0, turn=turn, N=len(B), H=H, bf=bf, A=A, logits=out.logits[0, -1].float())

    @torch.no_grad()
    def run(self, inp, writes):
        self.ST = dict(writes=writes)
        out = self.v.model(**inp, use_cache=False, return_dict=True, **self.v.forward_extra)
        self.ST = {}
        return out.logits[0, -1].float()


def cand_ids(v, ans):
    if ans["type"] == "word":
        return [v.tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in ans["cands"]]
    return [v.tok(c, add_special_tokens=False)["input_ids"][0] for c in ans["cands"]]


def calibrate(M, nproto=16, seed=60400):
    """Ordinal prototypes (L_r, L_b), detector (chart / full-state), frame anchors at L_b (start/end/last/final)."""
    v = M.v
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    ord_ids = {tok1(" " + o): i + 1 for i, o in enumerate(ORD[:C])}
    rng = random.Random(seed)
    protos = {i: [] for i in range(1, C + 1)}; protos_b = {i: [] for i in range(1, C + 1)}
    all_b, other_b, other_r = [], [], []
    anchors = collections.defaultdict(list)
    phrases = [("", None), (" from the start", "start"), (" from the beginning", "beginning"), (" from the end", "end"),
               (" from the last", "last")]
    for i in range(1, C + 1):
        for phr, an in phrases:
            for _ in range(nproto if an is None else max(4, nproto // 4)):
                N = rng.randint(max(3, i), 10)
                pics = [load_image(f"proc:{rng.randint(0, 59_999)}") for _ in range(N)]
                fmt = rng.choice(["q", "s", "o"])
                if fmt == "o":
                    ks = rng.sample(range(1, N + 1), min(4, N))
                    if i not in ks:
                        ks[0] = i
                    body = "Which image shows a circle?\nSelect from the following choices.\n" + "\n".join(
                        f"{chr(65 + a)}: The {ORD[k-1]} image{phr}" for a, k in enumerate(ks)) + "\nAnswer with the option's letter from the given choices directly."
                elif fmt == "q":
                    body = f"What is in the {ORD[i-1]} image{phr}? Answer with only one word from: circle, star, ring."
                else:
                    body = f"Look at the {ORD[i-1]} image{phr}. What is in it? Answer with only one word from: circle, star, ring."
                c = M.capture(body, pics)
                span = c["ids"][c["s0"]:c["turn"]]
                for jx, t in enumerate(span):
                    if t in ord_ids:
                        all_b.append(c["H"][M.Lb][jx])
                        if ord_ids[t] == i and an is None:
                            protos[i].append(c["H"][M.Lr][jx]); protos_b[i].append(c["H"][M.Lb][jx])
                    else:
                        other_b.append(c["H"][M.Lb][jx]); other_r.append(c["H"][M.Lr][jx])
                    if an is not None and v.tok.decode([t]).strip() == an and jx >= 2 and v.tok.decode([span[jx - 2]]).strip() == "from":
                        anchors[an].append(c["H"][M.Lb][jx])
    Mr = torch.stack([torch.stack(protos[i]).mean(0) for i in range(1, C + 1)])
    Mb = torch.stack([torch.stack(protos_b[i]).mean(0) for i in range(1, C + 1)])
    Pbb = torch.linalg.svd(Mb - Mb.mean(0), full_matrices=False)[2][:C - 1].T
    AB, OB = torch.stack(all_b), torch.stack(other_b)
    th_chart = float(torch.quantile(torch.cdist(AB @ Pbb, Mb @ Pbb).min(1)[0], .99))
    fp_chart = float((torch.cdist(OB @ Pbb, Mb @ Pbb).min(1)[0] <= th_chart).float().mean())
    th_full = float(torch.quantile(torch.cdist(AB, Mb).min(1)[0], .99))
    fp_full = float((torch.cdist(OB, Mb).min(1)[0] <= th_full).float().mean())
    det_space = "chart" if fp_chart <= .01 else "full"
    mu = {a: torch.stack(x).mean(0) for a, x in anchors.items()}
    thr = {}
    for a in ("end", "last"):
        own = (torch.stack(anchors[a]) - mu[a]).norm(dim=1); do = (OB - mu[a]).norm(dim=1)
        omax, dmin = float(own.max()), float(do.min())
        thr[a] = (omax + dmin) / 2 if dmin > omax else float(torch.quantile(own, .99))
    cal = dict(Mr=Mr, Mb=Mb, Pbb=Pbb, th_chart=th_chart, th_full=th_full, det_space=det_space,
               frame=dict(mu_start=mu["start"], mu={a: mu[a] for a in ("end", "last")}, thr=thr))
    print(f"[calib] 탐지 차트 {th_chart:.2f}/오탐 {fp_chart:.4f} · 전체 {th_full:.2f}/오탐 {fp_full:.4f} → {det_space} · 앵커 {dict((a, len(x)) for a, x in anchors.items())}"
          f" · 기준 문턱 { {a: round(t_, 2) for a, t_ in thr.items()} }", flush=True)
    return cal


def bind_calibrate(M, n=24, seed=60900):
    """S609: binding heads from canonical calibration sentences only (S607 / S607b rules) on A3 dev scenes.
    post-nominal: "What is in the k-th image from the end/start?" -> attention cue -> ordinal;
    fronted: "From the end/start, what is in the k-th image?" -> attention head noun (token after the ordinal) -> cue."""
    v = M.v
    r3_bind.install(v)
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(seed)
    rec_a, rec_f, caps = [], [], []
    M.att_layers = "all"
    for _ in range(n):
        N = rng.randint(3, 6); k = rng.randint(1, N)
        pics = [load_image(f"proc:{p}") for p in rng.sample(pids, N)]
        for side in ("end", "start"):
            for kind, body in (("after", f"What is in the {ORD[k-1]} image from the {side}? Answer with only one word from: circle, star, ring."),
                               ("front", f"From the {side}, what is in the {ORD[k-1]} image? Answer with only one word from: circle, star, ring.")):
                c = M.capture(body, pics)
                toks = [v.tok.decode([t]) for t in c["ids"][c["s0"]:c["turn"]]]
                s_ = toks.index(f" {ORD[k-1]}"); ci = [i for i, t in enumerate(toks) if t == f" {side}"]
                if kind == "after":
                    c_ = [i for i in ci if i > s_][0]
                    rec_a.append({l: c["A"][l][:, c_, s_] for l in c["A"]})
                else:
                    c_ = [i for i in ci if i < s_][0]
                    rec_f.append({l: c["A"][l][:, s_ + 1, c_] for l in c["A"]})
                qm = next(i for i in range(s_, len(toks)) if "?" in toks[i])
                caps.append(dict(kind=kind, A=c["A"], s=s_, c=c_, qm=qm, n=len(toks)))
    M.att_layers = None
    ha, ca = r3_bind.select_heads(rec_a); hf, cf = r3_bind.select_heads(rec_f)
    bind = dict(heads_after=ha, heads_front=hf, c_after=ca, c_front=cf, capL=32, capR=32, mode=BIND_MODE)
    # S609c thresholds: midpoint between the cue bindings (5th pct) and non-cue bindings (95th pct) of each side
    pa, na, pf, nf = [], [], [], []
    for cp in caps:
        MA, MF = r3_bind._mean_maps(cp["A"], bind)
        s_, c_ = cp["s"], cp["c"]
        if cp["kind"] == "after":
            pa.append(float(MA[c_, s_])); na += [float(MA[q, s_]) for q in range(cp["qm"] + 1, cp["n"])]
            nf += [float(MF[s_ + 1, q]) for q in range(0, s_)]
        else:
            pf.append(float(MF[s_ + 1, c_])); nf += [float(MF[s_ + 1, q]) for q in range(c_ + 2, s_)]
    def mid(pos, neg):
        p5, n95 = float(np.percentile(pos, 5)), float(np.percentile(neg, 95))
        return ((p5 + n95) / 2 if p5 > n95 else p5), p5, n95
    bind["th_after"], pa5, na95 = mid(pa, na); bind["th_front"], pf5, nf95 = mid(pf, nf)
    if BIND_TH == "noncue95":                                                      # S609d: bound more than 95% of unbound tokens
        bind["th_after"], bind["th_front"] = na95, nf95
    bind["th_rule"] = BIND_TH
    print(f"[bind] 뒤 결속 머리 {ha} (보정 결속 {ca:.3f}) · 앞머리 결속 머리 {hf} ({cf:.3f})", flush=True)
    print(f"[bind] 문턱: 뒤 {bind['th_after']:.3f} (단서 5% {pa5:.3f} / 비단서 95% {na95:.3f}) · 앞 {bind['th_front']:.3f} ({pf5:.3f} / {nf95:.3f})", flush=True)
    return bind


def dseg_calibrate(M, n=24, seed=61900):
    """S619: DeltaNet heads with the largest contribution of the ordinal to its post-nominal cue (canonical sentences only)."""
    import r3_dseg
    v = M.v
    r3_dseg.install(v)
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(seed)
    recs, caps = [], []
    M.dseg_layers = "all"
    for _ in range(n):
        N = rng.randint(3, 6); k = rng.randint(1, N)
        pics = [load_image(f"proc:{p}") for p in rng.sample(pids, N)]
        for side in ("end", "start"):
            c = M.capture(f"What is in the {ORD[k-1]} image from the {side}? Answer with only one word from: circle, star, ring.", pics)
            toks = [v.tok.decode([t]) for t in c["ids"]]
            s_ = max(i for i in range(c["turn"]) if toks[i] == f" {ORD[k-1]}"); cue = next(i for i in range(s_, c["turn"]) if toks[i] == f" {side}")
            qm = next(i for i in range(cue, c["turn"]) if "?" in toks[i])
            recs.append(r3_dseg.per_head(c["A"], s_, cue)); caps.append((pics, k, side))
            del c
    heads, lvl = r3_dseg.select(recs)
    M.dseg_layers = sorted({l for l, h in heads})
    out = dict(mode="dseg", dheads=heads, capL=0, capR=32)
    if DSEG_TH:                                                                    # S619b: contribution threshold (second pass,
        pos, neg = [], []                                                         # selected layers only -- memory)
        for pics, k, side in caps:
            c = M.capture(f"What is in the {ORD[k-1]} image from the {side}? Answer with only one word from: circle, star, ring.", pics)
            toks = [v.tok.decode([t]) for t in c["ids"]]
            s_ = max(i for i in range(c["turn"]) if toks[i] == f" {ORD[k-1]}"); cue = next(i for i in range(s_, c["turn"]) if toks[i] == f" {side}")
            qm = next(i for i in range(cue, c["turn"]) if "?" in toks[i]); turn = c["turn"]
            C = r3_dseg.contrib(c["A"], heads, s_, turn)
            peak = float(C[1:].max()) if DSEG_TH in ("rel", "rel95") else 1.0          # S619c/d: relative to the source's own peak
            pos.append(float(C[cue - s_]) / peak); neg += [float(C[t - s_]) / peak for t in range(qm + 1, turn)]
            del c
        p5, n95 = float(np.percentile(pos, 5)), float(np.percentile(neg, 95))
        val = n95 if DSEG_TH == "rel95" else ((p5 + n95) / 2 if p5 > n95 else p5)    # S619d: instruction 95th percentile
        out["th" if DSEG_TH == "abs" else "rho"] = val
        print(f"[dseg] {DSEG_TH} 문턱 {val:.4f} (단서 5% {p5:.4f} / 지시문 95% {n95:.4f})", flush=True)
    print(f"[dseg] DeltaNet 머리 {heads} (정준 C {lvl:.3f})", flush=True)
    return out


def window(M, c, jx, srcs=None):
    """the source's candidate tokens and their binding prior (S609), the binding window (S609c), or the clause window."""
    if M.bind is not None and M.bind.get("mode") == "dseg":                         # S619: tokens the DeltaNet memory gives to jx
        import r3_dseg
        s0 = c["s0"]; srcs = [s0 + x for x in (srcs or [jx])]
        hi = min(c["turn"], s0 + jx + 1 + M.bind["capR"])
        own = r3_dseg.owners(c["A"], M.bind["dheads"], srcs, hi)
        win = [q - s0 for q in range(s0 + jx + 1, hi) if own.get(q) == s0 + jx]
        if M.bind.get("th") is not None:                                           # S619b
            C = r3_dseg.contrib(c["A"], M.bind["dheads"], s0 + jx, hi)
            win = [q for q in win if float(C[q - jx]) >= M.bind["th"]]
        if M.bind.get("rho") is not None and win:                                  # S619c: relative to the source's peak
            C = r3_dseg.contrib(c["A"], M.bind["dheads"], s0 + jx, hi)
            peak = max(float(C[q - jx]) for q in win)
            win = [q for q in win if float(C[q - jx]) >= M.bind["rho"] * peak]
        return win, None
    if M.bind is not None and M.bind.get("mode") == "hard":
        win, _ = r3_bind.window_hard(c["A"], jx, c["turn"] - c["s0"], M.bind, srcs or [jx], M.bind["capL"], M.bind["capR"])
        return win, None
    if M.bind is not None:
        win, pr = r3_bind.prior(c["A"], jx, c["turn"] - c["s0"], M.bind, M.bind["capL"], M.bind["capR"])
        return win, pr
    L, Rr = clause_window(M.v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
    return [q - c["s0"] for q in L + Rr], None


def reads(M, cal, c):
    Hb = c["H"][M.Lb]
    if cal["det_space"] == "full":
        dist, kh0 = torch.cdist(Hb, cal["Mb"]).min(1); th = cal["th_full"]
    else:
        dist, kh0 = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]).min(1); th = cal["th_chart"]
    X = torch.cat([c["bf"][l] for l in M.Rlayers], 1)
    PR = reader_probs(M.RP, X)
    E = PR @ torch.arange(1, C + 1).float()
    e = PR[int(E.argmax())]
    return dict(src=(dist <= th).nonzero().flatten().tolist(), kh=(kh0 + 1).tolist(), ehat=int(e.argmax()) + 1, codes=(PR.argmax(1) + 1).tolist())


def frame_write(M, cal, c, jx, frame_words=("end", "last")):
    """(position, delta) of the frame carrier of source jx: a calibrated anchor inside its two-sided clause."""
    v = M.v
    L, Rr = clause_window(v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
    Hb = c["H"][M.Lb]
    fr = cal["frame"]
    best = None
    for q in L + Rr:
        qq = q - c["s0"]
        for a in frame_words:
            d = float((Hb[qq] - fr["mu"][a]).norm())
            if d <= fr["thr"][a] and (best is None or abs(q - (c["s0"] + jx)) < abs(best[0] - (c["s0"] + jx))):
                best = (q, fr["mu_start"] - fr["mu"][a])
    return best


def ref_sources(M, prog, c, rd):
    """Map the program's references and nuisance ordinals to detected source indices (relative to s0)."""
    v = M.v
    span = c["ids"][c["s0"]:c["turn"]]
    words = [v.tok.decode([t]).strip().lower() for t in span]
    roles = {}
    if prog["format"] == "options":
        for i, r in enumerate(prog["refs"]):
            lt = f"{chr(65 + i)}".lower()
            # the option line "X: The <ord> image ..." -> the first ordinal after the line letter
            for jx in range(len(words) - 2):
                if words[jx] == lt and words[jx + 1] == ":" and jx + 3 < len(words) and words[jx + 3] == ORD[r["k"] - 1]:
                    roles[jx + 3] = ("ref", i)
                    break
    else:
        r = prog["refs"][0]
        pos = [jx for jx, w_ in enumerate(words) if w_ == ORD[r["k"] - 1]]
        if pos:
            roles[pos[-1]] = ("ref", 0)                                             # the reference NP comes last in the body
    for jx in rd["src"]:
        if jx not in roles:
            roles[jx] = ("identity", None)                                          # context / frame-phrase ordinals
    return {jx: rl for jx, rl in roles.items() if jx in rd["src"]}


def discover(M, cal, prog, rho=.5, teacher="label"):
    v = M.v
    pics = [load_image(k) for k in prog["images"]]
    c = M.capture(prog["text"], pics)
    rd = reads(M, cal, c)
    cand = cand_ids(v, prog["answer"])
    gold = prog["answer"]["cands"].index(prog["answer"]["gold"])
    if teacher == "self":                                                           # A2: the model's own forward answer is the teacher
        j = prog["targets"][0]
        fwd = f"What is in the {ORD[j-1]} image? Answer with only one word from: {', '.join(prog['answer']['cands'])}."
        gold = int(M.capture(fwd, pics)["logits"][cand].argmax())
    base_p = float(torch.softmax(c["logits"][cand], -1)[gold])
    roles = ref_sources(M, prog, c, rd)
    labels = {}
    for jx, (role, ri) in roles.items():
        kh = rd["kh"][jx]
        if role == "identity":
            labels[jx] = dict(Z=[kh - 1], role="identity", kh=kh)
            continue
        r = prog["refs"][ri]
        j = prog["targets"][ri]
        if prog["format"] == "options" and ri != prog["gold_option"]:
            labels[jx] = dict(Z=[j - 1], role="option_other", kh=kh, ri=ri, j=j)   # referent (answer-invariant, S588u)
            continue
        fw = frame_write(M, cal, c, jx) if r["frame"] == "END" else None
        P = []
        for t in range(1, c["N"] + 1):
            if t == kh:
                lg = c["logits"]
            else:
                w_ = [(c["s0"] + jx, cal["Mb"][t - 1] - cal["Mb"][kh - 1])] + ([fw] if fw is not None else [])
                lg = M.run(c["inp"], w_)
            P.append(float(torch.softmax(lg[cand], -1)[gold]))
        Z = [t - 1 for t, p in enumerate(P, 1) if p >= rho]
        labels[jx] = dict(Z=Z, role="ref", kh=kh, P=P, frame=r["frame"], fw=fw is not None, ri=ri, j=j)
    return dict(labels=labels, base_p=base_p, N=c["N"])


def run_build(v, tag, data_dir, Lb, Lr, Lrel, reader, lam=.1, epochs=30, lr=1e-3, seed=60500, balance=True, max_train=None,
              teacher="label", a3=False, calib_from=None, scenes=None, freeze_counter=False, cache_from=None):
    os.makedirs(OUT, exist_ok=True)
    M = Model(v, Lb, Lr, Lrel, reader)
    cpath = f"{OUT}/{calib_from or tag}_calib.pt"
    cal = torch.load(cpath, weights_only=False) if os.path.exists(cpath) else calibrate(M)
    torch.save(cal, f"{OUT}/{tag}_calib.pt")
    if BIND and BIND_MODE == "dseg":                                              # S619
        M.bind = dseg_calibrate(M)
        FIT_EXTRA["bind"] = M.bind; FIT_EXTRA["reader_tag"] = reader
    elif BIND:                                                                     # S609: binding heads, then capture their layers
        M.bind = bind_calibrate(M)
        M.att_layers = sorted({l for l, h in M.bind["heads_after"] + M.bind["heads_front"]})
        FIT_EXTRA["bind"] = M.bind; FIT_EXTRA["reader_tag"] = reader
    if a3:                                                                         # protocol A3: no discovery, no reverse target
        caches = {}
        for split in ("train", "dev"):
            progs = [json.loads(l) for l in open(f"{data_dir}/{split}.jsonl")]
            if split == "train" and scenes:
                progs = [p for p in progs if p["scene"] < scenes]
            items = []
            for n_, prog in enumerate(progs):
                items += cache_a3(M, cal, prog)
            caches[split] = dict(items=items)
            print(f"[{tag}] A3 {split}: 프로그램 {len(progs)} · 원천 예 {len(items)}", flush=True)
        return fit(M, cal, caches, tag, lam=0, epochs=epochs, lr=lr, seed=seed, balance=False, a3=True)
    t0 = time.time()
    caches = {}
    for split in ("train", "dev"):
        kp = f"{OUT}/{cache_from or tag}_cache_{split}.pt"
        if os.path.exists(kp):
            caches[split] = torch.load(kp, weights_only=False); continue
        src = f"{data_dir}/{split}.tierb.jsonl" if os.path.exists(f"{data_dir}/{split}.tierb.jsonl") else f"{data_dir}/{split}.jsonl"
        progs = [json.loads(l) for l in open(src)]
        if max_train and split == "train":
            progs = progs[:max_train]
        items, dropped, stats = [], 0, collections.Counter()
        for n_, prog in enumerate(progs):
            d = discover(M, cal, prog, teacher=teacher)
            disc_roles = {("ref", lab["ri"]): lab for lab in d["labels"].values() if lab["role"] in ("ref", "option_other")}
            if not any(l_["role"] == "ref" and l_["Z"] for l_ in d["labels"].values()):
                dropped += 1
                continue
            for l_ in d["labels"].values():
                if l_["role"] == "ref":
                    stats["refs"] += 1
                    stats["tstar_eq_j"] += int(int(np.argmax(l_["P"])) + 1 == l_["j"])
                    stats["j_in_Z"] += int((l_["j"] - 1) in l_["Z"])
            reals = prog.get("realizations") or [dict(text=prog["text"], tier="A")]
            for ri_, real in enumerate(reals):
                its = cache_identity_aware(M, cal, prog, real["text"], disc_roles)
                for it in its:
                    it["tier"] = real["tier"]; it["real"] = ri_
                items += its
            if (n_ + 1) % 100 == 0:
                print(f"[{tag}] {split} 발견·캐시 {n_+1}/{len(progs)} · 버림 {dropped} · 원천 예 {len(items)} · argmax p = j {stats['tstar_eq_j']}/{stats['refs']}  ({time.time()-t0:.0f}s)", flush=True)
        caches[split] = dict(items=items, dropped=dropped, stats=dict(stats))
        torch.save(caches[split], kp)
        print(f"[{tag}] {split}: 프로그램 {len(progs)} · 버림 {dropped} · 예 {len(items)} · argmax p = j {stats['tstar_eq_j']}/{stats['refs']}", flush=True)
    mod = fit(M, cal, caches, tag, lam=lam, epochs=epochs, lr=lr, seed=seed, balance=balance, freeze_counter=freeze_counter)
    return mod


def cache_a3(M, cal, prog):
    """A3: one capture pass; the reference source carries its frame label; START -> identity target, END -> no target."""
    v = M.v
    pics = [load_image(k) for k in prog["images"]]
    c = M.capture(prog["text"], pics)
    rd = reads(M, cal, c)
    roles = ref_sources(M, prog, c, rd)
    hrel = torch.cat([c["H"][l] for l in M.Lrel], 1)
    items = []
    for jx, (role, ri) in roles.items():
        win, pr = window(M, c, jx, rd["src"])
        kh = rd["kh"][jx]
        if role == "ref":
            fr = prog["refs"][ri]["frame"]
            Z = [kh - 1] if fr == "START" else []
            g = 0.0 if fr == "START" else 1.0
        else:
            Z, g, fr = [kh - 1], 0.0, "START"
        items.append(dict(hs=hrel[jx].clone().half(), hw=(hrel[win] if win else hrel[jx:jx + 1]).clone().half(), kh=kh, N=c["N"],
                          codes=rd["codes"], ehat=rd["ehat"], Z=Z, gate=g, role=role, fmt=prog["format"], frame=fr,
                          pid=prog["pid"], src=f"{role}:{ri}", scene=prog.get("scene"),
                          pr=(pr if (pr is not None and win) else None)))
    return items


def cache_identity_aware(M, cal, prog, text, disc_roles):
    """cache_realization with identity sources labelled by their own pointer."""
    v = M.v
    pics = [load_image(k) for k in prog["images"]]
    c = M.capture(text, pics)
    rd = reads(M, cal, c)
    p2 = dict(prog); p2["text"] = text
    roles = ref_sources(M, p2, c, rd)
    hrel = torch.cat([c["H"][l] for l in M.Lrel], 1)
    items = []
    for jx, (role, ri) in roles.items():
        if role == "identity":
            lab = dict(Z=[rd["kh"][jx] - 1], role="identity")
        else:
            lab = disc_roles.get(("ref", ri))
            if lab is None or not lab["Z"]:
                continue
        L, Rr = clause_window(v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
        win = [q - c["s0"] for q in L + Rr]
        items.append(dict(hs=hrel[jx].clone().half(), hw=(hrel[win] if win else hrel[jx:jx + 1]).clone().half(),
                          kh=rd["kh"][jx], N=c["N"], codes=rd["codes"], ehat=rd["ehat"], Z=lab["Z"], role=lab["role"],
                          fmt=prog["format"], frame=lab.get("frame"), pid=prog["pid"], src=f"{role}:{ri}"))
    return items


def q_of(mod, it, dev):
    pi, g = mod(it["hs"].float().to(dev), it["hw"].float().to(dev), it["N"], it["kh"], prior=it.get("pr"))
    codes = torch.tensor(it["codes"], device=dev).long()
    q = torch.zeros(C, device=dev).index_add(0, (codes - 1).clamp(0, C - 1), pi)
    return q / (q.sum() + 1e-8), g


def fit(M, cal, caches, tag, lam=.1, epochs=30, lr=1e-3, seed=60500, balance=True, a3=False, freeze_counter=False):
    dev = M.v.device
    tr, va = caches["train"]["items"], caches["dev"]["items"]
    d_ = tr[0]["hs"].shape[-1]
    torch.manual_seed(seed)
    mod = Scanner(d_, warm=True, mix="prob").to(dev).float()
    if a3 or freeze_counter:                                                       # A3.1: the counting algorithm is fixed (exact counter);
        for nm_ in ("vin", "lam", "wp"):                                          # only the interface (relation head, gate) and tau learn
            getattr(mod, nm_).requires_grad_(False)
        with torch.no_grad():
            mod.g.bias.zero_()                                                     # neutral gate start
    opt = torch.optim.Adam([p_ for p_ in mod.parameters() if p_.requires_grad], lr=lr, weight_decay=1e-4)
    grp = lambda x: f"{x['fmt']}:{x['role']}"
    gc = collections.Counter(grp(x) for x in tr)
    wgt = {g: len(tr) / (len(gc) * c_) for g, c_ in gc.items()} if balance else {g: 1.0 for g in gc}
    bykey = collections.defaultdict(list)
    for i, x in enumerate(tr):
        bykey[(x["pid"], x["src"])].append(i)
    pairs = [ix for ix in bykey.values() if len(ix) >= 2]
    print(f"[{tag}] 맞춤: 예 {len(tr)} · 묶음 {dict(gc)} · 일관성 쌍 묶음 {len(pairs)} · λ {lam}", flush=True)

    def acc(items):
        with torch.no_grad():
            by = collections.defaultdict(list)
            for x in items:
                q, g = q_of(mod, x, dev)
                if a3:                                                             # selection: forward target + gate only (no reverse outcome)
                    by["gate"].append((float(g) > .5) == (x["gate"] > .5))
                    if x["Z"]:
                        by["fwd_target"].append(int(q.argmax()) in x["Z"])
                else:
                    by[grp(x)].append(int(q.argmax()) in x["Z"])
            return float(np.mean([np.mean(v_) for v_ in by.values()])), {g: round(float(np.mean(v_)), 3) for g, v_ in by.items()}
    rng = random.Random(seed + 1)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        order = list(range(len(tr))); rng.shuffle(order); tot = 0.0
        for i in order:
            x = tr[i]
            q, g = q_of(mod, x, dev)
            if a3 and BIND_MODE in ("hard", "dseg"):                               # S609c/S619: logit BCE (no dead clamp)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(mod.last_glogit.reshape(()), torch.tensor(x["gate"], device=dev))
                if x["Z"]:
                    loss = loss - torch.log(q[torch.tensor(x["Z"], device=dev)].sum() + 1e-6)
            elif a3:
                loss = torch.nn.functional.binary_cross_entropy(g.clamp(1e-5, 1 - 1e-5).reshape(()), torch.tensor(x["gate"], device=dev))
                if x["Z"]:
                    loss = loss - torch.log(q[torch.tensor(x["Z"], device=dev)].sum() + 1e-6)
            else:
                loss = -torch.log(q[torch.tensor(x["Z"], device=dev)].sum() + 1e-6) * wgt[grp(x)]
            if lam > 0 and not a3:
                sib = [j for j in bykey[(x["pid"], x["src"])] if j != i]
                if sib:
                    y = tr[rng.choice(sib)]
                    with torch.no_grad():
                        q2, g2 = q_of(mod, y, dev)
                    loss = loss + lam * (q2 * (torch.log(q2 + 1e-8) - torch.log(q + 1e-8))).sum() + lam * (g - g2) ** 2
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0); opt.step(); tot += float(loss)
        va_acc, va_by = acc(va)
        print(f"[{tag}] 에폭 {ep+1} 손실 {tot/len(tr):.4f} · DEV 표적 정확(묶음 평균) {va_acc:.3f} {va_by}", flush=True)
        if va_acc > best + 1e-4:
            best, best_state, bad = va_acc, {k_: v_.detach().clone() for k_, v_ in mod.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= (10 if (a3 or freeze_counter) else 5):
                break
    mod.load_state_dict(best_state)
    ck = dict(mod=mod.state_dict(), variant="scan", scan_mix="prob", Lb=M.Lb, Lr=M.Lr, Lrel=M.Lrel, reader=None,
              cal=cal, twosided=True, tau=.9, dev_acc=best, left_scope=LEFT_SCOPE, **FIT_EXTRA)
    if BIND:
        ck["fwd_safe"] = True                                                      # S609 (4): rebind only on an end-frame gate
    torch.save(ck, f"{OUT}/{tag}_r3.pt")
    print(f"[{tag}] -> {OUT}/{tag}_r3.pt (DEV {best:.3f})", flush=True)
    return mod


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--data", required=True)
    ap.add_argument("--Lb", type=int, required=True); ap.add_argument("--Lr", type=int, required=True)
    ap.add_argument("--Lrel", type=int, nargs="+", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--lam", type=float, default=.1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--teacher", default="label", choices=["label", "self"])
    ap.add_argument("--a3", action="store_true")
    ap.add_argument("--calib_from", default=None)
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--freeze_counter", action="store_true")
    ap.add_argument("--cache_from", default=None)
    ap.add_argument("--left_scope", default=None, choices=[None, "comma"])
    ap.add_argument("--bind", action="store_true")
    ap.add_argument("--bind_mode", default="prior", choices=["prior", "hard", "dseg"])
    ap.add_argument("--bind_th", default="mid", choices=["mid", "noncue95"])
    ap.add_argument("--dseg_th", default=None, choices=[None, "abs", "rel", "rel95"])
    a = ap.parse_args()
    DSEG_TH = a.dseg_th
    LEFT_SCOPE = a.left_scope
    BIND = a.bind; BIND_MODE = a.bind_mode; BIND_TH = a.bind_th
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    run_build(v, a.tag, a.data, a.Lb, a.Lr, a.Lrel, a.reader, lam=a.lam, epochs=a.epochs, max_train=a.max_train, teacher=a.teacher,
              a3=a.a3, calib_from=a.calib_from, scenes=a.scenes, freeze_counter=a.freeze_counter, cache_from=a.cache_from)
