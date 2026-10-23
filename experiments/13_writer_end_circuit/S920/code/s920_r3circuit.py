"""S920 collector (registered in knowledge/DECISIONS.md before the data; advisor reviews 2-3) - R³ against the END circuit on the same scenes:
in runs R³ repairs, is the band's END-specific rewriting gone and do the late read outputs approach direct START(t)'s?  What do its two writes
(27B R³-v2, S713: at layer 3 the ordinal's address Mb[t] - Mb[k_hat] and the frame carrier END -> START; t = k_hat gets the frame write only)
do separately?  Natural runs at one N (fresh scenes), k in --ks (target t = N + 1 - k): natural END(k), START(q) for every q (batched).  Per k:
three R³ plans (batch-1 capture forwards; READ ledger kept), then one batch of 10 with per-sample R³ writes (R3Batch): E (END(k)), Sa (START(a_k),
a_k = END(k)'s natural L47 anchor), E_B (E + Sa's band 46-49 read-row outputs: the natural writer-free run), St (START(t)), R3 (both writes),
R3_B (R3 + St's band outputs), R3A (address write only), R3A_B (R3A + St's band), R3F (frame write only), Sk (START(k)).  Stored per run: answer,
menu logits, matcher-head mean block scores of every cap row at L43, L47, L51-63, every head's L47 scores; per batch the cosine of every run's
late full-attention output (layers 51-63, rows B1 / B4 / answer) with St's and with E's.  The R³ object is not thread-safe: --threads 1.
Usage: python s920_r3circuit.py MODEL TAG --N 8 --ks 2,3,4,5,6 [--n 32] [--seed S] [--device_map balanced] [--threads 1] [--tokens 1]"""
from __future__ import annotations

import argparse, collections, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from r3_min_md import R3MinMD


class R3Batch(R3MinMD):
    """R3MinMD with per-sample writes in one batched forward: ST['bwrites'] = [list of (position, vector) for each batch element]."""
    def _hook(self, l):
        base = R3MinMD._hook(self, l)

        def h(m, i, o):
            ST = self.ST
            if not ST or ST.get("bwrites") is None:
                return base(m, i, o)
            if l != self.Lw:
                return o
            t = o[0] if isinstance(o, tuple) else o
            t2 = t.clone(); dev = t.device
            for b, ws in enumerate(ST["bwrites"]):
                for p_, dlt in ws:
                    t2[b, p_] = (t[b, p_].float() + dlt.to(dev)).to(t.dtype)
            return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
        return h

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s920_{}_r3circuit.json"


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    N = a.N; KS = tuple(int(x) for x in a.ks.split(","))
    assert os.environ.get("R3_READ") == "max"
    import c_tritonlock; c_tritonlock.install()                                                    # Triton autotuner is not thread-safe
    dev = os.environ.get("S863_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    REP = R3Batch(v, "/vlm/results/r3/s709_r327b_cal.pt", v2=True)                                  # 27B R³-v2 (S713): writes at layer 3
    ALLC = [(l, kd) for l in (46, 47, 48, 49) for kd in ("mix", "mlp")]                              # the band's components (BANDR)
    LO = {}                                                                                          # late FA read outputs (rows B1 / B4 / answer)
    def _late(l):
        def h(mod, args, kwargs, out):
            if LO.get("on"):
                t = out[0] if isinstance(out, tuple) else out
                LO[l] = t[:, LO["rows"]].detach().float().cpu()
            return None
        return h
    LHOOKS = []
    band = [46, 47, 48, 49]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tv = lambda s: (v.tok(s, add_special_tokens=False)["input_ids"][0], v.tok(" " + s, add_special_tokens=False)["input_ids"][0])
    def sc(lg, s):
        t0, t1 = tv(s); return torch.maximum(lg[t0], lg[t1])
    def t1ids(ws):
        return [tv(w)[0] for w in ws] + [tv(w)[1] for w in ws]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id, end_id = T("<|vision_start|>"), T("<|vision_end|>"), T("<|im_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=[])
    for l_ in LATE:                                                                                  # after TOOL's hooks: the outputs actually written
        LHOOKS.append(v.layers[l_].self_attn.register_forward_hook(_late(l_), with_kwargs=True))

    class _Tokens:                                                                               # memory tokens; take(cap) = run alone (priority)
        def __init__(self, n):
            self.n = n; self.cap = n; self.excl = 0; self.c = threading.Condition()
        def take(self, k):
            with self.c:
                if k >= self.cap:
                    self.excl += 1
                    try:
                        while self.n < self.cap:
                            self.c.wait()
                    finally:
                        self.excl -= 1
                    self.n -= self.cap; return
                while self.n < k or self.excl > 0:
                    self.c.wait()
                self.n -= k
        def give(self, k):
            with self.c:
                self.n += k; self.c.notify_all()
    TOK = _Tokens(a.tokens)
    _vis = getattr(v.model.model, "visual", None); VLOCK = threading.Lock(); _vis_fwd = None
    if _vis is not None:
        _vis_fwd = _vis.forward
        def _vis_locked(*args_, **kw_):
            with VLOCK:
                return _vis_fwd(*args_, **kw_)
        _vis.forward = _vis_locked

    QUEST = {"E": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "S": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None)
                    return lg, st
                except torch.OutOfMemoryError:
                    if attempt == 2:
                        raise
                    TOK.give(held); held = 0
                    gc.collect(); torch.cuda.empty_cache()
                    TOK.take(TOK.cap); held = TOK.cap
                    print(f"[{a.tag}] out of memory: retrying alone", flush=True)
        finally:
            TOK.give(held)

    cpu_ = lambda e: {k: (t.cpu() if torch.is_tensor(t) else t) for k, t in e.items()}           # encoded inputs wait on the host (GPU0 memory)
    rng = random.Random(a.seed); T1 = []; CNT = [0]; CLOCK = threading.Lock()
    SC = []
    for si in range(a.n):
        labs = rng.sample(labels, max(9, N))
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        names = {**{f"E{q}": ("E", q) for q in KS}, **{f"S{q}": ("S", q) for q in range(1, N + 1)}}
        SC.append(dict(si=si, labs=labs, menu=menu, pics=pics[:N], text={nm: prompt(f, q, menu, N) for nm, (f, q) in names.items()},
                       enc={nm: cpu_(v.encode(prompt(f, q, menu, N), pics[:N])) for nm, (f, q) in names.items()}))
    # ---- positions (identical layout in every scene and prompt); END(k) and START(k) differ only at the frame word
    def layout(sc_):
        ids = {nm: e["input_ids"][0].tolist() for nm, e in sc_["enc"].items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = next(iter(ids.values())); S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in sc_["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        rowsC = [qm] + list(range(uend, uend + 5)) + A4
        return dict(S=S_, qm=qm, uend=uend, owner=owner, opt=opt, rowsC=rowsC)
    for sc_ in SC:
        sc_["lay"] = layout(sc_)
    L0 = SC[0]["lay"]
    for sc_ in SC:
        assert (sc_["lay"]["S"], sc_["lay"]["rowsC"]) == (L0["S"], L0["rowsC"]), "layouts differ across scenes"
    rowsC = L0["rowsC"]
    capq = lambda sc_, **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=sc_["lay"]["owner"], nb=N, opt=sc_["lay"]["opt"]), **kw)

    RECS = [dict(si=si, labs=SC[si]["labs"], menu=SC[si]["menu"], land={}, mlog={}, mm={}) for si in range(a.n)]
    r4 = lambda t: np.round(t.float().cpu().numpy(), 4).tolist()

    def store(si, key, lgb, stb):
        T1.append(int(int(lgb.argmax()) in t1ids(SC[si]["menu"])))
        w = SC[si]["menu"][int(torch.stack([sc(lgb, c) for c in SC[si]["menu"]]).argmax())]
        RECS[si]["land"][key] = SC[si]["labs"].index(w) + 1
        RECS[si]["mlog"][key] = [round(float(sc(lgb, x)), 4) for x in SC[si]["labs"][:N]]
        RECS[si]["mm"][key] = {str(li): r4(stb[("cap", li)][:, hm(li)].float().mean(1)) for li in (43, 47) + tuple(LATE)}
        RECS[si].setdefault("qk47h", {})[key] = r4(stb[("cap", 47)][[2, 5, 9]])

    def anchor_of(q):
        q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
        c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]

    def task(si):
        sc_ = SC[si]
        # natural runs (one batch per form): END(k), START(q) for every q
        for grp in ([f"E{k}" for k in KS], [f"S{q}" for q in range(1, N + 1)]):
            lg, st = run1([sc_["enc"][nm] for nm in grp], [capq(sc_) for _ in grp])
            for b, nm in enumerate(grp):
                store(si, f"N_{nm}", lg[b], st[b])
            with CLOCK:
                CNT[0] += len(grp)
        for k in KS:
            t = N + 1 - k; ak = anchor_of(RECS[si]["mm"][f"N_E{k}"]["47"])
            # R³ plans (batch-1 capture forwards, unmeasured): both writes, address only (no_frame), frame only (no_ord)
            W = {}
            for arm, abl in (("R3", None), ("R3A", "no_frame"), ("R3F", "no_ord")):
                REP.ablate = abl
                _, writes = REP.plan(sc_["text"][f"E{k}"], sc_["pics"])
                W[arm] = writes
                RECS[si].setdefault("plan", {})[f"{arm}_{k}"] = dict(detail=[{kk: (vv if not isinstance(vv, tuple) else list(vv)) for kk, vv in d_.items()} for d_ in REP.last_detail],
                                                                      n_writes=len(writes))
            REP.ablate = None
            # one batch: E, START(a_k), E_B (E + START(a_k)'s band), START(t), R3, R3_B (+ START(t)'s band), R3A, R3A_B, R3F, START(k)
            items = [("E", f"E{k}", None, None), ("Sa", f"S{ak}", None, None), ("E_B", f"E{k}", None, 1), ("St", f"S{t}", None, None),
                     ("R3", f"E{k}", "R3", None), ("R3_B", f"E{k}", "R3", 3), ("R3A", f"E{k}", "R3A", None), ("R3A_B", f"E{k}", "R3A", 3),
                     ("R3F", f"E{k}", "R3F", None), ("Sk", f"S{k}", None, None)]
            cfgs = []
            for nm, ek, wk, src in items:
                c = capq(sc_)
                if src is not None:
                    c["cpatch"] = dict(src=src, comps=ALLC)
                cfgs.append(c)
            REP.ST = dict(bwrites=[(W[wk] if wk else []) for _, _, wk, _ in items])
            LO.clear(); LO.update(on=True, rows=[rowsC[i] for i in (2, 5, 9)])
            try:
                lg, st = run1([sc_["enc"][ek] for _, ek, _, _ in items], cfgs)
            finally:
                REP.ST = {}; LO["on"] = False
            for b, (nm, _, _, _) in enumerate(items):
                store(si, f"{nm}_{k}", lg[b], st[b])
            # late read outputs: cosine of every run's output with START(t)'s and with END(k)'s (layer, rows B1 / B4 / answer)
            cs = {}
            for l_ in LATE:
                O = LO[l_]                                                                             # [batch, 3, hidden]
                for b, (nm, _, _, _) in enumerate(items):
                    cs.setdefault(nm, {})[str(l_)] = dict(St=[round(float(x), 5) for x in torch.nn.functional.cosine_similarity(O[b], O[3], dim=-1)],
                                                          E=[round(float(x), 5) for x in torch.nn.functional.cosine_similarity(O[b], O[0], dim=-1)])
            RECS[si].setdefault("lateout", {})[str(k)] = cs
            RECS[si].setdefault("anchor", {})[str(k)] = ak
            with CLOCK:
                CNT[0] += len(items)
        gc.collect()

    WORK = list(range(a.n))
    for d_ in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d_)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        TOOL.remove()
        for h_ in LHOOKS:
            h_.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, ks=list(KS), r3=dict(cal="s709_r327b_cal.pt", Lb=REP.Lb, Lw=REP.Lw, v2=True), read_rows=rowsC,
                   qm=L0["qm"], read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   early_mean={str(li): hm(li) for li in EARLY}, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=92008); ap.add_argument("--N", type=int, default=8); ap.add_argument("--ks", default="2,3,4,5,6")
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
