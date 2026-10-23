"""S918 collector (registered in knowledge/DECISIONS.md before the data; advisor review 2, 10-22: address vs reading strength) - at a fixed early
address, does how the address is read (the picture-vs-text attention mass and the output gate at the read rows of the early full-attention
layers 35, 39, 43, 47) carry the END-conditioned departure?  Natural runs at N8 (fresh scenes), k in --ks (2-6), at END(k)'s natural L47 anchor
a_k.  Phase A: natural END(k) and START(q) - each a donor of its read-row query (address), picture mass and output gate (per layer, row, head).
Phase B per k, every run pinned at the four layers (exp/c_qpath_rm.py, float32 recompute of the read rows): the read-row query <- the address
donor's (S829 part='all': every head's picture scores = the donor's); the picture mass per row and head <- the reading donor's (the
distributions within the pictures and within the text kept) and the output gate <- the reading donor's, or the state's own.  Cells named
state-address-reading over {E = END(k), S = START(a_k)}: EEE, EES (+ EESm mass only, EESg gate only), ESE, ESS, SEE (+ SEEm, SEEg), SES, SSE, SSS.
Stored per run: answer, menu logits, block scores (every head, rows B1 / B4 / answer) at L43, L47, L51, the matcher-head mean of every cap row at
L47-63, and the picture mass actually used after the edit ('img2') at the four layers.
Usage: python s918_reading.py MODEL TAG --N 8 --ks 2,3,4,5,6 [--n 32] [--seed S] [--device_map balanced] [--threads 2] [--tokens 2]"""
from __future__ import annotations

import argparse, collections, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_rm import QPathRM

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s918_{}_reading.json"


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
    P47 = list(EARLY)
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
    TOOL = QPathRM(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=[])

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
        SC.append(dict(si=si, labs=labs, menu=menu, enc={nm: cpu_(v.encode(prompt(f, q, menu, N), pics[:N])) for nm, (f, q) in names.items()}))
    # ---- positions (identical layout in every scene and prompt: fixed picture size, single-token names and frame words)
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
        diff = sorted({i for i in range(S_) if len({ids[nm][i] for nm in ids}) > 1})
        return dict(S=S_, qm=qm, uend=uend, owner=owner, opt=opt, rowsC=rowsC, diff=diff, vs=vs, ve=ve)
    for sc_ in SC:
        sc_["lay"] = layout(sc_)
    L0 = SC[0]["lay"]
    for sc_ in SC:
        assert (sc_["lay"]["S"], sc_["lay"]["rowsC"], sc_["lay"]["diff"]) == (L0["S"], L0["rowsC"], L0["diff"]), "layouts differ across scenes"
    assert len(L0["diff"]) == 2 and L0["diff"][-1] < L0["qm"], f"prompts must differ only at the ordinal and the frame word: {L0['diff']}"
    IMGP = [i for i, o in enumerate(L0["owner"]) if o >= 0]
    rowsC = L0["rowsC"]
    capq = lambda sc_, **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=sc_["lay"]["owner"], nb=N, opt=sc_["lay"]["opt"]), **kw)

    RECS = [dict(si=si, labs=SC[si]["labs"], menu=SC[si]["menu"], land={}, mlog={}, mm={}, meta={}) for si in range(a.n)]
    r4 = lambda t: np.round(t.float().cpu().numpy(), 4).tolist()

    def store(si, key, lgb, stb):
        T1.append(int(int(lgb.argmax()) in t1ids(SC[si]["menu"])))
        w = SC[si]["menu"][int(torch.stack([sc(lgb, c) for c in SC[si]["menu"]]).argmax())]
        RECS[si]["land"][key] = SC[si]["labs"].index(w) + 1
        RECS[si]["mlog"][key] = [round(float(sc(lgb, x)), 4) for x in SC[si]["labs"][:N]]
        for li in (43, 47, 51):                                                                 # every head, rows B1 / B4 / answer, magnitudes kept
            RECS[si].setdefault(f"qk{li}h", {})[key] = r4(stb[("cap", li)][[2, 5, 9]])
        RECS[si]["mm"][key] = {str(li): r4(stb[("cap", li)][:, hm(li)].float().mean(1)) for li in (47,) + tuple(LATE)}

    def anchor_of(q):
        q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
        c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]

    def task(si):
        own = SC[si]["lay"]["owner"]; QDs = {}; RD = {}
        def ed(addr, read, parts=("mass", "gate")):
            """per pinned layer: the full read-row query <- addr's; the picture mass and / or the output gate <- read's (None: the state's own)"""
            out = {}
            for li in P47:
                e_ = dict(qsub=dict(owner=own, nb=N, part="all", donors=[(1.0, QDs[addr][li])]))
                if read is not None and "mass" in parts:
                    e_["mass"] = dict(target=RD[read][li]["img"], img=IMGP)
                if read is not None and "gate" in parts:
                    e_["gate"] = RD[read][li]["g"]
                out[li] = e_
            return out
        def mk(nm, ek, edit=None, grab=False, meta=None):
            q = capq(SC[si], edit=edit) if edit is not None else capq(SC[si])
            if grab:
                q["grabq"] = set(EARLY); q["grabg"] = set(EARLY)
            RECS[si]["meta"][nm] = meta or dict(state=ek)
            return (nm, SC[si]["enc"][ek], q, grab)
        def go(items):
            for i0 in range(0, len(items), 12):                                                   # sub-passes of at most 12 runs (GPU0 memory)
                part = items[i0:i0 + 12]
                lg, st = run1([x[1] for x in part], [x[2] for x in part])
                for b, x in enumerate(part):
                    if x[3]:
                        QDs[x[0]] = {li: st[b][("q", li)] for li in EARLY}
                        RD[x[0]] = {li: dict(img=st[b][("img", li)].float(), g=st[b][("g", li)].float()) for li in EARLY}
                        RECS[si].setdefault("img_nat", {})[x[0]] = {str(li): r4(st[b][("img", li)]) for li in EARLY}
                    else:
                        RECS[si].setdefault("img2", {})[x[0]] = {str(li): r4(st[b][("img2", li)]) for li in EARLY if ("img2", li) in st[b]}
                    store(si, x[0], lg[b], st[b])
                with CLOCK:
                    CNT[0] += len(part)
        # phase A: natural END(k) and START(q) - donors of address (query), picture mass and output gate
        go([mk(f"N_E{k}", f"E{k}", grab=True) for k in KS]); go([mk(f"N_S{q}", f"S{q}", grab=True) for q in range(1, N + 1)])
        anc = {k: anchor_of(RECS[si]["mm"][f"N_E{k}"]["47"]) for k in KS}
        RECS[si]["anchor"] = {str(k): x for k, x in anc.items()}
        # phase B: per k the 2 x 2 x 2 of state x address x reading (+ mass-only / gate-only for two cells); names: state-address-reading
        it = []
        for k in KS:
            E_, S_ = f"N_E{k}", f"N_S{anc[k]}"; sE, sS = f"E{k}", f"S{anc[k]}"
            cells = [("EEE", sE, E_, None, ("mass", "gate")), ("EES", sE, E_, S_, ("mass", "gate")), ("EESm", sE, E_, S_, ("mass",)), ("EESg", sE, E_, S_, ("gate",)),
                     ("ESE", sE, S_, None, ("mass", "gate")), ("ESS", sE, S_, S_, ("mass", "gate")),
                     ("SEE", sS, E_, E_, ("mass", "gate")), ("SEEm", sS, E_, E_, ("mass",)), ("SEEg", sS, E_, E_, ("gate",)), ("SES", sS, E_, None, ("mass", "gate")),
                     ("SSE", sS, S_, E_, ("mass", "gate")), ("SSS", sS, S_, None, ("mass", "gate"))]
            for nm, st_, addr, read, parts in cells:
                it.append(mk(f"{nm}_{k}", st_, edit=ed(addr, read, parts), meta=dict(state=st_, address=addr, reading=read or "state", parts=list(parts))))
        go(it)
        QDs.clear(); RD.clear(); gc.collect()

    WORK = list(range(a.n))
    for d_ in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d_)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        TOOL.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, ks=list(KS), P47=P47, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, early_mean={str(li): hm(li) for li in EARLY}, rowsC=rowsC,
                   diff=L0["diff"], readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0],
                   rows=RECS), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=91808); ap.add_argument("--N", type=int, default=8); ap.add_argument("--ks", default="2,3,4,5,6")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
