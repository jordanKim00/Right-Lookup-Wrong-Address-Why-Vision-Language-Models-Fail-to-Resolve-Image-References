"""S917 collector (registered in knowledge/DECISIONS.md before the data) - is the writer's effect on the answer transmitted through the late
full-attention reads?  (S914 post hoc: the L51 block profile's link to the answer differs with and without the writer.)  Natural runs at N8
(fresh scenes), k in --ks (2-6), at END(k)'s natural L47 anchor a.  Per scene and k one batch of 12: E (END(k)), R (START(a): the band source),
B (E_BANDR - the band 46-49 read-row outputs <- R's: the writer-free run); late read swaps with the head-patch tool (every head's gated output
at the read rows of the late full-attention layers <- the other run's, same pass): E_SW (layers 51-63 <- B's), B_SW (51-63 <- E's), E_SW51
(51 only), E_SWL (55-63), E_SWself (E <- E: exactness); late read removals (S891 FAAttnEdit: every head's attention mass on the positions set
to zero, the rest rescaled): E_NI / B_NI (all picture tokens, 51-63), E_NM / B_NM (menu name tokens, 51-63; descriptive - in the smoke the
menu removal broke half of the writer-free answers, so ratios under it are not interpretable).  Phase A: natural END(k) (anchors), START(q).
Stored per run: answer, menu logits, the matcher-head mean block scores of every cap row at L43, L47, L51-63.
Usage: python s917_lateroute.py MODEL TAG --N 8 --ks 2,3,4,5,6 [--n 40] [--seed S] [--device_map balanced] [--threads 2] [--tokens 2]"""
from __future__ import annotations

import argparse, collections, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from s891_attnpredict import FAAttnEdit
from c_headpatch import FAHeadPatchPinned

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s917_{}_lateroute.json"


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
    ALLC = [(l, kd) for l in (46, 47, 48, 49) for kd in ("mix", "mlp")]                             # the band's components (E_BANDR)
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
    LA51 = FAAttnEdit(v, [51]); LALATE = FAAttnEdit(v, [55, 59, 63])                                # late read-route removal (after TOOL's hooks)
    HP = FAHeadPatchPinned(v, [51, 55, 59, 63])                                                      # late read swaps (every head's gated output <- a source's)

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
    def run1(inps, qs, l51s=None, lates=None, hps=None):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs); LA51.set(l51s); LALATE.set(lates); HP.set(hps)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None); LA51.set(None); LALATE.set(None); HP.set(None)
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
    # ---- positions (identical layout in every scene and prompt: fixed picture size, single-token names, ordinals and frame words)
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

    def anchor_of(q):
        q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
        c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]

    ROUTES = dict(NI="img", NM="menu")                                                            # late read routes removed (all 24 heads, read rows)

    def task(si):
        lay = SC[si]["lay"]; IMG = [i for i, o in enumerate(lay["owner"]) if o >= 0]; MENU = list(lay["opt"])
        blk = dict(img=IMG, menu=MENU)
        def la(route):                                                                           # (layer-51 cfg, layers 55-63 cfg)
            if route is None:
                return None, None
            c = dict(rows=rowsC, edit={}, prow=rowsC, aedit=dict(heads=list(range(24)), blk=blk[route], kind="scale", factor=0.0))
            return c, c
        # phase A: natural END(k) (anchors) and START(q)
        items = [(f"N_E{k}", f"E{k}", capq(SC[si]), None) for k in KS] + [(f"N_S{q}", f"S{q}", capq(SC[si]), None) for q in range(1, N + 1)]
        for i0_ in range(0, len(items), 12):
            part = items[i0_:i0_ + 12]
            lg, st = run1([SC[si]["enc"][x[1]] for x in part], [x[2] for x in part])
            for b, x in enumerate(part):
                store(si, x[0], lg[b], st[b])
            with CLOCK:
                CNT[0] += len(part)
        anc = {k: anchor_of(RECS[si]["mm"][f"N_E{k}"]["47"]) for k in KS}; RECS[si]["anchor"] = {str(k): x for k, x in anc.items()}
        # phase B: one batch of 12 per k - E, R = START(a) (band source), B = E_BANDR; late read swaps (SW: the late full-attention layers' gated
        # outputs of the read rows <- the other run's, every head; SW51: layer 51 only; SWL: 55-63 only; SWself: E <- E, the exactness check);
        # the late picture reads removed (NI) and the late menu-name reads removed (NM, descriptive: it can break the writer-free answer)
        for k in KS:
            ak = anc[k]
            hp = lambda src, layers: dict(rows=rowsC, edit={}, prow=rowsC, hedits={l: [dict(heads=list(range(24)), op="swap", src=src)] for l in layers})
            LL = [51, 55, 59, 63]
            items = [(f"E{k}", f"E{k}", capq(SC[si]), None, None), (f"R{k}", f"S{ak}", capq(SC[si]), None, None),
                     (f"B{k}", f"E{k}", capq(SC[si], cpatch=dict(src=1, comps=ALLC)), None, None),
                     (f"E{k}_NI", f"E{k}", capq(SC[si]), "img", None), (f"B{k}_NI", f"E{k}", capq(SC[si], cpatch=dict(src=1, comps=ALLC)), "img", None),
                     (f"E{k}_NM", f"E{k}", capq(SC[si]), "menu", None), (f"B{k}_NM", f"E{k}", capq(SC[si], cpatch=dict(src=1, comps=ALLC)), "menu", None),
                     (f"E{k}_SW", f"E{k}", capq(SC[si]), None, hp(2, LL)),
                     (f"B{k}_SW", f"E{k}", capq(SC[si], cpatch=dict(src=1, comps=ALLC)), None, hp(0, LL)),
                     (f"E{k}_SW51", f"E{k}", capq(SC[si]), None, hp(2, [51])), (f"E{k}_SWL", f"E{k}", capq(SC[si]), None, hp(2, [55, 59, 63])),
                     (f"E{k}_SWself", f"E{k}", capq(SC[si]), None, hp(0, LL))]
            cf = [la(x[3]) for x in items]
            base = dict(rows=rowsC, edit={}, prow=rowsC, hedits={})                                    # no edit; every run a valid swap source
            lg, st = run1([SC[si]["enc"][x[1]] for x in items], [x[2] for x in items], [c[0] for c in cf], [c[1] for c in cf],
                          [x[4] if x[4] is not None else base for x in items])
            for b, x in enumerate(items):
                store(si, x[0], lg[b], st[b])
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
        TOOL.remove(); LA51.remove(); LALATE.remove(); HP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, ks=list(KS), routes=ROUTES, band=ALLC, read_rows=rowsC, qm=L0["qm"], read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   early_mean={str(li): hm(li) for li in EARLY}, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=91708); ap.add_argument("--N", type=int, default=8); ap.add_argument("--ks", default="2,3,4,5,6")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
