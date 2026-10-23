"""S915 collector (registered in knowledge/DECISIONS.md before the data; the advisor's sufficiency demand, 10-22) - is the early-address profile a
sufficient state for the later computation?  Hold the early address fixed and swap the remaining hidden state.  Natural runs at one N: END(k)
for every k ('from the last'; with --forms last,end also 'from the end'), START(q) for every q; every natural run's L35-47 read-row queries are
kept as donors.  Pinned runs replace every head's read-row query at the pinned layers by the donor's (S829 qsub part='all'; the picture keys
are the same in every prompt of a scene, so the read rows' block scores and picture attention at those layers become the donor's - the early
lookup is held exactly; a smoke check showed that replacing only the address part U left the anchor of graded END profiles unheld in ~12% of
runs); P47 = layers 35, 39, 43, 47, P43 = 35, 39, 43 (pre-band boundary):
  SP    self-pins (state = donor; the pin machinery control): every natural run, P47;
  XP    phrase crossing (P47): END_end(k) <- END_last(k)'s address (XP_X{k}), END_last(k) <- END_end(k)'s (XP_E{k});
  KC    k crossing (P47): END_last(k2) <- END_last(k)'s address (KC_E{k2}_{k}); --kc all: every k2 != k, nb: |k - k2| = 1;
  FC    frame crossing (P47): START(a_k) <- END(k)'s address (FC_S{k}); END(k) <- START(a_k)'s (FC_E{k}); a_k = END(k)'s natural L47 anchor;
  P43   (--p43 1): SP43_E{k}, KC43_E{k2}_{k} (|k - k2| = 1), XP43_X{k};
  P47U  (amendment 10-22, the advisor's review: the full query also carries within-block token selection and text matching, so holding it is
        not holding the address profile alone) - only the address part U of the read-row queries <- the donor's (S808 part='U'): SPU_E{k},
        XPU_X{k} / XPU_E{k}, KCU_E{k2}_{k} (|k - k2| = 1), FCU_S{k} / FCU_E{k};
  NR    natural re-runs of every natural prompt in phase B (the instrument's batch-noise floor).
Instrument check per non-natural run (against its donor; NR against its first run) at L35, 39, 43, 47: per-head block scores of the query
actually used (rows B1 / B4 / answer, centred) - max / median |difference|, min / mean per-head correlation; max picture-mass difference; max
relative difference of the block-mean post-RoPE picture keys.
Stored per run: answer, menu logits, block scores of every head (rows B1 / B4 / answer) at L43, L47, L51, and the matcher-head mean of every cap
row at L47-63 (4 decimals); 'inst' as above.
Usage: python s915_cross.py MODEL TAG --N 8 --forms last,end --kc all --p43 1 [--n 32] [--seed S] [--device_map balanced] [--threads 2] [--tokens 2]"""
from __future__ import annotations

import argparse, collections, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s915_{}_cross.json"


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    N = a.N; KS = tuple(range(1, N + 1)); FORMS = a.forms.split(",")
    assert os.environ.get("R3_READ") == "max" and FORMS[0] == "last" and set(FORMS) <= {"last", "end"} and a.kc in ("all", "nb")
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
    P47 = list(EARLY); P43 = [l for l in EARLY if l != EARLY[-1]]
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

    QUEST = {"E": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "X": lambda q: f"What is in the {ORDW[q-1]} image from the end?",
             "S": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}
    FL = ["E"] + (["X"] if "end" in FORMS else []) + ["S"]

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
        names = {f"{f}{q}": (f, q) for f in FL for q in KS}
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
        own = SC[si]["lay"]["owner"]; QDs = {}; REF = {}
        def pq(dn, layers, part="all"):
            return {li: dict(qsub=dict(owner=own, nb=N, part=part, donors=[(1.0, QDs[dn][li])])) for li in layers}
        def mk(nm, ek, edit=None, grab=False, meta=None):
            q = capq(SC[si], edit=edit) if edit is not None else capq(SC[si])
            q["grabkb"] = set(EARLY)                                                              # block-mean post-RoPE picture keys (instrument check)
            if grab:
                q["grabq"] = set(EARLY)
            RECS[si]["meta"][nm] = meta or dict(state=ek)
            return (nm, SC[si]["enc"][ek], q, grab)
        def inst(stb, ref):
            """the run's early lookup against a reference run (its donor, or for a natural re-run its first run) at every pinned layer:
            per-head block scores of the query actually used (rows B1 / B4 / answer, centred over blocks) - max and median |difference|,
            min and mean per-head correlation over blocks; max |picture attention mass difference| per row and head; max relative
            difference of the block-mean picture keys."""
            out = {}
            for li in EARLY:
                c = stb[("cap", li)][[2, 5, 9]].float(); c0 = ref[li]["cap"]
                cc = c - c.mean(-1, keepdim=True); cc0 = c0 - c0.mean(-1, keepdim=True); d = (cc - cc0).abs()
                r = ((cc / (cc.norm(dim=-1, keepdim=True) + 1e-9)) * (cc0 / (cc0.norm(dim=-1, keepdim=True) + 1e-9))).sum(-1)
                m = (stb[("img", li)][[2, 5, 9]].float() - ref[li]["img"]).abs()
                kb = stb[("kb", li)].float(); kb0 = ref[li]["kb"]; dk = (kb - kb0).norm(dim=-1) / (kb0.norm(dim=-1) + 1e-9)
                out[str(li)] = [round(float(x_), 5) for x_ in (d.max(), d.median(), r.min(), r.mean(), m.max(), dk.max())]
            return out
        def go(items):
            for i0 in range(0, len(items), 12):                                                   # sub-passes of at most 12 runs (GPU0 memory)
                part = items[i0:i0 + 12]
                lg, st = run1([x[1] for x in part], [x[2] for x in part])
                for b, x in enumerate(part):
                    if x[3]:
                        QDs[x[0]] = {li: st[b][("q", li)] for li in EARLY}
                        REF[x[0]] = {li: dict(cap=st[b][("cap", li)][[2, 5, 9]].float(), img=st[b][("img", li)][[2, 5, 9]].float(),
                                              kb=st[b][("kb", li)].float()) for li in EARLY}
                    else:
                        m_ = RECS[si]["meta"][x[0]]
                        RECS[si].setdefault("inst", {})[x[0]] = inst(st[b], REF[m_.get("donor") or m_["ref"]])
                    store(si, x[0], lg[b], st[b])
                with CLOCK:
                    CNT[0] += len(part)
        # phase A: natural runs (every one a donor)
        for f in FL:
            go([mk(f"N_{f}{q}", f"{f}{q}", grab=True) for q in KS])
        anc = {f: {k: anchor_of(RECS[si]["mm"][f"N_{f}{k}"]["47"]) for k in KS} for f in FL if f != "S"}
        RECS[si]["anchor"] = {f: {str(k): x for k, x in d.items()} for f, d in anc.items()}
        # phase B: natural re-runs (the instrument's batch-noise floor) and pinned runs
        it = []
        for f in FL:
            for q in KS:
                it.append(mk(f"NR_{f}{q}", f"{f}{q}", meta=dict(state=f"{f}{q}", ref=f"N_{f}{q}", pins="none")))
        for f in FL:
            for q in KS:
                it.append(mk(f"SP_{f}{q}", f"{f}{q}", edit=pq(f"N_{f}{q}", P47), meta=dict(state=f"{f}{q}", donor=f"N_{f}{q}", pins="P47")))
        if "X" in FL:
            for k in KS:
                it.append(mk(f"XP_X{k}", f"X{k}", edit=pq(f"N_E{k}", P47), meta=dict(state=f"X{k}", donor=f"N_E{k}", pins="P47")))
                it.append(mk(f"XP_E{k}", f"E{k}", edit=pq(f"N_X{k}", P47), meta=dict(state=f"E{k}", donor=f"N_X{k}", pins="P47")))
        for k in KS:
            for k2 in KS:
                if k2 != k and (a.kc == "all" or abs(k - k2) == 1):
                    it.append(mk(f"KC_E{k2}_{k}", f"E{k2}", edit=pq(f"N_E{k}", P47), meta=dict(state=f"E{k2}", donor=f"N_E{k}", pins="P47")))
        for k in KS:
            ak = anc["E"][k]
            it.append(mk(f"FC_S{k}", f"S{ak}", edit=pq(f"N_E{k}", P47), meta=dict(state=f"S{ak}", donor=f"N_E{k}", pins="P47")))
            it.append(mk(f"FC_E{k}", f"E{k}", edit=pq(f"N_S{ak}", P47), meta=dict(state=f"E{k}", donor=f"N_S{ak}", pins="P47")))
        for k in KS:                                                                              # P47U: only the address part U <- the donor's
            ak = anc["E"][k]
            it.append(mk(f"SPU_E{k}", f"E{k}", edit=pq(f"N_E{k}", P47, "U"), meta=dict(state=f"E{k}", donor=f"N_E{k}", pins="P47U")))
            if "X" in FL:
                it.append(mk(f"XPU_X{k}", f"X{k}", edit=pq(f"N_E{k}", P47, "U"), meta=dict(state=f"X{k}", donor=f"N_E{k}", pins="P47U")))
                it.append(mk(f"XPU_E{k}", f"E{k}", edit=pq(f"N_X{k}", P47, "U"), meta=dict(state=f"E{k}", donor=f"N_X{k}", pins="P47U")))
            for k2 in (k - 1, k + 1):
                if 1 <= k2 <= N:
                    it.append(mk(f"KCU_E{k2}_{k}", f"E{k2}", edit=pq(f"N_E{k}", P47, "U"), meta=dict(state=f"E{k2}", donor=f"N_E{k}", pins="P47U")))
            it.append(mk(f"FCU_S{k}", f"S{ak}", edit=pq(f"N_E{k}", P47, "U"), meta=dict(state=f"S{ak}", donor=f"N_E{k}", pins="P47U")))
            it.append(mk(f"FCU_E{k}", f"E{k}", edit=pq(f"N_S{ak}", P47, "U"), meta=dict(state=f"E{k}", donor=f"N_S{ak}", pins="P47U")))
        if a.p43:
            for k in KS:
                it.append(mk(f"SP43_E{k}", f"E{k}", edit=pq(f"N_E{k}", P43), meta=dict(state=f"E{k}", donor=f"N_E{k}", pins="P43")))
                for k2 in (k - 1, k + 1):
                    if 1 <= k2 <= N:
                        it.append(mk(f"KC43_E{k2}_{k}", f"E{k2}", edit=pq(f"N_E{k}", P43), meta=dict(state=f"E{k2}", donor=f"N_E{k}", pins="P43")))
                if "X" in FL:
                    it.append(mk(f"XP43_X{k}", f"X{k}", edit=pq(f"N_E{k}", P43), meta=dict(state=f"X{k}", donor=f"N_E{k}", pins="P43")))
        go(it)
        QDs.clear(); REF.clear(); gc.collect()

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
    json.dump(dict(tag=a.tag, model=a.model, N=N, ks=list(KS), forms=FORMS, kc=a.kc, p43=a.p43, P47=P47, P43=P43, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, early_mean={str(li): hm(li) for li in EARLY}, rowsC=rowsC,
                   diff=L0["diff"], readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0],
                   rows=RECS), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=91508); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--forms", default="last,end"); ap.add_argument("--kc", default="all"); ap.add_argument("--p43", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
