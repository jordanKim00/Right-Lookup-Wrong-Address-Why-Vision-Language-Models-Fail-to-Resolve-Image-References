"""S913 (registered in knowledge/DECISIONS.md before its data; the advisor's final test, 10-22) - frozen prediction of single natural END(k)
answers from the run's own early-address profile.  Data collector: one N per run (N 5 and 6 are unseen by the frozen models, N 8 is seen with
fresh scenes); one scene = nine pictures with all nine names as the menu, the first N shown; phase A only - natural END(k) for every k = 1..N
and START(q) for every q (no pins); stored per run: the L47 matcher-head block scores of the ten read rows (the predictors' input), late block
scores, menu logits, answers.  New scenes per N (seed 91300 + N), 48 per N.
Usage: python s913_grid.py MODEL TAG --N 6 --ks 1,2,3,4,5,6 [--n 48] [--device_map balanced] [--threads 2] [--tokens 2]"""
from __future__ import annotations

import argparse, collections, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import FAHeadPatchPinned
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s913_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5
KS = (2, 3, 4, 5, 6)                                                                        # overridden by --N / --ks


class FA47QueryPath:
    """Path patch into FA47's read-row queries / gates from layer 46's components (thread-local; see the module doc).  cfg per sample:
    rows (the QPath rows), edit (pins), prow (rows to patch), qsrc=batch index or None, qcomps=subset of ('mix46', 'mlp46'), out=dict for the
    per-head block attention mass at mrows (positions) with parts (block positions)."""

    def __init__(self, v):
        self._tl = threading.local(); self.v = v; self.hooks = []
        L46, L47 = v.layers[46], v.layers[47]
        self.hooks.append(L46.linear_attn.register_forward_hook(self._grab("mix46"), with_kwargs=True))
        self.hooks.append(L46.mlp.register_forward_hook(self._grab("mlp46"), with_kwargs=True))
        self.hooks.append(L47.register_forward_pre_hook(self._res, with_kwargs=True))
        self.hooks.append(L47.self_attn.register_forward_hook(self._post, with_kwargs=True))
        self.ln = L47.input_layernorm
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))
    st = property(lambda self: getattr(self._tl, "st", None), lambda self, x: setattr(self._tl, "st", x))

    def set(self, cfgs):
        self.cfgs = cfgs; self.st = {}

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _grab(self, name):
        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            rows = next(c["prow"] for c in self.cfgs if c is not None)
            self.st[name] = t[:, rows].float()
            return None
        return h

    def _res(self, mod, args, kwargs):
        if self.cfgs is None:
            return None
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        rows = next(c["prow"] for c in self.cfgs if c is not None)
        self.st["res47"] = x[:, rows].float()
        return None

    def _post(self, mod, args, kwargs, out):
        from c_qpath import QPath
        cfgs = self.cfgs
        if cfgs is None:
            return None
        t = out[0] if isinstance(out, tuple) else out
        t2 = None
        nh, hd = mod.config.num_attention_heads, mod.head_dim
        Wo = mod.o_proj.weight.float().view(-1, nh, hd)
        for b, c in enumerate(cfgs):
            if c is None:
                continue
            rows, prow = c["rows"], c["prow"]; ai = [rows.index(x_) for x_ in prow]
            hs = kwargs["hidden_states"][b].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
            S = hs.shape[0]; rd = cos.shape[-1]; R = len(rows)
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
            kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
            K = k[:, kh]; V = vv[:, kh]
            ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
            def heads(xrows):
                qg = torch.nn.functional.linear(xrows, mod.q_proj.weight.float()).view(R, nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
                qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
                ed = c.get("edit", {}).get(47)
                for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):
                    if "qsub" in e_:
                        qe = QPath._qsub(QPath, qe, ke, e_["qsub"], cos[rows], sin[rows], rd)
                qe = qe[ai]; sg = torch.sigmoid(gate[ai])
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(prow, device=hs.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
                return att, sg * torch.einsum("rhs,shd->rhd", att, V)
            if c.get("qsrc") is None:                                                             # measurement only
                att_used, _ = heads(hs[rows])
            else:                                                                                 # both from the stored residual (an identity source adds exactly zero)
                res0 = self.st["res47"][b]; res = res0.clone()
                if c["qsrc"] != b:
                    for nm in c["qcomps"]:
                        res = res - self.st[nm][b] + self.st[nm][c["qsrc"]]
                att0, o0 = heads(self._ln32(res0)); att1, o1 = heads(self._ln32(res))
                corr = torch.einsum("rhd,ohd->ro", o1 - o0, Wo)
                if t2 is None:
                    t2 = t.clone()
                t2[b, prow] = (t[b, prow].float() + corr).to(t2.dtype)
                att_used = att1
            mi = [prow.index(x_) for x_ in c["mrows"]]
            c["out"]["mass"] = torch.stack([att_used[mi][:, :, torch.tensor(pp, device=hs.device)].sum(-1) for pp in c["parts"]], -1).cpu()   # [3, H, 8]
        if t2 is None:
            return None
        return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2

    def _ln32(self, x):
        w = self.ln.weight.float(); eps = getattr(self.ln, "eps", 1e-6)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def unbits(a):
    return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main(a):
    global N, KS
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
    band = [46, 47, 48, 49]
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
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
    FHP = FAHeadPatchPinned(v, [47]); QP = FA47QueryPath(v)                                          # after TOOL: pins, component patches, head swaps, then the query path
    _TL = threading.local()
    hooks = []

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

    LATEQ = [51, 55, 59, 63]

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, qps):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs); FHP.set(fhs); QP.set(qps)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None); FHP.set(None); QP.set(None)
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
    for si in range(a.n):                                                                        # S872's scenes, drawn in its order
        labs = rng.sample(labels, max(9, N))
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        names = {**{f"END{k}": ("END", k) for k in KS}, **{f"START{q}": ("START", q) for q in range(1, N + 1)}}
        SC.append(dict(si=si, labs=labs, menu=menu, enc={nm: cpu_(v.encode(prompt(f, q, menu, N), pics[:N])) for nm, (f, q) in names.items()}))
    # ---- positions (identical layout in every scene: fixed picture size, single-token names)
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
        tail0 = min(i for i in range(S_) if len({ids[nm][i] for nm in ids}) > 1)
        return dict(S=S_, qm=qm, uend=uend, owner=owner, opt=opt, rowsC=rowsC, tail0=tail0, POS=list(range(tail0, S_)), vs=vs, ve=ve)
    for sc_ in SC:
        sc_["lay"] = layout(sc_)
    L0 = SC[0]["lay"]
    for sc_ in SC:
        assert (sc_["lay"]["S"], sc_["lay"]["tail0"], sc_["lay"]["rowsC"]) == (L0["S"], L0["tail0"], L0["rowsC"]), "layouts differ across scenes"
    POS, rowsC = L0["POS"], L0["rowsC"]
    capq = lambda sc_, **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=sc_["lay"]["owner"], nb=N, opt=sc_["lay"]["opt"]), **kw)

    LAND0 = [dict() for _ in range(a.n)]

    RECS = [dict(si=si, labs=SC[si]["labs"], menu=SC[si]["menu"], land=dict(LAND0[si]), mlog={}, zl={}, qk47={}, subpass={}) for si in range(a.n)]

    def store(si, key, lgb, stb):
        T1.append(int(int(lgb.argmax()) in t1ids(SC[si]["menu"])))
        w = SC[si]["menu"][int(torch.stack([sc(lgb, c) for c in SC[si]["menu"]]).argmax())]
        RECS[si]["land"][key] = SC[si]["labs"].index(w) + 1
        RECS[si]["mlog"][key] = [float(sc(lgb, x)) for x in SC[si]["labs"][:N]]
        RECS[si]["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
        RECS[si]["qk47"][key] = stb[("cap", 47)][:, hm(47)].float().mean(1).tolist() if ("cap", 47) in stb else None

    R3 = [rowsC[i] for i in (2, 5, 9)]

    def task(si):
        lay = SC[si]["lay"]; blk = [[i for i, o in enumerate(lay["owner"]) if o == b_] for b_ in range(N)]
        def mk(nm, ek, cp=None, fh=None, qsrc=None, qcomps=()):
            q = capq(SC[si])
            if cp is not None:
                q["cpatch"] = cp
            o = {}
            return (nm, SC[si]["enc"][ek], q, dict(rows=rowsC, edit={}, prow=rowsC, hedits=fh or {}),
                    dict(rows=rowsC, edit={}, prow=rowsC, qsrc=qsrc, qcomps=list(qcomps), mrows=R3, parts=blk, out=o))
        def go(items):
            lg, st = run1([x[1] for x in items], [x[2] for x in items], [x[3] for x in items], [x[4] for x in items])
            for b, x in enumerate(items):
                store(si, x[0], lg[b], st[b])
                RECS[si].setdefault("mass47", {})[x[0]] = x[4]["out"]["mass"].tolist()
            with CLOCK:
                CNT[0] += len(items)
        # phase A: natural runs
        if N >= 9:                                                                                 # two batches at long layouts (GPU0 memory)
            go([mk(f"N_E{k}", f"END{k}") for k in KS]); go([mk(f"N_S{q}", f"START{q}") for q in range(1, N + 1)])
        else:
            go([mk(f"N_E{k}", f"END{k}") for k in KS] + [mk(f"N_S{q}", f"START{q}") for q in range(1, N + 1)])
        anchors = {k: anchor_of(RECS[si]["qk47"][f"N_E{k}"]) for k in KS}
        RECS[si]["anchor"] = {str(k): v for k, v in anchors.items()}
        # phase B: per k, references at the natural anchor
        items = []
        if not a.phaseB:
            gc.collect(); return
        for k in KS:
            a_ = anchors[k]
            if not (2 <= a_ <= N):
                continue
            base = len(items)
            items.append(mk(f"B{k}_E", f"END{k}")); items.append(mk(f"B{k}_R", f"START{a_}"))
            iR = base + 1
            items.append(mk(f"B{k}_E_BANDR", f"END{k}", cp=dict(src=iR, comps=ALLC)))
            items.append(mk(f"B{k}_ALL47R", f"END{k}", fh={47: [dict(heads=list(range(24)), op="swap", src=iR)]}))
            items.append(mk(f"B{k}_QM", f"END{k}", qsrc=iR, qcomps=("mlp46",)))
        if items:
            go(items)
        gc.collect()

    def anchor_of(q):
        q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
        c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]

    WORK = list(range(a.n))
    for d_ in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d_)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); QP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY,
                   selfaddr=[1, 4, 12, 17, 20], anti=[0, 2, 5, 13, 15], ks=list(KS),
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "grid"), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'grid')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--phaseB", type=int, default=0)
    ap.add_argument("--N", type=int, default=7); ap.add_argument("--ks", default="3,4,5")
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=3)
    a_ = ap.parse_args(); a_.seed = a_.seed if a_.seed is not None else 91300 + a_.N
    main(a_)
