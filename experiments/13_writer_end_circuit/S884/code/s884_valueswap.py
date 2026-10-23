"""S884 (registered in knowledge/DECISIONS.md before its data; after S881 / S878 post hoc p21: the writer's late query change has the sign
structure of (predecessor content - anchor content) in key space, and the late matcher heads find pictures by content (S880 DUP: START's own
late T1 collapses between identical pictures)) - the query side, causally: does the late address follow the CONTENT the read rows read at
FA47?  Value transplant at the read rows only: FA47's attention of the ten read rows is recomputed (float32, pins re-applied, keys and
attention pattern unchanged) with the VALUES of one picture's 64 image-patch tokens replaced token by token by another picture's; image tokens and every
other position keep their own computation, so the late image keys are untouched.  Scenes = S872's (seed 87200, 48); END(5) (E) and START(p)
(R) pinned at p in {3, 4, 5}; far slot f = 7 for p 3, 1 for p 4 / 5 (S880's), second far slot f2 = 8 for p 3, 7 for p 4 / 5.
Per (scene, pin) one pass of 23 runs: E, R, E_BANDR (band read-row outputs <- R) and, for each value edit X, E_X, R_X, E_BANDR_X (band <-
R_X, its FA47 part re-copied after R_X's edit):
  VID  identity map (instrument: must reproduce E / E_BANDR bit for bit)
  VPF  predecessor block (p - 1) <- far block f's values      VAF  anchor block (p) <- f's values
  VSW  predecessor <-> anchor values exchanged                 VFF  far block f <- f2's values (control: a picture the writer barely reads)
  VPA  predecessor block <- the anchor's values (the writer-side counterpart of S880's DUP: the late keys keep two different pictures)
plus EARLY (E, R): the anchor block's values <- f's at every pinned early-lookup layer FA35 / 39 / 43 / 47 (does the late lookup dereference the
content the early lookup copied?).  Late block scores (rows B1 / B4 / answer, every head) at 51-63 and the L47 head-mean block scores stored.
Usage: python s884_valueswap.py MODEL TAG [--n 48] [--device_map balanced] [--threads 2] [--tokens 2]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import FAHeadPatchPinned
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s884_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5
FAR = {3: 7, 4: 1, 5: 1}; FAR2 = {3: 8, 4: 7, 5: 7}
EARLY_V = (35, 39, 43, 47)


class FAValueEdit(FAHeadPatchPinned):
    """Read-row attention outputs of a full-attention layer recomputed with edited values (keys, queries with the sample's pins re-applied, and
    the attention pattern unchanged): the written rows are corrected by sum_h W_O[:, h] (o_new,h - o_this,h), o = sigmoid(gate) * (att V), with
    V_new[dst] = V[src] for each (dst, src) position list.  cfg keys: rows / edit / prow as FAHeadPatchPinned, vmap={layer: [(dst, src), ...]},
    recopy={layer: batch index} (after the edits, this sample's prow rows <- that sample's edited rows: keeps a component patch consistent
    with an edited source).  An identity map adds exactly zero."""

    def _two(self, mod, l, b, kwargs, prow, vmap):
        from c_qpath import QPath
        c = self.cfgs[b]; rows = c["rows"]
        ai = [rows.index(t) for t in prow]
        hs = kwargs["hidden_states"][b].float()
        cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
        S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]; R = len(rows)
        qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
        k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
        vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
        vv2 = vv.clone()
        for dst, src in vmap:
            vv2[torch.tensor(dst, device=hs.device)] = vv[torch.tensor(src, device=hs.device)]
        kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
        K = k[:, kh]; V = vv[:, kh]; V2 = vv2[:, kh]
        qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
        ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
        ed = c.get("edit", {}).get(l)
        for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):
            if "qsub" not in e_:
                continue
            sub = e_.get("erows")
            if sub is None:
                qe = QPath._qsub(QPath, qe, ke, e_["qsub"], cos[rows], sin[rows], rd)
            else:
                qe = qe.clone(); cr_, sr_ = cos[rows][sub], sin[rows][sub]
                qe[sub] = QPath._qsub(QPath, qe[sub], ke, e_["qsub"], cr_, sr_, rd)
        qe = qe[ai]; gt = gate[ai]
        lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
        mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(prow, device=hs.device)[:, None]
        att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
        sg = torch.sigmoid(gt)
        return torch.einsum("rhs,shd->rhd", att, V) * sg, torch.einsum("rhs,shd->rhd", att, V2) * sg

    def _post(self, l):
        def h(mod, args, kwargs, out):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and l in c.get("vmap", {})]
            rec = [(b, c) for b, c in enumerate(cfgs) if c is not None and l in c.get("recopy", {})]
            if not todo and not rec:
                return None
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            nh, hd = mod.config.num_attention_heads, mod.head_dim
            Wo = mod.o_proj.weight.float().view(-1, nh, hd)
            for b, c in todo:
                prow = c["prow"]
                o0, o1 = self._two(mod, l, b, kwargs, prow, c["vmap"][l])
                corr = torch.einsum("rhd,ohd->ro", o1 - o0, Wo)
                t2[b, prow] = (t[b, prow].float() + corr).to(t2.dtype)
            for b, c in rec:
                s_ = c["recopy"][l]; t2[b, c["prow"]] = t2[s_, c["prow"]]
            self.calls += 1
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def unbits(a):
    return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main(a):
    assert os.environ.get("R3_READ") == "max"
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
    VED = FAValueEdit(v, [35, 39, 43, 47])                                                         # after TOOL: pins, then component patches, then value edits
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
    def run1(inps, qs, fhs):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs); VED.set(fhs)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None); VED.set(None)
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
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        names = {"END5": ("END", KEND), **{f"START{p}": ("START", p) for p in PINS}}
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

    # ---- pin donors for every scene (natural START passes)
    QD = {}
    def donors(si):
        sc_ = SC[si]
        lg0, st0 = run1([sc_["enc"][f"START{p}"] for p in PINS], [capq(sc_, grabq=set(EARLY)) for _ in PINS], [None] * len(PINS))
        QD[si] = {p: {li: st0[b][("q", li)] for li in EARLY} for b, p in enumerate(PINS)}
        land = {}
        for b, p in enumerate(PINS):
            w = SC[si]["menu"][int(torch.stack([sc(lg0[b], c) for c in SC[si]["menu"]]).argmax())]
            land[f"START{p}"] = SC[si]["labs"].index(w) + 1
            T1.append(int(int(lg0[b].argmax()) in t1ids(SC[si]["menu"])))
        return land
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        LAND0 = list(ex.map(donors, range(a.n)))
    pin = lambda si, p: {li: dict(qsub=dict(owner=SC[si]["lay"]["owner"], nb=N, part="U", donors=[(1.0, QD[si][p][li])])) for li in EARLY}

    RECS = [dict(si=si, labs=SC[si]["labs"], menu=SC[si]["menu"], land=dict(LAND0[si]), mlog={}, zl={}, qk47={}, subpass={}) for si in range(a.n)]

    def store(si, key, lgb, stb):
        T1.append(int(int(lgb.argmax()) in t1ids(SC[si]["menu"])))
        w = SC[si]["menu"][int(torch.stack([sc(lgb, c) for c in SC[si]["menu"]]).argmax())]
        RECS[si]["land"][key] = SC[si]["labs"].index(w) + 1
        RECS[si]["mlog"][key] = [float(sc(lgb, x)) for x in SC[si]["labs"][:N]]
        RECS[si]["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
        RECS[si]["qk47"][key] = stb[("cap", 47)][:, hm(47)].float().mean(1).tolist() if ("cap", 47) in stb else None

    def bpos(si, b1):                                                                              # image-patch positions of picture b1 (1-based; no vs / ve)
        return list(range(SC[si]["lay"]["vs"][b1 - 1] + 1, SC[si]["lay"]["ve"][b1 - 1]))

    def task(arg):
        si, p = arg
        f, f2 = FAR[p], FAR2[p]
        P_, A_, F_, F2_ = bpos(si, p - 1), bpos(si, p), bpos(si, f), bpos(si, f2)
        assert len(P_) == len(A_) == len(F_) == len(F2_)
        EDITS = {"VID": {47: [(P_, P_)]}, "VPF": {47: [(P_, F_)]}, "VAF": {47: [(A_, F_)]}, "VSW": {47: [(P_, A_), (A_, P_)]}, "VFF": {47: [(F_, F2_)]},
                 "VPA": {47: [(P_, A_)]}}
        inps, qs, fhs, tag = [], [], [], []
        def add(nm, ek, cp=None, vm=None, rc=None):
            inps.append(SC[si]["enc"][ek]); q = capq(SC[si], edit=pin(si, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q)
            fh = dict(rows=rowsC, edit=pin(si, p), prow=rowsC)
            if vm is not None:
                fh["vmap"] = vm
            if rc is not None:
                fh["recopy"] = rc
            fhs.append(fh if (vm is not None or rc is not None) else None); tag.append(nm)
            return len(inps) - 1
        iE = add(f"Ep{p}", "END5"); iR = add(f"Rp{p}", f"START{p}")
        add(f"Ep{p}_E_BANDR", "END5", cp=dict(src=iR, comps=ALLC))
        for nm, vm in EDITS.items():
            add(f"Ep{p}_{nm}", "END5", vm=vm); iRX = add(f"Rp{p}_{nm}", f"START{p}", vm=vm)
            add(f"Ep{p}_{nm}_E_BANDR", "END5", cp=dict(src=iRX, comps=ALLC), rc={47: iRX})
        ev = {l: [(A_, F_)] for l in EARLY_V}
        RECS[si].setdefault("vmaps", {})[f"p{p}"] = dict(VID=[[p - 1, p - 1]], VPF=[[p - 1, f]], VAF=[[p, f]], VSW=[[p - 1, p], [p, p - 1]], VFF=[[f, f2]], VPA=[[p - 1, p]],
                                                        EARLY=[[p, f]], layers=dict(edits=[47], early=list(EARLY_V)), n_tokens=len(P_))
        add(f"Ep{p}_EARLY", "END5", vm=ev); add(f"Rp{p}_EARLY", f"START{p}", vm=ev)
        lg, st = run1(inps, qs, fhs)
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
        with CLOCK:
            CNT[0] += len(tag)
        gc.collect()

    WORK = [(si, p) for si in range(a.n) for p in PINS]
    for d_ in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d_)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                if (i_ + 1) % len(PINS) == 0:
                    print(f"[{a.tag}] scene {(i_ + 1) // len(PINS)}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); VED.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY,
                   far=FAR, far2=FAR2, early_v=list(EARLY_V),
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "valueswap"), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'valueswap')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
