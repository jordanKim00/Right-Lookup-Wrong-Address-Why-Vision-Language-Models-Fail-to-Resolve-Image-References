"""S906 (registered in knowledge/DECISIONS.md before its data) - is the END mode vector the same for every ordinal?  S904: at the layer-35
input the read rows' END(5) - START(5) difference is one scene-independent direction per row, and adding it to START(5) produces END(5)'s late
shift and one-step-back answers.  If END(k) = START(k)'s state + a k-independent mode vector, the k = 5 vector added to START(2) / START(3)
should produce END(2) / END(3)'s own answers (the end-relative position computed downstream from the ordinal already in the state).
Natural runs (no pins).  Calibration: 24 scenes (seed 87310, disjoint from the test scenes), natural END(k) and START(k) for k in {2, 3, 5}:
d_k = mean(h_END(k) - h_START(k)) at the layer-35 input, read rows rowsC.  Test: 48 new scenes (seed 89600); per scene one pass of 19 runs:
for k in {2, 3, 5}: E_k, S_k, S_k + d_k, S_k + r_k (random direction per row with d_k's row norms), E_k - d_k; for k in {2, 3}: S_k + d_5,
E_k - d_5.
Usage: python s906_modek.py MODEL TAG [--n 48] [--ncal 24] [--device_map balanced] [--threads 3] [--tokens 3]"""
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
OUT = "/vlm/results/r3/s906_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5
KS = (2, 3, 5)


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


class RowAdd:
    """Thread-local pre-hook at one decoder layer: cfgs[b] = dict(rows=[positions], vec=float32 tensor [len(rows), D] (any device), coef=float) adds
    coef * vec to sample b's residual at those rows at the layer input (coef 0 adds exactly zero: the row is rewritten with its own value);
    cap=True stores the rows' input residual (float32, host) in cfgs[b]['out']."""

    def __init__(self, v, layer):
        self._tl = threading.local(); self.hooks = [v.layers[layer].register_forward_pre_hook(self._pre, with_kwargs=True)]
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _pre(self, mod, args, kwargs):
        cfgs = self.cfgs
        if cfgs is None:
            return None
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        x2 = None
        for b, c in enumerate(cfgs):
            if c is None:
                continue
            r_ = torch.tensor(c["rows"], device=x.device)
            if c.get("cap"):
                c["out"] = x[b, r_].float().cpu()
            if "vec" in c:
                if x2 is None:
                    x2 = x.clone()
                x2[b, r_] = (x[b, r_].float() + float(c["coef"]) * c["vec"].to(x.device)).to(x.dtype)
        if x2 is None:
            return None
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs); kwargs["hidden_states"] = x2; return args, kwargs
        return (x2,) + tuple(args[1:]), kwargs


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
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=[35])
    RADD = RowAdd(v, 35)
    VED = FAValueEdit(v, [47])                                                                     # unused here (no value edits); kept for the shared run1
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
    def run1(inps, qs, fhs, adds=None):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs); VED.set(fhs); RADD.set(adds)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None); VED.set(None); RADD.set(None)
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
        names = {**{f"END{k}": ("END", k) for k in KS}, **{f"START{k}": ("START", k) for k in KS}}
        SC.append(dict(si=si, labs=labs, menu=menu, enc={nm: cpu_(v.encode(prompt(f, q, menu, N), pics[:N])) for nm, (f, q) in names.items()}))
    rngc = random.Random(a.cal_seed); CAL = []
    for ci in range(a.ncal):                                                                     # calibration scenes (disjoint seed)
        labs = rngc.sample(labels, 9)
        pics = [proc_image(rngc.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rngc.shuffle(menu)
        CAL.append({nm: cpu_(v.encode(prompt(f, q, menu, N), pics[:N])) for nm, (f, q) in {**{f"END{k}": ("END", k) for k in KS}, **{f"START{k}": ("START", k) for k in KS}}.items()})
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

    # ---- no pins (natural runs); START landings come from the test passes
    LAND0 = [dict() for _ in range(a.n)]

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

    # ---- calibration: the mean END(k) - START(k) difference at the layer-35 input, read rows (natural runs, no pins)
    assert all(len(c_["END5"]["input_ids"][0]) == L0["S"] for c_ in CAL), "calibration layout differs"
    def calib(ci):
        cfg = [dict(rows=rowsC, cap=True) for _ in range(2 * len(KS))]
        run1([CAL[ci][f"{f}{k}"] for k in KS for f in ("END", "START")], [capq(SC[0]) for _ in cfg], [None] * len(cfg), cfg)
        return {k: cfg[2 * i]["out"] - cfg[2 * i + 1]["out"] for i, k in enumerate(KS)}
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        DIFF = list(ex.map(calib, range(a.ncal)))
    DV, RV, CALINFO = {}, {}, dict(energy_share={}, cos_scene_to_mean={}, cos_between={})
    g = torch.Generator().manual_seed(906)
    for k in KS:
        Dst = torch.stack([d_[k] for d_ in DIFF]); DV[k] = Dst.mean(0)
        CALINFO["energy_share"][str(k)] = float((DV[k] ** 2).sum(-1).mean() / (Dst ** 2).sum(-1).mean())
        CALINFO["cos_scene_to_mean"][str(k)] = float(torch.nn.functional.cosine_similarity(Dst.reshape(a.ncal, -1), DV[k].reshape(1, -1)).mean())
        rnd = torch.randn(DV[k].shape, generator=g); RV[k] = rnd / rnd.norm(dim=-1, keepdim=True) * DV[k].norm(dim=-1, keepdim=True)
    for i, k in enumerate(KS):
        for k2 in KS[i + 1:]:
            CALINFO["cos_between"][f"{k}_{k2}"] = float(torch.nn.functional.cosine_similarity(DV[k].reshape(1, -1), DV[k2].reshape(1, -1)))
            CALINFO["cos_between"][f"{k}_{k2}_rows"] = torch.nn.functional.cosine_similarity(DV[k], DV[k2], dim=-1).tolist()
    CALINFO["row_norm"] = {str(k): DV[k].norm(dim=-1).tolist() for k in KS}
    torch.save({"d": {k: DV[k] for k in KS}, "r": {k: RV[k] for k in KS}, "rows": rowsC, "layer": 35},
               OUT.format(a.tag, "dvec").replace(".json", ".pt"))                                   # the vectors themselves, for weight-level reads
    print(f"[{a.tag}] calibration " + json.dumps({k_: v_ for k_, v_ in CALINFO.items() if k_ != "row_norm"})[:400], flush=True)

    def task(si):
        inps, qs, adds, tag = [], [], [], []
        def add(nm, ek, vec=None, coef=0.0):
            inps.append(SC[si]["enc"][ek]); qs.append(capq(SC[si])); adds.append(dict(rows=rowsC, vec=vec, coef=coef) if vec is not None else None); tag.append(nm)
        for k in KS:
            add(f"E{k}", f"END{k}"); add(f"S{k}", f"START{k}"); add(f"S{k}_ADD", f"START{k}", DV[k], 1.0)
            add(f"S{k}_RAND", f"START{k}", RV[k], 1.0); add(f"E{k}_SUB", f"END{k}", DV[k], -1.0)
            if k != 5:
                add(f"S{k}_ADD5", f"START{k}", DV[5], 1.0); add(f"E{k}_SUB5", f"END{k}", DV[5], -1.0)
        lg, st = run1(inps, qs, [None] * len(inps), adds)
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
        with CLOCK:
            CNT[0] += len(tag)
        gc.collect()

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
        TOOL.remove(); VED.remove(); RADD.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY,
                   ks=list(KS), calibration=CALINFO, ncal=a.ncal, cal_seed=a.cal_seed,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "modek"), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'modek')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=89600); ap.add_argument("--ncal", type=int, default=24); ap.add_argument("--cal_seed", type=int, default=87310)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
