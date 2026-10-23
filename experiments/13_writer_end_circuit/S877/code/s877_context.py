"""S877 (registered in knowledge/DECISIONS.md before its data; follows S873 A1 'the writer magnitude follows the context scene' and the post hoc
p19 'the context dependence is already present at L51') - which part of the context sets the per-scene writer magnitude: the band's processing
of the scene's pictures (layers 46-49 read the image positions through FA47 and the DeltaNet memory) or the late lookup's key geometry (L51 on)?
Scenes = S872's (seed 87200, 48); END(5) (E) / START(p) (R) pinned at p in {3, 4, 5} as S872 / S873; donors d = c + 1, c + 17 (mod 48).
The prefix = every position before the first differing question token (system text + the eight image blocks + the stated count); it is the
same in E_d, R_d and E_d_BANDR (positions before the question do not see it), so one donor run carries it.
Edits (per run, at a decoder layer's input; positions in the prefix only):
  H46     E_c with the prefix at layer 46's input <- scene d's (the band and everything after see d's pictures; c's question / menu rows keep
          their own states); H46R = R_c with the same edit; H46_BANDR = H46 with the band's read-row outputs <- H46R's (the writer removed in the
          same hybrid context);
  H51     E_c with the prefix at layer 51's input <- d's (the band ran in c's context; the late lookup reads d's picture keys / values);
          H51_BANDR = E_c_BANDR with the same edit.
Outcomes: address-level writer effects W_L = T1_L(X) - T1_L(X_BANDR) at L51 / 55 / 59 / 63 (T1 = z_(p-1) - z_p, matcher-head mean block
scores, rows B1 / B4 / answer) - content-agnostic (block scores by position), so they stay defined when the pictures are another scene's;
natural W_c, W_d from E / E_BANDR of the same pass.  Per (scene, pin, donor) one sub-pass of 11 runs: E_c, R_c, E_c_BANDR, R_d (donor), E_d,
E_d_BANDR, H46, H46R, H46_BANDR, H51, H51_BANDR.  Pins x donors run in --threads threads; no-grad forwards share --tokens memory tokens.
Usage: python s877_context.py MODEL TAG [--n 48] [--device_map balanced] [--threads 3] [--tokens 4]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s877_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5
SHIFTS = (1, 17)


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
    FHP = FAHeadPatchPinned(v, [47])
    MIP = MLPInputPath(v, [47])
    _TL = threading.local()
    hooks = []

    class _Tokens:
        def __init__(self, n):
            self.n = n; self.c = threading.Condition()
        def take(self, k):
            with self.c:
                while self.n < k:
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

    def pre_prefix(layer):
        def h(mod, args, kwargs):
            ed = getattr(_TL, "ed", None)
            if not ed:
                return None
            todo = [(b_, e) for b_, e in ed.items() if e["layer"] == layer]
            if not todo:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            x = hs.clone(); n0 = _TL.pre
            for b_, e in todo:
                x[b_, :n0] = hs[e["src"], :n0]
            if args:
                return (x,) + tuple(args[1:]), kwargs
            kw = dict(kwargs); kw["hidden_states"] = x
            return args, kw
        return h
    hooks.append(v.layers[46].register_forward_pre_hook(pre_prefix(46), with_kwargs=True))
    hooks.append(v.layers[51].register_forward_pre_hook(pre_prefix(51), with_kwargs=True))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps, ed=None, pre=None):
        TOK.take(1)
        try:
            TOOL.set(qs); FHP.set(fhs); MIP.set(mps); _TL.ed = ed; _TL.pre = pre
            try:
                lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                st = TOOL.store
            finally:
                TOOL.set(None); FHP.set(None); MIP.set(None); _TL.ed = None
        finally:
            TOK.give(1)
        return lg, st

    rng = random.Random(a.seed); T1 = []; CNT = [0]; CLOCK = threading.Lock()
    SC = []
    for si in range(a.n):                                                                        # S872's scenes, drawn in its order
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        names = {"END5": ("END", KEND), **{f"START{p}": ("START", p) for p in PINS}}
        SC.append(dict(si=si, labs=labs, menu=menu, enc={nm: v.encode(prompt(f, q, menu, N), pics[:N]) for nm, (f, q) in names.items()}))
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
        return dict(S=S_, qm=qm, uend=uend, owner=owner, opt=opt, rowsC=rowsC, tail0=tail0, POS=list(range(tail0, S_)))
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
        lg0, st0 = run1([sc_["enc"][f"START{p}"] for p in PINS], [capq(sc_, grabq=set(EARLY)) for _ in PINS], [None] * len(PINS), [None] * len(PINS))
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

    def task(arg):
        si, p, j = arg
        c = si; dd = (si + SHIFTS[j]) % a.n
        inps, qs, fhs, tag, ed = [], [], [], [], {}
        def add(nm, sj, ek, cp=None):
            inps.append(SC[sj]["enc"][ek]); q = capq(SC[sj], edit=pin(sj, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q); fhs.append(dict(rows=rowsC, edit=pin(sj, p), prow=rowsC, hedits={})); tag.append(nm)
            return len(inps) - 1
        iE = add(f"Ep{p}@{j}", c, "END5"); iR = add(f"Rp{p}@{j}", c, f"START{p}")
        add(f"Ep{p}_d{j + 1}_E_BANDR", c, "END5", cp=dict(src=iR, comps=ALLC))
        iRd = add(f"Ep{p}_d{j + 1}_Rd", dd, f"START{p}"); iEd = add(f"Ep{p}_d{j + 1}_Ed", dd, "END5")
        add(f"Ep{p}_d{j + 1}_Ed_BANDR", dd, "END5", cp=dict(src=iRd, comps=ALLC))
        i46 = add(f"Ep{p}_d{j + 1}_H46", c, "END5"); ed[i46] = dict(layer=46, src=iRd)
        i46r = add(f"Ep{p}_d{j + 1}_H46R", c, f"START{p}"); ed[i46r] = dict(layer=46, src=iRd)
        i_ = add(f"Ep{p}_d{j + 1}_H46_BANDR", c, "END5", cp=dict(src=i46r, comps=ALLC)); ed[i_] = dict(layer=46, src=iRd)
        i_ = add(f"Ep{p}_d{j + 1}_H51", c, "END5"); ed[i_] = dict(layer=51, src=iRd)
        i_ = add(f"Ep{p}_d{j + 1}_H51_BANDR", c, "END5", cp=dict(src=iR, comps=ALLC)); ed[i_] = dict(layer=51, src=iRd)
        RECS[si].setdefault("donors", {})[f"p{p}_d{j + 1}"] = dd
        lg, st = run1(inps, qs, fhs, [None] * len(inps), ed=dict(ed), pre=POS[0])
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
            RECS[si]["subpass"][nm] = j
        with CLOCK:
            CNT[0] += len(tag)
        gc.collect()

    WORK = [(si, p, j) for si in range(a.n) for p in PINS for j in range(len(SHIFTS))]
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                if (i_ + 1) % (len(PINS) * len(SHIFTS)) == 0:
                    print(f"[{a.tag}] scene {(i_ + 1) // (len(PINS) * len(SHIFTS))}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), shifts=list(SHIFTS), band=band, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "context"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'context')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=4)
    main(ap.parse_args())
