"""S873 (registered in knowledge/DECISIONS.md before its data; causal tests of S872's Jacobian lens - advisor review 4 'lens readouts must match
causal patching' and review 3 'same address, different writer'; the swap test of Gurnee et al. 2026) - does each scene's writer effect follow
the pre-writer END - START state difference or the scene it acts in, do the lens's address directions carry the writer's input, and are the
lens's address coordinates causally used?  Scenes = S872's (seed 87200, 48 scenes: nine pictures, all nine names as the menu, the first eight
shown); END(5) (E) and START(p) (R) pinned at p in {3, 4, 5} with each scene's own START(p) donors (U part at 35-47), as S872.
Lens vectors (read at start from S872 v4's per-scene npz, --s872 TAG): for a scene of one half (0-23 / 24-47) the other half's mean E-pass
  gradient at layer 46's input (every position from the first differing question token on, POS) of the 16 objectives (z_1..8 = L51 block
  scores, L_1..8 = menu logits; S872's definitions) for the same pin.
Edits at the input of layer 46 (POS rows only; positions before POS are identical in every prompt of a scene):
  X_cd    R_c + (E_d - R_d): context scene c (its START(p) prompt and pins) given the END - START difference of scene d (d = c for the control
          X_cc = E_c; donors d1 = c + 1, d2 = c + 17 mod 48, their E and R run in the same pass with their own prompts and pins);
  X_SPAN  R_c + P (E_c - R_c), P = the orthogonal projector on the 16 lens vectors;  X_PERP  R_c + (I - P)(E_c - R_c);
  SWAP    E_c's state h -> h + a V (sigma(c) - c), V = [v_z(p-1), v_z(p)] (lens vectors), c = argmin |V c - h|, sigma swaps the two coordinates
          (Gurnee et al.'s coordinate swap; a = 1 and 2 = SWAP, SWAP2); CFAR the same with [v_z(p+1), v_z(p+2)]; CRAND with two seeded Gaussian
          directions of the same norms; R_SWAP the swap on R_c.
  *_BANDR the same run with the band's read-row outputs (mix / mlp 46-49) <- R_c's (the writer removed; S862-S872's E_BANDR).
Per (scene, pin) two equal-size sub-passes of 12 runs, each holding E_c and R_c (bit-identity refs).  Pins run in --threads threads; no-grad
forwards share --tokens memory tokens; the vision tower runs one thread at a time.  Saved per run: answer, menu logits, L47 / L51-63 caps.
Usage: python s873_transplant.py MODEL TAG [--n 48] [--s872 s872v4_27b] [--device_map balanced] [--threads 3] [--tokens 4]"""
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
OUT = "/vlm/results/r3/s873_{}_{}.json"
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

    def pre46(mod, args, kwargs):
        """per-thread edits of layer 46's input at POS: xfer (base + T(dE - dR)) and cswap (coordinate swap)."""
        ed = getattr(_TL, "ed", None)
        if not ed:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        pos = _TL.pos
        x = hs.clone()
        src = {b_: hs[b_, pos].float() for b_ in range(hs.shape[0])}
        for b_, e in ed.items():
            if e["kind"] == "xfer":
                d = src[e["dE"]] - src[e["dR"]]
                if e.get("P") is not None:
                    Q = e["P"].to(d.device)                                                     # [K, P*E] orthonormal rows
                    pr = (Q.T @ (Q @ d.reshape(-1))).reshape(d.shape)
                    d = pr if e["proj"] == "span" else d - pr
                x[b_, pos] = (src[e["base"]] + d).to(x.dtype)
            elif e["kind"] == "cswap":
                h = src[b_]; V = e["V"].to(h.device)                                           # [2, P, E]
                Vf = V.reshape(2, -1); hf = h.reshape(-1)
                G = Vf @ Vf.T; c = torch.linalg.solve(G, Vf @ hf)
                cs = torch.stack([c[1], c[0]])
                x[b_, pos] = (h + e["alpha"] * ((cs - c) @ Vf).reshape(h.shape)).to(x.dtype)
        if args:
            return (x,) + tuple(args[1:]), kwargs
        kw = dict(kwargs); kw["hidden_states"] = x
        return args, kw
    hooks.append(v.layers[46].register_forward_pre_hook(pre46, with_kwargs=True))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps, ed=None, pos=None):
        TOK.take(1)
        try:
            TOOL.set(qs); FHP.set(fhs); MIP.set(mps); _TL.ed = ed; _TL.pos = pos
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

    # ---- lens vectors from S872 v4 (other-half means of the E-pass gradients at layer 46's input)
    S872D = f"/vlm/results/r3/s872_{a.s872}_scenes"
    J72 = json.load(open(f"/vlm/results/r3/s872_{a.s872}_jlens.json"))
    assert [r["labs"] for r in J72["rows"]][:a.n] == [sc_["labs"] for sc_ in SC], "S873 scenes must be S872's"
    half = lambda s: 0 if s < 24 else 1
    SUMG = {(h, p): None for h in (0, 1) for p in PINS}; CNTG = {h: 0 for h in (0, 1)}
    for s in range(len(J72["rows"])):
        Z = np.load(f"{S872D}/scene{s:02d}.npz"); CNTG[half(s)] += 1
        for p in PINS:
            g = unbits(Z[f"p{p}_g_E_in46"]).astype(np.float64)                                   # [16, P, E]
            SUMG[(half(s), p)] = g if SUMG[(half(s), p)] is None else SUMG[(half(s), p)] + g
        del Z
    LV = {}
    for h in (0, 1):
        for p in PINS:
            hs_ = 1 - h if CNTG[1 - h] > 0 else h                                                   # (a software smoke may hold one half only)
            Gm = SUMG[(hs_, p)] / CNTG[hs_]                                                         # the other half's mean
            Qm, _ = np.linalg.qr(Gm.reshape(16, -1).T)                                             # orthonormal basis of the 16 lens vectors
            LV[(h, p)] = dict(V=torch.tensor(Gm[:8], dtype=torch.float32), Q=torch.tensor(Qm.T, dtype=torch.float32))
    del SUMG
    rgen = torch.Generator().manual_seed(8730)
    E_ = LV[(0, PINS[0])]["V"].shape[-1]
    RAND = {p: torch.randn(2, len(POS), E_, generator=rgen) for p in PINS}
    print(f"[{a.tag}] lens vectors from {S872D}: halves {CNTG} · POS {len(POS)} · read {READ} · late {LATE}", flush=True)

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
        si, p = arg
        c = si; d1, d2 = (si + SHIFTS[0]) % a.n, (si + SHIFTS[1]) % a.n
        lv = LV[(half(si), p)]
        inps, qs, fhs, tag, ed = [], [], [], [], {}
        def add(nm, sj, ek, cp=None):
            inps.append(SC[sj]["enc"][ek]); q = capq(SC[sj], edit=pin(sj, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q); fhs.append(dict(rows=rowsC, edit=pin(sj, p), prow=rowsC, hedits={})); tag.append(nm)
            return len(inps) - 1
        def flush(sp):
            lg, st = run1(inps, qs, fhs, [None] * len(inps), ed=dict(ed), pos=POS)
            for b, nm in enumerate(tag):
                keys = [nm] + ([nm.split("@")[0]] if nm.endswith("@0") else [])
                for key in keys:
                    store(si, key, lg[b], st[b])
                RECS[si]["subpass"][nm] = sp
            with CLOCK:
                CNT[0] += len(tag)
        # sub-pass 0: transplants
        iE = add(f"Ep{p}@0", c, "END5"); iR = add(f"Rp{p}@0", c, f"START{p}")
        add(f"Ep{p}_E_BANDR", c, "END5", cp=dict(src=iR, comps=ALLC))
        i_ = add(f"Ep{p}_X_cc", c, f"START{p}"); ed[i_] = dict(kind="xfer", base=iR, dE=iE, dR=iR)
        for j, dd in enumerate((d1, d2)):
            jE = add(f"Ep{p}_E_d{j + 1}", dd, "END5"); jR = add(f"Ep{p}_R_d{j + 1}", dd, f"START{p}")
            i_ = add(f"Ep{p}_X_d{j + 1}", c, f"START{p}"); ed[i_] = dict(kind="xfer", base=iR, dE=jE, dR=jR)
            i_ = add(f"Ep{p}_X_d{j + 1}_BANDR", c, f"START{p}", cp=dict(src=iR, comps=ALLC)); ed[i_] = dict(kind="xfer", base=iR, dE=jE, dR=jR)
        RECS[si].setdefault("donors", {})[f"p{p}"] = [d1, d2]
        flush(0)
        # sub-pass 1: lens span / complement, coordinate swaps
        inps.clear(); qs.clear(); fhs.clear(); tag.clear(); ed.clear()
        iE = add(f"Ep{p}@1", c, "END5"); iR = add(f"Rp{p}@1", c, f"START{p}")
        for proj in ("span", "perp"):
            i_ = add(f"Ep{p}_X_{proj.upper()}", c, f"START{p}"); ed[i_] = dict(kind="xfer", base=iR, dE=iE, dR=iR, P=lv["Q"], proj=proj)
            i_ = add(f"Ep{p}_X_{proj.upper()}_BANDR", c, f"START{p}", cp=dict(src=iR, comps=ALLC))
            ed[i_] = dict(kind="xfer", base=iR, dE=iE, dR=iR, P=lv["Q"], proj=proj)
        Vsw = torch.stack([lv["V"][p - 2], lv["V"][p - 1]]); Vfar = torch.stack([lv["V"][p], lv["V"][p + 1]])
        nr = Vsw.reshape(2, -1).norm(dim=1)
        Rr = RAND[p].reshape(2, -1); Vrand = (Rr / Rr.norm(dim=1, keepdim=True) * nr[:, None]).reshape(Vsw.shape)
        for nm, V, al, ek in (("SWAP", Vsw, 1.0, "END5"), ("SWAP2", Vsw, 2.0, "END5"), ("CFAR", Vfar, 1.0, "END5"), ("CRAND", Vrand, 1.0, "END5"),
                              ("R_SWAP", Vsw, 1.0, f"START{p}")):
            i_ = add(f"Ep{p}_{nm}", c, ek); ed[i_] = dict(kind="cswap", V=V, alpha=al)
        add(f"PADp{p}@1_11", c, "END5")
        flush(1)
        gc.collect()

    WORK = [(si, p) for si in range(a.n) for p in PINS]
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                if (i_ + 1) % len(PINS) == 0:
                    print(f"[{a.tag}] scene {(i_ + 1) // len(PINS)}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), shifts=list(SHIFTS), s872=a.s872, band=band, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "transplant"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'transplant')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200); ap.add_argument("--s872", default="s872v4_27b")
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=4)
    main(ap.parse_args())
