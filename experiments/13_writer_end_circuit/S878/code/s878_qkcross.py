"""S878 (registered in knowledge/DECISIONS.md before its data; advisor reviews 5 and 6, 10-21: the Query-Key crossed counterfactual) - is the
scene dependence of the writer's address effect (S873 A1; L51 post hoc p19) made by the writer's query update or by the scene's image keys?
Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R) and E_BANDR (E with the band's read-row outputs <- R) pinned at p in {3, 4, 5}.
One forward per scene holds the nine runs (3 pins x E, R, E_BANDR).  At the late full-attention layers 51, 55, 59, 63 a forward hook recomputes
from the layer's own inputs (input-normed hidden states, q_norm / k_norm, partial RoPE, GQA, as exp/c_qpath_b.py): per run the post-RoPE
queries of every head and the sigmoid output gate at the rows B1 / B4 / answer, and the post-RoPE keys and the values of every key-value head
at the nine menu-word positions; once per scene (they precede the question, so they are the same in every run - checked) the keys and values
at every image position.  Caps (late block scores) as S872.  Offline (exp/s878_analyze.py) the writer's address effect is recomputed for any
pair (query source A, key source B): W(qA, KB) = T1(q_A^E, K_B) - T1(q_A^BANDR, K_B), T1 = the matcher-head mean over rows of
LSE_{t in block p-1} - LSE_{t in block p} of the scaled q . k (token level, exactly the readout's block score).
Saved: results/r3/s878_<tag>_scenes/scene<si>.npz (float16; queries float32) and results/r3/s878_<tag>_qk.json (answers, caps, layout).
Usage: python s878_qkcross.py MODEL TAG [--n 48] [--device_map balanced] [--threads 2] [--tokens 3]"""
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
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s878_{}_{}.json"
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

    LATEQ = [51, 55, 59, 63]

    def rope(x, cos, sin, rd):
        return torch.cat([x[..., :rd] * cos[:, None, :] + rotate_half(x[..., :rd]) * sin[:, None, :], x[..., rd:]], -1)

    def post_cap(li):
        def h(mod, args, kwargs, out):
            cap = getattr(_TL, "cap", None)
            if not cap:
                return None
            hs = kwargs["hidden_states"]; cos_, sin_ = kwargs["position_embeddings"]
            hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos_.shape[-1]
            for b_ in range(hs.shape[0]):
                x = hs[b_].float(); cos, sin = cos_[b_].float(), sin_[b_].float(); S_ = x.shape[0]
                rr = cap["rows3"]
                qg = torch.nn.functional.linear(x[rr], mod.q_proj.weight.float()).view(len(rr), nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = torch.sigmoid(qg[..., hd:])
                qe = rope(q, cos[rr], sin[rr], rd)
                need = cap["img"] if b_ == 0 else []
                pos = need + cap["opt"]
                k = mod.k_norm(torch.nn.functional.linear(x[pos], mod.k_proj.weight.float()).view(len(pos), -1, hd).to(mod.k_norm.weight.dtype)).float()
                vv = torch.nn.functional.linear(x[pos], mod.v_proj.weight.float()).view(len(pos), -1, hd)
                ke = rope(k, cos[pos], sin[pos], rd)
                st = cap["out"].setdefault(b_, {})
                st[("q", li)] = qe.cpu(); st[("g", li)] = gate.to(torch.float16).cpu()
                n_ = len(need)
                if n_:
                    st[("kimg", li)] = ke[:n_].to(torch.float16).cpu(); st[("vimg", li)] = vv[:n_].to(torch.float16).cpu()
                st[("kopt", li)] = ke[n_:].to(torch.float16).cpu(); st[("vopt", li)] = vv[n_:].to(torch.float16).cpu()
                if b_ == 1 and cap.get("check"):                                          # the image keys are the same in every run
                    k1 = mod.k_norm(torch.nn.functional.linear(x[cap["img"]], mod.k_proj.weight.float()).view(len(cap["img"]), -1, hd).to(mod.k_norm.weight.dtype)).float()
                    cap["out"]["kdiff"] = max(cap["out"].get("kdiff", 0.0), float((rope(k1, cos[cap["img"]], sin[cap["img"]], rd).cpu() - cap["out"][0][("kimg", li)].float()).abs().max()))
            return None
        return h
    for li in LATEQ:
        hooks.append(v.layers[li].self_attn.register_forward_hook(post_cap(li), with_kwargs=True))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps, cap=None):
        TOK.take(1)
        try:
            TOOL.set(qs); FHP.set(fhs); MIP.set(mps); _TL.cap = cap
            try:
                lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                st = TOOL.store
            finally:
                TOOL.set(None); FHP.set(None); MIP.set(None); _TL.cap = None
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

    OUTD = OUT.format(a.tag, "scenes").replace(".json", "")
    os.makedirs(OUTD, exist_ok=True)
    def task(si):
        inps, qs, fhs, tag = [], [], [], []
        def add(nm, p, ek, cp=None):
            inps.append(SC[si]["enc"][ek]); q = capq(SC[si], edit=pin(si, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q); fhs.append(dict(rows=rowsC, edit=pin(si, p), prow=rowsC, hedits={})); tag.append(nm)
            return len(inps) - 1
        idx = {}
        for p in PINS:
            iE = add(f"Ep{p}", p, "END5"); iR = add(f"Rp{p}", p, f"START{p}")
            iB = add(f"Ep{p}_E_BANDR", p, "END5", cp=dict(src=iR, comps=ALLC)); idx[p] = (iE, iR, iB)
        own = SC[si]["lay"]["owner"]; img = [t for t in range(len(own)) if own[t] >= 0]
        cap = dict(rows3=[rowsC[2], rowsC[5], rowsC[9]], img=img, opt=SC[si]["lay"]["opt"], out={}, check=True)
        lg, st = run1(inps, qs, fhs, [None] * len(inps), cap=cap)
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
        O = cap["out"]; NPZ = dict(img_owner=np.array([own[t] for t in img], dtype=np.int8), opt=np.array(SC[si]["lay"]["opt"]))
        for li in LATEQ:
            NPZ[f"kimg_{li}"] = O[0][("kimg", li)].numpy(); NPZ[f"vimg_{li}"] = O[0][("vimg", li)].numpy()
            for p in PINS:
                for nm, b in zip(("E", "R", "B"), idx[p]):
                    NPZ[f"q_{nm}{p}_{li}"] = O[b][("q", li)].numpy(); NPZ[f"g_{nm}{p}_{li}"] = O[b][("g", li)].numpy()
                    NPZ[f"kopt_{nm}{p}_{li}"] = O[b][("kopt", li)].numpy(); NPZ[f"vopt_{nm}{p}_{li}"] = O[b][("vopt", li)].numpy()
        RECS[si]["kdiff"] = O.get("kdiff")
        np.savez(f"{OUTD}/scene{si:02d}.npz", **NPZ)
        with CLOCK:
            CNT[0] += len(tag)
        gc.collect()

    WORK = list(range(a.n))
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs · image-key check {RECS[i_].get('kdiff')}", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY, npz_dir=OUTD,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "qk"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'qk')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=4)
    main(ap.parse_args())
