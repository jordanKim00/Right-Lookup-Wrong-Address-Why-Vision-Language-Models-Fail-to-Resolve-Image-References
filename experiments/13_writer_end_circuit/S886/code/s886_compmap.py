"""S886 (registered in knowledge/DECISIONS.md before its data; after S881 / p21 (the writer's late query change carries predecessor content (+)
and anchor content (-)), S883 (the anchor heads' END part is what they read; attention changes do not set the per-scene size) and S885): which band
component writes which content?  Capture, at the read rows B1 / B4 / answer, the outputs of every band component - the token mixer (DN46, FA47,
DN48, DN49) and the MLP of layers 46-49 - in E (END(5)) and R (START(p)) pinned at p in {3, 4, 5}, the residual entering layer 51 in E and
E_BANDR (the late query map's operating point), and once per scene the post-RoPE L51 image keys.  Scenes = S872's (seed 87200, 48; the same
scenes and pins as S878 / S881 / S883-S885).  Per scene one pass of 9 runs (E, R, E_BANDR per pin).  Encoded inputs wait on the host.
Usage: python s886_compmap.py MODEL TAG [--n 48] [--device_map balanced] [--threads 3] [--tokens 2]"""
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
OUT = "/vlm/results/r3/s886_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5


class CompCapture:
    """Outputs of the band components (token mixer and MLP of each layer) at prow, for flagged samples (forward hooks after every patch tool)."""

    def __init__(self, v, layers):
        self._tl = threading.local(); self.hooks = []
        for l in layers:
            L = v.layers[l]
            mix = L.self_attn if getattr(L, "self_attn", None) is not None else L.linear_attn
            self.hooks.append(mix.register_forward_hook(self._h(l, "mix"), with_kwargs=True))
            self.hooks.append(L.mlp.register_forward_hook(self._h(l, "mlp"), with_kwargs=True))
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _h(self, l, kind):
        def h(mod, args, kwargs, out):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            for b, c in enumerate(cfgs):
                if c is not None and c.get("comp"):
                    c["out"][f"{kind}{l}"] = t[b, c["prow"]].float().cpu()
            return None
        return h


class L51Capture:
    """Residual entering layer 51 at prow (pre-hook) and, once per scene, the post-RoPE L51 keys of the image positions."""

    def __init__(self, v, layer=51):
        self._tl = threading.local(); self.v = v; self.hooks = []
        L = v.layers[layer]
        self.hooks.append(L.register_forward_pre_hook(self._pre, with_kwargs=True))
        self.hooks.append(L.self_attn.register_forward_hook(self._att, with_kwargs=True))
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
        for b, c in enumerate(cfgs):
            if c is not None:
                c["out"]["h51"] = x[b, c["prow"]].float().cpu()
        return None

    def _att(self, mod, args, kwargs, out):
        cfgs = self.cfgs
        if cfgs is None:
            return None
        for b, c in enumerate(cfgs):
            if c is None or not c.get("once"):
                continue
            hs = kwargs["hidden_states"][b].float(); cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
            S = hs.shape[0]; hd = mod.head_dim; rd = cos.shape[-1]
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            ke = torch.cat([k[..., :rd] * cos[:, None, :] + rotate_half(k[..., :rd]) * sin[:, None, :], k[..., rd:]], -1)
            ip = torch.tensor(c["img_pos"], device=hs.device)
            c["out"].update(k51=ke[ip].half().cpu(), cos51=cos[c["prow"]].cpu(), sin51=sin[c["prow"]].cpu())
        return None


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
    CAPC = CompCapture(v, band); CAP51 = L51Capture(v, 51)                                          # after TOOL (pins, component patches)
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
    def run1(inps, qs, fhs, c51=None):
        held = 1; TOK.take(1)
        try:
            for attempt in range(3):                                                             # on GPU memory exhaustion: retry alone (same batch, same outputs)
                try:
                    TOOL.set(qs); CAPC.set(fhs); CAP51.set(c51)
                    try:
                        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in batch_inputs(inps).items()}
                        lg = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                        st = TOOL.store
                    finally:
                        TOOL.set(None); CAPC.set(None); CAP51.set(None)
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
    R3 = [rowsC[i] for i in (2, 5, 9)]
    NPD = f"/vlm/results/r3/s886_{a.tag}_scenes"; os.makedirs(NPD, exist_ok=True)

    def task(si):
        lay = SC[si]["lay"]
        blk = [[i for i, o in enumerate(lay["owner"]) if o == b_] for b_ in range(N)]                # vs..ve of each picture
        img_pos = [i for b_ in range(N) for i in blk[b_]]
        inps, qs, fhs, c51, tag = [], [], [], [], []
        first = [True]
        def add(nm, ek, p, cp=None):
            inps.append(SC[si]["enc"][ek]); q = capq(SC[si], edit=pin(si, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q)
            o = {}; fh = dict(prow=R3, out=o, comp=(cp is None))
            c5 = dict(prow=R3, out=o)
            if first[0]:
                c5.update(once=True, img_pos=img_pos); first[0] = False
            fhs.append(fh); c51.append(c5); tag.append(nm)
            return len(inps) - 1
        for p in PINS:
            add(f"Ep{p}", "END5", p); iR = add(f"Rp{p}", f"START{p}", p)
            add(f"Ep{p}_E_BANDR", "END5", p, cp=dict(src=iR, comps=ALLC))
        lg, st = run1(inps, qs, fhs, c51)
        NPZ = dict(img_owner=np.array([lay["owner"][i] for i in img_pos], dtype=np.int8), img_pos=np.array(img_pos), rows3=np.array(R3))
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
            o = fhs[b]["out"]
            for kk, vv_ in o.items():
                if kk.startswith("mix") or kk.startswith("mlp"):
                    NPZ[f"{kk}_{nm}"] = vv_.numpy()
            NPZ[f"h51_{nm}"] = o["h51"].numpy()
            if "k51" in o:
                NPZ["k51"] = o["k51"].numpy(); NPZ["cos51"] = o["cos51"].numpy(); NPZ["sin51"] = o["sin51"].numpy()
        np.savez(f"{NPD}/scene{si:02d}.npz", **NPZ)
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
        TOOL.remove(); CAPC.remove(); CAP51.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY,
                   npz_dir=NPD, rows3=[rowsC[i] for i in (2, 5, 9)],
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "compmap"), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'compmap')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=1)
    main(ap.parse_args())
