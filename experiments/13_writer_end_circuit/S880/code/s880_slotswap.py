"""S880 (registered in knowledge/DECISIONS.md before its data; follows S873 A1 'the writer magnitude follows the context scene') - which pictures of
the scene set the writer's magnitude?  Hybrid scenes at the input: scene c with one picture slot b replaced by a photo of c's unshown ninth shape (the
name pool has nine shapes and every scene shows eight of them; the slot's old shape becomes the unshown one, the menu of nine names is unchanged),
everything else c's.  Scenes = S872's (seed 87200, 48; nine pictures, first eight shown, N 8); END(5) (E)
and START(p) (R) pinned at p in {3, 4, 5} with each (hybrid) scene's own natural START(p) donors (U part at 35-47), as S872 / S873.
Slots per pin: b in {p - 1 (predecessor), p (anchor), p + 1 (successor), far} (far = 7 for p 3, 1 for p 4 and 5).  Two neighbour-similarity variants (the
content-coupled writer test, S877 post hoc): DUP = the predecessor slot p - 1 holds an exact copy of the anchor picture p; SAME = it holds another
photo of the anchor's name.  In DUP / SAME the menu keeps c's nine names (the predecessor's original name stays as an unshown distractor);
answers are located by name (0 = a name not shown).
Runs per (scene, pin): base E, R, E_BANDR (the band's read-row outputs <- R) and per slot the hybrid's E, R, E_BANDR; the START(p) donors of the
base and of every hybrid in a pass of their own.  Writer effects W = T(E) - T(E_BANDR) at the address level (T1, L51-63) and the logit level
(T3) in exp/s880_analyze.py.  Pins run in --threads threads; no-grad forwards share --tokens memory tokens; the vision tower runs one thread at a
time.
Usage: python s880_slotswap.py MODEL TAG [--n 48] [--device_map balanced] [--threads 3] [--tokens 3]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s880_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5


def slots_of(p):
    return [p - 1, p, p + 1, 7 if p == 3 else 1]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


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
    FHP = FAHeadPatchPinned(v, [47]); MIP = MLPInputPath(v, [47])

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
    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps):
        TOK.take(1)
        try:
            TOOL.set(qs); FHP.set(fhs); MIP.set(mps)
            try:
                lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                st = TOOL.store
            finally:
                TOOL.set(None); FHP.set(None); MIP.set(None)
        finally:
            TOK.give(1)
        return lg, st

    rng = random.Random(a.seed); T1 = []; CNT = [0]; CLOCK = threading.Lock()
    BASE = []
    for si in range(a.n):                                                                        # S872's scenes, drawn in its order
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        BASE.append(dict(labs=labs, pics=pics, menu=menu))

    def scene_of(labs, pics, menu):
        enc = {"END5": v.encode(prompt("END", KEND, menu, N), pics[:N]), **{f"START{p}": v.encode(prompt("START", p, menu, N), pics[:N]) for p in PINS}}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
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
        for w_ in menu:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        rowsC = [qm] + list(range(uend, uend + 5)) + A4
        return dict(labs=labs, menu=menu, enc=enc, owner=owner, opt=opt, rowsC=rowsC, S=S_)

    def hybrid(c, b):
        """scene c with picture slot b (1-indexed) <- a pool photo of c's unshown ninth name (the name pool has nine shapes and every scene uses
        all nine; the slot's old name becomes the unshown one, so the menu stays valid)."""
        base = BASE[c]; labs = list(base["labs"]); old = labs[b - 1]; new_ = labs[8]
        rr = random.Random(7919 * c + 13 * b)
        pics = list(base["pics"]); pics[b - 1] = proc_image(rr.choice(pool[new_]))[0].resize((a.pixels, a.pixels))
        labs[b - 1] = new_; labs[8] = old
        return -1, labs, pics, list(base["menu"])

    def neighbour(c, p, kind):
        """slot p - 1 <- the anchor picture itself (dup) or another photo of the anchor's name (same)."""
        base = BASE[c]; labs = list(base["labs"]); pics = list(base["pics"]); lab = labs[p - 1]
        if kind == "dup":
            pics[p - 2] = base["pics"][p - 1]
        else:
            alt = [pid for pid in pool[lab]]
            rr = random.Random(10007 * c + 31 * p)
            pics[p - 2] = proc_image(rr.choice(alt))[0].resize((a.pixels, a.pixels))
        labs[p - 2] = lab
        return labs, pics, list(base["menu"])

    RECS = [dict(si=si, labs=BASE[si]["labs"], menu=BASE[si]["menu"], land={}, mlog={}, zl={}, qk47={}, donor={}) for si in range(a.n)]

    def task(arg):
        si, p = arg
        variants = [("base", None, scene_of(BASE[si]["labs"], BASE[si]["pics"], BASE[si]["menu"]))]
        for b in slots_of(p):
            d, labs, pics, menu = hybrid(si, b)
            variants.append((f"s{b}", d, scene_of(labs, pics, menu)))
            RECS[si]["donor"][f"p{p}_s{b}"] = dict(scene=d, label=labs[b - 1])
        for kind in ("dup", "same"):
            labs, pics, menu = neighbour(si, p, kind)
            variants.append((kind, None, scene_of(labs, pics, menu)))
        for _, _, sv in variants:
            assert (sv["S"], sv["rowsC"]) == (variants[0][2]["S"], variants[0][2]["rowsC"])
        rowsC = variants[0][2]["rowsC"]
        capq = lambda sv, **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=sv["owner"], nb=N, opt=sv["opt"]), **kw)
        lg0, st0 = run1([sv["enc"][f"START{p}"] for _, _, sv in variants], [capq(sv, grabq=set(EARLY)) for _, _, sv in variants],
                        [None] * len(variants), [None] * len(variants))
        pins = []
        for b_, (nm, _, sv) in enumerate(variants):
            qd = {li: st0[b_][("q", li)] for li in EARLY}
            pins.append({li: dict(qsub=dict(owner=sv["owner"], nb=N, part="U", donors=[(1.0, qd[li])])) for li in EARLY})
            w = sv["menu"][int(torch.stack([sc(lg0[b_], c) for c in sv["menu"]]).argmax())]
            RECS[si]["land"][f"p{p}_{nm}_START"] = (sv["labs"].index(w) + 1) if w in sv["labs"][:N] else 0
            T1.append(int(int(lg0[b_].argmax()) in t1ids(sv["menu"])))
        del st0
        inps, qs, fhs, tag, owners = [], [], [], [], []
        for b_, (nm, _, sv) in enumerate(variants):
            pq = pins[b_]
            iE = len(inps); inps.append(sv["enc"]["END5"]); qs.append(capq(sv, edit=pq)); tag.append((f"p{p}_{nm}_E", sv))
            iR = len(inps); inps.append(sv["enc"][f"START{p}"]); qs.append(capq(sv, edit=pq)); tag.append((f"p{p}_{nm}_R", sv))
            q_ = capq(sv, edit=pq); q_["cpatch"] = dict(src=iR, comps=ALLC); inps.append(sv["enc"]["END5"]); qs.append(q_); tag.append((f"p{p}_{nm}_E_BANDR", sv))
            fhs += [dict(rows=rowsC, edit=pq, prow=rowsC, hedits={}) for _ in range(3)]
        lg, st = run1(inps, qs, fhs, [None] * len(inps))
        for b_, (key, sv) in enumerate(tag):
            lgb, stb = lg[b_], st[b_]
            T1.append(int(int(lgb.argmax()) in t1ids(sv["menu"])))
            w = sv["menu"][int(torch.stack([sc(lgb, c) for c in sv["menu"]]).argmax())]
            RECS[si]["land"][key] = (sv["labs"].index(w) + 1) if w in sv["labs"][:N] else 0
            RECS[si]["mlog"][key] = [float(sc(lgb, x)) for x in sv["labs"][:N]]
            RECS[si]["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
            RECS[si]["qk47"][key] = stb[("cap", 47)][:, hm(47)].float().mean(1).tolist() if ("cap", 47) in stb else None
        with CLOCK:
            CNT[0] += len(tag) + len(variants)
        del st; gc.collect()

    WORK = [(si, p) for si in range(a.n) for p in PINS]
    SL = {p: slots_of(p) for p in PINS}
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · slots {SL}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                if (i_ + 1) % len(PINS) == 0:
                    print(f"[{a.tag}] scene {(i_ + 1) // len(PINS)}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        TOOL.remove(); FHP.remove(); MIP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), slots={str(p): slots_of(p) for p in PINS}, band=band, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "slotswap"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'slotswap')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=3)
    main(ap.parse_args())
