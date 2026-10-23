"""S894 (registered in knowledge/DECISIONS.md before its data; predictions frozen and hashed before the run; after the post hoc finding that END
pulls FA47's self-address heads one block end-ward - anchor -> successor - although the address part of the query is pinned to START's, and
S891: the composition map predicts FA47 attention interventions) - is the end-ward pull of the self-address heads part of the writer's step?
FA47 read-row attention edits (exp/s891 FAAttnEdit plus a partial move): R_SA2S40 / R_SA2S100 (START: 40% / all of the self-address heads'
anchor mass moved onto the successor block, the successor's own distribution scaled up), E_SA_S2A (END: the self-address heads' successor mass
moved back onto the anchor), E_ALL_S2A (END: all heads' successor mass -> anchor), R_ID.  Scenes = S872's (seed 87200, 48; S885's), END(5) and
START(p) pinned at p in {3, 4, 5}.  Per (scene, pin) one pass of 8 runs: R, E, E_BANDR, R_SA2S40, R_SA2S100, E_SA_S2A, E_ALL_S2A, R_ID.
Usage: python s894_endpull.py MODEL TAG [--n 48] [--device_map balanced] [--threads 3] [--tokens 2]"""
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
OUT = "/vlm/results/r3/s894_{}_{}.json"
N = 8; PINS = (3, 4, 5); KEND = 5
SA = [1, 4, 12, 17, 20]; ANTI = [0, 2, 5, 13, 15]; H16 = [16]


class FAAttnEdit(FAHeadPatchPinned):
    """Read-row attention of a full-attention layer recomputed (float32, pins re-applied) with the attention mass of chosen heads on a source
    block rescaled or moved: cfg aedit = dict(heads=[...], blk=positions of the anchor block, kind='scale' (factor f; the other positions are
    rescaled to keep the total) or 'move' (to=positions of the target block; the target's own distribution is scaled up by the moved mass)).
    The rows get t + sum_h W_O[:, h] (o_new,h - o_this,h); factor 1 adds exactly zero."""

    def _post(self, l):
        def h(mod, args, kwargs, out):
            from c_qpath import QPath
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and "aedit" in c]
            if not todo:
                return None
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            nh, hd = mod.config.num_attention_heads, mod.head_dim
            Wo = mod.o_proj.weight.float().view(-1, nh, hd)
            for b, c in todo:
                rows, prow = c["rows"], c["prow"]; ai = [rows.index(x_) for x_ in prow]
                hs = kwargs["hidden_states"][b].float()
                cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
                S = hs.shape[0]; rd = cos.shape[-1]; R = len(rows)
                qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
                k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
                vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
                kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
                K = k[:, kh]; V = vv[:, kh]
                qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
                ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
                ed = c.get("edit", {}).get(l)
                for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):
                    if "qsub" in e_:
                        qe = QPath._qsub(QPath, qe, ke, e_["qsub"], cos[rows], sin[rows], rd)
                qe = qe[ai]; sg = torch.sigmoid(gate[ai])
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(prow, device=hs.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)             # [P, H, S]
                ae = c["aedit"]; hh = torch.tensor(ae["heads"], device=hs.device); blk = torch.tensor(ae["blk"], device=hs.device)
                att2 = att.clone(); sub = att[:, hh]                                                  # [P, h, S]
                mB = sub[:, :, blk].sum(-1, keepdim=True)                                             # [P, h, 1]
                if ae["kind"] == "pmove":                                                             # move a fraction phi of the block's mass onto the target block
                    phi = float(ae["phi"]); to = torch.tensor(ae["to"], device=hs.device); new = sub.clone()
                    mT = sub[:, :, to].sum(-1, keepdim=True)
                    new[:, :, blk] = sub[:, :, blk] * (1 - phi)
                    new[:, :, to] = sub[:, :, to] * (1 + phi * mB / mT.clamp_min(1e-12))
                elif ae["kind"] == "scale":
                    f = float(ae["factor"]); new = sub.clone()
                    new[:, :, blk] = sub[:, :, blk] * f
                    oth = torch.ones(S, dtype=torch.bool, device=hs.device); oth[blk] = False
                    mO = sub[:, :, oth].sum(-1, keepdim=True)
                    new[:, :, oth] = sub[:, :, oth] * (1 + (1 - f) * mB / mO.clamp_min(1e-12))
                else:
                    to = torch.tensor(ae["to"], device=hs.device); new = sub.clone()
                    mT = sub[:, :, to].sum(-1, keepdim=True)
                    new[:, :, blk] = 0.0
                    new[:, :, to] = sub[:, :, to] * (1 + mB / mT.clamp_min(1e-12))
                att2[:, hh] = new
                o0 = sg * torch.einsum("rhs,shd->rhd", att, V); o1 = sg * torch.einsum("rhs,shd->rhd", att2, V)
                corr = torch.einsum("rhd,ohd->ro", o1 - o0, Wo)
                t2[b, prow] = (t[b, prow].float() + corr).to(t2.dtype)
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
    VED = FAAttnEdit(v, [47])                                                                      # after TOOL: pins, then component patches, then the attention edit
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

    def blkpos(si, b1):                                                                            # vs..ve positions of picture b1 (1-based)
        return [i for i, o in enumerate(SC[si]["lay"]["owner"]) if o == b1 - 1]

    def task(arg):
        si, p = arg
        A_, S_ = blkpos(si, p), blkpos(si, p + 1)
        inps, qs, fhs, tag = [], [], [], []
        def add(nm, ek, cp=None, ae=None):
            inps.append(SC[si]["enc"][ek]); q = capq(SC[si], edit=pin(si, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q)
            fhs.append(dict(rows=rowsC, edit=pin(si, p), prow=rowsC, aedit=ae) if ae is not None else None); tag.append(nm)
            return len(inps) - 1
        iE = add(f"Ep{p}", "END5"); iR = add(f"Rp{p}", f"START{p}")
        add(f"Ep{p}_E_BANDR", "END5", cp=dict(src=iR, comps=ALLC))
        add(f"Rp{p}_SA2S40", f"START{p}", ae=dict(heads=SA, blk=A_, kind="pmove", phi=.4, to=S_))
        add(f"Rp{p}_SA2S100", f"START{p}", ae=dict(heads=SA, blk=A_, kind="move", to=S_))
        add(f"Ep{p}_SA_S2A", "END5", ae=dict(heads=SA, blk=S_, kind="move", to=A_))
        add(f"Ep{p}_ALL_S2A", "END5", ae=dict(heads=list(range(24)), blk=S_, kind="move", to=A_))
        add(f"Rp{p}_ID", f"START{p}", ae=dict(heads=SA, blk=A_, kind="scale", factor=1.0))
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
                   selfaddr=SA, anti=ANTI, h16=H16,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "endpull"), "w"))
    print(f"[{a.tag}] peak GB " + " ".join(f"{d_}:{torch.cuda.max_memory_allocated(d_) / 2**30:.1f}" for d_ in range(torch.cuda.device_count())), flush=True)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'endpull')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--tokens", type=int, default=2)
    main(ap.parse_args())
