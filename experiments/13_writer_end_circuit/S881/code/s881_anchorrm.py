"""S881 (registered in knowledge/DECISIONS.md before its data; S878 post hoc p20: the writer's late query update = a shared front-ward shift + a
component along the anchor picture's own key that sets the per-scene magnitude, r .72-.84) - causal test: remove only that component.
Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R), E_BANDR (the band's read-row outputs <- R) pinned at p in {3, 4, 5}.  At every late
full-attention layer (51, 55, 59, 63) a forward hook recomputes the read rows' attention (rows qm, B0-B4, A4; float32, q_norm / k_norm, partial
RoPE, GQA, output gate, o_proj - as exp/c_qpath_b.py) of the edited runs and writes it back:
  E_AR   q' = q_E - ((q_E - q_B) . u_p) u_p, per row and head: the writer-induced query difference (this run minus E_BANDR of the same pass, at
         the same layer) loses its component along u_p = the unit, block-centred block-mean key of the anchor picture p (post-RoPE, the
         head's key-value head) - the anchor-suppression component;
  E_PR   the same along u_(p-1) (the predecessor's key: the boost component);
  E_FR   the same along a far picture's key (picture 7 for p 3, 1 for p 4 and 5; control);
  E_ID   the recompute with nothing removed (exactness check: must reproduce E).
The edited runs' block scores (LSE over each picture's tokens, every head, rows B1 / B4 / answer) are recomputed with the edited queries and
stored per layer.  One forward per scene holds 3 pins x (E, R, E_BANDR, E_AR, E_PR, E_FR, E_ID).  Scenes run in --threads threads; no-grad
forwards share --tokens memory tokens; the vision tower runs one thread at a time.
Usage: python s881_anchorrm.py MODEL TAG [--n 48] [--device_map balanced] [--threads 2] [--tokens 2]"""
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
OUT = "/vlm/results/r3/s881_{}_{}.json"
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

    def post_edit(li):
        def h(mod, args, kwargs, out):
            ed = getattr(_TL, "ed", None)
            if not ed:
                return None
            hs = kwargs["hidden_states"]; cos_, sin_ = kwargs["position_embeddings"]
            hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos_.shape[-1]; grp = mod.num_key_value_groups
            rows = _TL.rows; R = len(rows); own = _TL.own
            t0 = (out[0] if isinstance(out, tuple) else out).clone()
            Wo = mod.o_proj.weight.float()
            qcache = {}
            def qof(b_):
                if b_ not in qcache:
                    x = hs[b_].float(); cos, sin = cos_[b_].float(), sin_[b_].float()
                    qg = torch.nn.functional.linear(x[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
                    q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
                    qcache[b_] = (rope(q, cos[rows], sin[rows], rd), qg[..., hd:])
                return qcache[b_]
            for b_, e in ed.items():
                x = hs[b_].float(); cos, sin = cos_[b_].float(), sin_[b_].float(); S_ = x.shape[0]
                k = mod.k_norm(torch.nn.functional.linear(x, mod.k_proj.weight.float()).view(S_, -1, hd).to(mod.k_norm.weight.dtype)).float()
                vv = torch.nn.functional.linear(x, mod.v_proj.weight.float()).view(S_, -1, hd)
                kh = torch.arange(nh, device=x.device) // grp
                ke = rope(k, cos, sin, rd)[:, kh]; V = vv[:, kh]                                 # [S, H, hd]
                qe, gate = qof(b_)
                if e["kind"] != "id":
                    ot = torch.tensor(own, device=x.device)
                    kb = torch.stack([ke[ot == bb].mean(0) for bb in range(N)])                  # [8, H, hd]
                    kb = kb - kb.mean(0, keepdim=True); u = kb[e["blk"]]; u = u / (u.norm(dim=-1, keepdim=True) + 1e-9)
                    qr, _ = qof(e["ref"])
                    dq = qe - qr                                                                # [R, H, hd]
                    qe = qe - (dq * u[None]).sum(-1, keepdim=True) * u[None]
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                mask = torch.arange(S_, device=x.device)[None, :] > torch.tensor(rows, device=x.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
                o = torch.sigmoid(gate) * torch.einsum("ths,shd->thd", att, V)
                new = torch.nn.functional.linear(o.reshape(R, -1), Wo, None if mod.o_proj.bias is None else mod.o_proj.bias.float())
                t0[b_, rows] = new.to(t0.dtype)
                ot = torch.tensor(own, device=x.device)
                z = torch.stack([torch.logsumexp(lg[[2, 5, 9]][:, :, ot == bb], -1) for bb in range(N)], -1)      # [3, H, 8]
                _TL.zout.setdefault(b_, {})[li] = z.cpu()
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h
    for li in LATEQ:
        hooks.append(v.layers[li].self_attn.register_forward_hook(post_edit(li), with_kwargs=True))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps, ed=None, rows=None, own=None):
        TOK.take(1)
        try:
            TOOL.set(qs); FHP.set(fhs); MIP.set(mps); _TL.ed = ed; _TL.rows = rows; _TL.own = own; _TL.zout = {}
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

    def task(si):
        inps, qs, fhs, tag = [], [], [], []
        def add(nm, p, ek, cp=None):
            inps.append(SC[si]["enc"][ek]); q = capq(SC[si], edit=pin(si, p))
            if cp is not None:
                q["cpatch"] = cp
            qs.append(q); fhs.append(dict(rows=rowsC, edit=pin(si, p), prow=rowsC, hedits={})); tag.append(nm)
            return len(inps) - 1
        ed = {}
        for p in PINS:
            iE = add(f"Ep{p}", p, "END5"); iR = add(f"Rp{p}", p, f"START{p}")
            iB = add(f"Ep{p}_E_BANDR", p, "END5", cp=dict(src=iR, comps=ALLC))
            far = 7 if p == 3 else 1
            for nm, blk in (("AR", p - 1), ("PR", p - 2), ("FR", far - 1)):
                ed[add(f"Ep{p}_{nm}", p, "END5")] = dict(kind="rm", blk=blk, ref=iB)
            ed[add(f"Ep{p}_ID", p, "END5")] = dict(kind="id")
        lg, st = run1(inps, qs, fhs, [None] * len(inps), ed=ed, rows=rowsC, own=SC[si]["lay"]["owner"])
        zo = _TL.zout
        for b, nm in enumerate(tag):
            store(si, nm, lg[b], st[b])
            if b in zo:
                RECS[si].setdefault("zedit", {})[nm] = {str(li): zo[b][li].float().tolist() for li in zo[b]}
        with CLOCK:
            CNT[0] += len(tag)
        gc.collect()

    WORK = list(range(a.n))
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, WORK)):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), lateq=LATEQ, band=band, read=READ, early=EARLY,
                   late=LATE, late_mean={str(li): hm(li) for li in LATE}, pos0=POS[0], rowsC=rowsC, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "anchorrm"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'anchorrm')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200)
    ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=4)
    main(ap.parse_args())
