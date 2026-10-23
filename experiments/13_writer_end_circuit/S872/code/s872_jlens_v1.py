"""S872 (registered in knowledge/DECISIONS.md before its data; advisor review 4, 10-21: an address Jacobian lens / writer-effect lens, after the
Jacobian lens of Gurnee et al. 2026) - does the model's own sensitivity, applied to each scene's pre-writer END - START state difference, compute
that scene's writer effect?  One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown).
END(5) (E) and START(p) (R) pinned at p in {3, 4, 5} (U part at 35-47 <- the natural START(p)); p 4 = identity (target 4), p 5 = one step.
Objectives (16): z_b = the L51 block log-sum-exp score of picture b (matcher-head mean, rows B1 / B4 / answer averaged; b 1..8) and L_i = the
  answer logit of picture i's menu word (i 1..8; the spacing variant that is larger in E).  T1 = z_(p-1) - z_p, T2 = z_p - z_(p+1) and
  T3 = L_(p-1) - L_p (S869's d1 pair) are linear in these.
No-grad passes per (scene, pin): (i) [E, R] with the residual captured at every position from the first differing question token on (POS) at the
  inputs of layers 35..51 and after the token mixer of layers 35..50 (mid), and FA47's per-head outputs at the ten read rows; (ii) causal arms in
  equal-size sub-passes (each holding E and R): the band's read-row outputs <- R (E_BANDR, the writer of S862-S871) or at every POS position
  (E_BANDR_POS); single band components <- R at the read rows; FA47 head groups <- R; the residual at layer 46's input <- R at the read rows, the
  question tail, the menu, or every POS position (the last is R itself, a control); pre-band module groups (35-45) <- R at POS.
Gradient passes (vector-Jacobian products through the model's own bf16 computation, pins active, batch 1): E (leaf = layer 35's input of the E
  prompt), R (leaf = layer 35's input of the R prompt), M (E prompt; leaf = layer 46's input with POS set to the midpoint of E's and R's states);
  for every objective, torch.autograd.grad to every captured residual point, and the read-row band pathway J_B,rows^T g50 (the gradient of
  g50(rows) . (h50 - h46)(rows) with g50 held fixed).  The pin's U basis is built from detached keys: the image keys precede POS, so the exact
  gradient through it is zero and forward values are unchanged.
Saved per scene (results/r3/s872_<tag>/scene<si>.npz; bf16 bit patterns as int16): E and R states at 13 points; gradients of the 16 objectives at
  those points (E, R) and at the 10 points from layer 46 on (M); the read-row band pathway (E, R, M); FA47 per-head output differences E - R;
  scene-specific attribution scalars for every module and point of 35..51 and every position group.  JSON: answers, menu logits, block scores.
Usage: python s872_jlens.py MODEL TAG [--n 48] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
import c_qpath
from c_qpath_k import QPathK
from c_tools import rotate_half
from c_headpatch import FAHeadPatchPinned, MLPInputPath

_qsub0 = c_qpath.QPath._qsub


def _qsub_det(self, qe, ke, spec, cr, sr, rd):                 # same values; no gradient through the key-built U basis (keys precede POS)
    return _qsub0(self, qe, ke.detach(), spec, cr, sr, rd)


c_qpath.QPath._qsub = _qsub_det

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s872_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]
REST47 = [h for h in range(24) if h not in ANTI + H16 + SELFADDR]
N = 8; PINS = (3, 4, 5); KEND = 5
INL = list(range(35, 52)); MIDL = list(range(35, 51))
PTS = [("in", l) for l in INL] + [("mid", l) for l in MIDL]                                  # 33 residual points
SAVE = [("in", 35), ("in", 39), ("in", 43), ("in", 46), ("mid", 46), ("in", 47), ("mid", 47), ("in", 48), ("mid", 48), ("in", 49), ("mid", 49),
        ("in", 50), ("in", 51)]
SAVE_M = [q for q in SAVE if q[1] >= 46]
MODS = [("mix", l) for l in MIDL] + [("mlp", l) for l in MIDL]                               # 32 modules: output point mid_l / in_(l+1)
GROUPS = ("tail", "qm", "menu", "B", "A", "all")
SPECS = [("E_BANDR", "BAND"), ("E_BANDR_POS", "BANDPOS"), ("E_MIX46", "MIX46"), ("E_MLP46", "MLP46"), ("E_FA47", "MIX47"), ("E_MLP47", "MLP47"),
         ("E_MIX48", "MIX48"), ("E_MLP48", "MLP48"), ("E_MIX49", "MIX49"), ("E_MLP49", "MLP49"), ("E_H16", "H16"), ("E_ANTI", "ANTI"),
         ("E_SELFADDR", "SELFADDR"), ("E_REST47", "REST47"), ("E_RES46ROWS", "RESROWS"), ("E_RES46TAIL", "RESTAIL"), ("E_RES46MENU", "RESMENU"),
         ("E_RES46POS", "RESPOS"), ("E_PREFA", "PREFA"), ("E_PREDN", "PREDN"), ("E_PREMLP", "PREMLP")]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def bits(t):
    return t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy()


def rope(x, cos, sin, rd):
    return torch.cat([x[..., :rd] * cos[:, None, :] + rotate_half(x[..., :rd]) * sin[:, None, :], x[..., rd:]], -1)


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S863_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    band = a.band
    F47 = [l for l in band if l in FA_L][0]; L51 = LATE[0]
    assert (F47, L51, band) == (47, 51, [46, 47, 48, 49]) and EARLY[0] == 35
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    PRE = list(range(35, 46))
    COMPS = {"BAND": ALLC, **{f"{kd.upper()}{l}": [(l, kd)] for l in band for kd in ("mix", "mlp")}}
    HEADS = {"H16": H16, "ANTI": ANTI, "SELFADDR": SELFADDR, "REST47": REST47}
    PREG = {"PREFA": [(l, "mix") for l in PRE if l in FA_L], "PREDN": [(l, "mix") for l in PRE if l not in FA_L], "PREMLP": [(l, "mlp") for l in PRE]}
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    assert NH == 24
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
    assert len(labels) >= 9
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=list(range(35, 50)), rpatch_layers=[46])
    FHP = FAHeadPatchPinned(v, [F47])                                                     # after TOOL: corrects the rows its pins wrote
    MIP = MLPInputPath(v, [F47])
    GR = dict(mode=None, t={}, st={}, z=None)
    hooks = []

    def pre_in(l):
        def h(mod, args, kwargs):
            m = GR["mode"]
            if m is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            if m == "grab":
                GR["st"][("in", l)] = hs[:2, GR["pos"]].detach().to(torch.bfloat16).cpu()
                return None
            if l < GR["leaf"]:
                return None
            if l == GR["leaf"]:
                x = hs.detach().clone()
                if GR.get("setpos") is not None:
                    x[0, GR["pos"]] = GR["setpos"].to(x.device, x.dtype)
                x.requires_grad_(True)
                GR["t"][("in", l)] = x
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                kw = dict(kwargs); kw["hidden_states"] = x
                return args, kw
            GR["t"][("in", l)] = hs
            return None
        return h

    def pre_mid(l):
        def h(mod, args):
            m = GR["mode"]
            if m == "grab":
                GR["st"][("mid", l)] = args[0][:2, GR["pos"]].detach().to(torch.bfloat16).cpu()
            elif m == "grad" and l >= GR["leaf"]:
                GR["t"][("mid", l)] = args[0]
            return None
        return h

    def pre_z(mod, args, kwargs):
        if GR["mode"] != "grad":
            return None
        hsn = kwargs["hidden_states"][0].float()
        cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
        hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]; S_ = hsn.shape[0]
        zr = GR["zrows"]
        qg = F.linear(hsn[zr], mod.q_proj.weight.float()).view(len(zr), nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
        k = mod.k_norm(F.linear(hsn, mod.k_proj.weight.float()).view(S_, -1, hd).to(mod.k_norm.weight.dtype)).float()
        kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
        hsel = torch.tensor(GR["zheads"], device=q.device)
        qe = rope(q, cos[zr], sin[zr], rd)[:, hsel]; ke = rope(k[:, kh], cos, sin, rd)[:, hsel]
        lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
        own = GR["own_t"].to(q.device)
        GR["z"] = torch.stack([torch.logsumexp(lg[:, :, own == b], -1).mean() for b in range(N)])
        return None

    def post47(mod, args, kwargs, out):
        if GR["mode"] != "grab":
            return None
        Wo = mod.o_proj.weight.float().view(-1, mod.config.num_attention_heads, mod.head_dim)
        t0 = out[0] if isinstance(out, tuple) else out
        ph = []
        for b in (0, 1):
            o = FHP._heads_out(mod, F47, b, kwargs, GR["rows"])
            ph.append(torch.einsum("rhd,ehd->rhe", o, Wo))
            GR["hchk"].append(float((ph[-1].sum(1) - t0[b, GR["rows"]].float()).abs().max()))
        GR["dho"] = (ph[0] - ph[1]).to(torch.bfloat16).cpu()
        return None

    for l in INL:
        hooks.append(v.layers[l].register_forward_pre_hook(pre_in(l), with_kwargs=True))
    for l in MIDL:
        hooks.append(v.layers[l].post_attention_layernorm.register_forward_pre_hook(pre_mid(l)))
    hooks.append(v.layers[L51].self_attn.register_forward_pre_hook(pre_z, with_kwargs=True))
    hooks.append(v.layers[F47].self_attn.register_forward_hook(post47, with_kwargs=True))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps):
        TOOL.set(qs); FHP.set(fhs); MIP.set(mps)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); FHP.set(None); MIP.set(None)
        return lg, st

    rng = random.Random(a.seed); nrun = 0; recs = []; T1 = []
    OUTD = OUT.format(a.tag, "scenes").replace(".json", "")
    os.makedirs(OUTD, exist_ok=True)
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · FA {F47} · L51 heads {hm(L51)}", flush=True)
    try:
        for si in range(a.n):
            labs = rng.sample(labels, 9)
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = list(labs); rng.shuffle(menu)
            rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={}, qk={}, zl={}, subpass={}, gz={}, gl={}, hchk={}, tokvar={})
            names = {"END5": ("END", KEND), **{f"START{p}": ("START", p) for p in PINS}}
            enc = {nm: v.encode(prompt(f, q, menu, N), pics[:N]) for nm, (f, q) in names.items()}
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1, "prompts must align"
            ref = next(iter(ids.values())); S_ = len(ref)
            ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
            uend = max(i for i, x in enumerate(ref) if x == end_id)
            for nm in ids:
                assert "?" in v.tok.decode([ids[nm][qm]])
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
            tail0 = min(i for i in range(S_) if len({ids[nm][i] for nm in ids}) > 1)            # the first question token that differs
            assert ve[-1] < tail0 < qm
            POS = list(range(tail0, S_)); P_ = len(POS)
            ix = lambda ps: [POS.index(t) for t in ps]
            GIX = {"tail": ix(range(tail0, qm)), "qm": ix([qm]), "menu": ix(range(qm + 1, uend)), "B": ix(range(uend, uend + 5)), "A": ix(A4),
                   "all": list(range(P_))}
            TAILQ = list(range(tail0, qm)); MENU = list(range(qm + 1, uend))
            capq = lambda **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=owner, nb=N, opt=opt), **kw)
            GR.update(zrows=[rowsC[2], rowsC[5], rowsC[9]], zheads=hm(L51), own_t=torch.tensor(owner), rows=rowsC)
            rec.update(S=S_, tail0=tail0, qm=qm, uend=uend, POS0=tail0, groups={g: GIX[g] for g in GROUPS})

            # ---- pin donors: one START pass (natural)
            inps = [enc[f"START{p}"] for p in PINS]
            qs = [capq(grabq=set(EARLY)) for _ in PINS]
            lg0, st0 = run1(inps, qs, [None] * len(PINS), [None] * len(PINS))
            qd = {}
            for b, p in enumerate(PINS):
                qd[p] = {li: st0[b][("q", li)] for li in EARLY}
                w = menu[int(torch.stack([sc(lg0[b], c) for c in menu]).argmax())]
                rec["land"][f"START{p}"] = labs.index(w) + 1
                T1.append(int(int(lg0[b].argmax()) in t1ids(menu)))
            nrun += len(PINS)
            del st0; gc.collect()

            def pin(p):
                return {li: dict(qsub=dict(owner=owner, nb=N, part="U", donors=[(1.0, qd[p][li])])) for li in EARLY}

            def store_run(key, lgb, stb):
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][key] = labs.index(w) + 1
                rec["mlog"][key] = [float(sc(lgb, x)) for x in labs[:N]]
                rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}

            def group(p, size):
                """refs E, R in every equal-size sub-pass; the arms of SPECS (recipient E, donor R)."""
                nonlocal nrun
                pq = pin(p)
                chunks = [SPECS[i:i + size] for i in range(0, len(SPECS), size)]
                for sp, chunk in enumerate(chunks):
                    chunk = chunk + [("PAD", "pad")] * max(0, size - len(chunk))
                    inps, qs, fhs, tag = [], [], [], []
                    def add(nm, ek, cp=None, fh=None, cpr=None, rp=None):
                        inps.append(enc[ek]); q = capq(edit=pq)
                        if cp is not None:
                            q["cpatch"] = cp
                        if cpr is not None:
                            q["cpatch_rows"] = cpr
                        if rp is not None:
                            q["rpatch"] = rp
                        qs.append(q); fhs.append(dict(rows=rowsC, edit=pq, prow=rowsC, hedits=fh or {})); tag.append(nm)
                        return len(inps) - 1
                    iE = add(f"Ep{p}@{sp}", "END5"); iR = add(f"Rp{p}@{sp}", f"START{p}")
                    for arm, X in chunk:
                        nm = f"Ep{p}_{arm}"
                        if X == "pad":
                            add(f"PADp{p}@{sp}_{len(inps)}", "END5")
                        elif X == "BAND" or X in ("MIX46", "MLP46", "MIX47", "MLP47", "MIX48", "MLP48", "MIX49", "MLP49"):
                            add(nm, "END5", cp=dict(src=iR, comps=COMPS[X]))
                        elif X == "BANDPOS":
                            add(nm, "END5", cpr=dict(src=iR, comps=ALLC, rows=POS))
                        elif X in HEADS:
                            add(nm, "END5", fh={F47: [dict(heads=HEADS[X], op="swap", src=iR)]})
                        elif X.startswith("RES"):
                            rws = {"RESROWS": rowsC, "RESTAIL": TAILQ, "RESMENU": MENU, "RESPOS": POS}[X]
                            add(nm, "END5", rp={46: dict(src=iR, rows=rws)})
                        elif X in PREG:
                            add(nm, "END5", cpr=dict(src=iR, comps=PREG[X], rows=POS))
                        else:
                            raise ValueError(X)
                    lg, st = run1(inps, qs, fhs, [None] * len(inps))
                    for b, nm in enumerate(tag):
                        keys = [nm] + ([nm.split("@")[0]] if nm.endswith("@0") else [])
                        for key in keys:
                            store_run(key, lg[b], st[b])
                        rec["subpass"][nm] = sp
                        nrun += 1
                    del st; gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            def grad_pass(ek, pq, leaf, setpos, pts, tokvar):
                """one forward with the leaf at layer 'leaf'; 16 objectives -> gradients at pts (POS rows, bf16 on cpu), the read-row band
                pathway at in46, and the objectives' values."""
                GR.update(mode="grad", leaf=leaf, setpos=setpos, pos=POS, t={}, z=None)
                TOOL.set([dict(rows=rowsC, edit=pq)]); FHP.set([None]); MIP.set([None])
                G = {q: [] for q in pts}; GB = []
                try:
                    with torch.enable_grad():
                        out = v.model(**batch_inputs([enc[ek]]), use_cache=False, return_dict=True, **v.forward_extra)
                        lgt = out.logits[0, -1].float(); z = GR["z"]
                        objs = [z[b] for b in range(N)] + [lgt[tokvar[i]] for i in range(N)]
                        ins = [GR["t"][q] for q in pts]
                        i46, i50 = pts.index(("in", 46)), pts.index(("in", 50))
                        for ob in objs:
                            gs = torch.autograd.grad(ob, ins, retain_graph=True, allow_unused=True)
                            for q, g in zip(pts, gs):
                                G[q].append(torch.zeros(P_, ins[0].shape[-1], dtype=torch.bfloat16) if g is None else g[0, POS].detach().to(torch.bfloat16).cpu())
                            g50r = gs[i50][0, rowsC].detach().float()
                            d50 = ins[i50].device
                            o2 = (g50r * (ins[i50][0, rowsC].float() - ins[i46][0, rowsC].float().to(d50))).sum()
                            gb = torch.autograd.grad(o2, ins[i46], retain_graph=True, allow_unused=True)[0]
                            GB.append(torch.zeros(P_, ins[0].shape[-1], dtype=torch.bfloat16) if gb is None else gb[0, POS].detach().to(torch.bfloat16).cpu())
                        vals = dict(z=z.detach().float().cpu().tolist(), L=[float(lgt[tokvar[i]]) for i in range(N)])
                        del out, lgt, z, objs, ins
                finally:
                    TOOL.set(None); FHP.set(None); MIP.set(None); GR.update(mode=None, t={}, z=None, setpos=None)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return {q: torch.stack(G[q]) for q in pts}, torch.stack(GB), vals

            NPZ = {}
            for p in PINS:
                pq = pin(p)
                # ---- (i) E and R states
                GR.update(mode="grab", pos=POS, st={}, hchk=[], dho=None)
                lg, st = run1([enc["END5"], enc[f"START{p}"]], [capq(edit=pq), capq(edit=pq)],
                              [dict(rows=rowsC, edit=pq, prow=rowsC, hedits={}) for _ in range(2)], [None, None])
                GR["mode"] = None
                ST = dict(GR["st"]); rec["hchk"][f"p{p}"] = GR["hchk"]; NPZ[f"p{p}_dho47"] = bits(GR["dho"])
                for b, nm in enumerate((f"Ep{p}", f"Rp{p}")):
                    store_run(nm + "@g", lg[b], st[b]); nrun += 1
                tokvar = [tv(w)[0] if float(lg[0][tv(w)[0]]) >= float(lg[0][tv(w)[1]]) else tv(w)[1] for w in labs[:N]]
                rec["tokvar"][f"p{p}"] = tokvar
                del st
                for q in SAVE:
                    NPZ[f"p{p}_st_E_{q[0]}{q[1]}"] = bits(ST[q][0]); NPZ[f"p{p}_st_R_{q[0]}{q[1]}"] = bits(ST[q][1])
                # ---- (ii) causal arms
                group(p, a.size)
                # ---- (iii) gradients
                dH = {q: (ST[q][0].float() - ST[q][1].float()) for q in PTS}
                dO = {}
                for kd, l in MODS:
                    dO[(kd, l)] = (dH[("mid", l)] - dH[("in", l)]) if kd == "mix" else (dH[("in", l + 1)] - dH[("mid", l)])
                outp = {("mix", l): ("mid", l) for l in MIDL}; outp.update({("mlp", l): ("in", l + 1) for l in MIDL})
                mid = (ST[("in", 46)][0].float() + ST[("in", 46)][1].float()) / 2
                for tagp, ek, leaf, setpos, pts in (("E", "END5", 35, None, PTS), ("R", f"START{p}", 35, None, PTS),
                                                    ("M", "END5", 46, mid, [q for q in PTS if q[1] >= 46])):
                    G, GB, vals = grad_pass(ek, pq, leaf, setpos, pts, tokvar)
                    rec["gz"][f"p{p}_{tagp}"] = vals["z"]; rec["gl"][f"p{p}_{tagp}"] = vals["L"]
                    for q in (SAVE if tagp != "M" else SAVE_M):
                        NPZ[f"p{p}_g_{tagp}_{q[0]}{q[1]}"] = bits(G[q])
                    NPZ[f"p{p}_gband_{tagp}"] = bits(GB)
                    # scene-specific attribution scalars [16 objectives, 33 points + 32 modules, 6 groups]
                    AT = np.full((2 * N, len(PTS) + len(MODS), len(GROUPS)), np.nan, dtype=np.float32)
                    for j, q in enumerate(PTS):
                        if q in G:
                            pr = (G[q].float() * dH[q][None]).sum(-1)                                   # [16, P]
                            for gi, g in enumerate(GROUPS):
                                AT[:, j, gi] = pr[:, GIX[g]].sum(-1).numpy()
                    for j, m in enumerate(MODS):
                        q = outp[m]
                        if q in G and (tagp != "M" or m[1] >= 46):
                            pr = (G[q].float() * dO[m][None]).sum(-1)
                            for gi, g in enumerate(GROUPS):
                                AT[:, len(PTS) + j, gi] = pr[:, GIX[g]].sum(-1).numpy()
                    pr = (GB.float() * dH[("in", 46)][None]).sum(-1)
                    NPZ[f"p{p}_attr_{tagp}"] = AT
                    NPZ[f"p{p}_bandlens_{tagp}"] = np.stack([pr[:, GIX[g]].sum(-1).numpy() for g in GROUPS], -1).astype(np.float32)
                    del G, GB
                    nrun += 1
                del ST, dH, dO
                gc.collect()
            np.savez(f"{OUTD}/scene{si:02d}.npz", **NPZ)
            del NPZ; gc.collect()
            recs.append(rec)
            print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · pinned head-patch calls {FHP.calls} · head check "
                  f"{max(max(x) for x in rec['hchk'].values()):.4f}", flush=True)
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        c_qpath.QPath._qsub = _qsub0
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), specs=[list(x) for x in SPECS], anti=ANTI, h16=H16, selfaddr=SELFADDR,
                   rest47=REST47, band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   pts=[list(q) for q in PTS], save=[list(q) for q in SAVE], save_m=[list(q) for q in SAVE_M], mods=[list(m) for m in MODS],
                   groups=list(GROUPS), objectives=[f"z{b + 1}" for b in range(N)] + [f"L{i + 1}" for i in range(N)], npz_dir=OUTD,
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "jlens"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'jlens')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--size", type=int, default=7)
    main(ap.parse_args())
