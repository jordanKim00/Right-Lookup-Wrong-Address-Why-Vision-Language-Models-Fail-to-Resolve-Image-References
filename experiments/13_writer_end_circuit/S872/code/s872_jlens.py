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
v2 (10-21 14:4x, the user asked to keep every card busy; implementation only, the measurements of v1 = exp/s872_jlens_v1.py unchanged): the
  E / R state capture rides on the first causal sub-pass (its items 0 and 1 are the refs E and R); sub-passes of --size arms; the 16 objectives'
  vector-Jacobian products are taken in one batched backward (torch.autograd.grad, is_grads_batched=True; the first (scene, pin) also takes
  two objectives by the plain loop and records the largest difference); pins run in --threads threads (hook state thread-local), with at most
  one gradient pass at a time (memory); scenes are drawn and encoded in v1's order and then run --scene_par at a time; no-grad forwards share
  a semaphore of --ngpar; --threads 1 --scene_par 1 is the sequential order of v1.
Usage: python s872_jlens.py MODEL TAG [--n 48] [--device_map balanced] [--threads 3] [--scene_par 2] [--ngpar 3] [--size 11]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
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
INL = list(range(46, 52)); MIDL = list(range(46, 51))                                       # v3: every gradient pass has its leaf at 46
PTS = [("in", l) for l in INL] + [("mid", l) for l in MIDL]                                  # 11 residual points
SAVE = [("in", 46), ("mid", 46), ("in", 47), ("mid", 47), ("in", 48), ("mid", 48), ("in", 49), ("mid", 49), ("in", 50), ("in", 51)]
SAVE_M = list(SAVE)
MODS = [("mix", l) for l in MIDL] + [("mlp", l) for l in MIDL]                               # 10 modules: output point mid_l / in_(l+1)
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
    import c_tritonlock; c_tritonlock.install()                                                    # Triton autotuner is not thread-safe (10-22)
    dev = os.environ.get("S863_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    CUDNN0 = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = a.cudnn == 1                                            # (--cudnn 0 is ~10x slower: native Conv3d in the vision tower)
    _vis = getattr(v.model.model, "visual", None)
    VLOCK = threading.Lock()
    if _vis is not None:                                                                   # v4: the vision tower's convolutions run one thread at a time
        _vis_fwd = _vis.forward                                                            # (concurrent cuDNN Conv3d raised CUDNN_STATUS_INTERNAL_ERROR)
        def _vis_locked(*args_, **kw_):
            with VLOCK:
                return _vis_fwd(*args_, **kw_)
        _vis.forward = _vis_locked
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
    _TL = threading.local()
    class _GRP:                                                        # per-thread hook state (pins run in threads)
        def __getitem__(self, k):
            return self.d[k]
        def __setitem__(self, k, x):
            self.d[k] = x
        def get(self, k, dflt=None):
            return self.d.get(k, dflt)
        def update(self, **kw):
            self.d.update(kw)
        @property
        def d(self):
            g = getattr(_TL, "g", None)
            if g is None:
                g = _TL.g = dict(mode=None, t={}, st={}, z=None, want=[0, 1])
            return g
    GR = _GRP()
    class _Tokens:                                                                     # memory governor: a gradient graph costs a.gtok, a no-grad
        def __init__(self, n):                                                         # forward one token, a.tokens in all (GPU3 holds layers 48-63)
            self.n = n; self.c = threading.Condition()
        def take(self, k):
            with self.c:
                while self.n < k:
                    self.c.wait()
                self.n -= k
        def give(self, k):
            with self.c:
                self.n += k; self.c.notify_all()
    TOK = _Tokens(a.tokens); CLOCK = threading.Lock()
    class _Hold:
        def __init__(self, k):
            self.k = k
        def __enter__(self):
            TOK.take(self.k)
        def __exit__(self, *e):
            TOK.give(self.k)
    GSEM = _Hold(a.gtok)
    hooks = []

    def pre_in(l):
        def h(mod, args, kwargs):
            m = GR["mode"]
            if m is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            if m == "grab":
                GR["st"][("in", l)] = hs[GR["want"]][:, GR["pos"]].detach().to(torch.bfloat16).cpu()
                return None
            if l < GR["leaf"]:
                return None
            if l == GR["leaf"]:
                x = hs.detach().clone()
                if GR.get("setpos") is not None:
                    for b_, val in GR["setpos"].items():
                        x[b_, GR["pos"]] = val.to(x.device, x.dtype)
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
                GR["st"][("mid", l)] = args[0][GR["want"]][:, GR["pos"]].detach().to(torch.bfloat16).cpu()
            elif m == "grad" and l >= GR["leaf"]:
                GR["t"][("mid", l)] = args[0]
            return None
        return h

    def pre_z(mod, args, kwargs):
        if GR["mode"] != "grad":
            return None
        zs = []
        for bi in range(kwargs["hidden_states"].shape[0]):
            hsn = kwargs["hidden_states"][bi].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[bi].float(), sin[bi].float()
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
            zs.append(torch.stack([torch.logsumexp(lg[:, :, own == b], -1).mean() for b in range(N)]))
        GR["z"] = torch.stack(zs)                                                                # [B, 8]
        return None

    def post47(mod, args, kwargs, out):
        if GR["mode"] != "grab":
            return None
        Wo = mod.o_proj.weight.float().view(-1, mod.config.num_attention_heads, mod.head_dim)
        t0 = out[0] if isinstance(out, tuple) else out
        ph = []
        for b in GR["want"]:
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

    NGSEM = _Hold(1)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps):
        with NGSEM:
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
    SCN = []                                                                                        # scenes drawn in v1's order
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        names = {"END5": ("END", KEND), **{f"START{p}": ("START", p) for p in PINS}}
        SCN.append((si, labs, menu, {nm: v.encode(prompt(f, q, menu, N), pics[:N]) for nm, (f, q) in names.items()}))

    DONE = {}

    def bump(k=1):
        nonlocal nrun
        with CLOCK:
            nrun += k

    def scene_task(si, labs, menu, enc):
            rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={}, qk={}, zl={}, subpass={}, gz={}, gl={}, hchk={}, tokvar={})
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
            bump(len(PINS))
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
                """refs E, R in every equal-size sub-pass; the arms of SPECS (recipient E, donor R); sub-pass 0 also captures E's and R's states
                (its items 0 and 1) - v1 took them in a pass of their own."""
                pq = pin(p)
                chunks = [SPECS[i:i + size] for i in range(0, len(SPECS), size)]
                grab = None
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
                    if sp == 0:
                        GR.update(mode="grab", pos=POS, st={}, hchk=[], dho=None, want=[iE, iR])
                    try:
                        lg, st = run1(inps, qs, fhs, [None] * len(inps))
                    finally:
                        GR["mode"] = None
                    if sp == 0:
                        grab = dict(st=dict(GR["st"]), hchk=list(GR["hchk"]), dho=GR["dho"], lgE=lg[iE].clone())
                    for b, nm in enumerate(tag):
                        keys = [nm] + ([nm.split("@")[0]] if nm.endswith("@0") else [])
                        if nm.endswith("@0") and b in (iE, iR):
                            keys.append(nm.split("@")[0] + "@g")                                   # the state-capture runs
                        for key in keys:
                            store_run(key, lg[b], st[b])
                        rec["subpass"][nm] = sp
                        bump()
                    del st
                return grab

            def grad_pass(ek, pq, setpos, tokvar, check=False):
                """v4: one batch-1 forward with the leaf at layer 46's input (setpos: POS of the leaf <- that state, for M); the 16 objectives'
                VJPs in batched backward calls of a.vchunk cotangents -> gradients at PTS (POS rows, bf16 on cpu), the read-row band pathway
                J_B,rows^T g50(rows) at in46, and the objectives' values."""
                GR.update(mode="grad", leaf=46, setpos=(None if setpos is None else {0: setpos}), pos=POS, t={}, z=None)
                TOOL.set([dict(rows=rowsC, edit=pq)]); FHP.set([None]); MIP.set([None])
                info = {}
                try:
                    with torch.enable_grad():
                        out = v.model(**batch_inputs([enc[ek]]), use_cache=False, return_dict=True, **v.forward_extra)
                        lgt = out.logits[0, -1].float(); z = GR["z"][0]; dv = lgt.device
                        ov = torch.cat([z.to(dv), lgt[torch.tensor(tokvar, device=dv)]])                     # [16]
                        ins = [GR["t"][q] for q in PTS]
                        i46, i50 = PTS.index(("in", 46)), PTS.index(("in", 50))
                        E_ = ins[0].shape[-1]; K = 2 * N; ch = a.vchunk
                        eye = torch.eye(K, device=dv, dtype=ov.dtype)
                        G = {q: [] for q in PTS}
                        for c0 in range(0, K, ch):
                            gs = torch.autograd.grad(ov, ins, grad_outputs=eye[c0:c0 + ch], retain_graph=True, allow_unused=True, is_grads_batched=True)
                            n_ = min(ch, K - c0)
                            for q, g in zip(PTS, gs):
                                G[q].append(torch.zeros(n_, P_, E_, dtype=torch.bfloat16) if g is None else g[:, 0, POS].detach().to(torch.bfloat16).cpu())
                            del gs
                        G = {q: torch.cat(x, 0) for q, x in G.items()}
                        if check:
                            for j in (0, N + 2):
                                gl = torch.autograd.grad(ov[j], ins[i46], retain_graph=True)[0][0, POS].float().cpu()
                                info[f"vmap_vs_loop_obj{j}"] = float((gl - G[("in", 46)][j].float()).abs().max() / (gl.abs().max() + 1e-12))
                        d50, d46 = ins[i50].device, ins[i46].device
                        rP = [POS.index(t) for t in rowsC]
                        GB = []
                        for c0 in range(0, K, ch):
                            n_ = min(ch, K - c0)
                            cot = torch.zeros((n_,) + tuple(ins[i50].shape), dtype=ins[i50].dtype, device=d50)
                            cot[:, 0, rowsC] = G[("in", 50)][c0:c0 + n_][:, rP].to(d50, cot.dtype)
                            gb = torch.autograd.grad(ins[i50], ins[i46], grad_outputs=cot, retain_graph=True, allow_unused=True, is_grads_batched=True)[0]
                            gb = gb.float() - cot.to(d46).float()                                         # J_B^T cot (the identity part removed)
                            GB.append(gb[:, 0, POS].to(torch.bfloat16).cpu())
                            del gb, cot
                        GB = torch.cat(GB, 0)
                        vals = dict(z=z.detach().float().cpu().tolist(), L=[float(lgt[tokvar[i]]) for i in range(N)])
                        del out, lgt, z, ov, ins
                finally:
                    TOOL.set(None); FHP.set(None); MIP.set(None); GR.update(mode=None, t={}, z=None, setpos=None)
                    gc.collect()
                return G, GB, vals, info

            def pin_task(p):
                GR.update(zrows=[rowsC[2], rowsC[5], rowsC[9]], zheads=hm(L51), own_t=torch.tensor(owner), rows=rowsC)
                pq = pin(p); NP = {}
                grab = group(p, a.size)                                                            # (i) + (ii)
                ST = grab["st"]; rec["hchk"][f"p{p}"] = grab["hchk"]; NP[f"p{p}_dho47"] = bits(grab["dho"])
                lgE = grab["lgE"]
                tokvar = [tv(w)[0] if float(lgE[tv(w)[0]]) >= float(lgE[tv(w)[1]]) else tv(w)[1] for w in labs[:N]]
                rec["tokvar"][f"p{p}"] = tokvar
                for q in SAVE:
                    NP[f"p{p}_st_E_{q[0]}{q[1]}"] = bits(ST[q][0]); NP[f"p{p}_st_R_{q[0]}{q[1]}"] = bits(ST[q][1])
                # ---- (iii) gradients
                dH = {q: (ST[q][0].float() - ST[q][1].float()) for q in PTS}
                dO = {}
                for kd, l in MODS:
                    dO[(kd, l)] = (dH[("mid", l)] - dH[("in", l)]) if kd == "mix" else (dH[("in", l + 1)] - dH[("mid", l)])
                outp = {("mix", l): ("mid", l) for l in MIDL}; outp.update({("mlp", l): ("in", l + 1) for l in MIDL})
                mid = (ST[("in", 46)][0].float() + ST[("in", 46)][1].float()) / 2
                for tagp, ek, setpos in (("E", "END5", None), ("R", f"START{p}", None), ("M", "END5", mid)):
                    with GSEM:                                                                     # bounded gradient graphs (memory)
                        G, GB, vals, info = grad_pass(ek, pq, setpos, tokvar, check=(si == 0 and p == PINS[0] and tagp == "E"))
                    if info:
                        rec.setdefault("vmapchk", {}).update(info)
                    rec["gz"][f"p{p}_{tagp}"] = vals["z"]; rec["gl"][f"p{p}_{tagp}"] = vals["L"]
                    for q in SAVE:
                        NP[f"p{p}_g_{tagp}_{q[0]}{q[1]}"] = bits(G[q])
                    NP[f"p{p}_gband_{tagp}"] = bits(GB)
                    # scene-specific attribution scalars [16 objectives, 11 points + 10 modules, 6 groups]
                    AT = np.full((2 * N, len(PTS) + len(MODS), len(GROUPS)), np.nan, dtype=np.float32)
                    for j, q in enumerate(PTS):
                        pr = (G[q].float() * dH[q][None]).sum(-1)                                       # [16, P]
                        for gi, g in enumerate(GROUPS):
                            AT[:, j, gi] = pr[:, GIX[g]].sum(-1).numpy()
                    for j, m in enumerate(MODS):
                        pr = (G[outp[m]].float() * dO[m][None]).sum(-1)
                        for gi, g in enumerate(GROUPS):
                            AT[:, len(PTS) + j, gi] = pr[:, GIX[g]].sum(-1).numpy()
                    pr = (GB.float() * dH[("in", 46)][None]).sum(-1)
                    NP[f"p{p}_attr_{tagp}"] = AT
                    NP[f"p{p}_bandlens_{tagp}"] = np.stack([pr[:, GIX[g]].sum(-1).numpy() for g in GROUPS], -1).astype(np.float32)
                    bump()
                    del G, GB
                del ST, dH, dO
                return NP

            NPZ = {}
            if a.threads > 1:
                with ThreadPoolExecutor(max_workers=a.threads) as ex:
                    for o in ex.map(pin_task, PINS):
                        NPZ.update(o)
            else:
                for p in PINS:
                    NPZ.update(pin_task(p))
            gc.collect()
            if torch.cuda.is_available():
                rec["maxmem_gb"] = [round(torch.cuda.max_memory_allocated(d_) / 2 ** 30, 2) for d_ in range(torch.cuda.device_count())]
            np.savez(f"{OUTD}/scene{si:02d}.npz", **NPZ)
            del NPZ; gc.collect()
            with CLOCK:                                                                    # a crash late in the run keeps the finished scenes
                DONE[si] = rec
                json.dump(dict(tag=a.tag, partial=True, n_done=len(DONE), rows=[DONE[k] for k in sorted(DONE)]),
                          open(OUT.format(a.tag, "jlens_partial"), "w"))
            print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · pinned head-patch calls {FHP.calls} · head check "
                  f"{max(max(x) for x in rec['hchk'].values()):.4f} · max mem GB {rec.get('maxmem_gb')} · vmap check {rec.get('vmapchk')}", flush=True)
            return rec

    n_resumed = 0
    if a.resume:                                                                     # 10-21: continue a run that died; finished scenes are kept
        P0 = json.load(open(OUT.format(a.tag, "jlens_partial")))
        for r_ in P0["rows"]:
            if os.path.exists(f"{OUTD}/scene{r_['si']:02d}.npz"):
                DONE[r_["si"]] = r_
        n_resumed = len(DONE)
        SCN = [x for x in SCN if x[0] not in DONE]
        print(f"[{a.tag}] resume: {n_resumed} finished scenes kept, {len(SCN)} to run", flush=True)
    try:
        if a.scene_par > 1:
            with ThreadPoolExecutor(max_workers=a.scene_par) as ex:
                list(ex.map(lambda x: scene_task(*x), SCN))
        else:
            for x in SCN:
                scene_task(*x)
        recs = [DONE[k] for k in sorted(DONE)]
    finally:
        for h_ in hooks:
            h_.remove()
        TOOL.remove(); FHP.remove(); MIP.remove()
        c_qpath.QPath._qsub = _qsub0
        torch.backends.cudnn.enabled = CUDNN0
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, N=N, k=KEND, pins=list(PINS), specs=[list(x) for x in SPECS], anti=ANTI, h16=H16, selfaddr=SELFADDR,
                   rest47=REST47, band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   pts=[list(q) for q in PTS], save=[list(q) for q in SAVE], save_m=[list(q) for q in SAVE_M], mods=[list(m) for m in MODS],
                   groups=list(GROUPS), objectives=[f"z{b + 1}" for b in range(N)] + [f"L{i + 1}" for i in range(N)], npz_dir=OUTD,
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs,
                   resumed_scenes=n_resumed),
              open(OUT.format(a.tag, "jlens"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'jlens')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87200); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--size", type=int, default=11); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--vchunk", type=int, default=4)
    ap.add_argument("--scene_par", type=int, default=2); ap.add_argument("--ngpar", type=int, default=3); ap.add_argument("--cudnn", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0); ap.add_argument("--gpar", type=int, default=1); ap.add_argument("--tokens", type=int, default=4); ap.add_argument("--gtok", type=int, default=2)
    main(ap.parse_args())
