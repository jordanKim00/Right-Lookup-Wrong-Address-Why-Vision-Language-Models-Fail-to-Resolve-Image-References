"""S758: through which reads do the ordinal (k) and the frame word enter the END address?  Complete-route read interchange.
Pre-registration in knowledge/DECISIONS.md (2026-10-13 12:3x) before any S758 data; analysis exp/s758_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt.  Per scene END(2), END(3), END(5), START(2), START(5), same images and menu.
Read interchange of a source token p (the ordinal o, or the frame word f) at a layer set B: at every layer in B the sequence mixer (full attention
or Gated DeltaNet) receives the donor's normalized input at p, so every later row reads the donor's p through every route (attention keys and
values; DeltaNet memory entry q, k, v, g, beta and the 4-tap causal conv into p+1..p+3), while the mixer output at row p is reset to the
recipient's own value, so that p and every earlier row stay the recipient's (S752 edited the delta-rule inputs after the conv and left the conv
taps from p open).  Route-split arms at the DeltaNet layers of Lt: conv only (the taps from p carry the donor's pre-conv column, the memory entry
at p stays the recipient's) and memory only (the memory entry and g, beta at p are the donor's, the taps carry the recipient's column).
Bands (27B): E = L0-22, Lt = L23-26, P = L27-63; mapped by relative depth for other models.
Arms (recipient <- donor):
    naturals E2 E3 E5 S2 S5;  instrument NOOP (E5 <- E5, all layers), T22 / T26 (rows after o copied after L22 / L26, E5 <- E2), K:E+Lsplit
    K   (E5 <- E2, p = o):   E L P EL EP LP ELP;  E+L23 E+L24 E+L25 E+L26 E+Lconv E+Lmem;  L23 L24 L25 L26 Lconv Lmem
    K3  (E3 <- E2, p = o):   E L EL
    F2  (E2 <- S2, p = f):   E L P EL EP LP ELP
    F5  (E5 <- S5, p = f):   E L P EL EP LP ELP
Readouts: the menu-restricted answer, the menu logits, and the fixed-bank block scores of every head at L35, 39, 43, 47 (27B) for the rows
"?", tail1 and the answer (as S757).
Usage: python s758_paths.py MODEL TAG [--start 0] [--n 40] [--device_map balanced] [--check]
"""
from __future__ import annotations

import argparse, os, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s758_{}_paths"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    DN_L = [i for i, L in enumerate(v.layers) if getattr(L, "linear_attn", None) is not None]
    mix = [L.self_attn if i in FA_L else L.linear_attn for i, L in enumerate(v.layers)]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    e_hi, l_hi = mp(22), mp(26)
    BAND = {"E": list(range(0, e_hi + 1)), "L": list(range(e_hi + 1, l_hi + 1)), "P": list(range(l_hi + 1, nl))}
    LT = BAND["L"]; LT_DN = [l for l in LT if l in DN_L]
    R_ = lambda r: int(round(r * (nl - 1)))
    CAP = sorted({min(FA_L, key=lambda l: abs(l - R_(x / 63))) for x in (35, 39, 43, 47)})
    RES = [e_hi, l_hi]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    def aid(s):                                          # answer readout (S783): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return tid(s)
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    T1 = []                                              # S783: is the vocabulary-wide top-1 a bare menu token?
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)

    X = {}

    def clear():
        X.clear(); X.update(grab=False, gpos=[], p=None, swap={}, reset={}, conv={}, cur=None, rows=None,
                            g={}, res_grab=False, res_store={}, res_rep={}, cap_on=False, cap={})
    clear()

    def pre(li):
        def h(mod, args, kwargs):
            X["cur"] = li
            hs = kwargs["hidden_states"]
            if X["grab"]:
                for q in X["gpos"]:
                    X["g"][("in", q, li)] = hs[0, q].detach().clone()
            d = X["swap"].get(li)
            if d is None:
                return None
            hs = hs.clone(); hs[0, X["p"]] = d.to(hs.device, hs.dtype)
            kw = dict(kwargs); kw["hidden_states"] = hs
            return args, kw
        return h

    def post(li):
        def h(mod, args, kwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            if X["grab"]:
                for q in X["gpos"]:
                    X["g"][("out", q, li)] = t[0, q].detach().clone()
            r = X["reset"].get(li)
            if r is None:
                return None
            t = t.clone(); t[0, X["p"]] = r.to(t.device, t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return h

    orig_conv = QM.causal_conv1d_fn

    def conv_fn(hidden_states, weight, bias=None, activation=None, **kw):
        li = X["cur"]; c = X["conv"].get(li); x = hidden_states
        if c is not None:                                          # c = (pre-conv column at p, post-activation column at p)
            x = x.clone(); x[0, :, X["p"]] = c[0].to(x.device, x.dtype)
        y = orig_conv(x, weight, bias, activation=activation, **kw)
        if X["grab"]:
            for q in X["gpos"]:
                X["g"][("x", q, li)] = hidden_states[0, :, q].detach().clone(); X["g"][("y", q, li)] = y[0, :, q].detach().clone()
        if c is not None:
            y = y.clone(); y[0, :, X["p"]] = c[1].to(y.device, y.dtype)
        return y
    QM.causal_conv1d_fn = conv_fn

    def res_hook(li):
        def h(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            if X["res_grab"]:
                X["res_store"][li] = t[0, X["rows"]].detach().clone()
            rep = X["res_rep"].get(li)
            if rep is None:
                return None
            t = t.clone(); t[0, X["rows"]] = rep.to(t.device, t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return h

    def cap_hook(li):
        def h(mod, args, kwargs, out):
            if not X["cap_on"]:
                return None
            hs = kwargs["hidden_states"][0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            rows = X["crow"]; img = torch.tensor(X["img"], device=hs.device)
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(),
                                            None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
            k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[img][:, None] + rotate_half(k[..., :rd]) * sin[img][:, None], k[..., rd:]], -1)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe, ke[:, kh]) * mod.scaling
            own = torch.tensor(X["owner"], device=q.device)
            X["cap"][li] = torch.stack([torch.logsumexp(lg[:, :, own == b], -1) for b in range(N)], -1).cpu().numpy().astype(np.float32)
            return None
        return h

    hooks = [mix[li].register_forward_pre_hook(pre(li), with_kwargs=True) for li in range(nl)]
    hooks += [mix[li].register_forward_hook(post(li), with_kwargs=True) for li in range(nl)]
    hooks += [v.layers[li].register_forward_hook(res_hook(li)) for li in RES]
    hooks += [v.layers[li].self_attn.register_forward_hook(cap_hook(li), with_kwargs=True) for li in CAP]

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, ids, labs, menu, extra, setup):
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, len(ids)) if "?" in v.tok.decode([ids[i]]))
        uend = max(i for i, t in enumerate(ids) if t == T("<|im_end|>"))
        clear(); setup()
        X.update(cap_on=True, crow=[qm, uend + 1, len(ids) - 1], img=[p for b in range(N) for p in range(vs[b], ve[b] + 1)],
                 owner=[b for b in range(N) for _ in range(vs[b], ve[b] + 1)], cap={})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        out = dict(cap=X["cap"], g=X["g"], res=X["res_store"])
        clear()
        lab = menu[int(torch.stack([lg[aid(c)] for c in menu]).argmax())]
        T1.append(int(int(lg.argmax()) in [aid(c) for c in menu]))
        return (labs.index(lab) + 1 if lab in labs else 0), [float(lg[aid(l)]) for l in labs + extra], out

    print(f"[{a.tag}] bands E {BAND['E'][0]}-{BAND['E'][-1]} · Lt {LT} · P {BAND['P'][0]}-{BAND['P'][-1]} · Lt DN {LT_DN} · CAP {CAP} · RES {RES}",
          flush=True)
    rng = random.Random(75800)
    meta, arrays = [], {}
    for si in range(a.start + a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        if si < a.start:
            continue
        runs = {"E2": ("E", 2), "E3": ("E", 3), "E5": ("E", 5), "S2": ("S", 2), "S5": ("S", 5)}
        inp = {nm: v.encode(prompt(fr, k, menu), [pic[l] for l in labs]) for nm, (fr, k) in runs.items()}
        ids = {nm: x["input_ids"][0].tolist() for nm, x in inp.items()}
        S_ = len(ids["E2"]); assert all(len(x) == S_ for x in ids.values())
        dif = lambda A, B: [i for i, (x, y) in enumerate(zip(ids[A], ids[B])) if x != y]
        o = dif("E2", "E5"); f = dif("E2", "S2")
        assert len(o) == 1 and len(f) == 1 and dif("E2", "E3") == o and dif("S2", "S5") == o and dif("E5", "S5") == f, (o, f)
        o, f = o[0], f[0]; assert o < f
        rows = list(range(o + 1, S_))
        rec = dict(si=si, pos=dict(o=o, f=f, S=S_), land={}, ll={})
        nat = {}

        def keep(nm, res):
            rec["land"][nm], rec["ll"][nm], o_ = res
            for li, sc in o_["cap"].items():
                arrays[f"{si}|{nm}|{li}"] = sc
            return o_

        for nm in runs:
            def setup(nm=nm):
                X.update(grab=True, gpos=[o, f], rows=rows, res_grab=(nm == "E2"))
            nat[nm] = keep(nm, run(inp[nm], ids[nm], labs, menu, extra, setup))
        G = lambda nm, kind, q, li: nat[nm]["g"][(kind, q, li)]

        def oii(rcp, don, q, full=(), conv=(), mem=(), both=()):
            """full: layers whose mixer reads the donor's p; conv / mem: DeltaNet layers with one route only; both: split hooks, both routes."""
            def setup():
                X["p"] = q
                for li in list(full) + list(mem) + list(both):
                    X["swap"][li] = G(don, "in", q, li)
                for li in list(full) + list(conv) + list(mem) + list(both):
                    X["reset"][li] = G(rcp, "out", q, li)
                for li in conv:
                    X["conv"][li] = (G(don, "x", q, li), G(rcp, "y", q, li))
                for li in mem:
                    X["conv"][li] = (G(rcp, "x", q, li), G(don, "y", q, li))
                for li in both:
                    X["conv"][li] = (G(don, "x", q, li), G(don, "y", q, li))
            return run(inp[rcp], ids[rcp], labs, menu, extra, setup)

        def bands(s):
            return [li for b in s for li in BAND[b]]

        keep("NOOP", oii("E5", "E5", o, full=range(nl)))
        for L, nm in zip(RES, ("T22", "T26")):
            def setup(L=L):
                X.update(rows=rows, res_rep={L: nat["E2"]["res"][L]})
            keep(nm, run(inp["E5"], ids["E5"], labs, menu, extra, setup))
        lt_fa = [l for l in LT if l not in LT_DN]
        keep("K:E+Lsplit", oii("E5", "E2", o, full=bands("E") + lt_fa, both=LT_DN))
        for s in ("E", "L", "P", "EL", "EP", "LP", "ELP"):
            keep(f"K:{s}", oii("E5", "E2", o, full=bands(s)))
        for base, pre_ in (("E+", bands("E")), ("", [])):
            for li in LT:
                keep(f"K:{base}L{li}", oii("E5", "E2", o, full=pre_ + [li]))
            keep(f"K:{base}Lconv", oii("E5", "E2", o, full=pre_, conv=LT_DN))
            keep(f"K:{base}Lmem", oii("E5", "E2", o, full=pre_, mem=LT_DN))
        for s in ("E", "L", "EL"):
            keep(f"K3:{s}", oii("E3", "E2", o, full=bands(s)))
        for fam, rcp, don in (("F2", "E2", "S2"), ("F5", "E5", "S5")):
            for s in ("E", "L", "P", "EL", "EP", "LP", "ELP"):
                keep(f"{fam}:{s}", oii(rcp, don, f, full=bands(s)))
        if a.check:                                                  # software-only exactness arms (smoke test)
            li = LT_DN[0]
            keep("CHK:conv_self", oii("E5", "E5", o, conv=[li]))
            keep("CHK:split1", oii("E5", "E2", o, both=[li]))
            keep("CHK:full1", oii("E5", "E2", o, full=[li]))
        meta.append(rec)
        del nat
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['land'])} arms · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
              flush=True)
    for h in hooks:
        h.remove()
    QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), BAND={k: [x[0], x[-1]] for k, x in BAND.items()}, LT=LT, LT_DN=LT_DN, CAP=CAP, RES=RES,
                   rows=["qm", "tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=0); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--check", action="store_true")
    main(ap.parse_args())
