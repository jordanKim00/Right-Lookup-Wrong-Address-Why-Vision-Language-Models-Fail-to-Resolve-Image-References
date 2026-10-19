"""S759: where after layer 26 is the frame word read, and through which route?  Complete-route read interchange of the frame word.
Pre-registration in knowledge/DECISIONS.md (2026-10-13 13:2x) before any S759 data; analysis exp/s759_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt.  Per scene END(2), END(5), START(2), START(5), same images and menu.
Same read interchange as S758 (exp/s758_paths.py): at every layer of a set the sequence mixer receives the donor's normalized input at the source
token p, so every later row reads the donor's p through every route, and the mixer output at row p is reset to the recipient's own value.
S758 found that the frame word f (" last" / " first") is used only at L27-63.  Sub-bands of P (27B): P1 = L27-34, P2 = L35-47, P3 = L48-63.
Arms (recipient <- donor, p = f):  F2 (END(2) <- START(2)) and F5 (END(5) <- START(5)):  P, P1, P2, P3, P12, P13, P23, and route-only arms over
P, P1 and P2 (fa = full-attention layers only, conv = DeltaNet conv taps only, mem = DeltaNet memory entry only).  Ordinal (p = o, END(5) <- END(2)):
K:EL, K:EL+P1, K:EL+P2, K:EL+P3, K:ELP.  Instrument: NOOP (END(5) <- END(5), p = o, all layers), NOOPF (END(2) <- END(2), p = f, all layers).
Readouts: the menu-restricted answer, the menu logits, and the fixed-bank block scores of every head at L35, 39, 43, 47 (rows "?", tail1, answer).
Usage: python s759_frame.py MODEL TAG [--start 0] [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s759_{}_frame"                   # S766 replication reuses this script with --seed 76600 and its own tag


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
    p1_hi, p2_hi = mp(34), mp(47)
    SB = {"1": [l for l in BAND["P"] if l <= p1_hi], "2": [l for l in BAND["P"] if p1_hi < l <= p2_hi], "3": [l for l in BAND["P"] if l > p2_hi]}
    LT = BAND["L"]; LT_DN = [l for l in LT if l in DN_L]
    R_ = lambda r: int(round(r * (nl - 1)))
    CAP = sorted({min(FA_L, key=lambda l: abs(l - R_(x / 63))) for x in (35, 39, 43, 47)})
    RES = [e_hi, l_hi]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
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
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), [float(lg[tid(l)]) for l in labs + extra], out

    print(f"[{a.tag}] P1 {SB['1'][0]}-{SB['1'][-1]} · P2 {SB['2'][0]}-{SB['2'][-1]} · P3 {SB['3'][0]}-{SB['3'][-1]} · CAP {CAP}", flush=True)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.start + a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        if si < a.start:
            continue
        runs = {"E2": ("E", 2), "E5": ("E", 5), "S2": ("S", 2), "S5": ("S", 5)}
        inp = {nm: v.encode(prompt(fr, k, menu), [pic[l] for l in labs]) for nm, (fr, k) in runs.items()}
        ids = {nm: x["input_ids"][0].tolist() for nm, x in inp.items()}
        S_ = len(ids["E2"]); assert all(len(x) == S_ for x in ids.values())
        dif = lambda A, B: [i for i, (x, y) in enumerate(zip(ids[A], ids[B])) if x != y]
        o = dif("E2", "E5"); f = dif("E2", "S2")
        assert len(o) == 1 and len(f) == 1 and dif("S2", "S5") == o and dif("E5", "S5") == f, (o, f)
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

        fa_of = lambda Ls: [l for l in Ls if l in FA_L]
        dn_of = lambda Ls: [l for l in Ls if l in DN_L]
        sub = lambda st: [li for c in st for li in SB[c]]
        keep("NOOP", oii("E5", "E5", o, full=range(nl)))
        keep("NOOPF", oii("E2", "E2", f, full=range(nl)))
        for fam, rcp, don in (("F2", "E2", "S2"), ("F5", "E5", "S5")):
            keep(f"{fam}:P", oii(rcp, don, f, full=BAND["P"]))
            for st in ("1", "2", "3", "12", "13", "23"):
                keep(f"{fam}:P{st}", oii(rcp, don, f, full=sub(st)))
            for nm, Ls in (("P", BAND["P"]), ("P1", SB["1"]), ("P2", SB["2"])):
                keep(f"{fam}:{nm}fa", oii(rcp, don, f, full=fa_of(Ls)))
                keep(f"{fam}:{nm}conv", oii(rcp, don, f, conv=dn_of(Ls)))
                keep(f"{fam}:{nm}mem", oii(rcp, don, f, mem=dn_of(Ls)))
        keep("K:EL", oii("E5", "E2", o, full=bands("EL")))
        for st in ("1", "2", "3"):
            keep(f"K:EL+P{st}", oii("E5", "E2", o, full=bands("EL") + sub(st)))
        keep("K:ELP", oii("E5", "E2", o, full=range(nl)))
        meta.append(rec)
        del nat
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['land'])} arms · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
              flush=True)
    for h in hooks:
        h.remove()
    QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, BAND={k: [x[0], x[-1]] for k, x in BAND.items()}, SB={k: [x[0], x[-1]] for k, x in SB.items()}, CAP=CAP,
                   rows=["qm", "tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=0); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--seed", type=int, default=75900)
    main(ap.parse_args())
