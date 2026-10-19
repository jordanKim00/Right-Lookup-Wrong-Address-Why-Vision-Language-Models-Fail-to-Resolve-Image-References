"""S781: the same-scene positive control for S779 / S780.  Can late queries formed in a scene point to that scene's own position?  R3 is run
afresh in the rearranged scenes B (images ta and N exchanged) and B2 (B's image ta replaced by the first extra word's picture), and its queries
(and START(ta)'s from the same scene) are given to that scene's END(5); A's R3 queries into B and B2 replicate S779's donor-content arm; A's R3
write plan applied unchanged to B and B2 checks the fixed plan.
Pre-registration in knowledge/DECISIONS.md (2026-10-14 15:1x) before any S781 data; analysis exp/s781_analyze.py.
Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count, wording "last"; cells N = 7 and N = 8, 40 new scenes each.
Query transplant (exp/c_qpath.py) at every question row of the full-attention layers L35-39 (ALL39), L35-47 (ALL47) or L35-59 (ALL59).
Readouts: the menu-restricted answer (word, position) and every menu word's logit; block scores and menu-word weights at L35-59 (rows tail1, answer).
Usage: python s781_posctl.py MODEL TAG --N 7 [--n 40] [--device_map balanced] [--r3cal FILE]
"""
from __future__ import annotations

import argparse, os, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
from c_qpath import QPath
from c_dmtg import DMTG
import r3_select
from r3_min_md import R3MinMD
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
PHR = {"A": ("last", "first"), "B": ("end", "start")}
OUT = "/vlm/results/r3/s781_{}_posctl"


def main(a, v=None):
    global N
    N = a.N
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
                            g={}, res_grab=False, res_store={}, res_rep={}, cap_on=False, cap={}, srows=None, sgrab=set())
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
            if li in X["sgrab"]:                                          # sender outputs at the question rows (S771)
                X["g"][("mix", li)] = t[0, X["srows"]].detach().clone()
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

    def mlp_hook(li):
        def h(mod, args, out):
            if li in X["sgrab"]:
                X["g"][("mlp", li)] = out[0, X["srows"]].detach().clone()
            return None
        return h
    hooks += [v.layers[li].mlp.register_forward_hook(mlp_hook(li)) for li in range(nl)]
    RB = {"early": [l for l in FA_L if mp(35) <= l <= mp(39)], "mid": [l for l in FA_L if mp(43) <= l <= mp(47)],
          "late": [l for l in FA_L if mp(51) <= l <= mp(59)]}
    EM = RB["early"] + RB["mid"]; RD = EM + RB["late"]
    TOOL = QPath(v, RD, [])
    DT = DMTG(v, RD)
    if a.r3cal:
        rep = R3MinMD(v, a.r3cal, v2=True, det_rule="margin", det_param=.45, anchor_scale=1.2, n_mode="last")      # smoke only
    else:
        rep = r3_select.build(v, a.r3base, a.r3rc)

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, ws, menu, cfg=None, dcfg=None, writes=None):
        clear(); TOOL.set(cfg); DT.set(dcfg); rep.ST = dict(writes=writes) if writes else {}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); dst = dict(DT.store); clear(); TOOL.set(None); DT.set(None); rep.ST = {}
        lab = menu[int(torch.stack([lg[aid(c)] for c in menu]).argmax())]
        return dict(t1m=int(int(lg.argmax()) in [aid(c) for c in menu]), word=lab, pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[aid(w)]) for w in ws]), st, dst

    @torch.no_grad()
    def r3plan(text, pics):
        TOOL.set(None); DT.set(None)
        rep.on = True
        inp, writes = rep.plan(text, pics)
        det = [dict(kh=x["kh"], t=x["t"], N_hat=x["N_hat"], act=x["act"], fonly=x["fonly"]) for x in rep.last_detail]
        return inp, writes, det

    ta = N - 4
    print(f"[{a.tag}] N {N} · END(5) target {ta} · readers {RB} · R3 write layer {rep.Lw}", flush=True)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        r2 = random.Random(a.seed * 1000 + si)
        x0 = proc_image(r2.choice(pool[extra[0]]))[0].resize((a.pixels, a.pixels))
        ws = labs + extra
        imA = [pic[l] for l in labs]
        imB = list(imA); imB[ta - 1], imB[N - 1] = imA[N - 1], imA[ta - 1]
        labsB = list(labs); labsB[ta - 1], labsB[N - 1] = labs[N - 1], labs[ta - 1]
        imB2 = list(imB); imB2[ta - 1] = x0; labsB2 = list(labsB); labsB2[ta - 1] = extra[0]
        tE, tS = prompt("E", 5, menu), prompt("S", ta, menu)
        inp = dict(E5_A=v.encode(tE, imA), E5_B=v.encode(tE, imB), E5_B2=v.encode(tE, imB2), S_B=v.encode(tS, imB), S_B2=v.encode(tS, imB2))
        ids = {k: x["input_ids"][0].tolist() for k, x in inp.items()}
        S_ = len(ids["E5_A"]); assert all(len(x) == S_ for x in ids.values())
        vs = [i for i, t in enumerate(ids["E5_A"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E5_A"]) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5_A"][i]]))
        uend = max(i for i, t in enumerate(ids["E5_A"]) if t == T("<|im_end|>"))
        rows = list(range(qm, S_))
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        opt = [next(i for i in range(qm, uend) if ids["E5_A"][i] == tid(w)) for w in ws]
        CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
        rec = dict(si=si, labs=labs, extra=extra, menu=menu, labsB=labsB, labsB2=labsB2, ans={}, r3={})
        base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
        st = {}

        def keep(nm, res):
            rec["ans"][nm] = res[0]
            for li in RD:
                arrays[f"{si}|{nm}|{li}"] = res[1][("cap", li)].numpy().astype(np.float16)
                arrays[f"{si}|{nm}|{li}|opt"] = res[1][("opt", li)].numpy().astype(np.float16)
            return res
        LB = {"E5_A": labs, "E5_B": labsB, "E5_B2": labsB2, "S_B": labsB, "S_B2": labsB2}
        for nm in ("E5_A", "E5_B", "E5_B2", "S_B", "S_B2"):
            _, st[nm], _ = keep(nm, run(inp[nm], LB[nm], ws, menu, base(grabq=set(RD))))
        keep("NOOP|E5_B", run(inp["E5_B"], labsB, ws, menu, base(noop=set(RD))))
        plans = {}
        for sc, im, lb in (("A", imA, labs), ("B", imB, labsB), ("B2", imB2, labsB2)):
            inR, wR, det = r3plan(tE, im)
            plans[sc] = (inR, wR); rec["r3"][sc] = det
            _, st[f"R3_{sc}"], _ = keep(f"R3_{sc}", run(inR, lb, ws, menu, base(grabq=set(RD)), writes=wR))
        for sc, lb in (("B", labsB), ("B2", labsB2)):
            keep(f"R3fixA_{sc}", run(inp[f"E5_{sc}"], lb, ws, menu, base(), writes=plans["A"][1]))
        LV = {"ALL39": RB["early"], "ALL47": EM, "ALL59": RD}

        def qcfg(don, V):
            return base(edit={li: dict(qd=st[don][("q", li)]) for li in LV[V]}, noop=set(RD) - set(LV[V]))
        for V in LV:
            for sc, lb in (("B", labsB), ("B2", labsB2)):
                keep(f"E5_{sc}<R3_{sc}:{V}", run(inp[f"E5_{sc}"], lb, ws, menu, qcfg(f"R3_{sc}", V)))
                keep(f"E5_{sc}<R3_A:{V}", run(inp[f"E5_{sc}"], lb, ws, menu, qcfg("R3_A", V)))
                keep(f"E5_{sc}<S_{sc}:{V}", run(inp[f"E5_{sc}"], lb, ws, menu, qcfg(f"S_{sc}", V)))
        meta.append(rec)
        del st
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); DT.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, N=N, ta=ta, RB=RB, r3=dict(Lw=rep.Lw, Lb=rep.Lb), cap_rows=["tail1", "ans"], meta=meta),
              open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--N", type=int, required=True); ap.add_argument("--phr", default="A")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--r3base", default="r327b"); ap.add_argument("--r3rc", default="rc_27b"); ap.add_argument("--r3cal", default=None)
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else 78100 + (a.N - 7)
    main(a)
