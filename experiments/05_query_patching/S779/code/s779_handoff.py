"""S779: the address-to-answer handoff.  Does the repairing query route the lookup to a physical image position (address) or carry the donor's
answer (content), does the paper's repair R3 work through the same queries and parts, and do natural END errors and R3 answers read the image they
name?  Pre-registration in knowledge/DECISIONS.md (2026-10-14 14:3x) before any S779 data; analysis exp/s779_analyze.py.
Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count, wording "last".  E5 = END(5) (target ta = N - 4), S = START(ta).
Scenes per draw: A (original); B = A with images ta and N exchanged (B's correct END(5) answer is A's image-N word; A's image-ta word now sits at
B's image N); B2 = B with image ta replaced by a picture of the first extra menu word; A_rep_i = A with image i replaced by a picture of the second
extra word, i in {j = ta, a = END(5)'s answered image (if correct: the highest-logit other image), r = the lowest-logit remaining image}; A_rev = A with
the menu order reversed.  Every scene keeps the prompt's tokens and positions (A_rev permutes the menu words).
R3 = the paper's repair (exp/r3_select.build: model-specific READ, RESOLVE t = N_hat + 1 - k_hat, REBIND writes at its write layer).
P1 (query address vs content): the queries of S in A at the full-attention layers (exp/c_qpath.py; keys, values, gates of the recipient) given to E5 in
A (repair), in B and in B2, for the rows V = boundary rows at L35-47 (B47), every question row at L35-47 (ALL47) or at L35-59 (ALL59).
P2 (R3 mediation): R3 in A; R3 with E5's queries at ALL47 / ALL59 (necessity); E5 with R3's queries at ALL59 (sufficiency); E5 in B and in B2 with
A's R3 queries at ALL59 (R3's address vs content); R3 with E5's image part (D, M) or text part (T) of the L35-59 attention output (exp/c_dmtg.py).
P3 (visual source): E5 and R3 in A_rep_j, A_rep_a, A_rep_r and A_rev.
Readouts: the menu-restricted answer (word and position) and the logits of every menu word; block scores and menu-word weights of every head at the
full-attention layers L35-59 (rows tail1, answer); R3's plan (whether it wrote, k_hat, t, N_hat).
Usage: python s779_handoff.py MODEL TAG --N 7 [--n 40] [--device_map balanced] [--r3base r327b --r3rc rc_27b | --r3cal FILE]
"""
from __future__ import annotations

import argparse, gc, json, random, resource, sys
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
OUT = "/vlm/results/r3/s779_{}_handoff"


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
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(word=lab, pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(w)]) for w in ws]), st, dst

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
        x0 = proc_image(r2.choice(pool[extra[0]]))[0].resize((a.pixels, a.pixels)); x1 = proc_image(r2.choice(pool[extra[1]]))[0].resize((a.pixels, a.pixels))
        ws = labs + extra                                                     # logits kept for every menu word, in this order
        imA = [pic[l] for l in labs]
        imB = list(imA); imB[ta - 1], imB[N - 1] = imA[N - 1], imA[ta - 1]
        labsB = list(labs); labsB[ta - 1], labsB[N - 1] = labs[N - 1], labs[ta - 1]
        imB2 = list(imB); imB2[ta - 1] = x0; labsB2 = list(labsB); labsB2[ta - 1] = extra[0]
        tE, tS = prompt("E", 5, menu), prompt("S", ta, menu)
        inp = dict(E5_A=v.encode(tE, imA), S_A=v.encode(tS, imA), E5_B=v.encode(tE, imB), E5_B2=v.encode(tE, imB2))
        ids = {k: x["input_ids"][0].tolist() for k, x in inp.items()}
        S_ = len(ids["E5_A"]); assert all(len(x) == S_ for x in ids.values())
        vs = [i for i, t in enumerate(ids["E5_A"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E5_A"]) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5_A"][i]]))
        uend = max(i for i, t in enumerate(ids["E5_A"]) if t == T("<|im_end|>"))
        assert v.tok.decode([ids["E5_A"][uend + 3]]).strip() == "assistant"
        rows = list(range(qm, S_)); ix = {p_: i for i, p_ in enumerate(rows)}
        Bix = torch.tensor([ix[p_] for p_ in range(uend, uend + 5)])
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        IMG = [p_ for p_ in range(S_) if owner[p_] >= 0]
        opt = [next(i for i in range(qm, uend) if ids["E5_A"][i] == tid(w)) for w in ws]
        CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
        rec = dict(si=si, labs=labs, extra=extra, menu=menu, labsB=labsB, labsB2=labsB2, ans={}, r3={})
        base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
        st, dst = {}, {}

        def keep(nm, res, lab_=None):
            rec["ans"][nm] = res[0]
            for li in RD:
                arrays[f"{si}|{nm}|{li}"] = res[1][("cap", li)].numpy().astype(np.float16)
                arrays[f"{si}|{nm}|{li}|opt"] = res[1][("opt", li)].numpy().astype(np.float16)
            return res
        img_mask = [owner[p_] >= 0 for p_ in range(S_)]
        dcap = lambda: dict(rows=rows, img=img_mask, grab=set(RD))
        for nm, lb in (("E5_A", labs), ("S_A", labs), ("E5_B", labsB), ("E5_B2", labsB2)):
            _, st[nm], dst[nm] = keep(nm, run(inp[nm], lb, ws, menu, base(grabq=set(RD)), dcap() if nm == "E5_A" else None))
        keep("NOOP|E5_A", run(inp["E5_A"], labs, ws, menu, base(noop=set(RD))))
        inR, wR, det = r3plan(tE, imA)
        rec["r3"]["A"] = det
        _, st["R3_A"], _ = keep("R3_A", run(inR, labs, ws, menu, base(grabq=set(RD)), writes=wR))
        LV = {"B47": (EM, Bix), "ALL47": (EM, None), "ALL59": (RD, None)}

        def qcfg(rcp, don, V):
            Ls, rr = LV[V]
            ed = {}
            for li in Ls:
                qd = st[don][("q", li)].clone() if rr is None else st[rcp][("q", li)].clone()
                if rr is not None:
                    qd[rr] = st[don][("q", li)][rr]
                ed[li] = dict(qd=qd)
            return base(edit=ed, noop=set(RD) - set(Ls))
        for V in LV:
            for rcp, lb in (("E5_A", labs), ("E5_B", labsB), ("E5_B2", labsB2)):
                keep(f"{rcp}<S:{V}", run(inp[rcp], lb, ws, menu, qcfg(rcp, "S_A", V)))
        for V in ("ALL47", "ALL59"):
            keep(f"R3_A<E5:{V}", run(inR, labs, ws, menu, qcfg("R3_A", "E5_A", V), writes=wR))
        for rcp, lb in (("E5_A", labs), ("E5_B", labsB), ("E5_B2", labsB2)):
            keep(f"{rcp}<R3:ALL59", run(inp[rcp], lb, ws, menu, qcfg(rcp, "R3_A", "ALL59")))
        for nm, comps in (("IMG", ("D", "M")), ("TXT", ("T",))):
            dc = dict(rows=rows, img=img_mask, give={li: (dst["E5_A"][li], comps) for li in RD})
            keep(f"R3_A<E5:{nm}", run(inR, labs, ws, menu, base(), dc, writes=wR))
        lgA = np.array(rec["ans"]["E5_A"]["ll"][:N])
        a_img = rec["ans"]["E5_A"]["pos"] if rec["ans"]["E5_A"]["pos"] not in (0, ta) else int(np.argsort(-np.where(np.arange(1, N + 1) == ta, -1e9, lgA))[0]) + 1
        rest_ = [i for i in range(1, N + 1) if i not in (ta, a_img)]
        r_img = min(rest_, key=lambda i: lgA[i - 1])
        rec["p3"] = dict(j=ta, a=a_img, r=r_img, a_is_answer=bool(rec["ans"]["E5_A"]["pos"] == a_img))
        for tag_, i_ in (("j", ta), ("a", a_img), ("r", r_img)):
            imR = list(imA); imR[i_ - 1] = x1; lbR = list(labs); lbR[i_ - 1] = extra[1]
            keep(f"E5_rep{tag_}", run(v.encode(tE, imR), lbR, ws, menu, base()))
            inRr, wRr, detr = r3plan(tE, imR); rec["r3"][f"rep{tag_}"] = detr
            keep(f"R3_rep{tag_}", run(inRr, lbR, ws, menu, base(), writes=wRr))
        mrev = list(reversed(menu)); tEr = prompt("E", 5, mrev)
        cap_rev = dict(CAPC); ids_r = v.encode(tEr, imA)["input_ids"][0].tolist()
        cap_rev["opt"] = [next(i for i in range(qm, uend) if ids_r[i] == tid(w)) for w in ws]
        keep("E5_rev", run(v.encode(tEr, imA), labs, ws, mrev, dict(rows=rows, cap=cap_rev)))
        inRv, wRv, detv = r3plan(tEr, imA); rec["r3"]["rev"] = detv
        keep("R3_rev", run(inRv, labs, ws, mrev, dict(rows=rows, cap=cap_rev), writes=wRv))
        meta.append(rec)
        del st, dst
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · R3 wrote {sum(x['act'] or x['fonly'] for x in det)} · "
              f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
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
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else 77900 + (a.N - 7)
    main(a)
