"""S778: are the turn-boundary rows reading sites?  S777: in START the boundary row reads the target image at L35-47 and carries the answer's
content, in END it does not.  Row-specific query transplant (sufficiency) and attention knockout (necessity) at the full-attention layers L35-47.
Pre-registration in knowledge/DECISIONS.md (2026-10-14 12:2x) before any S778 data; analysis exp/s778_analyze.py.
Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count, wording "last"; cells N = 7 and N = 8, 40 new scenes each.
Per scene E5 (target ta = N - 4), Sa = START(ta), E2 (target tb = N - 1), Sb = START(tb).  Row groups among the question rows: Q = "?" to the
row before <|im_end|> (question end and menu), B = the five turn-boundary rows, R = the rows after B (think markers and the answer row), ALL.
FIX@G: E5's rows G use Sa's queries at L35, L39, L43, L47 (keys, values, gates and the other rows E5's own; exp/c_qpath.py).
KO@G: rows G may not attend to any image token at L35-47 (the others unchanged), in Sa and in E2;  KOlate@R: the same for R at L51-59.
Readouts: the menu-restricted answer and menu logits; block scores of every head at L35-59 and their menu-word weights (rows tail1, answer).
Usage: python s778_readrows.py MODEL TAG --N 7 [--n 40] [--device_map balanced]
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
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
PHR = {"A": ("last", "first"), "B": ("end", "start")}
OUT = "/vlm/results/r3/s778_{}_readrows"


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

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, menu, extra, cfg=None):
        clear(); TOOL.set(cfg)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); clear(); TOOL.set(None)
        lab = menu[int(torch.stack([lg[aid(c)] for c in menu]).argmax())]
        return dict(t1m=int(int(lg.argmax()) in [aid(c) for c in menu]), word=lab, pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[aid(l)]) for l in labs + extra]), st

    ta, tb = N - 4, N - 1
    QS = {"E5": ("E", 5), "Sa": ("S", ta), "E2": ("E", 2), "Sb": ("S", tb)}
    print(f"[{a.tag}] N {N} · ta {ta} · tb {tb} · readers {RB}", flush=True)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        inp = {nm: v.encode(prompt(fr, k, menu), [pic[l] for l in labs]) for nm, (fr, k) in QS.items()}
        ids = {nm: x["input_ids"][0].tolist() for nm, x in inp.items()}
        S_ = len(ids["E5"]); assert all(len(x) == S_ for x in ids.values())
        vs = [i for i, t in enumerate(ids["E5"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E5"]) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5"][i]]))
        uend = max(i for i, t in enumerate(ids["E5"]) if t == T("<|im_end|>"))
        assert v.tok.decode([ids["E5"][uend + 3]]).strip() == "assistant"
        rows = list(range(qm, S_)); ix = {p_: i for i, p_ in enumerate(rows)}
        GRP = {"Q": list(range(qm, uend)), "B": list(range(uend, uend + 5)), "R": list(range(uend + 5, S_))}
        GRP["BR"] = GRP["B"] + GRP["R"]; GRP["ALL"] = rows
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        IMG = [p_ for p_ in range(S_) if owner[p_] >= 0]
        opt = [next(i for i in range(qm, uend) if ids["E5"][i] == tid(w)) for w in labs + extra]
        CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
        rec = dict(si=si, labs=labs, extra=extra, pos=dict(qm=qm, uend=uend, S=S_), ans={})
        base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
        st = {}

        def keep(nm, res):
            rec["ans"][nm] = res[0]
            for li in RD:
                arrays[f"{si}|{nm}|{li}"] = res[1][("cap", li)].numpy().astype(np.float16)
                arrays[f"{si}|{nm}|{li}|opt"] = res[1][("opt", li)].numpy().astype(np.float16)
            return res
        for nm in QS:
            _, st[nm] = keep(nm, run(inp[nm], labs, menu, extra, base(grabq=set(EM))))
        for nm in ("E5", "Sa", "E2"):
            keep(f"NOOP|{nm}", run(inp[nm], labs, menu, extra, base(noop=set(RD))))

        def fix(g):
            ii = torch.tensor([ix[p_] for p_ in GRP[g]])
            ed = {}
            for li in EM:
                qd = st["E5"][("q", li)].clone(); qd[ii] = st["Sa"][("q", li)][ii]
                ed[li] = dict(qd=qd)
            return base(edit=ed, noop=set(RD) - set(EM))

        def ko(g, layers):
            kr = [ix[p_] for p_ in GRP[g]]
            return base(edit={li: dict(ko=(kr, IMG)) for li in layers}, noop=set(RD) - set(layers))
        for g in ("Q", "B", "R", "ALL"):
            keep(f"FIX@{g}|E5", run(inp["E5"], labs, menu, extra, fix(g)))
        for nm in ("Sa", "E2"):
            for g in ("B", "R", "BR", "ALL"):
                keep(f"KO@{g}|{nm}", run(inp[nm], labs, menu, extra, ko(g, EM)))
            keep(f"KOlate@R|{nm}", run(inp[nm], labs, menu, extra, ko("R", RB["late"])))
        meta.append(rec)
        del st
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, N=N, ta=ta, tb=tb, RB=RB, cap_rows=["tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--N", type=int, required=True); ap.add_argument("--phr", default="A")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else 77800 + (a.N - 7)
    main(a)
