"""S773: which sublayer writes the address the L43-59 readers use - near END(2): which part of L35-42 adds the END(2) address to the message
(S771: L34 message .325 of the disruption, L42 message .875); far END(5): which part of L43-50 builds the late readers' error-carrying query
(S771: L42 message .20 of the error reproduction, the late queries .60).  Sender-specific query-path patching (exp/c_qpath.py).
Pre-registration in knowledge/DECISIONS.md (2026-10-14 01:4x) before any S773 data; analysis exp/s773_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt, new scenes (seed 77300).  Per scene END(5), START(5), END(2), START(2); F5P1 as S771.
Message as of layer b = residual at the input of layer b + 1 (M34 = L35 input, M42 = L43 input, M50 = L51 input).  Sublayer writes: the summed
donor-minus-recipient outputs, at the question rows, of the full-attention mixer of one layer (FA35, FA39, FA43, FA47), of all DeltaNet mixers of a
band (DN35-42: L36-38, L40-42; DN43-50: L44-46, L48-50) or of all MLPs of a band (MLP35-42, MLP43-50).
Near (readers L43-59, every head): Nc = END(2) <- START(2) and B = END(5) <- END(2): DIR_M34, DIR_M42, M34 + each L35-42 write; Nc also the
L35-42 writes alone (no M34).  Far (late readers L51-59 only; L43, L47 recomputed unchanged): C = END(2) <- END(5) and A = END(5) <- F5P1:
QTOT@late, DIR_M42, DIR_M50, M42 + each L43-50 write.  Instrument: NOOP and DIR0 for END(5), END(2).
Readouts: the menu-restricted answer and menu logits; at every reader layer the block scores of every head and the weights on every menu word
(rows tail1, answer).
Usage: python s773_writers.py MODEL TAG [--n 40] [--device_map balanced]
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
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s773_{}_writers"


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
    RB = {"mid": [l for l in FA_L if mp(43) <= l <= mp(47)], "late": [l for l in FA_L if mp(51) <= l <= mp(59)]}
    RB["ML"] = RB["mid"] + RB["late"]
    BND = {"M34": mp(35), "M42": mp(43), "M50": mp(51)}
    TOOL = QPath(v, RB["ML"], list(BND.values()))
    ML = RB["ML"]
    E1 = [l for l in range(mp(35), mp(43))]; E2_ = [l for l in range(mp(43), mp(51))]           # L35-42, L43-50
    W = {"FA35": [("mix", l) for l in E1 if l in FA_L][:1], "FA39": [("mix", l) for l in E1 if l in FA_L][1:2],
         "DN35-42": [("mix", l) for l in E1 if l in DN_L], "MLP35-42": [("mlp", l) for l in E1],
         "FA43": [("mix", l) for l in E2_ if l in FA_L][:1], "FA47": [("mix", l) for l in E2_ if l in FA_L][1:2],
         "DN43-50": [("mix", l) for l in E2_ if l in DN_L], "MLP43-50": [("mlp", l) for l in E2_]}
    W["ALL35-42"] = W["FA35"] + W["FA39"] + W["DN35-42"] + W["MLP35-42"]

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, menu, extra, setup, cfg=None):
        clear(); setup(); TOOL.set(cfg)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        g = X["g"]; st = dict(TOOL.store); clear(); TOOL.set(None)
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs + extra]), g, st

    print(f"[{a.tag}] readers {RB} · boundaries {BND} · writers { {k: [x[1] for x in v_] for k, v_ in W.items()} }", flush=True)
    P1 = [l for l in BAND["P"] if l <= mp(34)]
    SG = set(E1 + E2_)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        runs = {"E5": ("E", 5), "S5": ("S", 5), "E2": ("E", 2), "S2": ("S", 2)}
        inp = {nm: v.encode(prompt(fr, k, menu), [pic[l] for l in labs]) for nm, (fr, k) in runs.items()}
        ids = {nm: x["input_ids"][0].tolist() for nm, x in inp.items()}
        S_ = len(ids["E5"]); assert all(len(x) == S_ for x in ids.values())
        f = [i for i, (x, y) in enumerate(zip(ids["E5"], ids["S5"])) if x != y]; assert len(f) == 1; f = f[0]
        vs = [i for i, t in enumerate(ids["E5"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E5"]) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5"][i]]))
        uend = max(i for i, t in enumerate(ids["E5"]) if t == T("<|im_end|>"))
        rows = list(range(qm, S_))
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        opt = [next(i for i in range(qm, uend) if ids["E5"][i] == tid(w)) for w in labs + extra]
        CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
        rec = dict(si=si, labs=labs, extra=extra, pos=dict(f=f, qm=qm, S=S_), ans={})
        base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
        gcfg = lambda: base(grab=set(BND.values()), grabq=set(ML))
        nat, st = {}, {}

        def keep(nm, res):
            rec["ans"][nm] = res[0]
            for li in ML:
                arrays[f"{si}|{nm}|{li}"] = res[2][("cap", li)].numpy().astype(np.float32)
                arrays[f"{si}|{nm}|{li}|opt"] = res[2][("opt", li)].numpy().astype(np.float16)
            return res
        for nm in ("E5", "S5"):
            def setup(nm=nm):
                X.update(grab=True, gpos=[f], srows=rows, sgrab=SG)
            _, nat[nm], st[nm] = keep(nm, run(inp[nm], labs, menu, extra, setup, gcfg()))
        G_ = lambda nm, kind, li: nat[nm][(kind, f, li)]

        def f5p1():
            def setup():
                X["p"] = f
                for li in P1:
                    X["swap"][li] = G_("S5", "in", li); X["reset"][li] = G_("E5", "out", li)
                X.update(srows=rows, sgrab=SG)
            return setup
        none = lambda: None
        sg = lambda: X.update(srows=rows, sgrab=SG)
        _, nat["F5P1"], st["F5P1"] = keep("F5P1", run(inp["E5"], labs, menu, extra, f5p1(), gcfg()))
        for nm in ("E2", "S2"):
            _, nat[nm], st[nm] = keep(nm, run(inp[nm], labs, menu, extra, sg, gcfg()))
        for nm in ("E5", "E2"):
            keep(f"NOOP_{nm}", run(inp[nm], labs, menu, extra, none, base(noop=set(ML))))
            z0 = {li: dict(delta=torch.zeros_like(st[nm][("res", BND["M34"])])) for li in ML}
            keep(f"DIR0_{nm}", run(inp[nm], labs, menu, extra, none, base(edit=z0)))
        resd = lambda rcp, don, b: st[don][("res", BND[b])] - st[rcp][("res", BND[b])]
        wd = lambda rcp, don, w: sum(nat[don][k].float().cpu() - nat[rcp][k].float().cpu() for k in W[w])

        def give(rcp, d, band):
            return base(edit={li: dict(delta=d) for li in RB[band]}, noop=set(ML) - set(RB[band]))
        for pr, rcp, don in (("Nc", "E2", "S2"), ("B", "E5", "E2")):
            keep(f"{pr}_DIR_M34", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M34"), "ML")))
            keep(f"{pr}_DIR_M42", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M42"), "ML")))
            for w in ("FA35", "FA39", "DN35-42", "MLP35-42"):
                keep(f"{pr}_M34+{w}", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M34") + wd(rcp, don, w), "ML")))
            if pr == "Nc":
                keep("Nc_ALL35-42", run(inp[rcp], labs, menu, extra, none, give(rcp, wd(rcp, don, "ALL35-42"), "ML")))
        for pr, rcp, don in (("C", "E2", "E5"), ("A", "E5", "F5P1")):
            keep(f"{pr}_QTOT@late", run(inp[rcp], labs, menu, extra, none,
                                        base(edit={li: dict(qd=st[don][("q", li)]) for li in RB["late"]}, noop=set(RB["mid"]))))
            keep(f"{pr}_DIR_M42", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M42"), "late")))
            keep(f"{pr}_DIR_M50", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M50"), "late")))
            for w in ("FA43", "FA47", "DN43-50", "MLP43-50"):
                keep(f"{pr}_M42+{w}", run(inp[rcp], labs, menu, extra, none, give(rcp, resd(rcp, don, "M42") + wd(rcp, don, w), "late")))
        meta.append(rec)
        del nat, st
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, RB=RB, BND=BND, W={k: [x[1] for x in v_] for k, v_ in W.items()}, cap_rows=["tail1", "ans"],
                   meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--seed", type=int, default=77300)
    main(ap.parse_args())
