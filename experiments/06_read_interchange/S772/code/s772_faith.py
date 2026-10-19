"""S772: faithfulness of the END circuit on new scenes, both image counts and a second END phrasing.  Pre-registration in knowledge/DECISIONS.md (2026-10-14 01:1x)
before any S772 data; analysis exp/s772_analyze.py.
Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count; phrasing A "from the last" / "from the first", phrasing B
"from the end" / "from the start" (one-token frame words, the ordinal four tokens before).  Per scene and k = 2..6: END(k), START(k) and
START(k') with k' = N + 1 - k (the image END(k) asks for; START(k') = START(k) when k' = k).
Arms per (scene, k):  F_P1 / F_P12 = END(k) whose rows after the frame word read START(k)'s frame word at L27-34 / L27-47 (S758 read
interchange, all routes);  R_P1 = START(k) reading END(k)'s frame word at L27-34;  Q_ES = START(k') using END(k)'s queries at every full-attention
layer of L43-59 (rows "?" to the answer, exp/c_qpath.py; keys, values, gates of the recipient);  Q_SE = END(k) using START(k')'s queries there;
D42 = START(k) whose L43-59 readers compute their query from (own residual + END(k) - START(k) residual difference at the input of L43): is the
END address already in the message after L42 (S771: yes for END(2), no for END(5))?
Readouts: the menu-restricted answer and menu logits; at every L43-59 reader layer the block scores of every head, the weights on every menu word
and the total image weight (rows tail1, answer).
Usage: python s772_faith.py MODEL TAG --N 7 --phr A [--n 80] [--device_map balanced]
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
OUT = "/vlm/results/r3/s772_{}_faith"


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
    RB = {"mid": [l for l in FA_L if mp(43) <= l <= mp(47)], "late": [l for l in FA_L if mp(51) <= l <= mp(59)]}
    RB["ML"] = RB["mid"] + RB["late"]
    B43 = mp(43)                                                         # message as of L42 = residual at the input of L43
    TOOL = QPath(v, RB["ML"], [B43])
    ML = RB["ML"]
    fwE, fwS = PHR[a.phr]

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{fwE if fr == 'E' else fwS}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, menu, extra, setup, cfg=None):
        clear(); setup(); TOOL.set(cfg)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        g = X["g"]; st = dict(TOOL.store); clear(); TOOL.set(None)
        lab = menu[int(torch.stack([lg[aid(c)] for c in menu]).argmax())]
        return dict(t1m=int(int(lg.argmax()) in [aid(c) for c in menu]), pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[aid(l)]) for l in labs + extra]), g, st

    P1 = [l for l in BAND["P"] if l <= mp(34)]
    P12 = [l for l in BAND["P"] if l <= mp(47)]
    KS = [k for k in range(2, 7) if k <= N - 1]
    print(f"[{a.tag}] N {N} · phrasing {a.phr} ({fwE}/{fwS}) · k {KS} · readers {RB} · P1 {P1[0]}-{P1[-1]} · P12 {P12[0]}-{P12[-1]}", flush=True)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, labs=labs, extra=extra, ans={})
        cache = {}

        def encode(fr, k):
            if (fr, k) not in cache:
                cache[(fr, k)] = v.encode(prompt(fr, k, menu), [pic[l] for l in labs])
            return cache[(fr, k)]
        for k in KS:
            kp = N + 1 - k
            iE, iS, iP = encode("E", k), encode("S", k), encode("S", kp)
            ids = {nm: x["input_ids"][0].tolist() for nm, x in (("E", iE), ("S", iS), ("P", iP))}
            S_ = len(ids["E"]); assert all(len(x) == S_ for x in ids.values())
            f = [i for i, (x, y) in enumerate(zip(ids["E"], ids["S"])) if x != y]; assert len(f) == 1, f; f = f[0]
            assert v.tok.decode([ids["E"][f - 4]]).strip() == ORDW[k - 1]
            vs = [i for i, t in enumerate(ids["E"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E"]) if t == ve_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E"][i]]))
            uend = max(i for i, t in enumerate(ids["E"]) if t == T("<|im_end|>"))
            rows = list(range(qm, S_))
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            opt = [next(i for i in range(qm, uend) if ids["E"][i] == tid(w)) for w in labs + extra]
            CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
            base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
            nat, st = {}, {}

            def keep(nm, res):
                rec["ans"][f"{nm}|{k}"] = res[0]
                for li in ML:
                    arrays[f"{si}|{k}|{nm}|{li}"] = res[2][("cap", li)].numpy().astype(np.float32)
                    arrays[f"{si}|{k}|{nm}|{li}|opt"] = res[2][("opt", li)].numpy().astype(np.float16)
                    arrays[f"{si}|{k}|{nm}|{li}|img"] = res[2][("img", li)].numpy().astype(np.float16)
                return res
            for nm, x in (("E", iE), ("S", iS)):
                def setup():
                    X.update(grab=True, gpos=[f])
                _, nat[nm], st[nm] = keep(nm, run(x, labs, menu, extra, setup, base(grabq=set(ML), grab={B43})))
            if kp != k:
                _, _, st["P"] = keep("P", run(iP, labs, menu, extra, lambda: None, base(grabq=set(ML))))
            else:
                st["P"] = st["S"]
            G_ = lambda nm, kind, li: nat[nm][(kind, f, li)]

            def frame(rcp, don, Ls):
                def setup():
                    X["p"] = f
                    for li in Ls:
                        X["swap"][li] = G_(don, "in", li); X["reset"][li] = G_(rcp, "out", li)
                return setup
            none = lambda: None
            keep("F_P1", run(iE, labs, menu, extra, frame("E", "S", P1), base()))
            keep("F_P12", run(iE, labs, menu, extra, frame("E", "S", P12), base()))
            keep("R_P1", run(iS, labs, menu, extra, frame("S", "E", P1), base()))
            keep("Q_ES", run(iP, labs, menu, extra, none, base(edit={li: dict(qd=st["E"][("q", li)]) for li in ML})))
            keep("Q_SE", run(iE, labs, menu, extra, none, base(edit={li: dict(qd=st["P"][("q", li)]) for li in ML})))
            d42 = st["E"][("res", B43)] - st["S"][("res", B43)]
            keep("D42", run(iS, labs, menu, extra, none, base(edit={li: dict(delta=d42) for li in ML})))
            del nat, st
        meta.append(rec)
        del cache
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, N=N, phr=a.phr, KS=KS, RB=RB, P1=P1, P12=P12, cap_rows=["tail1", "ans"], meta=meta),
              open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--N", type=int, required=True); ap.add_argument("--phr", required=True)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else 77200 + 10 * a.N + (0 if a.phr == "A" else 1)
    main(a)
