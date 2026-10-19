"""S771: the sender -> reader connection.  How does the message that sets the answer reach the full-attention queries at L43-59 (the readers that
control the answer, S767), and where does END(5)'s error enter it?  Query-path patching (exp/c_qpath.py).
Pre-registration in knowledge/DECISIONS.md (2026-10-14 00:2x) before any S771 data; analysis exp/s771_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt, new scenes (seed 77100).  Per scene END(5), START(5), END(2), START(2); F5P1 = END(5) whose rows
after the frame word read START(5)'s frame word at L27-34 only (S758 read interchange, S766 / S768).  Readers: every full-attention layer of L43-59
(mid L43, L47 + late L51, L55, L59 = ML), rows "?" to the answer, every head.
QTOT@band   the reader layers of the band use the donor's whole query (S767; keys, values, gates and all else of the recipient).
DIR_Mb@ML   path patch: at every ML reader the query is computed from (recipient residual + donor-minus-recipient residual at the input of layer b + 1)
            - the donor's message as of layer b, with none of the processing between b and the reader (b = 26: before the frame read; 34: after
            the L27-34 frame read; 42: after the early band L35-42).  DIR_M34c / DIR_M34r: only the three rows after the frame word ("?" and the
            next two, where the L27-34 DeltaNet convolution writes the frame word) / only the other rows.
DIR_P1DN / DIR_P1FA / DIR_P1MLP@ML: sender-specific path patch - the delta is the summed donor-minus-recipient output, at the question rows, of only
            the DeltaNet mixers / only the full-attention mixers / only the MLPs of L27-34 (DIR_M34 - DIR_M26 = the sum of the three, residual additivity).
Pairs: A = END(5) <- F5P1 (the frame read; sufficiency for the literal), B = END(5) <- END(2) (restore END(2)'s connection), C = END(2) <- END(5)
(error reproduction), Nc = END(2) <- START(2) (necessity: END(2)'s readers get a START message).  Every pair: QTOT@ML, DIR_M34, DIR_M42, DIR_P1DN,
DIR_P1FA, DIR_P1MLP; B, C, Nc also DIR_M26; A, B, C also QTOT@mid, QTOT@late; A also DIR_M34c / DIR_M34r, DIR_P1DNFA (both mixers, no MLP) and
DIR_M34 / DIR_P1DN given to the mid or the late readers only (the other band recomputed unchanged).  Instrument: NOOP and DIR0 (zero
difference) for END(5), END(2).  AMP (direction test): the readers of END(5) get alpha x (END(5) - START(5)) residual difference at M34
(alpha = 1, 2, -1) or M42 (alpha = 1) on top of their own state - is the END message a weak pointer to the correct image 3, or only a push away from
the literal?  Control AMP_E2_M34x1 (END(2), reference START(2)).  Readouts: the menu-restricted answer, the menu logits, and at every reader layer the block scores of every head
(rows tail1, answer) for the query each arm uses, with the weights on every menu word (labs + extra order) and the total image weight.
Usage: python s771_qpath.py MODEL TAG [--n 40] [--device_map balanced]
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
OUT = "/vlm/results/r3/s771_{}_qpath"


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
    BND = {"M26": mp(27), "M34": mp(35), "M42": mp(43)}                  # message as of layer b = residual at the input of layer b + 1
    TOOL = QPath(v, RB["ML"], list(BND.values()))

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

    print(f"[{a.tag}] readers {RB} · boundaries {BND} · frame layers {BAND['P'][:8]}", flush=True)
    P1 = [l for l in BAND["P"] if l <= mp(34)]
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
        ve = [i for i, t in enumerate(ids["E5"]) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5"][i]]))
        assert qm == f + 1, (qm, f)
        rows = list(range(qm, S_))
        conv_rows = [i < 3 for i in range(len(rows))]
        vs = [i for i, t in enumerate(ids["E5"]) if t == vs_id]
        uend = max(i for i, t in enumerate(ids["E5"]) if t == T("<|im_end|>"))
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        opt_w = labs + extra
        opt = [next(i for i in range(qm, uend) if ids["E5"][i] == tid(w)) for w in opt_w]           # menu-word positions, in labs + extra order
        CAPC = dict(rows=[uend + 1 - qm, S_ - 1 - qm], owner=owner, nb=N, opt=opt)
        rec = dict(si=si, labs=labs, extra=extra, pos=dict(f=f, qm=qm, S=S_), ans={})
        base = lambda **kw: dict(rows=rows, cap=CAPC, **kw)
        gcfg = lambda: base(grab=set(BND.values()), grabq=set(RB["ML"]))
        nat, st = {}, {}

        def keep(nm, res):
            rec["ans"][nm] = res[0]
            for li in RB["ML"]:
                arrays[f"{si}|{nm}|{li}"] = res[2][("cap", li)].numpy().astype(np.float32)
                arrays[f"{si}|{nm}|{li}|opt"] = res[2][("opt", li)].numpy().astype(np.float16)
                arrays[f"{si}|{nm}|{li}|img"] = res[2][("img", li)].numpy().astype(np.float16)
            return res
        for nm in ("E5", "S5"):
            def setup(nm=nm):
                X.update(grab=True, gpos=[f], srows=rows, sgrab=set(P1))
            _, nat[nm], st[nm] = keep(nm, run(inp[nm], labs, menu, extra, setup, gcfg()))
        G_ = lambda nm, kind, li: nat[nm][(kind, f, li)]

        def f5p1(grab=False):
            def setup():
                X["p"] = f
                for li in P1:
                    X["swap"][li] = G_("S5", "in", li); X["reset"][li] = G_("E5", "out", li)
                if grab:
                    X.update(srows=rows, sgrab=set(P1))
            return setup
        none = lambda: None
        sgrab = lambda: X.update(srows=rows, sgrab=set(P1))
        _, nat["F5P1"], st["F5P1"] = keep("F5P1", run(inp["E5"], labs, menu, extra, f5p1(True), gcfg()))
        for nm in ("E2", "S2"):
            _, nat[nm], st[nm] = keep(nm, run(inp[nm], labs, menu, extra, sgrab, gcfg()))
        ML = RB["ML"]
        for nm in ("E5", "E2"):
            keep(f"NOOP_{nm}", run(inp[nm], labs, menu, extra, none, base(noop=set(ML))))
            z0 = {li: dict(delta=torch.zeros_like(st[nm][("res", BND["M34"])])) for li in ML}
            keep(f"DIR0_{nm}", run(inp[nm], labs, menu, extra, none, base(edit=z0)))

        def qtot(don, band):
            return base(edit={li: dict(qd=st[don][("q", li)]) for li in RB[band]}, noop=set(ML) - set(RB[band]))

        def dirb(rcp, don, b, rmask=None):
            d = st[don][("res", BND[b])] - st[rcp][("res", BND[b])]
            return base(edit={li: dict(delta=d, rmask=rmask) for li in ML})

        def dirs(rcp, don, kind, band="ML"):                           # sender-specific: summed P1 sublayer output differences
            parts = {"DN": [("mix", l) for l in P1 if l in DN_L], "FA": [("mix", l) for l in P1 if l in FA_L], "MLP": [("mlp", l) for l in P1]}
            parts["DNFA"] = parts["DN"] + parts["FA"]
            d = sum(nat[don][k].float().cpu() - nat[rcp][k].float().cpu() for k in parts[kind])
            return base(edit={li: dict(delta=d) for li in RB[band]}, noop=set(ML) - set(RB[band]))

        def dirb_band(rcp, don, b, band):
            d = st[don][("res", BND[b])] - st[rcp][("res", BND[b])]
            return base(edit={li: dict(delta=d) for li in RB[band]}, noop=set(ML) - set(RB[band]))
        for pr, rcp, don in (("A", "E5", "F5P1"), ("B", "E5", "E2"), ("C", "E2", "E5"), ("Nc", "E2", "S2")):
            keep(f"{pr}_QTOT@ML", run(inp[rcp], labs, menu, extra, none, qtot(don, "ML")))
            if pr in ("A", "B", "C"):
                for band in ("mid", "late"):
                    keep(f"{pr}_QTOT@{band}", run(inp[rcp], labs, menu, extra, none, qtot(don, band)))
            for b in (("M34", "M42") if pr == "A" else ("M26", "M34", "M42")):
                keep(f"{pr}_DIR_{b}", run(inp[rcp], labs, menu, extra, none, dirb(rcp, don, b)))
            for kind in ("DN", "FA", "MLP"):
                keep(f"{pr}_DIR_P1{kind}", run(inp[rcp], labs, menu, extra, none, dirs(rcp, don, kind)))
            if pr == "A":
                keep("A_DIR_M34c", run(inp[rcp], labs, menu, extra, none, dirb(rcp, don, "M34", conv_rows)))
                keep("A_DIR_M34r", run(inp[rcp], labs, menu, extra, none, dirb(rcp, don, "M34", [not x for x in conv_rows])))
                keep("A_DIR_P1DNFA", run(inp[rcp], labs, menu, extra, none, dirs(rcp, don, "DNFA")))
                for band in ("mid", "late"):
                    keep(f"A_DIR_M34@{band}", run(inp[rcp], labs, menu, extra, none, dirb_band(rcp, don, "M34", band)))
                    keep(f"A_DIR_P1DN@{band}", run(inp[rcp], labs, menu, extra, none, dirs(rcp, don, "DN", band)))

        def amp(rcp, ref, b, alpha):                                   # alpha x (recipient - START reference) at boundary b, into the readers only
            d = alpha * (st[rcp][("res", BND[b])] - st[ref][("res", BND[b])])
            return base(edit={li: dict(delta=d) for li in ML})
        for b, alpha in (("M34", 1.0), ("M34", 2.0), ("M34", -1.0), ("M42", 1.0)):
            keep(f"AMP_E5_{b}x{alpha:g}", run(inp["E5"], labs, menu, extra, none, amp("E5", "S5", b, alpha)))
        keep("AMP_E2_M34x1", run(inp["E2"], labs, menu, extra, none, amp("E2", "S2", "M34", 1.0)))
        meta.append(rec)
        del nat, st
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, RB=RB, BND=BND, P1=P1, cap_rows=["tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--seed", type=int, default=77100)
    main(ap.parse_args())
