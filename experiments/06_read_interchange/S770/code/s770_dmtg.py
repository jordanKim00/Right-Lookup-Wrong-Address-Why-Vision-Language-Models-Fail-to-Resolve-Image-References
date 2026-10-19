"""S770: which part of the full-attention read at L35-59 carries the frame effect (S768's missing third) and the late query control of START
answers (S767 late .85, image selection alone 0)?  Output split o = G * (M * D + (1 - M) * T) (exp/c_dmtg.py): image mass M, image read D (which images),
text read T (where in the text and what it carries), output gate G.
Pre-registration in knowledge/DECISIONS.md (2026-10-13 23:4x) before any S770 data; analysis exp/s770_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt, new scenes (seed 77000).  Per scene END(5), START(5), START(3), START(6); F5P1 = END(5)
whose rows after the frame word read START(5)'s frame word at L27-34 only (S758 read interchange, as S768).  Rows "?" to the answer; bands mid L43, L47,
late L51, L55, L59, all L35-59; every band layer recomputed in float32 in every edited arm.
END arms (recipient END(5), donor F5P1): GIVE_X@{mid, late, all} for X in D, M, T, Tw (donor text weights on the recipient's values), G, DM, DG, DMT, DMTG;
OUT@all (donor module outputs, instrument comparison);  REV_X@all (recipient F5P1, donor END(5)) for X in D, M, T, G, DMTG;
E_Q@{early, mid, late, all}: END(5) uses F5P1's query (weights on its own keys and values, S767) - does the frame effect reach the answer through
the band queries;  REV_Q@all: F5P1 uses END(5)'s query at every band layer.
START arms (recipient START(3), donor START(6)): GIVE_X@{mid, late} for X in D, M, T, Tw, G, DMTG;  Q_X@{mid, late}: weights from START(6)'s query on
START(3)'s keys (S767), X in DMT (the whole query swap), M, T (Q_D is GIVE_D: the image keys are shared).  NOOP_E5, NOOP_F5P1, NOOP_S3.
Readouts: the menu-restricted answer and menu logits; for the natural runs the float16 attention of rows "?", tail1 and the answer at every band layer.
Usage: python s770_dmtg.py MODEL TAG [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, os, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
from c_dmtg import DMTG
import transformers.models.qwen3_5.modeling_qwen3_5 as QM

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s770_{}_dmtg"


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
    PB = {"early": [l for l in FA_L if mp(35) <= l <= mp(39)], "mid": [l for l in FA_L if mp(43) <= l <= mp(47)],
          "late": [l for l in FA_L if mp(51) <= l <= mp(59)]}
    PB["all"] = PB["early"] + PB["mid"] + PB["late"]
    AL = PB["all"]
    TOOL = DMTG(v, AL)
    COMP = {"D": ("D",), "M": ("M",), "T": ("T",), "Tw": ("Tw",), "G": ("G",), "DM": ("D", "M"), "DG": ("D", "G"), "DMT": ("D", "M", "T"),
            "DMTG": ("D", "M", "T", "G")}

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, menu, extra, setup, cfg=None):
        clear(); setup(); TOOL.set(cfg)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        g = X["g"]; st = dict(TOOL.store); clear(); TOOL.set(None)
        lab = menu[int(torch.stack([lg[aid(c)] for c in menu]).argmax())]
        return dict(t1m=int(int(lg.argmax()) in [aid(c) for c in menu]), pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[aid(l)]) for l in labs + extra]), g, st

    print(f"[{a.tag}] bands {PB} · frame layers {BAND['P'][:8]}", flush=True)
    P1 = [l for l in BAND["P"] if l <= mp(34)]
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        runs = {"E5": ("E", 5), "S5": ("S", 5), "S3": ("S", 3), "S6": ("S", 6)}
        inp = {nm: v.encode(prompt(fr, k, menu), [pic[l] for l in labs]) for nm, (fr, k) in runs.items()}
        ids = {nm: x["input_ids"][0].tolist() for nm, x in inp.items()}
        S_ = len(ids["E5"]); assert all(len(x) == S_ for x in ids.values())
        f = [i for i, (x, y) in enumerate(zip(ids["E5"], ids["S5"])) if x != y]; assert len(f) == 1; f = f[0]
        vs = [i for i, t in enumerate(ids["E5"]) if t == vs_id]; ve = [i for i, t in enumerate(ids["E5"]) if t == ve_id]
        assert ve[-1] < f - 4
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E5"][i]]))
        uend = max(i for i, t in enumerate(ids["E5"]) if t == T("<|im_end|>"))
        rows = list(range(qm, S_)); keep_rows = [0, uend + 1 - qm, S_ - 1 - qm]
        img = [False] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                img[p_] = True
        opt = {w: next(i for i in range(qm, uend) if ids["E5"][i] == tid(w)) for w in menu}
        rec = dict(si=si, labs=labs, extra=extra, menu=menu, pos=dict(o=f - 4, f=f, qm=qm, uend=uend, S=S_, vs=vs, ve=ve, opt=opt), ans={})
        base = lambda **kw: dict(rows=rows, img=img, **kw)
        gcfg = lambda: base(grab=set(AL), keep_att=set(AL), keep_rows=keep_rows)
        nat = {}
        st = {}
        for nm in ("E5", "S5"):
            def setup(nm=nm):
                X.update(grab=True, gpos=[f])
            rec["ans"][nm], nat[nm], st[nm] = run(inp[nm], labs, menu, extra, setup, gcfg())
        G_ = lambda nm, kind, li: nat[nm][(kind, f, li)]

        def f5p1():
            def setup():
                X["p"] = f
                for li in P1:
                    X["swap"][li] = G_("S5", "in", li); X["reset"][li] = G_("E5", "out", li)
            return setup
        none = lambda: None
        rec["ans"]["F5P1"], _, st["F5P1"] = run(inp["E5"], labs, menu, extra, f5p1(), gcfg())
        for nm in ("S3", "S6"):
            rec["ans"][nm], _, st[nm] = run(inp[nm], labs, menu, extra, none, gcfg())
        for nm in ("E5", "F5P1", "S3", "S6"):
            for li in AL:
                arrays[f"{si}|{nm}|{li}"] = st[nm][li]["att"].numpy()
        noop = lambda used: set(AL) - set(used)
        rec["ans"]["NOOP_E5"] = run(inp["E5"], labs, menu, extra, none, base(noop=set(AL)))[0]
        rec["ans"]["NOOP_F5P1"] = run(inp["E5"], labs, menu, extra, f5p1(), base(noop=set(AL)))[0]
        rec["ans"]["NOOP_S3"] = run(inp["S3"], labs, menu, extra, none, base(noop=set(AL)))[0]
        for b in ("mid", "late", "all"):
            for x in ("D", "M", "T", "Tw", "G", "DM", "DG", "DMT", "DMTG"):
                rec["ans"][f"GIVE_{x}@{b}"] = run(inp["E5"], labs, menu, extra, none,
                                                  base(give={li: (st["F5P1"][li], COMP[x]) for li in PB[b]}, noop=noop(PB[b])))[0]
        rec["ans"]["OUT@all"] = run(inp["E5"], labs, menu, extra, none, base(out={li: st["F5P1"][li]["out"] for li in AL}))[0]
        for x in ("D", "M", "T", "G", "DMTG"):
            rec["ans"][f"REV_{x}@all"] = run(inp["E5"], labs, menu, extra, f5p1(), base(give={li: (st["E5"][li], COMP[x]) for li in AL}))[0]
        for b in ("early", "mid", "late", "all"):
            rec["ans"][f"E_Q@{b}"] = run(inp["E5"], labs, menu, extra, none,
                                         base(qsrc={li: (st["F5P1"][li]["q"], COMP["DMT"]) for li in PB[b]}, noop=noop(PB[b])))[0]
        rec["ans"]["REV_Q@all"] = run(inp["E5"], labs, menu, extra, f5p1(), base(qsrc={li: (st["E5"][li]["q"], COMP["DMT"]) for li in AL}))[0]
        for b in ("mid", "late"):
            for x in ("D", "M", "T", "Tw", "G", "DMTG"):
                rec["ans"][f"S_GIVE_{x}@{b}"] = run(inp["S3"], labs, menu, extra, none,
                                                    base(give={li: (st["S6"][li], COMP[x]) for li in PB[b]}, noop=noop(PB[b])))[0]
            for x in ("DMT", "M", "T"):
                rec["ans"][f"S_Q_{x}@{b}"] = run(inp["S3"], labs, menu, extra, none,
                                                 base(qsrc={li: (st["S6"][li]["q"], COMP[x]) for li in PB[b]}, noop=noop(PB[b])))[0]
        meta.append(rec)
        del nat, st
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · {len(rec['ans'])} answers · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    TOOL.remove(); QM.causal_conv1d_fn = orig_conv
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, nl=nl, PB=PB, P1=P1, rows_kept=["qm", "tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--seed", type=int, default=77000)
    main(ap.parse_args())
