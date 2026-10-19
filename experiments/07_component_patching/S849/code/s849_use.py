"""S849-U (test A part 2; registered in knowledge/DECISIONS.md before any S849 data).  S848 found a weak goal-attainment ('match') signal in the
answer row at the band input (layer 46) and output (layer 50).  Is it used to stop the update?  On fresh scenes of the same grid (8 pictures,
k = 4, N_s 6-9, early lookup pinned to p 3-6), the answer row's residual is steered at the input of layer 46 or 50 along the frozen match
direction m (research/ces/outputs/s849_match_dirs.npz, from S848 only), by +ALPHA or -ALPHA within-cell SDs of its projection, or along a
random direction r in the same 50-PC subspace (same norm).  Residuals of the answer row at the inputs of layers 48, 50 and 52 are kept (the
steering must persist downstream).
Per scene and N_s: stage 1 = END(4), START(p) donors; stage 2 = QNOOP_E and per p: E_EARLY_p, PLUS46_p, MINUS46_p, RAND46_p, PLUS50_p, MINUS50_p,
RAND50_p.  24 scenes (seed 84910).
Usage: python s849_use.py MODEL TAG [--n 24] [--alpha 8] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_g import QPathG

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s849_{}_{}.json"
DIRS = "/vlm/research/ces/outputs/s849_match_dirs.npz"
CASES = [(8, 4, 5, ns) for ns in (6, 7, 8, 9)]                            # (shown N, k, shown target, stated N_s)
PSET = {8: [3, 4, 5, 6]}


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S849_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]                      # look layers + the last full-attention layers (59, 63)
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]                        # 51, 55, 59, 63 in the 27B
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE))      # smoke models: by relative depth
    assert band and all(b < min(LATE) for b in band)
    SITES = [46, 50] if nl == 64 else sorted({int(round(x * (nl - 1) / 63)) for x in (46, 50)})
    RES = [48, 50, 52] if nl == 64 else sorted({int(round(x * (nl - 1) / 63)) for x in (48, 50, 52)})
    ROW = 9                                                               # the answer row (index in the ten read rows)
    E_ = v.model.config.text_config.hidden_size if hasattr(v.model.config, "text_config") else v.model.config.hidden_size
    dz = np.load(DIRS)
    vec = {}
    for site, tag in zip(SITES, ("L46_r9", "L50_r9")):
        m_, r_, sg = dz[f"m_{tag}"], dz[f"r_{tag}"], float(dz[f"sigma_{tag}"])
        if m_.shape[0] != E_:                                             # smoke models: random unit directions of the right size
            g = np.random.default_rng(site); m_ = g.standard_normal(E_); m_ /= np.linalg.norm(m_); r_ = g.standard_normal(E_); r_ /= np.linalg.norm(r_); sg = 1.0
        vec[site] = dict(m=torch.tensor(m_, dtype=torch.float32), r=torch.tensor(r_, dtype=torch.float32), sigma=sg)
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    scaling = float(v.layers[LOOK[0]].self_attn.scaling)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def sc(lg, s):
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
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
    TOOL = QPathG(v, READ, res_layers=RES, comp_layers=(), patch_comps=(), add_layers=SITES)
    T1 = []

    def prompt(N, q, frame, menu, Ns):
        return v.chat([("user", f"{v.IMG*N}\nThere are {Ns} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs):
        TOOL.set(qs)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None)
        return lg, st

    def runs_for(gis, build):
        """One forward pass for the scenes gis (donor rows of same-pass patches live in the same pass); on out-of-memory split the scenes in
        halves (each sample's computation does not depend on its batch neighbours; src indices are rebuilt per pass)."""
        inps, qs, tag = build(gis)
        try:
            lg, st = run1(inps, qs)
        except torch.cuda.OutOfMemoryError:
            if len(gis) == 1:
                raise
            lg = None
        if lg is None:
            gc.collect(); torch.cuda.empty_cache()
            h_ = len(gis) // 2
            print(f"[{a.tag}] OOM at {len(gis)} scenes -> {h_} + {len(gis) - h_}", flush=True)
            return runs_for(gis[:h_], build) + runs_for(gis[h_:], build)
        return [(gi, nm, lg[b], st[b]) for b, (gi, nm) in enumerate(tag)]

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · res {RES} · matcher layers {sorted(MH)} · scaling {scaling}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe; one scene set for every N_s
        N, k, j = 8, 4, 5
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, N=N, k=k, j=j, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel = []
    for N, k, j, Ns in CASES:
        SC = scenes
        G1 = a.group or max(1, a.max_images // ((1 + len(PSET[N])) * N))
        G2 = a.group2 or max(1, a.max_images // ((1 + 7 * len(PSET[N])) * N))
        print(f"[{a.tag}] N{N} Ns{Ns}: {G1} / {G2} scenes per batch (stage 1 / 2)", flush=True)
        P = []
        for s in SC:
            qk_ = dict(END=(k, "last"), **{f"START_p{p_}": (p_, "first") for p_ in PSET[N]})
            enc = {nm: v.encode(prompt(N, q, fr, s["menu"], Ns), s["pics"]) for nm, (q, fr) in qk_.items()}
            enc["START"] = enc[f"START_p{k}"]
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
            S_ = len(ids["END"])
            ve = [i for i, x in enumerate(ids["END"]) if x == ve_id]; vs = [i for i, x in enumerate(ids["END"]) if x == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["END"][i]]))
            uend = max(i for i, x in enumerate(ids["END"]) if x == end_id)
            assert v.tok.decode([ids["END"][uend + 3]]).strip() == "assistant"
            Fw = [i for i, (x, y) in enumerate(zip(ids["START"], ids["END"])) if x != y]
            assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
            opt = []
            for w_ in s["menu"]:
                pos_ = [i for i in range(qm + 1, uend) if ids["END"][i] == ids1(w_)[0]]
                assert len(pos_) == 1, f"menu word {w_} must be one token once"
                opt.append(pos_[0])
            P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4))
        assert len({p["S"] for p in P}) == 1, "scenes of one case must align for batching"
        capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=N, opt=p["opt"]), **kw)
        recs = [dict(si=p["s"]["si"], N=N, k=k, j=j, Ns=Ns, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={},
                     menu_pos=[(p["s"]["labs"].index(w_) + 1 if w_ in p["s"]["labs"] else 0) for w_ in p["s"]["menu"]]) for p in P]
        def keep(gi, nm, lgb, stb):
            rec = recs[gi]; menu = rec["menu"]
            T1.append(int(int(lgb.argmax()) in t1ids(menu)))
            w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
            rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
            rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
            hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))     # matcher heads where listed (35-55), else all heads (59, 63)
            rec["qk"][nm] = {str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
            rec.setdefault("mass", {})[nm] = {str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist()
                                              for li in READ if ("img", li) in stb}
            key = f"Ns{Ns}_s{rec['si']}_{nm}"
            if ("res", RES[0]) in stb:
                npz[key + "_res"] = np.stack([stb[("res", li)][ROW].float().numpy() for li in RES]).astype(np.float32)  # answer row [RES, E]
        # stage 1: natural runs (END, START(p) donors)
        NAT = ["END"] + [f"START_p{p_}" for p_ in PSET[N]]
        def build1(gis):
            inps, qs, tag = [], [], []
            for gi in gis:
                for nm in NAT:
                    inps.append(P[gi]["enc"][nm]); tag.append((gi, nm)); qs.append(capq(P[gi], grabq=set(EARLY), grabkb=set(LOOK)))
            return inps, qs, tag
        don = {}
        for g0 in range(0, len(P), G1):
            for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G1, len(P)))), build1):
                keep(gi, nm, lgb, stb); nrun += 1
                don[(gi, nm)] = dict(q={li: stb[("q", li)] for li in EARLY}, kb={li: stb[("kb", li)] for li in LOOK})
        for gi in range(len(P)):
            recs[gi]["land"]["START"] = recs[gi]["land"][f"START_p{k}"]; recs[gi]["mlog"]["START"] = recs[gi]["mlog"][f"START_p{k}"]
            kE, kS = don[(gi, "END")]["kb"], don[(gi, f"START_p{k}")]["kb"]
            kr = max(float((kE[li] - kS[li]).abs().max() / kE[li].abs().max()) for li in LOOK); keyrel.append(kr)
            recs[gi]["key_rel_diff"] = kr
        # stage 2: pinned runs and the band swaps (same pass)
        BA = list(range(10))
        def build2(gis):
            inps, qs, tag = [], [], []
            for gi in gis:
                p = P[gi]
                def ed(p_, p=p, gi=gi, **kw):
                    dq = don[(gi, f"START_p{p_}")]["q"]
                    return capq(p, edit={li: [dict(qsub=dict(owner=p["owner"], nb=N, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}, **kw)
                inps.append(p["enc"]["END"]); qs.append(capq(p, noop=set(LOOK))); tag.append((gi, "QNOOP_E"))
                GR = set(RES)
                def radd(site, kind, sign):
                    t = torch.zeros(10, E_, dtype=torch.float32)
                    t[ROW] = sign * a.alpha * vec[site]["sigma"] * vec[site][kind]
                    return {site: t}
                for p_ in PSET[N]:
                    inps.append(p["enc"]["END"]); qs.append(ed(p_, grabres=GR)); tag.append((gi, f"E_EARLY_p{p_}"))
                    for site in SITES:
                        sn = 46 if site == SITES[0] else 50
                        for nm, kind, sign in (("PLUS", "m", 1.0), ("MINUS", "m", -1.0), ("RAND", "r", 1.0)):
                            inps.append(p["enc"]["END"]); qs.append(ed(p_, grabres=GR, radd=radd(site, kind, sign))); tag.append((gi, f"{nm}{sn}_p{p_}"))
            return inps, qs, tag
        for g0 in range(0, len(P), G2):
            for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G2, len(P)))), build2):
                keep(gi, nm, lgb, stb); nrun += 1
            print(f"[{a.tag}] N{N} Ns{Ns} {min(g0 + G2, len(P))}/{len(P)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        out.extend(recs)
        del don, P
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "use"
    json.dump(dict(tag=a.tag, model=a.model, pset=PSET, band=band, comps=COMPS, readout=os.environ.get("R3_READ"), read=READ, look=LOOK, early=EARLY, late=LATE,
                   res_layers=RES, sites=SITES, row=ROW, alpha=a.alpha, sigma={str(s_): vec[s_]["sigma"] for s_ in SITES},
                   matcher_layers=sorted(MH), matchers={str(l): h for l, h in MH.items()}, n_layer=nl, scaling=scaling,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), key_rel_diff_max=max(keyrel), rows=out), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--alpha", type=float, default=8.0); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=84910); ap.add_argument("--group", type=int, default=0); ap.add_argument("--group2", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower
    main(ap.parse_args())
