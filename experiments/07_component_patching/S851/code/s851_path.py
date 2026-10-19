"""S851 (pre-registered in knowledge/DECISIONS.md before any S851 data; user 10-17 late evening, minimal path test after S850): do the late
full-attention queries (layers 51-63) carry the correction that replacing the boundary state (FULLB) produces, and does the steered run (PLUS16)
recover when only its late queries are corrected?  The S850 grid and arms on fresh scenes (8 pictures, k = 4, N_s 6-8, p 3-5 pinned; 30 scenes,
seed 85100).  Per p in one pass: E_EARLY_p (natural END pinned), S_EARLY_p, FULLB_p (boundary rows B0-B4 at the input of layer 46 <- S_EARLY_p),
PLUS16_p (B1 / B4 steered +16 SD at layer 46 along the frozen S848 directions), PLUSQ_p (PLUS16 with the queries of every read row at the late
full-attention layers 51, 55, 59, 63 replaced by FULLB's, same pass: exp/c_qpath_d.py's residual delta at each reader's own layer makes its query
exactly the source run's), FULLBQE_p (FULLB with those late queries replaced by E_EARLY's).  QNOOP_E as the instrument check.
Usage: python s851_path.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_h import QPathH

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s851_{}_{}.json"
DIRS = "/vlm/research/ces/outputs/s850_boundary_dirs.npz"
CASES = [(8, 4, 5, ns) for ns in (6, 7, 8)]                               # (shown N, k, shown target, stated N_s)
PSET = {8: [3, 4, 5]}


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S851_DEVICE", "cuda:0")
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
    SITE = 46 if nl == 64 else int(round(46 * (nl - 1) / 63))           # the writer's input
    RES = [43, 44, 45, 46, 47, 48, 49] if nl == 64 else sorted({int(round(x * (nl - 1) / 63)) for x in (43, 44, 45, 46, 47, 48, 49)})
    SROWS = [2, 5]                                                        # boundary rows B1, B4 (indices in the ten read rows)
    KROWS = [2, 5, 9]                                                     # rows kept for band outputs and late queries (B1, B4, answer)
    LQ = [l for l in LATE if l / (nl - 1) <= .875]                        # late readers 51, 55
    E_ = v.model.config.text_config.hidden_size if hasattr(v.model.config, "text_config") else v.model.config.hidden_size
    dz = np.load(DIRS)
    vec = {}
    for row, tag in zip(SROWS, ("L46_r2", "L46_r5")):
        m_, r_, sg = dz[f"m_{tag}"], dz[f"r_{tag}"], float(dz[f"sigma_{tag}"])
        if m_.shape[0] != E_:                                             # smoke models: random unit directions of the right size
            g = np.random.default_rng(row); m_ = g.standard_normal(E_); m_ /= np.linalg.norm(m_); r_ = g.standard_normal(E_); r_ /= np.linalg.norm(r_); sg = 1.0
        vec[row] = dict(m=torch.tensor(m_, dtype=torch.float32), r=torch.tensor(r_, dtype=torch.float32), sigma=sg)
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
    TOOL = QPathH(v, READ, res_layers=(), comp_layers=(), patch_comps=(), add_layers=[SITE], rpatch_layers=[SITE])
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
        G2 = a.group2 or max(1, a.max_images // ((1 + 6 * len(PSET[N])) * N))
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
                npz[key + "_res"] = np.stack([stb[("res", li)][SROWS].float().numpy() for li in RES]).astype(np.float16)    # [RES, B1/B4, E]
            if ("mix", band[0]) in stb:
                npz[key + "_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)  # [8, rows, E]
            if ("q", LQ[0]) in stb:
                npz[key + "_q"] = np.stack([stb[("q", li)][KROWS].float().numpy() for li in LQ]).astype(np.float16)      # [LQ, rows, H, hd]
            if ("qd", LQ[0]) in stb:
                npz[key + "_qd"] = np.stack([stb[("qd", li)][KROWS].float().numpy() for li in LQ]).astype(np.float16)    # predicted
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
                L4 = set(LATE)
                def svec(kind, alpha):
                    t = torch.zeros(10, E_, dtype=torch.float32)
                    for row in SROWS:
                        t[row] = alpha * vec[row]["sigma"] * vec[row][kind]
                    return t
                bpos = p["rowsC"][1:6]                                     # positions of the boundary rows B0-B4
                for p_ in PSET[N]:
                    iE = len(inps); inps.append(p["enc"]["END"]); qs.append(ed(p_, keepres=L4)); tag.append((gi, f"E_EARLY_p{p_}"))
                    iS = len(inps); inps.append(p["enc"]["START"]); qs.append(ed(p_)); tag.append((gi, f"S_EARLY_p{p_}"))
                    iF = len(inps); inps.append(p["enc"]["END"]); qs.append(ed(p_, rpatch={SITE: dict(src=iS, rows=bpos)}, keepres=L4)); tag.append((gi, f"FULLB_p{p_}"))
                    inps.append(p["enc"]["END"]); qs.append(ed(p_, radd={SITE: svec("m", 16.0)})); tag.append((gi, f"PLUS16_p{p_}"))
                    inps.append(p["enc"]["END"]); qs.append(ed(p_, radd={SITE: svec("m", 16.0)}, keepres=L4, delta={li: dict(src=iF, layer=li) for li in LATE})); tag.append((gi, f"PLUSQ_p{p_}"))
                    inps.append(p["enc"]["END"]); qs.append(ed(p_, rpatch={SITE: dict(src=iS, rows=bpos)}, keepres=L4, delta={li: dict(src=iE, layer=li) for li in LATE})); tag.append((gi, f"FULLBQE_p{p_}"))
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
    nm_ = "path"
    json.dump(dict(tag=a.tag, model=a.model, pset=PSET, band=band, comps=COMPS, readout=os.environ.get("R3_READ"), read=READ, look=LOOK, early=EARLY, late=LATE,
                   res_layers=RES, site=SITE, srows=SROWS, krows=KROWS, lq=LQ, sigma={str(r_): vec[r_]["sigma"] for r_ in SROWS},
                   matcher_layers=sorted(MH), matchers={str(l): h for l, h in MH.items()}, n_layer=nl, scaling=scaling,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), key_rel_diff_max=max(keyrel), rows=out), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85100); ap.add_argument("--group", type=int, default=0); ap.add_argument("--group2", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower
    main(ap.parse_args())
