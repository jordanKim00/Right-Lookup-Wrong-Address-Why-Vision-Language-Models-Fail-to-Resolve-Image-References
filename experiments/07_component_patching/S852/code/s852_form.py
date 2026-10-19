"""S852 (pre-registered in knowledge/DECISIONS.md before any S852 data; user 10-18 'continue' after the S851 report): how is the wrong late
query formed?  Two questions on fresh scenes with the early lookup pinned (address part U of the ten read rows set to START(p)'s at the early
look layers 35-47), eight pictures shown and stated (N_s = N = 8, so the end-relative target j* = 9 - k is unambiguous):
  (A) Does the END-specific write of the band carry the anchor?  The band = token-mixer and MLP outputs of layers 46-49 at the read rows.  The END
      write at anchor p is D(p) = band_E(p) - band_S(p) (END(k) minus START(k), both pinned at p, same pass).
        XD_p      E_EARLY_p with band outputs band_S(p) + D(p'), p' = PAIR[p]: the END write computed at another anchor, on the receiving run's
                  own anchor-related band content (linear component patch, exp/c_qpath_i.py)
        XB_p      E_EARLY_p with band_E(p'): the whole band of the other anchor (this also moves the layer-47 early read of p', since layer 47
                  is both a pinned early look layer and a band layer)
        BANDF_p   E_EARLY_p with band_S(p): no END write (positive control; S846: departures .46 -> .09)
        XDSELF_p  band_S(p) + D(p) = band_E(p): bit-identical to E_EARLY_p by construction (exactness check of the linear patch; k = 4 only)
  (B) What sets the direction of the late step?  E_EARLY_p for k in {3, 4, 5} (copy k, target 9 - k): a step toward the copy predicts a positive
      within-p slope of the move direction on k, a step toward the target a negative one, a k-independent step none.  S_EARLY_p (START(k)
      pinned at p, which returns to k) is the positive control of the slope measure.
Per scene: stage 1 = END(k) for k in {3, 4, 5} and START(p) for p in {3, 4, 5, 6} (START(k) = START_p{k}); stage 2 per k (one scene per pass, so
every source of a same-pass patch is in the pass) = QNOOP_E and per p: E_EARLY, S_EARLY, BANDF, XD, XB (+ XDSELF for k = 4).  30 scenes (seed
85200; main-series recipe: 8 labels + the first unused word, the same pictures for every k).  Readouts: answer, menu-word logits, positional
scores (matcher-head block scores of every read row at the read layers), picture mass; band outputs of E_EARLY / S_EARLY at rows B1, B4, answer.
Usage: python s852_form.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s852_{}_{}.json"
KS = (3, 4, 5)
PSET = [3, 4, 5, 6]
PAIR = {3: 5, 5: 3, 4: 6, 6: 4}
N = 8


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S852_DEVICE", "cuda:0")
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
    KROWS = [2, 5, 9]                                                     # rows kept for band outputs (B1, B4, answer)
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
    TOOL = QPathI(v, READ, res_layers=(), comp_layers=band, patch_comps=band)
    T1 = []

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
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
        """One forward pass for the scenes gis; on out-of-memory split the scenes in halves (a sample's computation does not depend on its
        batch neighbours; src indices are rebuilt per pass).  Stage 2 passes hold one scene, so every same-pass source is present."""
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

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · comps {COMPS} · matcher layers {sorted(MH)} · scaling {scaling}",
          flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe; one scene set for every k
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra, menu=menu, pics=pics))
    NAT = [f"END_k{k}" for k in KS] + [f"START_p{p_}" for p_ in PSET]
    P = []
    for s in scenes:
        qk_ = {**{f"END_k{k}": (k, "last") for k in KS}, **{f"START_p{p_}": (p_, "first") for p_ in PSET}}
        enc = {nm: v.encode(prompt(q, fr, s["menu"]), s["pics"]) for nm, (q, fr) in qk_.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
        ref = ids["END_k4"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
        for k in KS:
            Fw = [i for i, (x, y) in enumerate(zip(ids[f"START_p{k}"], ids[f"END_k{k}"])) if x != y]
            assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1, f"menu word {w_} must be one token once"
            opt.append(pos_[0])
        P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4))
    assert len({p["S"] for p in P}) == 1, "scenes must align for batching"
    capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=N, opt=p["opt"]), **kw)
    recs = {(p["s"]["si"], k): dict(si=p["s"]["si"], N=N, k=k, j=N + 1 - k, Ns=N, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"],
                                    land={}, mlog={}, qk={}, mass={},
                                    menu_pos=[(p["s"]["labs"].index(w_) + 1 if w_ in p["s"]["labs"] else 0) for w_ in p["s"]["menu"]])
            for p in P for k in KS}

    def measure(rec, lgb, stb):
        """One run's readouts (t1 counted once per run); the records of a scene share menu, labels and words."""
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))         # matcher heads where listed (35-55), else all heads (59, 63)
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]

    # stage 1: natural runs (END(k) for every k, START(p) donors; START(k) = START_p{k})
    G1 = a.group or max(1, a.max_images // (len(NAT) * N))
    don = {}; keyrel = []
    for g0 in range(0, len(P), G1):
        def build1(gis):
            inps, qs, tag = [], [], []
            for gi in gis:
                for nm in NAT:
                    inps.append(P[gi]["enc"][nm]); tag.append((gi, nm)); qs.append(capq(P[gi], grabq=set(EARLY), grabkb=set(LOOK)))
            return inps, qs, tag
        for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G1, len(P)))), build1):
            si = P[gi]["s"]["si"]
            m = measure(recs[(si, KS[0])], lgb, stb)
            for k in KS:                                                    # every k's record holds the natural runs it is compared with
                rec = recs[(si, k)]
                if nm == f"END_k{k}":
                    put(rec, "END", m)
                elif nm.startswith("START_p"):
                    put(rec, nm, m)
                    if nm == f"START_p{k}":
                        put(rec, "START", m)
            nrun += 1
            don[(gi, nm)] = dict(q={li: stb[("q", li)] for li in EARLY}, kb={li: stb[("kb", li)] for li in LOOK})
        print(f"[{a.tag}] stage 1 {min(g0 + G1, len(P))}/{len(P)} · {nrun} runs", flush=True)
    for gi in range(len(P)):
        for k in KS:
            kE, kS = don[(gi, f"END_k{k}")]["kb"], don[(gi, f"START_p{k}")]["kb"]
            kr = max(float((kE[li] - kS[li]).abs().max() / kE[li].abs().max()) for li in LOOK); keyrel.append(kr)
            recs[(P[gi]["s"]["si"], k)]["key_rel_diff"] = kr
    # stage 2: pinned runs and the band patches, one scene and one k per pass
    BA = list(range(10)); CB = set(band)
    for k in KS:
        for gi in range(len(P)):
            def build2(gis, k=k):
                inps, qs, tag = [], [], []
                for gj in gis:
                    p = P[gj]; eE, eS = p["enc"][f"END_k{k}"], p["enc"][f"START_p{k}"]
                    def ed(p_, p=p, gj=gj, **kw):
                        dq = don[(gj, f"START_p{p_}")]["q"]
                        return capq(p, edit={li: [dict(qsub=dict(owner=p["owner"], nb=N, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}, **kw)
                    inps.append(eE); qs.append(capq(p, noop=set(LOOK))); tag.append((gj, "QNOOP_E"))
                    iE, iS = {}, {}
                    for p_ in PSET:
                        iE[p_] = len(inps); inps.append(eE); qs.append(ed(p_, keepcomp=CB, grabcomp=CB)); tag.append((gj, f"E_EARLY_p{p_}"))
                        iS[p_] = len(inps); inps.append(eS); qs.append(ed(p_, keepcomp=CB, grabcomp=CB)); tag.append((gj, f"S_EARLY_p{p_}"))
                    for p_ in PSET:
                        q_ = PAIR[p_]
                        inps.append(eE); qs.append(ed(p_, cpatch=dict(src=iS[p_], comps=COMPS))); tag.append((gj, f"BANDF_p{p_}"))
                        inps.append(eE); qs.append(ed(p_, cpatch_lin=dict(terms=[(1.0, iS[p_]), (1.0, iE[q_]), (-1.0, iS[q_])], comps=COMPS)))
                        tag.append((gj, f"XD_p{p_}"))
                        inps.append(eE); qs.append(ed(p_, cpatch=dict(src=iE[q_], comps=COMPS))); tag.append((gj, f"XB_p{p_}"))
                        if k == 4:
                            inps.append(eE); qs.append(ed(p_, cpatch_lin=dict(terms=[(1.0, iS[p_]), (1.0, iE[p_]), (-1.0, iS[p_])], comps=COMPS)))
                            tag.append((gj, f"XDSELF_p{p_}"))
                return inps, qs, tag
            for gj, nm, lgb, stb in runs_for([gi], build2):
                rec = recs[(P[gj]["s"]["si"], k)]
                put(rec, nm, measure(rec, lgb, stb)); nrun += 1
                if ("mix", band[0]) in stb:
                    npz[f"k{k}_s{rec['si']}_{nm}_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
            if gi % 5 == 4 or gi == len(P) - 1:
                print(f"[{a.tag}] k{k} {gi + 1}/{len(P)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "form"
    json.dump(dict(tag=a.tag, model=a.model, ks=KS, pset=PSET, pair=PAIR, band=band, comps=COMPS, readout=os.environ.get("R3_READ"), read=READ,
                   look=LOOK, early=EARLY, late=LATE, krows=KROWS, matcher_layers=sorted(MH), matchers={str(l): h for l, h in MH.items()},
                   n_layer=nl, scaling=scaling, qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun,
                   rows=[recs[key] for key in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85200); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower
    main(ap.parse_args())
