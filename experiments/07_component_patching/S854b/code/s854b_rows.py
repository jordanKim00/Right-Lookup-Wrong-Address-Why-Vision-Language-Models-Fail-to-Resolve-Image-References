"""S854-2 (pre-registered in knowledge/DECISIONS.md before any S854-2 data; user 10-18 'S854-2 - input-row causal decomposition of the writer').
Which input rows carry the ordinal-related state into the writer (layers 46-49), and do they combine additively?  Eight pictures shown and stated,
early lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p)); natural states E1 (k 3, p 3), E2 (k 3, p 4),
E3 (k 5, p 3), E4 (k 5, p 4).  Input groups at the input of layer 46 (positions from the ordinal token on; earlier positions are identical in
every run): Q = question tail (ordinal token .. question mark), M = menu sentence, O = boundary rows B0 / B2 / B3, A = boundary rows B1 / B4;
T = the four thinking / answer positions (a separate control).  Same p, so donors and recipients share every early pin.
  pass A  E1-E4, QNOOP_E; 'up' = recipient E1 (k 3) <- donor E3 (k 5) for all 15 non-empty subsets of {Q, M, O, A}, T alone, and Q+M+O+A+T
          (= the donor exactly from layer 46 on); XSELF (all positions <- E1 itself) and LINSELF (linear patch E3 + E1 - E3) as exactness checks
  pass B  E1-E4; 'dn' = recipient E3 (k 5) <- donor E1 (k 3), the same 17 arms
  pass C  E1-E4; p 4: E2 <- E4 and E4 <- E2 for Q, M, O, A and Q+M+O+A; CF (user's Gap 3): for each corner c, the input of layer 46 at every
          position from the ordinal token on <- h(k-flip of c) + h(p-flip of c) - h(c) (linear residual patch, exp/c_qpath_j.py), with the
          layer-47 pin of the read rows set to the diagonal corner's p - the additive combination of the two changes, compared with the
          natural diagonal corner
Per scene: stage 1 = END(k 3), END(k 5), START(q) for q in {3, 4, 5}.  30 new scenes (seed 85500).  Readouts: answer, menu-word logits,
positional scores of every read row at the read layers, picture mass; writer outputs (layers 46-49, mixer and MLP) and late queries (51-63) at
rows B1, B4, answer; the residual at the input of layer 46 (rows B1, B4, answer) for E and CF.
Usage: python s854b_rows.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, itertools, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_j import QPathJ

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s854b_{}_{}.json"
N = 8
COND = {"E1": (3, 3), "E2": (3, 4), "E3": (5, 3), "E4": (5, 4)}                 # (k, p)
GROUPS = ("Q", "M", "O", "A")
SUBSETS = ["".join(g for g, b in zip(GROUPS, bits) if b) for bits in itertools.product((0, 1), repeat=4) if any(bits)]
KFLIP = {"E1": "E3", "E3": "E1", "E2": "E4", "E4": "E2"}
PFLIP = {"E1": "E2", "E2": "E1", "E3": "E4", "E4": "E3"}
DIAG = {"E1": "E4", "E4": "E1", "E2": "E3", "E3": "E2"}


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S854_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE))
    assert band and all(b < min(LATE) for b in band)
    SITE = 46 if nl == 64 else band[0]
    KROWS = [2, 5, 9]
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
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
    TOOL = QPathJ(v, READ, res_layers=[SITE], comp_layers=band, patch_comps=(), rpatch_layers=[SITE], rpatch_lin_layers=[SITE])
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

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · site {SITE} · subsets {SUBSETS}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra, menu=menu, pics=pics))
    NAT = ["END_k3", "END_k5", "START_q3", "START_q4", "START_q5"]
    P = []
    for s in scenes:
        qk_ = {"END_k3": (3, "last"), "END_k5": (5, "last"), "START_q3": (3, "first"), "START_q4": (4, "first"), "START_q5": (5, "first")}
        enc = {nm: v.encode(prompt(q, fr, s["menu"]), s["pics"]) for nm, (q, fr) in qk_.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
        ref = ids["END_k5"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
        for k in (3, 5):
            Fw = [i for i, (x, y) in enumerate(zip(ids[f"START_q{k}"], ids[f"END_k{k}"])) if x != y]
            assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
        Dw = [i for i, (x, y) in enumerate(zip(ids["END_k3"], ids["END_k5"])) if x != y]
        assert len(Dw) == 1 and Dw[0] < qm, "END(3) and END(5) differ in the ordinal token only"
        o = Dw[0]
        grp = {"Q": list(range(o, qm + 1)), "M": list(range(qm + 1, uend)), "O": [uend, uend + 2, uend + 3], "A": [uend + 1, uend + 4],
               "T": list(range(uend + 5, S_))}
        assert sum(len(x) for x in grp.values()) == S_ - o and len(grp["T"]) == 4
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_))
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1, f"menu word {w_} must be one token once"
            opt.append(pos_[0])
        P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, o=o, grp=grp, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4))
    assert len({p["S"] for p in P}) == 1 and len({p["o"] for p in P}) == 1, "scenes must align for batching"
    capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=N, opt=p["opt"]), **kw)
    recs = {p["s"]["si"]: dict(si=p["s"]["si"], N=N, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={}, mass={})
            for p in P}

    def measure(rec, lgb, stb):
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]

    G1 = a.group or max(1, a.max_images // (len(NAT) * N))
    don = {}
    for g0 in range(0, len(P), G1):
        def build1(gis):
            inps, qs, tag = [], [], []
            for gi in gis:
                for nm in NAT:
                    inps.append(P[gi]["enc"][nm]); tag.append((gi, nm)); qs.append(capq(P[gi], grabq=set(EARLY), grabkb=set(LOOK)))
            return inps, qs, tag
        for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G1, len(P)))), build1):
            rec = recs[P[gi]["s"]["si"]]; put(rec, nm, measure(rec, lgb, stb)); nrun += 1
            don[(gi, nm)] = dict(q={li: stb[("q", li)] for li in EARLY}, kb={li: stb[("kb", li)] for li in LOOK})
        print(f"[{a.tag}] stage 1 {min(g0 + G1, len(P))}/{len(P)} · {nrun} runs", flush=True)
    for gi in range(len(P)):
        kr = max(max(float((don[(gi, nm)]["kb"][li] - don[(gi, "END_k5")]["kb"][li]).abs().max() / don[(gi, "END_k5")]["kb"][li].abs().max())
                     for li in LOOK) for nm in NAT)
        keyrel.append(kr); recs[P[gi]["s"]["si"]]["key_rel_diff"] = kr
    BA = list(range(10)); CB = set(band); LQ = set(LATE)
    def pin(gi, p_, p47=None):
        qa = don[(gi, f"START_q{p_}")]["q"]; qb = don[(gi, f"START_q{p47}")]["q"] if p47 is not None else qa
        return {li: [dict(qsub=dict(owner=P[gi]["owner"], nb=N, part="U", donors=[(1.0, (qb if li == EARLY[-1] else qa)[li][BA])]), erows=BA)]
                for li in EARLY}
    def build_pass(gis, which):
        inps, qs, tag = [], [], []
        for gj in gis:
            p = P[gj]; idx = {}; tail = list(range(p["o"], p["S"]))
            for nm, (k, p_) in COND.items():
                idx[nm] = len(inps); inps.append(p["enc"][f"END_k{k}"])
                qs.append(capq(p, edit=pin(gj, p_), keepcomp=CB, grabcomp=CB, grabq=LQ, grabres={SITE})); tag.append((gj, nm))
            def arm(nm, rec_, rows, src):
                k, p_ = COND[rec_]
                inps.append(p["enc"][f"END_k{k}"])
                qs.append(capq(p, edit=pin(gj, p_), rpatch={SITE: dict(src=idx[src], rows=rows)}, keepcomp=CB, grabcomp=CB, grabq=LQ)); tag.append((gj, nm))
            rows_of = lambda sub: sorted(x for g in GROUPS if g in sub for x in p["grp"][g])
            if which in ("A", "B"):
                r_, d_ = ("E1", "E3") if which == "A" else ("E3", "E1"); dn = "up" if which == "A" else "dn"
                if which == "A":
                    inps.append(p["enc"]["END_k5"]); qs.append(capq(p, noop=set(LOOK))); tag.append((gj, "QNOOP_E"))
                    arm("XSELF", "E1", tail, "E1")
                    inps.append(p["enc"]["END_k3"])
                    qs.append(capq(p, edit=pin(gj, 3), rpatch_lin={SITE: dict(terms=[(1.0, idx["E3"]), (1.0, idx["E1"]), (-1.0, idx["E3"])], rows=tail)},
                                   keepcomp=CB, grabcomp=CB, grabq=LQ)); tag.append((gj, "LINSELF"))
                for sub in SUBSETS:
                    arm(f"{dn}_{sub}", r_, rows_of(sub), d_)
                arm(f"{dn}_T", r_, p["grp"]["T"], d_)
                arm(f"{dn}_QMOAT", r_, tail, d_)
            else:
                for r_, d_, dn in (("E2", "E4", "up4"), ("E4", "E2", "dn4")):
                    for sub in ("Q", "M", "O", "A", "QMOA"):
                        arm(f"{dn}_{sub}", r_, rows_of(sub), d_)
                for c in COND:
                    k, p_ = COND[c]; pd = COND[DIAG[c]][1]
                    inps.append(p["enc"][f"END_k{k}"])
                    qs.append(capq(p, edit=pin(gj, p_, pd), rpatch_lin={SITE: dict(terms=[(1.0, idx[KFLIP[c]]), (1.0, idx[PFLIP[c]]), (-1.0, idx[c])], rows=tail)},
                                   keepcomp=CB, grabcomp=CB, grabq=LQ, grabres={SITE})); tag.append((gj, f"CF_{c}"))
        return inps, qs, tag
    for gi in range(len(P)):
        for which in ("A", "B", "C"):
            for gj, nm, lgb, stb in runs_for([gi], lambda g, w=which: build_pass(g, w)):
                rec = recs[P[gj]["s"]["si"]]; key = nm if (which == "A" or nm not in COND) else f"{nm}_{which}"
                put(rec, key, measure(rec, lgb, stb)); nrun += 1
                if ("mix", band[0]) in stb:
                    npz[f"s{rec['si']}_{key}_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
                if ("q", LATE[0]) in stb:
                    npz[f"s{rec['si']}_{key}_q"] = np.stack([stb[("q", li)][KROWS].float().numpy() for li in LATE]).astype(np.float16)
                if ("res", SITE) in stb:
                    npz[f"s{rec['si']}_{key}_res"] = stb[("res", SITE)][KROWS].float().numpy().astype(np.float16)
        if gi % 5 == 4 or gi == len(P) - 1:
            print(f"[{a.tag}] stage 2 {gi + 1}/{len(P)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "rows"
    json.dump(dict(tag=a.tag, model=a.model, cond=COND, subsets=SUBSETS, groups=GROUPS, kflip=KFLIP, pflip=PFLIP, diag=DIAG, band=band, site=SITE,
                   comps=COMPS, readout=os.environ.get("R3_READ"), read=READ, look=LOOK, early=EARLY, late=LATE, krows=KROWS, matcher_layers=sorted(MH),
                   n_layer=nl, group_sizes={g: len(x) for g, x in P[0]["grp"].items()},
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun,
                   rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85500); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)
    main(ap.parse_args())
