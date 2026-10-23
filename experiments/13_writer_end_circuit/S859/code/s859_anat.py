"""S859 (descriptive anatomy; registered in knowledge/DECISIONS.md before its data) - which heads of the writer band (27B: DeltaNet 46 / 48 / 49,
full attention 47, the four MLPs) write the end-relative late update, from which source tokens, and what each head's write does to the late
lookup's block scores (FA 51 / 55 / 59 / 63) through the direct path.  Passive: the exp/c_anat.py hooks only read (two passes per batch, logits
of the two passes compared bit for bit).

Scenes: eight real pictures and a menu of nine names; the same scene is also shown with its first six pictures only (N_a 6, same menu); the
stated count always equals the shown count (N_s = N_a).  Per scene and layout:
  stage 1 (natural, no pins): N_a 8 - END(k) k 2..7, START(q) q 1..8;  N_a 6 - END(k) k 2..5, START(q) q 1..6
  stage 2 (early lookup pinned: address part U of the ten read rows at the early look layers 35-47 <- START(p), as S844-S858):
      N_a 8 - E(k, p) = END(k) and S(k, p) = START(k), k 2..6 x p {3, 4, 5, 7} (p = 7 = N - 1: the end anchor; passes k 2-3 + QNOOP, 4-5, 6)
      N_a 6 - E(k, p), S(k, p), k 2..4 x p {3, 4, 5} (p = 5 = N - 1)
  every pass also runs R(p) = START(p) pinned at its own address p for each p of the pass (a pure pointer at p: no conflict between the
  ordinal and the pin, so no late return to another picture); every E(k, p) is paired with S(k, p) ('pair': the frame-only contrast) and
  with R(p) ('ref': END(k) against a pointer at p), both projected at E's late statistics
Anatomy units: the band (FA 47 heads, DN 46 / 48 / 49 value heads, MLPs 46-49) and, as upstream references, the full-attention readers 35 /
39 / 43 (heads x groups; their direct path into the late queries).  Anatomy rows B1, B4 and the answer row; source groups: 0 pre (system / header), 1-8 the image blocks, 9 count sentence, 10 'What is in the',
11 ordinal, 12 ' image from the', 13 frame word, 14 '?', 15 instruction, 16 menu, 17 boundary rows B0-B4, 18 think / answer rows.
Readouts per run: answer, menu-word logits, head-mean block scores (matcher heads at 35-55, all heads at 59 / 63) of the ten read rows at every
read layer, per-head block scores of the anatomy rows at the late layers, picture mass, the eight band outputs and the late queries (rows B1, B4,
answer), and the anatomy arrays of exp/c_anat.py.
Usage: python s859_anat.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI
from c_anat import Anat

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s859_{}_{}.json"
NMAX = 8
LAYOUT = {8: dict(ends=[2, 3, 4, 5, 6, 7], starts=[1, 2, 3, 4, 5, 6, 7, 8], ks=[2, 3, 4, 5, 6], ps=[3, 4, 5, 7], split=[[2, 3], [4, 5], [6]]),
          6: dict(ends=[2, 3, 4, 5], starts=[1, 2, 3, 4, 5, 6], ks=[2, 3, 4], ps=[3, 4, 5], split=[[2, 3, 4]])}
GROUPS = ["pre"] + [f"img{b + 1}" for b in range(NMAX)] + ["count", "qhead", "ord", "qmid", "frame", "qm", "instr", "menu", "bnd", "think"]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S859_DEVICE", "cuda:0")
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
    FA_W = [l for l in band if l in FA_L]; DN_W = [l for l in band if l not in FA_L]
    FA_UP = [l for l in EARLY if l not in band]                                        # upstream full-attention readers (35 / 39 / 43)
    KROWS = [2, 5, 9]                                                                 # B1, B4, answer (indices into the ten read rows)
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
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
    TOOL = QPathI(v, READ, res_layers=(), comp_layers=band, patch_comps=(), rpatch_layers=())
    AN = Anat(v, FA_UP + FA_W, DN_W, band, LATE)                                      # after TOOL: its hooks run after the pins' write-back
    T1 = []

    def prompt(q, frame, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run2(inps, qs, specs):
        """pass 1 (pins + readouts + late statistics), pass 2 (same, anatomy); returns logits of pass 1, its store, anatomy, max |logit diff|.
        specs[b] = the sample's anatomy spec or None; the anatomy also sees the sample's QPath rows and pins (to re-apply them at FA 47)."""
        bi = batch_inputs(inps)
        anat = [None if sp is None else dict(anat=sp, rows=q["rows"], edit=q.get("edit", {})) for q, sp in zip(qs, specs)]
        TOOL.set(qs); AN.set(anat, "stat")
        try:
            lg1 = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None)
        TOOL.set(qs); AN.set(anat, "anat")
        try:
            lg2 = v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            res = {b: AN.finish(b) for b in AN.out}
        finally:
            TOOL.set(None); AN.set(None, None); AN.clear()
        return lg1, st, res, float((lg1 - lg2).abs().max())

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} (FA {FA_W}, DN {DN_W}) · matcher layers {sorted(MH)} · "
          f"units {AN.nU}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; lgdiff = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, NMAX); extra = [l for l in labels if l not in labs][:9 - NMAX]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra, menu=menu, pics=pics))

    def layout(s, n):
        """encodings and position bookkeeping of one scene shown with its first n pictures (stated count n)."""
        lay = LAYOUT[n]
        qk_ = {**{f"END_k{k}": (k, "last") for k in lay["ends"]}, **{f"START_q{q}": (q, "first") for q in lay["starts"]}}
        enc = {nm: v.encode(prompt(q, fr, s["menu"], n), s["pics"][:n]) for nm, (q, fr) in qk_.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "the prompts of a layout must align"
        ref = ids[f"END_k{lay['ends'][0]}"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        assert len(ve) == n and len(vs) == n
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
        Fw = [i for i, (x, y) in enumerate(zip(ids["START_q2"], ids["END_k2"])) if x != y]
        assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
        Ow = [i for i, (x, y) in enumerate(zip(ids["END_k2"], ids["END_k3"])) if x != y]
        assert len(Ow) == 1 and Ow[0] < Fw[0], "END prompts differ in the ordinal only"
        what = next(i for i in range(ve[-1] + 1, Ow[0]) if "What" in v.tok.decode([ref[i]]))
        owner = [-1] * S_
        for b in range(n):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1, f"menu word {w_} must be one token once"
            opt.append(pos_[0])
        grp = [-1] * S_
        for i in range(S_):
            if owner[i] >= 0:
                grp[i] = 1 + owner[i]
            elif i < vs[0]:
                grp[i] = 0
            elif ve[-1] < i < what:
                grp[i] = 9
            elif what <= i < Ow[0]:
                grp[i] = 10
            elif i == Ow[0]:
                grp[i] = 11
            elif Ow[0] < i < Fw[0]:
                grp[i] = 12
            elif i == Fw[0]:
                grp[i] = 13
            elif i == qm:
                grp[i] = 14
            elif qm < i < min(opt):
                grp[i] = 15
            elif min(opt) <= i < uend:
                grp[i] = 16
            elif uend <= i <= uend + 4:
                grp[i] = 17
            elif i >= uend + 5:
                grp[i] = 18
        assert all(x >= 0 for x in grp[vs[0]:]), "every position after the first picture has a group"
        rowsC = [qm] + list(range(uend, uend + 5)) + A4
        return dict(n=n, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=rowsC, grp=grp, ordpos=Ow[0], framepos=Fw[0])

    P = []
    for s in scenes:
        P.append({n: layout(s, n) for n in (8, 6)})
    for n in (8, 6):
        assert len({p[n]["S"] for p in P}) == 1, "scenes must align for batching"
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}, zl={},
                                            menu_pos=[(s["labs"].index(w_) + 1 if w_ in s["labs"] else 0) for w_ in s["menu"]])
            for s, p in zip(scenes, P)}

    def capq(L, **kw):
        return dict(rows=L["rowsC"], cap=dict(rows=list(range(10)), owner=L["owner"], nb=L["n"], opt=L["opt"]), **kw)

    def anat(L, pairs=None):
        return dict(rows=[L["rowsC"][i] for i in KROWS], group=L["grp"], ng=len(GROUPS), owner=L["owner"], nb=L["n"],
                    late_mean={li: hm(li) for li in LATE}, perhead=LATE[:2], gz_late=LATE[:2], pairs=pairs)

    def measure(rec, lgb, stb, n):
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("img", li) in stb},
                    zl={str(li): stb[("cap", li)][KROWS].float().tolist() for li in LATE if ("cap", li) in stb})

    def put(rec, nm, m):
        for key in ("land", "mlog", "qk", "mass", "zl"):
            rec[key][nm] = m[key]

    def save(si, nm, stb, an):
        if ("mix", band[0]) in stb:
            npz[f"s{si}_{nm}_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
        if ("q", LATE[0]) in stb:
            npz[f"s{si}_{nm}_q"] = np.stack([stb[("q", li)][KROWS].float().numpy() for li in LATE]).astype(np.float16)
        if an is not None:
            for k_, x in an.items():
                npz[f"s{si}_{nm}_{k_}"] = x

    don = {}; keyrel = []
    for gi, p in enumerate(P):
        si = scenes[gi]["si"]; rec = recs[si]
        for n in (8, 6):
            L = p[n]; lay = LAYOUT[n]
            # stage 1: natural runs, every one with anatomy (no pair)
            names = [f"END_k{k}" for k in lay["ends"]] + [f"START_q{q}" for q in lay["starts"]]
            inps = [L["enc"][nm] for nm in names]
            qs = [capq(L, grabq=set(EARLY) | set(LATE), grabkb=set(LOOK), keepcomp=set(band), grabcomp=set(band)) for _ in names]
            lg, st, res, dmax = run2(inps, qs, [anat(L) for _ in names])
            lgdiff.append(dmax)
            for b, nm in enumerate(names):
                key = f"N{n}_{nm}"
                put(rec, key, measure(rec, lg[b], st[b], n)); nrun += 1
                don[(gi, n, nm)] = dict(q={li: st[b][("q", li)] for li in EARLY}, kb={li: st[b][("kb", li)] for li in LOOK})
                save(si, key, st[b], res.get(b))
            print(f"[{a.tag}] scene {gi + 1} N{n} natural · {nrun} runs · pass diff {dmax:.1e}", flush=True)
            ref_kb = don[(gi, n, names[0])]["kb"]
            for nm in names:
                kb = don[(gi, n, nm)]["kb"]
                keyrel.append(max(float((kb[li] - ref_kb[li]).abs().max() / ref_kb[li].abs().max()) for li in LOOK))
            # stage 2: pinned E(k, p) / S(k, p) pairs
            def pin(p_):
                qr = don[(gi, n, f"START_q{p_}")]["q"]
                return {li: dict(qsub=dict(owner=L["owner"], nb=n, part="U", donors=[(1.0, qr[li])])) for li in EARLY}
            for part, ks in enumerate(lay["split"]):
                inps, qs, an_, tag = [], [], [], []
                iR = {}
                for p_ in lay["ps"]:                                                       # the pure-pointer reference R(p): START(p) pinned at p
                    iR[p_] = len(inps)
                    inps.append(L["enc"][f"START_q{p_}"]); tag.append(f"N{n}_R{part}_p{p_}")
                    qs.append(capq(L, edit=pin(p_), keepcomp=set(band), grabcomp=set(band), grabq=set(LATE)))
                    an_.append(anat(L))
                for k in ks:
                    for p_ in lay["ps"]:
                        iE = len(inps)
                        inps.append(L["enc"][f"END_k{k}"]); tag.append(f"N{n}_E_k{k}p{p_}")
                        qs.append(capq(L, edit=pin(p_), keepcomp=set(band), grabcomp=set(band), grabq=set(LATE)))
                        inps.append(L["enc"][f"START_q{k}"]); tag.append(f"N{n}_S_k{k}p{p_}")
                        qs.append(capq(L, edit=pin(p_), keepcomp=set(band), grabcomp=set(band), grabq=set(LATE)))
                        an_ += [anat(L, pairs=[("pair", iE + 1), ("ref", iR[p_])]), anat(L)]
                if n == 8 and part == 0:
                    inps.append(L["enc"]["END_k4"]); tag.append("N8_QNOOP_E4"); qs.append(capq(L, noop=set(LOOK))); an_.append(None)
                lg, st, res, dmax = run2(inps, qs, an_)
                lgdiff.append(dmax)
                print(f"[{a.tag}] scene {gi + 1} N{n} pinned part {part} ({len(inps)} runs) · pass diff {dmax:.1e}", flush=True)
                for b, nm in enumerate(tag):
                    put(rec, nm, measure(rec, lg[b], st[b], n)); nrun += 1
                    save(si, nm, st[b], res.get(b))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rec["key_rel_diff"] = max(keyrel[-(len(LAYOUT[8]["ends"]) + len(LAYOUT[8]["starts"]) + len(LAYOUT[6]["ends"]) + len(LAYOUT[6]["starts"])):])
        print(f"[{a.tag}] scene {gi + 1}/{len(P)} · {nrun} runs · key diff max {max(keyrel):.2e} · two-pass logit diff max {max(lgdiff):.2e}", flush=True)
    TOOL.remove(); AN.remove()
    lays = {str(n): dict(S=P[0][n]["S"], rowsC=P[0][n]["rowsC"], grp=P[0][n]["grp"], owner=P[0][n]["owner"], ordpos=P[0][n]["ordpos"],
                         framepos=P[0][n]["framepos"], **{k_: v_ for k_, v_ in LAYOUT[n].items()}) for n in (8, 6)}
    json.dump(dict(tag=a.tag, model=a.model, layouts=lays, groups=GROUPS, band=band, fa_w=FA_W, fa_up=FA_UP, fa_anat=AN.fa, dn_w=DN_W, comps=COMPS, units=AN.units,
                   readout=os.environ.get("R3_READ"), read=READ, look=LOOK, early=EARLY, late=LATE, krows=KROWS, matchers={str(l): h for l, h in MH.items()},
                   late_mean={str(li): hm(li) for li in LATE}, perhead=LATE[:2], gz_late=LATE[:2], n_layer=nl,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), two_pass_logit_diff_max=max(lgdiff),
                   n_runs=nrun, rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, "anat"), "w"))
    np.savez_compressed(OUT.format(a.tag, "anat").replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'anat')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85990); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
