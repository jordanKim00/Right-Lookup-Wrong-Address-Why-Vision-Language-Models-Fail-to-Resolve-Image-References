"""S847 (pre-registered in knowledge/DECISIONS.md before any S847 data; continues S845 after its PATH near miss): does the band's correction reach the
answer through the late full-attention queries once the last two full-attention layers (59, 63; S839: START's protection) are included, and are
those queries needed?  Writers = token mixers + MLPs of layers 46-49; readers = the queries of the ten read rows (question mark, boundary B0-B4,
answer-side four) at the late full-attention layers 51, 55, 59, 63.
  A (one pass per scene): START(k) (donor), END(k), A_BAND (END with the eight band outputs at the read rows replaced by START's; everything
    downstream sees it); late queries of END and A_BAND kept.
  B (one pass per scene): START(k) again (same-pass donor for the NEC arms); END_B and A_BAND_B (the references again, in the same pass as the
    arms - amendment 10-17 before data: the GPU smoke showed cross-batch drift between stages at n = 1); QNOOP; Q5155 / Q5963 / QALL (END with A_BAND's whole late queries at
    51 + 55 / 59 + 63 / all four); QALL_U (address part U only, all four); NEC (A_BAND with END's whole late queries at all four); NEC5155 (A_BAND
    with END's late queries at 51 + 55 only); TRES / TREM (amendment 10-17 before data, user request 11:48: the S842 text-key subspace T of the
    START - END query difference, rank 16, at the look layers 35-55, added to END (restore, S842 replication) or the END - START difference added
    to START (removal)).
Cases (N, k, j): identity (7, 4, 4), (9, 5, 5); non-identity (7, 5, 3), (9, 7, 3) (as S845).  Seed 84700 (new), 30 scenes per case.
Readouts: answer, menu-word logits, positional scores (matcher-head block scores at 35-55; all-head means at 59 / 63), picture mass.
Usage: python s847_close.py MODEL TAG --band 46 47 48 49 [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_f import QPathF
import c_qpath_m

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s847_{}_{}.json"
CASES = [(7, 4, 4), (9, 5, 5), (7, 5, 3), (9, 7, 3)]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S847_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    c_qpath_m.patch()                                                     # 'tsub' (S842 text-key subspace) for TRES / TREM
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]                      # look layers + the last full-attention layers (59, 63)
    LATE4 = [l for l in READ if l / (nl - 1) > .75]                       # 51, 55, 59, 63 in the 27B
    LOOK = [l for l in READ if l / (nl - 1) <= .875]                      # 35-55: the S842 layers of the text-key edit
    L12, L34 = LATE4[:len(LATE4) // 2], LATE4[len(LATE4) // 2:]          # 27B: (51, 55) and (59, 63); smoke models split their late layers in two
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE4))      # smoke models: by relative depth
    assert band and all(b < min(LATE4) for b in band) and (len(LATE4) == 4 if nl == 64 else len(LATE4) >= 2)
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    NRUN = 12                                                             # stage B runs per scene (stage A has 3)
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
    TOOL = QPathF(v, READ, res_layers=(), comp_layers=(), patch_comps=band)
    T1 = []

    def prompt(N, q, frame, menu):
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
        return [(gi, nm, lg[b], st[b], st[src]) for b, (gi, nm, src) in enumerate(tag)]

    print(f"[{a.tag}] read {READ} · late {LATE4} · band {band} · matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}; scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe, per case
        for N, k, j in CASES:
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            scenes.append(dict(si=si, N=N, k=k, j=j, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel = []
    for N, k, j in CASES:
        SC = [s for s in scenes if s["N"] == N and s["k"] == k]
        G = a.group or max(1, a.max_images // (NRUN * N))
        print(f"[{a.tag}] N{N} k{k}: {G} scenes per batch", flush=True)
        for g0 in range(0, len(SC), G):
            grp = SC[g0:g0 + G]
            P = []
            for s in grp:
                enc = {nm: v.encode(prompt(N, k, fr, s["menu"]), s["pics"]) for nm, fr in (("END", "last"), ("START", "first"))}
                ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
                assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
                S_ = len(ids["END"])
                ve = [i for i, x in enumerate(ids["END"]) if x == ve_id]; vs = [i for i, x in enumerate(ids["END"]) if x == vs_id]
                qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["END"][i]]))
                uend = max(i for i, x in enumerate(ids["END"]) if x == end_id)
                assert v.tok.decode([ids["END"][uend + 3]]).strip() == "assistant"
                Fw = [i for i, (x, y) in enumerate(zip(ids["START"], ids["END"])) if x != y]
                assert len(Fw) == 1 and Fw[0] < qm, "END and START differ in the frame word only"
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
            recs = [dict(si=p["s"]["si"], N=N, k=k, j=j, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={},
                         mass={}) for p in P]
            def hsel(li, H):
                return MH[li] if MH.get(li) else list(range(H))
            def keep(gi, nm, lgb, stb):
                rec = recs[gi]; menu = rec["menu"]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
                rec["qk"][nm] = {str(li): stb[("cap", li)][:, hsel(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                rec["mass"][nm] = {str(li): stb[("img", li)][:, hsel(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb}
                key = f"N{N}k{k}_s{rec['si']}_{nm}"
                npz[key + "_cap"] = np.stack([stb[("cap", li)].float().numpy() for li in READ]).astype(np.float16)   # [L, rows, H, nb]
                npz[key + "_img"] = np.stack([stb[("img", li)].float().numpy() for li in READ]).astype(np.float16)   # [L, rows, H]
            BA = list(range(10))
            def buildA(gis):
                inps, qs, tag = [], [], []
                for gi in gis:
                    p = P[gi]; src = len(inps)
                    runs = [("START", "START", capq(p, grabq=set(READ), grabkb={LATE4[0]})), ("END", "END", capq(p, grabq=set(READ), grabkb={LATE4[0]})),
                            ("A_BAND", "END", capq(p, grabq=set(LATE4), cpatch=dict(src=src, comps=COMPS)))]
                    for nm, rcp, q_ in runs:
                        inps.append(p["enc"][rcp]); qs.append(q_); tag.append((gi, nm, src))
                return inps, qs, tag
            qdon = {}
            for gi, nm, lgb, stb, sts in runs_for(list(range(len(P))), buildA):
                keep(gi, nm, lgb, stb); nrun += 1
                qdon[(gi, nm)] = {li: stb[("q", li)] for li in READ if ("q", li) in stb}
                if nm == "END":
                    kE, kS = stb[("kb", LATE4[0])], sts[("kb", LATE4[0])]
                    kr = float((kE - kS).abs().max() / kE.abs().max()); keyrel.append(kr); recs[gi]["key_rel_diff"] = kr
            def buildB(gis):
                inps, qs, tag = [], [], []
                for gi in gis:
                    p = P[gi]; src = len(inps)
                    def ed(part, dkey, layers, p=p, gi=gi, **kw):
                        dq = qdon[(gi, dkey)]
                        return capq(p, edit={li: [dict(qsub=dict(owner=p["owner"], nb=N, part=part, donors=[(1.0, dq[li][BA])]), erows=BA)]
                                             for li in layers}, **kw)
                    cp = dict(src=src, comps=COMPS)
                    uend = p["rowsC"][1]
                    def edT(dkey, p=p, gi=gi, uend=uend):
                        dq = qdon[(gi, dkey)]
                        return capq(p, edit={li: [dict(qsub=dict(owner=p["owner"], nb=N, part="tsub", donors=[(1.0, dq[li][BA])], text_end=uend, rank=16),
                                                       erows=BA)] for li in LOOK})
                    runs = [("START_B", "START", capq(p)), ("END_B", "END", capq(p)), ("A_BAND_B", "END", capq(p, cpatch=cp)),
                            ("QNOOP", "END", capq(p, noop=set(LATE4))),
                            ("Q5155", "END", ed("all", "A_BAND", L12)), ("Q5963", "END", ed("all", "A_BAND", L34)),
                            ("QALL", "END", ed("all", "A_BAND", LATE4)), ("QALL_U", "END", ed("U", "A_BAND", LATE4)),
                            ("NEC", "END", ed("all", "END", LATE4, cpatch=cp)), ("NEC5155", "END", ed("all", "END", L12, cpatch=cp)),
                            ("TRES", "END", edT("START")), ("TREM", "START", edT("END"))]
                    for nm, rcp, q_ in runs:
                        inps.append(p["enc"][rcp]); qs.append(q_); tag.append((gi, nm, src))
                return inps, qs, tag
            for gi, nm, lgb, stb, sts in runs_for(list(range(len(P))), buildB):
                keep(gi, nm, lgb, stb); nrun += 1
            out.extend(recs)
            del P, qdon
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{a.tag}] N{N} k{k} {min(g0 + G, len(SC))}/{len(SC)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
    TOOL.remove()
    json.dump(dict(tag=a.tag, model=a.model, cases=CASES, band=band, comps=COMPS, readout=os.environ.get("R3_READ"), read=READ, late=LATE4,
                   matcher_layers=sorted(MH), n_layer=nl, qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   qk_heads="matcher heads where listed (35-55), else all heads (59, 63)",
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), rows=out),
              open(OUT.format(a.tag, "close"), "w"))
    np.savez_compressed(OUT.format(a.tag, "close").replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'close')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=84700); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", required=True)
    ap.add_argument("--max_images", type=int, default=330)
    main(ap.parse_args())
