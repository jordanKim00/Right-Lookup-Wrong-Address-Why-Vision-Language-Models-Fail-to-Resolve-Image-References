"""S827 (pre-registered in knowledge/DECISIONS.md before any S827 data; design revised 10-16 19:5x-20:4x after the math review
(knowledge/agent_exchange_2026-10-15/MATH_S827_REVIEW.md, 19:17) and the user's advisor notes (19:3x, 20:0x, 20:1x); the unrun first draft is kept
as exp/s827_identity_draft1.py).  One question: when the end-relative reading spoils an identity reference (target j = k = (N + 1) / 2, no
transformation needed), does END write a WRONG address or only WEAKEN the right one, which part of END's change of the boundary rows' query does
it (H-A: a copy-block suppression; H-B: a predecessor (k - 1) update; the k + 1 part for N = 9's 6), and do the internal quantities predict the
natural errors and the effects of interventions before those are run?
Cases: N = 7 END(4) (S823 / S824: 4 .475-.50, 3 .20-.275, 2 .10) and N = 9 END(5) (S825: 4 .50, 5 .10, 6 .325).
Settings exactly as the main series: N = 7 as S705a / S801 / S814 / S823 (seven sampled labels + the first two unused labels as menu-only words,
shuffled), N = 9 as S825 (the nine labels, shuffled); procedural pictures 160 px from the r3cal_a3 pool; the same question sentence ('There are N
images in a row. What is in the <ordinal> image from the last|first? Answer with only one word from: ...'), think prefix, two-form menu readout.
Exact decomposition: the pictures precede the question, so END(k) and START(k) of one scene share every picture token's key at every layer
(checked per scene: block-mean post-RoPE keys, largest relative difference; 4B smoke 0.0); a picture token's score difference is exactly
(q_END - q_START).K / sqrt d; block scores (log-sum-exp over a picture's tokens) are computed exactly from the shared keys.
Read rows: the question mark, the five boundary rows (B0..B4 = <|im_end|> \\n <|im_start|> assistant \\n), the answer row; per look layer (full
attention 35-55) every head's block scores and its gated output (norm of what it writes, of its picture part, mean output gate).
STAGE 1 (natural runs only): END(k), START(k), END(k - 1), END(k + 1), START(k - 1), START(k + 1); saves the post-RoPE queries and block-mean keys
of END(k) / START(k).  Then exp/s827_predict.py fixes the predictions (readout law, held-out-scene internal address, every arm's internal
change and answer shares) in research/ces/outputs/s827_predictions.json before STAGE 2 runs.
STAGE 2 (the arms; END(k) / START(k) rerun as donors; edit rows = the question mark + the five boundary rows; every look layer; every head;
U = span of the centred block-mean picture keys; d = q_END - q_START, natural; local parts of d per head: pattern c = C d, a_i = c_i - mean of c
over the blocks outside {k - 1, k, k + 1}, part x_S = pinv(C) sum_{i in S} a_i e_i):
  QNOOP_E / QNOOP_S   float32 recompute of the read rows, no edit (exactness; reference of the arms)
  A_FULL / A_HALF     END(k): U part set to START(k)'s / to the mean of the two (100 % / 50 % of the change removed)
  A_DIP / A_DIP50     END(k): x_{k} removed (100 % / 50 %) - H-A's copy-block part
  A_PREV / A_PREV50   END(k): x_{k-1} removed (100 % / 50 %) - H-B's predecessor part
  A_NEXT              END(k): x_{k+1} removed;   A_FLANK  x_{k-1} + x_{k+1} removed
  A_UK                END(k): the part of d along u_k = Kbar_k - mean_{b != k} Kbar_b removed (the advisor's definition)
  R_FULL              START(k): U part set to END(k)'s;   R_DIP / R_PREV / R_NEXT  START(k) + x_{k} / x_{k-1} / x_{k+1}
Scenes are run in groups for large batches (--group, or by default as many scenes as keep a batch's pictures <= --max_images); a batch that
does not fit is split in halves (identical per-sample computation).
Seed 82700 (new), 40 scenes per case.
Usage: python s827_identity.py MODEL TAG --stage 1|2 [--n 40] [--group G | --max_images 330] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_b import QPathB

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s827_{}_{}.json"
CASES = [(7, 4), (9, 5)]
NAT1 = ["END", "START", "END_m1", "END_p1", "START_m1", "START_p1"]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S827_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    LOOK = [l for l in FA_L if .555 <= l / (nl - 1) <= .875]
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
    TOOL = QPathB(v, LOOK)
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

    def runb(inps, qs):
        """One batch; on out-of-memory split in halves (each sample's computation does not depend on its batch neighbours).  The retry runs
        outside the except block, so the failed attempt's activations (held by the traceback) are freed first."""
        try:
            return run1(inps, qs)
        except torch.cuda.OutOfMemoryError:
            if len(inps) == 1:
                raise
        gc.collect(); torch.cuda.empty_cache()
        h_ = len(inps) // 2
        print(f"[{a.tag}] OOM at batch {len(inps)} -> {h_} + {len(inps) - h_}", flush=True)
        l1, s1 = runb(inps[:h_], qs[:h_]); l2, s2 = runb(inps[h_:], qs[h_:])
        return torch.cat([l1, l2], 0), {**s1, **{b + h_: x for b, x in s2.items()}}

    print(f"[{a.tag}] stage {a.stage} · look {LOOK} · matcher layers {sorted(MH)} · scaling {scaling}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe, per case
        for N, k in CASES:
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            scenes.append(dict(si=si, N=N, k=k, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel = []
    for N, k in CASES:
        SC = [s for s in scenes if s["N"] == N]
        G = a.group or max(1, a.max_images // ((len(NAT1) if a.stage == 1 else 15) * N))   # scenes per batch: pictures per batch <= max_images
        print(f"[{a.tag}] N{N}: {G} scenes per batch", flush=True)
        for g0 in range(0, len(SC), G):
            grp = SC[g0:g0 + G]
            P = []
            for s in grp:
                qk_ = dict(END=(k, "last"), START=(k, "first"), END_m1=(k - 1, "last"), END_p1=(k + 1, "last"),
                           START_m1=(k - 1, "first"), START_p1=(k + 1, "first"))
                enc = {nm: v.encode(prompt(N, q, fr, s["menu"]), s["pics"]) for nm, (q, fr) in qk_.items()}
                ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
                assert len({len(x) for x in ids.values()}) == 1, "the six prompts of a scene must align"
                S_ = len(ids["END"])
                ve = [i for i, x in enumerate(ids["END"]) if x == ve_id]; vs = [i for i, x in enumerate(ids["END"]) if x == vs_id]
                qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["END"][i]]))
                uend = max(i for i, x in enumerate(ids["END"]) if x == end_id)
                assert v.tok.decode([ids["END"][uend + 3]]).strip() == "assistant"
                Fw = [i for i, (x, y) in enumerate(zip(ids["START"], ids["END"])) if x != y]
                assert Fw and max(Fw) < qm
                owner = [-1] * S_
                for b in range(N):
                    for p_ in range(vs[b], ve[b] + 1):
                        owner[p_] = b
                P.append(dict(s=s, enc=enc, S=S_, owner=owner, rowsC=[qm] + list(range(uend, uend + 5)) + [S_ - 1]))
            assert len({p["S"] for p in P}) == 1, "scenes of one case must align for batching"
            capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(7)), owner=p["owner"], nb=N, headout=True), **kw)
            recs = [dict(si=p["s"]["si"], N=N, k=k, j=N + 1 - k, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={})
                    for p in P]
            def keep(gi, nm, lgb, stb):
                rec = recs[gi]; menu = rec["menu"]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
                rec["qk"][nm] = {str(li): stb[("cap", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if ("cap", li) in stb and MH.get(li)}
                key = f"N{N}_s{rec['si']}_{nm}"
                npz[key + "_cap"] = np.stack([stb[("cap", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H, nb]
                npz[key + "_hout"] = np.stack([stb[("hout", li)].float().numpy() for li in LOOK]).astype(np.float16)  # [L, 3, rows, H]
            # natural runs (stage 1: all six; stage 2: END / START as donors)
            NAT = NAT1 if a.stage == 1 else ["END", "START"]
            inps, qs, tag = [], [], []
            for gi, p in enumerate(P):
                for nm in NAT:
                    inps.append(p["enc"][nm]); tag.append((gi, nm))
                    qs.append(capq(p, grabq=set(LOOK), grabqe=set(LOOK), grabkb=set(LOOK)) if nm in ("END", "START") else capq(p))
            lg, st = runb(inps, qs); nrun += len(inps)
            don = {}
            for b, (gi, nm) in enumerate(tag):
                keep(gi, nm, lg[b], st[b])
                if nm in ("END", "START"):
                    don[(gi, nm)] = dict(q={li: st[b][("q", li)] for li in LOOK}, qe={li: st[b][("qe", li)] for li in LOOK},
                                         kb={li: st[b][("kb", li)] for li in LOOK})
            for gi, p in enumerate(P):
                kE, kS = don[(gi, "END")]["kb"], don[(gi, "START")]["kb"]
                kr = max(float((kE[li] - kS[li]).abs().max() / kE[li].abs().max()) for li in LOOK); keyrel.append(kr)
                recs[gi]["key_rel_diff"] = kr
                if a.stage == 1:
                    key = f"N{N}_s{recs[gi]['si']}"
                    for nm, tg in (("END", "E"), ("START", "S")):
                        npz[f"{key}_qe{tg}"] = np.stack([don[(gi, nm)]["qe"][li].float().numpy() for li in LOOK]).astype(np.float32)  # [L, rows, H, hd]
                    npz[key + "_kb"] = np.stack([kE[li].float().numpy() for li in LOOK]).astype(np.float32)                         # [L, nb, H, hd]
            if a.stage == 2:
                inps, qs, tag = [], [], []
                for gi, p in enumerate(P):
                    qE, qS = don[(gi, "END")]["q"], don[(gi, "START")]["q"]
                    e6 = list(range(6)); loc = [k - 2, k - 1, k]
                    def qcfg(fn, p=p):
                        return capq(p, edit={li: dict(qsub=fn(li), erows=e6) for li in LOOK})
                    U = lambda dn, p=p: (lambda li: dict(owner=p["owner"], nb=N, part="U", donors=[(w, q[li][:6]) for w, q in dn]))
                    Lc = lambda sel, coef, p=p, qE=qE, qS=qS: (lambda li: dict(owner=p["owner"], nb=N, part="local",
                                                                                delta=[(1.0, qE[li][:6]), (-1.0, qS[li][:6])], loc=loc, sel=sel, coef=coef))
                    Pj = lambda coef, p=p, qE=qE, qS=qS: (lambda li: dict(owner=p["owner"], nb=N, part="proj", block=k - 1,
                                                                           delta=[(1.0, qE[li][:6]), (-1.0, qS[li][:6])], coef=coef))
                    arms = [("QNOOP_E", "END", capq(p, noop=set(LOOK))), ("QNOOP_S", "START", capq(p, noop=set(LOOK))),
                            ("A_FULL", "END", qcfg(U([(1.0, qS)]))), ("A_HALF", "END", qcfg(U([(0.5, qS), (0.5, qE)]))),
                            ("A_DIP", "END", qcfg(Lc([k - 1], -1.0))), ("A_DIP50", "END", qcfg(Lc([k - 1], -0.5))),
                            ("A_PREV", "END", qcfg(Lc([k - 2], -1.0))), ("A_PREV50", "END", qcfg(Lc([k - 2], -0.5))),
                            ("A_NEXT", "END", qcfg(Lc([k], -1.0))), ("A_FLANK", "END", qcfg(Lc([k - 2, k], -1.0))),
                            ("A_UK", "END", qcfg(Pj(-1.0))),
                            ("R_FULL", "START", qcfg(U([(1.0, qE)]))), ("R_DIP", "START", qcfg(Lc([k - 1], 1.0))),
                            ("R_PREV", "START", qcfg(Lc([k - 2], 1.0))), ("R_NEXT", "START", qcfg(Lc([k], 1.0)))]
                    for nm, rcp, q_ in arms:
                        inps.append(p["enc"][rcp]); qs.append(q_); tag.append((gi, nm))
                lg, st = runb(inps, qs); nrun += len(inps)
                for b, (gi, nm) in enumerate(tag):
                    keep(gi, nm, lg[b], st[b])
            out.extend(recs)
            del don, P
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{a.tag}] stage {a.stage} N{N} {min(g0 + G, len(SC))}/{len(SC)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
    TOOL.remove()
    nm_ = "natural" if a.stage == 1 else "identity"
    json.dump(dict(tag=a.tag, model=a.model, stage=a.stage, readout=os.environ.get("R3_READ"), look=LOOK, matcher_layers=sorted(MH),
                   matchers={str(l): h for l, h in MH.items()}, n_layer=nl, scaling=scaling,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   key_rel_diff_max=max(keyrel), rows=out), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=82700); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower; N = 7 x 48 runs (336 pictures) fit
    main(ap.parse_args())
