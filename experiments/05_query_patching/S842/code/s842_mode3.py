"""S842 (pre-registered in knowledge/DECISIONS.md before any S842 data): gap 1 continued - where in the non-address part of the queries is the END
reading mode?  S839: not the picture-vs-menu, picture-vs-all-text or norm directions (the repairing non-address edit works without the menu
direction).  Here the START-minus-END difference of the queries (look layers 35-55, question-mark + boundary + answer rows) is split into
orthogonal subspaces per head (exp/c_qpath_m.py): W = within-picture token pattern (top 16 directions of image-token keys minus their block
mean), T = text keys (top 16 directions of the centred keys of every non-image token before the boundary rows), R = the non-address rest; and
the late re-reading of the question words is blocked (the ten rows cannot attend to the ordinal word and / or the frame word).
Cases: identity N7 END(4), N9 END(5); seed 84200 (new), 30 scenes per case; settings as the main series.
Arms: QNOOP_E / QNOOP_S, P_FULL (non-address part <- START's, S829 / S839 replication), WSUB_FULL / TSUB_FULL / RSUB_FULL (only that subspace of
the START - END difference added), KO_Q_LATE / KO_F_LATE / KO_O_LATE (END: no attention to ordinal + frame / frame / ordinal word at layers 51-63),
KO_Q_ALL (the same at 35-63).
Usage: python s842_mode3.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_b import QPathB
import c_qpath_m
from pipe_run import pipelined

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s842_{}_{}.json"
CASES = [(7, 4, 4), (9, 5, 5)]
NAT1 = ["END", "START"]
RUNS2 = 12                                                                # END, START + 10 arms


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S842_DEVICE", "cuda:0")
    c_qpath_m.patch()
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
    READ = [l for l in FA_L if l / (nl - 1) >= .555]                      # look layers + the last full-attention layers (59, 63)
    TOOL = QPathB(v, READ)
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
            return pipelined(run1, inps, qs, parts=a.parts)
        except torch.cuda.OutOfMemoryError:
            if len(inps) == 1:
                raise
        gc.collect(); torch.cuda.empty_cache()
        h_ = len(inps) // 2
        print(f"[{a.tag}] OOM at batch {len(inps)} -> {h_} + {len(inps) - h_}", flush=True)
        l1, s1 = runb(inps[:h_], qs[:h_]); l2, s2 = runb(inps[h_:], qs[h_:])
        return torch.cat([l1, l2], 0), {**s1, **{b + h_: x for b, x in s2.items()}}

    print(f"[{a.tag}] look {LOOK} · read {READ} · matcher layers {sorted(MH)} · scaling {scaling}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe, per case
        for N, k, j in CASES:
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            scenes.append(dict(si=si, N=N, k=k, j=j, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel = []
    for N, k, j in CASES:
        SC = [s for s in scenes if s["N"] == N]
        G = a.group or max(1, a.max_images // ((RUNS2 - 2) * N))   # scenes per batch: pictures per batch <= max_images
        print(f"[{a.tag}] N{N}: {G} scenes per batch", flush=True)
        for g0 in range(0, len(SC), G):
            grp = SC[g0:g0 + G]
            P = []
            for s in grp:
                qk_ = dict(END=(k, "last"), START=(j, "first"))
                enc = {nm: v.encode(prompt(N, q, fr, s["menu"]), s["pics"]) for nm, (q, fr) in qk_.items()}
                ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
                assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
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
                A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
                opt = []
                for w_ in s["menu"]:
                    pos_ = [i for i in range(qm + 1, uend) if ids["END"][i] == ids1(w_)[0]]
                    assert len(pos_) == 1, f"menu word {w_} must be one token once"
                    opt.append(pos_[0])
                P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, frame_pos=Fw))
            assert len({p["S"] for p in P}) == 1, "scenes of one case must align for batching"
            capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=N, headout=True, opt=p["opt"]), **kw)
            recs = [dict(si=p["s"]["si"], N=N, k=k, j=j, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={},
                         menu_pos=[(p["s"]["labs"].index(w_) + 1 if w_ in p["s"]["labs"] else 0) for w_ in p["s"]["menu"]]) for p in P]
            def keep(gi, nm, lgb, stb):
                rec = recs[gi]; menu = rec["menu"]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
                rec["qk"][nm] = {str(li): stb[("cap", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if ("cap", li) in stb and MH.get(li)}
                rec.setdefault("mass", {})[nm] = {str(li): stb[("img", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if MH.get(li)}
                key = f"N{N}_s{rec['si']}_{nm}"
                npz[key + "_cap"] = np.stack([stb[("cap", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H, nb]
                npz[key + "_hout"] = np.stack([stb[("hout", li)].float().numpy() for li in LOOK]).astype(np.float16)  # [L, 3, rows, H]
                npz[key + "_img"] = np.stack([stb[("img", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H] picture mass
                npz[key + "_opt"] = np.stack([stb[("opt", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H, 9] menu words
            # natural runs (stage 1: all six; stage 2: END / START as donors)
            NAT = ["END", "START"]
            inps, qs, tag = [], [], []
            for gi, p in enumerate(P):
                for nm in NAT:
                    inps.append(p["enc"][nm]); tag.append((gi, nm))
                    qs.append(capq(p, grabq=set(READ), grabkb=set(LOOK)))
            lg, st = runb(inps, qs); nrun += len(inps)
            don = {}
            for b, (gi, nm) in enumerate(tag):
                keep(gi, nm, lg[b], st[b])
                if nm in ("END", "START"):
                    don[(gi, nm)] = dict(q={li: st[b][("q", li)] for li in READ}, kb={li: st[b][("kb", li)] for li in LOOK})
            for gi, p in enumerate(P):
                kE, kS = don[(gi, "END")]["kb"], don[(gi, "START")]["kb"]
                kr = max(float((kE[li] - kS[li]).abs().max() / kE[li].abs().max()) for li in LOOK); keyrel.append(kr)
                recs[gi]["key_rel_diff"] = kr
            if True:
                inps, qs, tag = [], [], []
                BA = list(range(10)); LATE63 = [l for l in READ if l / (nl - 1) > .75]
                for gi, p in enumerate(P):
                    qE, qS = don[(gi, "END")]["q"], don[(gi, "START")]["q"]
                    uend = p["rowsC"][1]
                    def qcfg(layers, fns, p=p, **kw):
                        return capq(p, edit={li: [dict(qsub=fn(li), erows=BA) for fn in fns] for li in layers}, **kw)
                    part = lambda pt, dn, p=p, **kw: (lambda li: dict(owner=p["owner"], nb=N, part=pt, donors=[(w, q[li][BA]) for w, q in dn], **kw))
                    idsE = p["enc"]["END"]["input_ids"][0].tolist()
                    ordp = [i for i in range(p["qm"]) if idsE[i] == ids1(ORDW[k - 1])[0]]; assert len(ordp) == 1, "the ordinal word is one token once"
                    frp = p["frame_pos"]
                    LATEQ = {l for l in READ if l / (nl - 1) > .75}
                    ko = lambda layers, pos: dict(layers=set(layers), rows=BA, pos=pos)
                    sub = lambda pt: part(pt, [(1.0, qS)], text_end=uend, rank=16)
                    arms = [("QNOOP_E", "END", capq(p, noop=set(LOOK))), ("QNOOP_S", "START", capq(p, noop=set(LOOK))),
                            ("P_FULL", "END", qcfg(LOOK, [part("perp", [(1.0, qS)])])),
                            ("WSUB_FULL", "END", qcfg(LOOK, [sub("wsub")])), ("TSUB_FULL", "END", qcfg(LOOK, [sub("tsub")])),
                            ("RSUB_FULL", "END", qcfg(LOOK, [sub("rsub")])),
                            ("KO_Q_LATE", "END", capq(p, ko=ko(LATEQ, ordp + frp))), ("KO_F_LATE", "END", capq(p, ko=ko(LATEQ, frp))),
                            ("KO_O_LATE", "END", capq(p, ko=ko(LATEQ, ordp))), ("KO_Q_ALL", "END", capq(p, ko=ko(READ, ordp + frp)))]
                    assert len(arms) == RUNS2 - 2
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
            print(f"[{a.tag}] N{N} {min(g0 + G, len(SC))}/{len(SC)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
    TOOL.remove()
    nm_ = "mode3"
    json.dump(dict(tag=a.tag, model=a.model, read=READ, readout=os.environ.get("R3_READ"), look=LOOK, matcher_layers=sorted(MH),
                   matchers={str(l): h for l, h in MH.items()}, n_layer=nl, scaling=scaling,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   key_rel_diff_max=max(keyrel), rows=out), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=84200); ap.add_argument("--group", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower; N = 7 x 48 runs (336 pictures) fit
    main(ap.parse_args())
