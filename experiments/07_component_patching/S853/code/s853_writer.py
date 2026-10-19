"""S853 (pre-registered in knowledge/DECISIONS.md before any S853 data; user 10-18 'GO - writer input decomposition').  What does the writer
(layers 46-49) take in, what does it compute, and how is the result carried?  Eight real pictures, the same menu, the END frame, the early
lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p, N_s)).  Writer-input states (natural pinned END
runs; the stated count N_s is a counterfactual probe when it differs from the eight pictures shown):
  E1 (k 3, p 3, N_s 8)  E2 (k 3, p 4, 8)  E3 (k 5, p 3, 8)  E4 (k 5, p 4, 8)      main 2 x 2 (ordinal k x early position p), j* = 9 - k
  E5 (k 3, p 3, 6)  E6 (k 5, p 3, 6)  E7 (k 4, p 3, 7)                             count / same-target set at p 3 (j* = N_s + 1 - k)
Transplants X[R <- D] (same scene, same pass): the residual of the boundary rows B1 and B4 at the input of layer 46 is replaced by the donor's,
and the layer-47 pin of those two rows uses the donor's START(p, N_s) (layer 47 is both a pinned early look layer and a writer layer); every
other row keeps the recipient's state and pin.
  K   k only      E1<-E3  E3<-E1  E2<-E4  E4<-E2
  P   p only      E1<-E2  E2<-E1  E3<-E4  E4<-E3
  KP  both        E1<-E4  E4<-E1  E2<-E3  E3<-E2       (interaction: is the both-swap the sum of the single swaps?)
  N   count only  E1<-E5  E5<-E1  E3<-E6  E6<-E3       (same k, different N_s)
  T   same target E3<-E5  E5<-E3                       (j* = 4 in both)
  SELF            E1<-E1  E3<-E3                       (bit-identical to E1 / E3 by construction)
  ALLR            all ten read rows (and all their layer-47 pins) from the donor, for E1<-E3, E3<-E1, E3<-E4, E4<-E3, E1<-E4, E1<-E5, E3<-E6,
                  E3<-E5 (how much of the all-rows effect B1 / B4 carry)
  FULLB           E1 / E3 with the boundary rows B0-B4 at the input of layer 46 <- START(k) pinned at the same p (S_E1 / S_E3; S850's
                  positive control of the residual patch: answers changed .46)
Per scene: stage 1 = END(k, N_s) for (3, 8), (5, 8), (3, 6), (5, 6), (4, 7) and START(q, N_s) for (3, 8), (4, 8), (5, 8), (3, 6), (5, 6),
(3, 7), (4, 7); stage 2 = pass A (E1-E7, QNOOP_E on END(5, 8), K, P, KP, N, T, SELF) and pass B (E1-E7 again, S_E1, S_E3, ALLR, FULLB).
30 scenes (seed 85300).  Readouts: answer, menu-word logits, positional scores of every read row at the read layers (35-63), picture mass;
writer outputs (layers 46-49, mixer and MLP) and late queries (51-63) at rows B1, B4, answer for the E, X and ALLR runs.
Usage: python s853_writer.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s853_{}_{}.json"
N = 8
COND = {"E1": (3, 3, 8), "E2": (3, 4, 8), "E3": (5, 3, 8), "E4": (5, 4, 8), "E5": (3, 3, 6), "E6": (5, 3, 6), "E7": (4, 3, 7)}   # (k, p, N_s)
ENDS = [(3, 8), (5, 8), (3, 6), (5, 6), (4, 7)]                                   # (k, N_s)
STARTS = [(3, 8), (4, 8), (5, 8), (3, 6), (5, 6), (3, 7), (4, 7)]                 # (q, N_s)
XA = [("XK_1_3", "E1", "E3"), ("XK_3_1", "E3", "E1"), ("XK_2_4", "E2", "E4"), ("XK_4_2", "E4", "E2"),
      ("XP_1_2", "E1", "E2"), ("XP_2_1", "E2", "E1"), ("XP_3_4", "E3", "E4"), ("XP_4_3", "E4", "E3"),
      ("XKP_1_4", "E1", "E4"), ("XKP_4_1", "E4", "E1"), ("XKP_2_3", "E2", "E3"), ("XKP_3_2", "E3", "E2"),
      ("XN_1_5", "E1", "E5"), ("XN_5_1", "E5", "E1"), ("XN_3_6", "E3", "E6"), ("XN_6_3", "E6", "E3"),
      ("XT_3_5", "E3", "E5"), ("XT_5_3", "E5", "E3"), ("XSELF_1", "E1", "E1"), ("XSELF_3", "E3", "E3")]
XB = [("AK_1_3", "E1", "E3"), ("AK_3_1", "E3", "E1"), ("AP_3_4", "E3", "E4"), ("AP_4_3", "E4", "E3"), ("AKP_1_4", "E1", "E4"),
      ("AN_1_5", "E1", "E5"), ("AN_3_6", "E3", "E6"), ("AT_3_5", "E3", "E5")]
RB14 = [2, 5]                                                                      # B1, B4 (indices into the ten read rows)
ROTHER = [i for i in range(10) if i not in RB14]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S853_DEVICE", "cuda:0")
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
    SITE = 46 if nl == 64 else band[0]                                     # the writer's input
    assert EARLY[-1] in band or nl != 64, "layer 47 is a pinned early look layer inside the band in the 27B"
    KROWS = [2, 5, 9]
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
    TOOL = QPathI(v, READ, res_layers=(), comp_layers=band, patch_comps=(), rpatch_layers=[SITE])
    T1 = []

    def prompt(q, frame, menu, Ns):
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

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · site {SITE} · matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra, menu=menu, pics=pics))
    NAT = [f"END_k{k}N{ns}" for k, ns in ENDS] + [f"START_q{q}N{ns}" for q, ns in STARTS]
    P = []
    for s in scenes:
        qk_ = {**{f"END_k{k}N{ns}": (k, "last", ns) for k, ns in ENDS}, **{f"START_q{q}N{ns}": (q, "first", ns) for q, ns in STARTS}}
        enc = {nm: v.encode(prompt(q, fr, s["menu"], ns), s["pics"]) for nm, (q, fr, ns) in qk_.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
        ref = ids["END_k5N8"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
        for k, ns in ENDS:
            Fw = [i for i, (x, y) in enumerate(zip(ids[f"START_q{k}N{ns}"], ids[f"END_k{k}N{ns}"])) if x != y]
            assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
        for (k1, n1), (k2, n2) in [((3, 8), (3, 6)), ((5, 8), (5, 6)), ((3, 8), (5, 8))]:
            Dw = [i for i, (x, y) in enumerate(zip(ids[f"END_k{k1}N{n1}"], ids[f"END_k{k2}N{n2}"])) if x != y]
            assert len(Dw) == 1 and Dw[0] < qm, "END prompts differ in one token (the stated count or the ordinal)"
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
    recs = {p["s"]["si"]: dict(si=p["s"]["si"], N=N, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={}, mass={},
                               menu_pos=[(p["s"]["labs"].index(w_) + 1 if w_ in p["s"]["labs"] else 0) for w_ in p["s"]["menu"]]) for p in P}

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

    # stage 1: natural runs (END(k, N_s), START(q, N_s): pin donors and START(k) checks)
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
            rec = recs[P[gi]["s"]["si"]]; put(rec, nm, measure(rec, lgb, stb)); nrun += 1
            don[(gi, nm)] = dict(q={li: stb[("q", li)] for li in EARLY}, kb={li: stb[("kb", li)] for li in LOOK})
        print(f"[{a.tag}] stage 1 {min(g0 + G1, len(P))}/{len(P)} · {nrun} runs", flush=True)
    for gi in range(len(P)):
        ref_kb = don[(gi, "END_k5N8")]["kb"]
        for nm in NAT:
            kb = don[(gi, nm)]["kb"]
            keyrel.append(max(float((kb[li] - ref_kb[li]).abs().max() / ref_kb[li].abs().max()) for li in LOOK))
        recs[P[gi]["s"]["si"]]["key_rel_diff"] = max(keyrel[-len(NAT):])
    # stage 2: two passes per scene; every donor of a same-pass patch (E1-E7, S_E1, S_E3) is in its pass
    CB = set(band); LQ = set(LATE)
    def pin(gi, p, pn, pd=None, pdn=None, all_from_donor=False):
        """Early-layer query pins: the ten rows <- START(p, pn); with a donor (pd, pdn) the layer-47 pin of B1 / B4 (or of every row) uses
        START(pd, pdn).  Every run uses the same two-entry structure (B1 / B4 and the other rows), so a self transplant is bit-identical."""
        qr = don[(gi, f"START_q{p}N{pn}")]["q"]; qd = don[(gi, f"START_q{pd}N{pdn}")]["q"] if pd is not None else qr
        ed = {}
        for li in EARLY:
            last = li == EARLY[-1]
            src_b = qd if (last and pd is not None) else qr
            src_o = qd if (last and pd is not None and all_from_donor) else qr
            ed[li] = [dict(qsub=dict(owner=P[gi]["owner"], nb=N, part="U", donors=[(1.0, src_o[li][ROTHER])]), erows=ROTHER),
                      dict(qsub=dict(owner=P[gi]["owner"], nb=N, part="U", donors=[(1.0, src_b[li][RB14])]), erows=RB14)]
        return ed
    def build_pass(gis, which):
        inps, qs, tag = [], [], []
        for gj in gis:
            p = P[gj]; idx = {}
            for nm, (k, p_, ns) in COND.items():
                idx[nm] = len(inps); inps.append(p["enc"][f"END_k{k}N{ns}"])
                qs.append(capq(p, edit=pin(gj, p_, ns), keepcomp=CB, grabcomp=CB, grabq=LQ)); tag.append((gj, nm))
            if which == "A":
                inps.append(p["enc"]["END_k5N8"]); qs.append(capq(p, noop=set(LOOK))); tag.append((gj, "QNOOP_E"))
                for nm, r_, d_ in XA:
                    k, p_, ns = COND[r_]; kd, pd, nd = COND[d_]
                    inps.append(p["enc"][f"END_k{k}N{ns}"])
                    qs.append(capq(p, edit=pin(gj, p_, ns, pd, nd), rpatch={SITE: dict(src=idx[d_], rows=[p["rowsC"][i] for i in RB14])},
                                   keepcomp=CB, grabcomp=CB, grabq=LQ)); tag.append((gj, nm))
            else:
                for nm, (k, ns) in (("S_E1", (3, 8)), ("S_E3", (5, 8))):
                    idx[nm] = len(inps); inps.append(p["enc"][f"START_q{k}N{ns}"]); qs.append(capq(p, edit=pin(gj, 3, ns))); tag.append((gj, nm))
                for nm, r_, d_ in XB:
                    k, p_, ns = COND[r_]; kd, pd, nd = COND[d_]
                    inps.append(p["enc"][f"END_k{k}N{ns}"])
                    qs.append(capq(p, edit=pin(gj, p_, ns, pd, nd, all_from_donor=True), rpatch={SITE: dict(src=idx[d_], rows=list(p["rowsC"]))},
                                   keepcomp=CB, grabcomp=CB, grabq=LQ)); tag.append((gj, nm))
                for nm, r_, s_ in (("FB_1", "E1", "S_E1"), ("FB_3", "E3", "S_E3")):
                    k, p_, ns = COND[r_]
                    inps.append(p["enc"][f"END_k{k}N{ns}"])
                    qs.append(capq(p, edit=pin(gj, p_, ns), rpatch={SITE: dict(src=idx[s_], rows=p["rowsC"][1:6])})); tag.append((gj, nm))
        return inps, qs, tag
    for gi in range(len(P)):
        for which in ("A", "B"):
            for gj, nm, lgb, stb in runs_for([gi], lambda g, w=which: build_pass(g, w)):
                rec = recs[P[gj]["s"]["si"]]; key = nm if (which == "A" or nm not in COND) else nm + "_B"
                put(rec, key, measure(rec, lgb, stb)); nrun += 1
                if ("mix", band[0]) in stb:
                    npz[f"s{rec['si']}_{key}_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
                if ("q", LATE[0]) in stb:
                    npz[f"s{rec['si']}_{key}_q"] = np.stack([stb[("q", li)][KROWS].float().numpy() for li in LATE]).astype(np.float16)
        if gi % 5 == 4 or gi == len(P) - 1:
            print(f"[{a.tag}] stage 2 {gi + 1}/{len(P)} · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "writer"
    json.dump(dict(tag=a.tag, model=a.model, cond=COND, xa=XA, xb=XB, band=band, site=SITE, comps=COMPS, readout=os.environ.get("R3_READ"),
                   read=READ, look=LOOK, early=EARLY, late=LATE, krows=KROWS, matcher_layers=sorted(MH), matchers={str(l): h for l, h in MH.items()},
                   n_layer=nl, scaling=scaling, qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun,
                   rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85300); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)
    main(ap.parse_args())
