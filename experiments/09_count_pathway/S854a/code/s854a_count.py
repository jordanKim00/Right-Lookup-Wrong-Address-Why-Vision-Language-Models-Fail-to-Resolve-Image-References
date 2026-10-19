"""S854-1 (pre-registered in knowledge/DECISIONS.md before any S854-1 data; user 10-18 'GO - first separate the ordinal from the image counts').
Does the late update follow the ordinal k, the target counted from the actual end (j_a = N_a + 1 - k) or the target counted from the stated count
(j_s = N_s + 1 - k)?  Full factorial per scene: actual pictures N_a in {6, 8} (the first 6 or all 8 pictures of the same scene, the same nine-word
menu in the same order), stated count N_s in {6, 8}, ordinal k in {3, 5}, early lookup pinned to p in {3, 4} (address part U of the ten read rows
at the early look layers 35-47 <- START(p, N_a, N_s)).  Conditions with N_s != N_a are counterfactual probes, not accuracy tests.  Runs of
different N_a differ in length and are never patched into each other; every comparison is between conditions' own runs.
Per scene and N_a: stage 1 = END(N_s, k) (4) and START(q, N_s) for q in {3, 4, 5} (6; pin donors q = 3, 4; START(k) checks q = 3, 5);
stage 2 = QNOOP_E (END(N_s = N_a, k = 5)), E (END(N_s, k) pinned at p; 8) and S (START(k) pinned at p, same N_s; 8 - the END-specific write is
D = band_E - band_S at the same N_a, N_s, k, p).  30 new scenes (seed 85400).  Readouts: answer, menu-word logits, positional scores of every
read row at the read layers (35-63), picture mass; for E and S the writer outputs (layers 46-49, mixer and MLP) at rows B1, B4, answer; for E
the residual at the input of layer 46 (rows B1, B4, answer) and the late queries (51-63).
Usage: python s854a_count.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s854a_{}_{}.json"
NAS = (8, 6); NSS = (6, 8); KS = (3, 5); PS = (3, 4); QS = (3, 4, 5)
NLAB = 8


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
    TOOL = QPathI(v, READ, res_layers=[SITE], comp_layers=band, patch_comps=())
    T1 = {na: [] for na in NAS}

    def prompt(na, q, frame, menu, ns):
        return v.chat([("user", f"{v.IMG*na}\nThere are {ns} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
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
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = {na: [] for na in NAS}
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, NLAB); extra = [l for l in labels if l not in labs][:9 - NLAB]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra, menu=menu, pics=pics))
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}, key_rel_diff={})
            for s in scenes}

    def measure(rec, na, lgb, stb):
        menu = rec["menu"]
        T1[na].append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]

    for na in NAS:
        P = []
        for s in scenes:
            nms = {**{f"END_s{ns}k{k}": (k, "last", ns) for ns in NSS for k in KS}, **{f"START_s{ns}q{q}": (q, "first", ns) for ns in NSS for q in QS}}
            enc = {nm: v.encode(prompt(na, q, fr, s["menu"], ns), s["pics"][:na]) for nm, (q, fr, ns) in nms.items()}
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene and N_a must align"
            ref = ids[f"END_s{NSS[-1]}k5"]; S_ = len(ref)
            ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
            assert len(ve) == na and len(vs) == na
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
            uend = max(i for i, x in enumerate(ref) if x == end_id)
            assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
            for ns in NSS:
                for k in KS:
                    Fw = [i for i, (x, y) in enumerate(zip(ids[f"START_s{ns}q{k}"], ids[f"END_s{ns}k{k}"])) if x != y]
                    assert len(Fw) == 1 and Fw[0] < qm, "START(k) and END(k) differ in the frame word only"
            Dw = [i for i, (x, y) in enumerate(zip(ids["END_s6k5"], ids["END_s8k5"])) if x != y]
            assert len(Dw) == 1 and Dw[0] < qm, "stated counts differ in one token"
            owner = [-1] * S_
            for b in range(na):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
            opt = []
            for w_ in s["menu"]:
                pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
                assert len(pos_) == 1, f"menu word {w_} must be one token once"
                opt.append(pos_[0])
            P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4))
        assert len({p["S"] for p in P}) == 1, "scenes of one N_a must align for batching"
        capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=na, opt=p["opt"]), **kw)
        NAT = list(P[0]["enc"])
        G1 = a.group or max(1, a.max_images // (len(NAT) * na))
        don = {}
        for g0 in range(0, len(P), G1):
            def build1(gis):
                inps, qs, tag = [], [], []
                for gi in gis:
                    for nm in NAT:
                        inps.append(P[gi]["enc"][nm]); tag.append((gi, nm)); qs.append(capq(P[gi], grabq=set(EARLY), grabkb=set(LOOK)))
                return inps, qs, tag
            for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G1, len(P)))), build1):
                rec = recs[P[gi]["s"]["si"]]; put(rec, f"{nm}_a{na}", measure(rec, na, lgb, stb)); nrun += 1
                don[(gi, nm)] = dict(q={li: stb[("q", li)] for li in EARLY}, kb={li: stb[("kb", li)] for li in LOOK})
            print(f"[{a.tag}] N_a {na} stage 1 {min(g0 + G1, len(P))}/{len(P)} · {nrun} runs", flush=True)
        for gi in range(len(P)):
            ref_kb = don[(gi, f"END_s{NSS[-1]}k5")]["kb"]
            kr = max(max(float((don[(gi, nm)]["kb"][li] - ref_kb[li]).abs().max() / ref_kb[li].abs().max()) for li in LOOK) for nm in NAT)
            keyrel[na].append(kr); recs[P[gi]["s"]["si"]]["key_rel_diff"][str(na)] = kr
        BA = list(range(10)); CB = set(band); LQ = set(LATE)
        def ed(gi, p_, ns):
            dq = don[(gi, f"START_s{ns}q{p_}")]["q"]
            return {li: [dict(qsub=dict(owner=P[gi]["owner"], nb=na, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}
        G2 = a.group2 or max(1, a.max_images // ((1 + 2 * len(NSS) * len(KS) * len(PS)) * na))
        for g0 in range(0, len(P), G2):
            def build2(gis):
                inps, qs, tag = [], [], []
                for gi in gis:
                    p = P[gi]
                    inps.append(p["enc"][f"END_s{na}k5"]); qs.append(capq(p, noop=set(LOOK))); tag.append((gi, "QNOOP_E"))
                    for ns in NSS:
                        for k in KS:
                            for p_ in PS:
                                inps.append(p["enc"][f"END_s{ns}k{k}"])
                                qs.append(capq(p, edit=ed(gi, p_, ns), keepcomp=CB, grabcomp=CB, grabq=LQ, grabres={SITE})); tag.append((gi, f"E_s{ns}k{k}p{p_}"))
                                inps.append(p["enc"][f"START_s{ns}q{k}"])
                                qs.append(capq(p, edit=ed(gi, p_, ns), keepcomp=CB, grabcomp=CB)); tag.append((gi, f"S_s{ns}k{k}p{p_}"))
                return inps, qs, tag
            for gi, nm, lgb, stb in runs_for(list(range(g0, min(g0 + G2, len(P)))), build2):
                rec = recs[P[gi]["s"]["si"]]; key = f"{nm}_a{na}"
                put(rec, key, measure(rec, na, lgb, stb)); nrun += 1
                if ("mix", band[0]) in stb:
                    npz[f"s{rec['si']}_{key}_comp"] = np.stack([stb[(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
                if ("q", LATE[0]) in stb:
                    npz[f"s{rec['si']}_{key}_q"] = np.stack([stb[("q", li)][KROWS].float().numpy() for li in LATE]).astype(np.float16)
                if ("res", SITE) in stb:
                    npz[f"s{rec['si']}_{key}_res"] = stb[("res", SITE)][KROWS].float().numpy().astype(np.float16)
            print(f"[{a.tag}] N_a {na} stage 2 {min(g0 + G2, len(P))}/{len(P)} · {nrun} runs · key diff max {max(keyrel[na]):.2e}", flush=True)
        del don, P
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "count"
    json.dump(dict(tag=a.tag, model=a.model, nas=NAS, nss=NSS, ks=KS, ps=PS, band=band, site=SITE, comps=COMPS, readout=os.environ.get("R3_READ"),
                   read=READ, look=LOOK, early=EARLY, late=LATE, krows=KROWS, matcher_layers=sorted(MH), n_layer=nl,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu={str(na): (sum(x) / len(x) if x else None) for na, x in T1.items()}, key_rel_diff_max={str(na): max(x) for na, x in keyrel.items()},
                   n_runs=nrun, rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85400); ap.add_argument("--group", type=int, default=0); ap.add_argument("--group2", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--max_images", type=int, default=330)
    main(ap.parse_args())
