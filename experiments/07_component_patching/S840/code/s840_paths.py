"""S840 (pre-registered in knowledge/DECISIONS.md before any S840 data): S837 found no single band component (layers 46-49) whose direct path to the
late queries carries the step (largest: DeltaNet 49, rho .17), the band's direct paths together .46, and the reading mode written by MLP 47.
(1) Which group carries the step on its direct paths - the mixers or the MLPs, layers 46-47 or 48-49?  (2) Does the step need the band's
outputs to be processed downstream, and is any single component a bottleneck?  Activation patches (exp/c_qpath_f.py 'cpatch'): END's output
of a component at the question mark, boundary and answer rows is replaced by START(k)'s of the same pass and every later layer sees it.
Cases as S836 / S837; seed 84000 (new), 30 scenes per case.  Per scene in one batch: START(k) (donor), END(k), DNOOP, D_LATE (the exact late-
query swap), C_BAND (direct, all eight; S837 replication), G_MIX / G_MLP / G_4647 / G_4849 (direct paths of the mixers 46-49 / the MLPs 46-49 /
all of 46-47 / all of 48-49), A_BAND (activation patch of all eight), A_l_mix / A_l_mlp for l in 46-49 (activation patch of one).
Usage: python s840_paths.py MODEL TAG --band 46 47 48 49 [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_f import QPathF

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s840_{}_{}.json"
CASES = [(7, 4, 4), (9, 5, 5), (7, 5, 3), (9, 7, 3)]
BOUNDS = []                                                               # S837: the band comes from --band


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S840_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    LOOK = [l for l in FA_L if .555 <= l / (nl - 1) <= .875]
    LATE = [l for l in LOOK if l / (nl - 1) > .75]
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE))      # smoke models: by relative depth
    assert band and all(b < min(LATE) for b in band)
    bounds = band
    NRUN = 4 + 5 + 1 + 2 * len(band)                                     # START, END, DNOOP, D_LATE, C_BAND + 4 groups, A_BAND, A_l_kind
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
    TOOL = QPathF(v, LOOK, res_layers=LATE, comp_layers=band, patch_comps=band)
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
        """One batch; on out-of-memory split in halves (each sample's computation does not depend on its batch neighbours)."""
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

    print(f"[{a.tag}] look {LOOK} · late {LATE} · band {band} · matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}; scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe, per case
        for N, k, j in CASES:
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            scenes.append(dict(si=si, N=N, k=k, j=j, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel, qchk = [], []
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
            def keep(gi, nm, lgb, stb):
                rec = recs[gi]; menu = rec["menu"]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
                rec["qk"][nm] = {str(li): stb[("cap", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if ("cap", li) in stb and MH.get(li)}
                rec["mass"][nm] = {str(li): stb[("img", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if ("img", li) in stb and MH.get(li)}
                key = f"N{N}k{k}_s{rec['si']}_{nm}"
                npz[key + "_cap"] = np.stack([stb[("cap", li)].float().numpy() for li in LOOK]).astype(np.float16)   # [L, rows, H, nb]
                npz[key + "_img"] = np.stack([stb[("img", li)].float().numpy() for li in LOOK]).astype(np.float16)   # [L, rows, H]
                npz[key + "_opt"] = np.stack([stb[("opt", li)].float().numpy() for li in LOOK]).astype(np.float16)   # [L, rows, H, 9]
            KEEP = set(LATE); KC = set(band)
            COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
            def build(gis):
                """inputs / configs of the scenes gis in one batch; deltas refer to the donor START(k) of the same scene in this batch."""
                inps, qs, tag = [], [], []
                for gi in gis:
                    p = P[gi]; src = len(inps)
                    runs = [("START", "START", capq(p, keepres=KEEP, keepcomp=KC, grabq=set(LATE), grabkb={LATE[0]})),
                            ("END", "END", capq(p, grabkb={LATE[0]})),
                            ("DNOOP", "END", capq(p, keepres=KEEP, delta={li: dict(vec=torch.zeros(10, v.model.config.get_text_config().hidden_size))
                                                                          for li in LATE})),
                            ("D_LATE", "END", capq(p, keepres=KEEP, delta={li: dict(src=src, layer=li) for li in LATE}))]
                    half = len(band) // 2
                    groups = {"C_BAND": COMPS, "G_MIX": [c_ for c_ in COMPS if c_[1] == "mix"], "G_MLP": [c_ for c_ in COMPS if c_[1] == "mlp"],
                              "G_4647": [c_ for c_ in COMPS if c_[0] in band[:half]], "G_4849": [c_ for c_ in COMPS if c_[0] in band[half:]]}
                    runs += [(g_, "END", capq(p, keepres=KEEP, keepcomp=KC, delta={li: dict(src=src, comps=cs) for li in LATE})) for g_, cs in groups.items()]
                    runs += [("A_BAND", "END", capq(p, cpatch=dict(src=src, comps=COMPS)))]
                    runs += [(f"A_{l_}_{kd}", "END", capq(p, cpatch=dict(src=src, comps=[(l_, kd)]))) for l_, kd in COMPS]
                    for nm, rcp, q_ in runs:
                        inps.append(p["enc"][rcp]); qs.append(q_); tag.append((gi, nm, src))
                return inps, qs, tag
            def runs_for(gis):
                inps, qs, tag = build(gis)
                try:
                    lg, st = run1(inps, qs)
                except torch.cuda.OutOfMemoryError:
                    if len(gis) == 1:
                        raise
                    gc.collect(); torch.cuda.empty_cache()
                    h_ = len(gis) // 2
                    print(f"[{a.tag}] OOM at {len(gis)} scenes -> {h_} + {len(gis) - h_}", flush=True)
                    return runs_for(gis[:h_]) + runs_for(gis[h_:])
                res_ = []
                for b, (gi, nm, src) in enumerate(tag):
                    res_.append((gi, nm, lg[b], st[b], st[src]))
                return res_
            for gi, nm, lgb, stb, sts in runs_for(list(range(len(P)))):
                keep(gi, nm, lgb, stb); nrun += 1
                if nm == "END":
                    kE, kS = stb[("kb", LATE[0])], sts[("kb", LATE[0])]
                    kr = float((kE - kS).abs().max() / kE.abs().max()); keyrel.append(kr); recs[gi]["key_rel_diff"] = kr
            out.extend(recs)
            del P
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{a.tag}] N{N} k{k} {min(g0 + G, len(SC))}/{len(SC)} · {nrun} runs · key diff max {max(keyrel):.2e}",
                  flush=True)
    TOOL.remove()
    json.dump(dict(tag=a.tag, model=a.model, cases=CASES, band=band, readout=os.environ.get("R3_READ"), look=LOOK, late=LATE,
                   matcher_layers=sorted(MH), n_layer=nl, qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), rows=out),
              open(OUT.format(a.tag, "paths"), "w"))
    np.savez_compressed(OUT.format(a.tag, "paths").replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'paths')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=84000); ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--band", type=int, nargs="+", required=True)
    ap.add_argument("--max_images", type=int, default=330)
    main(ap.parse_args())
