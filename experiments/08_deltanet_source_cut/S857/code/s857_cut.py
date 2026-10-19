"""S857 (P2, user 10-18 plan; pre-registered in knowledge/DECISIONS.md before any S857 data).  How do pictures 7-8 reach the late query?
Same scenes and prompt as S856 form A ('There are 8 images in a row.'), 8 pictures (X8) or the first 6 (X6), k {3, 5}, early lookup pinned to
p {3, 4} (address part U over the shared pictures 1-6 <- START(p) of the same configuration, layers 35-47).  The source = pictures 7-8 (picture
7's vision-start .. picture 8's vision-end); cuts remove it from every text row after it (exp/c_cut.py: full-attention mask, DeltaNet state
segmenting, convolution splice), per layer window:  E = layers 0..EARLY[0]-1 (0-34), M = EARLY[0]..LATE[0]-1 (35-50), L = LATE[0].. (51-63).
X8 arms (one batch of 17 per k, p; the cut instrument on for all of them):
  N8     natural (no cut)
  CA     cut at every layer, every route                       (the 6-picture state expected: positive control)
  CAsw   CA with pictures 7 and 8 swapped                       (must equal CA bit for bit: leak check)
  CQ     cut in E+M (all routes); late layers L open             (only the direct late readout of pictures 7-8 left)
  CR     cut in L only                                          (the query path open, the direct late readout removed)
  CQF    full-attention cut in E+M only (DeltaNet open); L open
  CQD    DeltaNet cut (state + convolution) in E+M only (full attention open); L open
  O1     open only E (cut in M+L)        O2  open only M (cut in E+L)
  RW     CA + the residual of rows B1, B4 at the input of layer WL (46) <- N8's (same pass)
  RT     CA + the residual of the question tail (ordinal word .. '?') at the input of WL <- N8's
  RQ     CA + the late queries (layers 51-63, all ten read rows, pre-RoPE) <- N8's (same pass)
  KQ     N8 + the late queries <- CA's (same pass)
  QSELF  N8 + the late queries <- its own (must equal N8 bit for bit)
  SW     pictures 7 and 8 swapped (no cut)    G78  pictures 7-8 uniform grey    D78  pictures 7-8 = copies of pictures 1 and 5
Other batches: START(3), START(4) for every source configuration (pin donors; one batch of 24), X6 START (2) and X6 END N6 (4; natural kernels),
N8std (N8 without the instrument, 4; INSTR check).  Readouts as S856 (answer, menu-word logits, block scores of every read row at 35-63, picture
mass; per head at 51-63 for rows B1, B4, answer for the main arms).  30 new scenes (seed 85700 + 200 = 85900).
Usage: python s857_cut.py MODEL TAG [--n 30] [--wl 46] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_j import QPathJ
from c_cut import SourceCut, QFrom

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s857_{}_{}.json"
KS = (3, 5); PS = (3, 4)
X8ARMS = ("N8", "CA", "CAsw", "CQ", "CR", "CQF", "CQD", "O1", "O2", "RW", "RT", "RQ", "KQ", "QSELF", "SW", "G78", "D78")
CONFIGS = ("N8", "CA", "CAsw", "CQ", "CR", "CQF", "CQD", "O1", "O2", "SW", "G78", "D78")          # source configurations (own START donors)
BASE = {"RW": "CA", "RT": "CA", "RQ": "CA", "KQ": "N8", "QSELF": "N8"}
HEADS_ARMS = ("N8", "CA", "CQ", "CR", "CQF", "CQD", "RQ", "KQ")


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S857_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    DN_L = [i for i, L in enumerate(v.layers) if getattr(L, "linear_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    KR = [2, 5, 9]
    WL = a.wl
    assert EARLY[0] <= WL <= EARLY[-1], "restore layer inside the early band"
    W_E = set(range(0, EARLY[0])); W_M = set(range(EARLY[0], LATE[0])); W_L = set(range(LATE[0], nl))
    FAs, DNs = set(FA_L), set(DN_L)
    CUT = {"CA": dict(fa=FAs, dn=DNs), "CAsw": dict(fa=FAs, dn=DNs),
           "CQ": dict(fa=FAs & (W_E | W_M), dn=DNs & (W_E | W_M)), "CR": dict(fa=FAs & W_L, dn=DNs & W_L),
           "CQF": dict(fa=FAs & (W_E | W_M), dn=set()), "CQD": dict(fa=set(), dn=DNs & (W_E | W_M)),
           "O1": dict(fa=FAs & (W_M | W_L), dn=DNs & (W_M | W_L)), "O2": dict(fa=FAs & (W_E | W_L), dn=DNs & (W_E | W_L))}
    for x in ("RW", "RT", "RQ"):
        CUT[x] = CUT["CA"]
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
    TOOL = QPathJ(v, READ, res_layers=(), comp_layers=(), patch_comps=(), rpatch_layers=(WL,))
    CUTTER = SourceCut(v)
    QF = QFrom(v, LATE, TOOL)
    T1 = []

    def prompt(na, q, frame, menu):
        return v.chat([("user", f"{v.IMG*na}\nThere are 8 images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, cutcfg=None):
        TOOL.set(qs); CUTTER.set(cutcfg)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); CUTTER.set(None)
        return lg, st

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · restore layer {WL} · windows E 0-{EARLY[0]-1} M {EARLY[0]}-{LATE[0]-1} "
          f"L {LATE[0]}-{nl-1} · FA {len(FA_L)} DN {len(DN_L)}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, 8); extra_w = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra_w; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra_w, menu=menu, pics=pics))
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}) for s in scenes}
    GREY = Image.new("RGB", (a.pixels, a.pixels), (128, 128, 128))

    def measure(rec, lgb, stb):
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m, stb=None, keep_heads=False):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]
        if keep_heads and stb is not None:
            npz[f"s{rec['si']}_{nm}_capH"] = np.stack([stb[("cap", li)][KR].float().numpy() for li in LATE]).astype(np.float16)
            npz[f"s{rec['si']}_{nm}_imgH"] = np.stack([stb[("img", li)][KR].float().numpy() for li in LATE]).astype(np.float16)
            if ("hout", LATE[0]) in stb:
                npz[f"s{rec['si']}_{nm}_hout"] = np.stack([stb[("hout", li)].float().numpy() for li in LATE]).astype(np.float16)

    def pics_for(s, cfg):
        p = list(s["pics"])
        if cfg in ("CAsw", "SW"):
            p[6], p[7] = p[7], p[6]
        elif cfg == "G78":
            p[6], p[7] = GREY, GREY
        elif cfg == "D78":
            p[6], p[7] = p[0], p[4]
        return p

    def build(s, na, cfg="N8"):
        pics = pics_for(s, cfg)[:na]
        nms = {**{f"END_k{k}": (k, "last") for k in KS}, **{f"START_q{q}": (q, "first") for q in PS}}
        enc = {nm: v.encode(prompt(na, q, fr, s["menu"]), pics) for nm, (q, fr) in nms.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = ids["END_k5"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        assert len(ve) == na and len(vs) == na
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        dk = [i for i, (x, y) in enumerate(zip(ids["END_k3"], ids["END_k5"])) if x != y]
        assert len(dk) == 1 and dk[0] < qm
        owner = [-1] * S_
        for b in range(na):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        owner6 = [b if 0 <= b < 6 else -1 for b in owner]
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        src = (vs[6], ve[7]) if na == 8 else None
        return dict(enc=enc, S=S_, owner=owner, owner6=owner6, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, na=na, src=src,
                    Q=list(range(dk[0], qm + 1)), B14=[uend + 1, uend + 4], ids=ref)

    BA = list(range(10)); LQ = set(LATE)
    try:
        for gi, s in enumerate(scenes):
            rec = recs[s["si"]]
            P = {cfg: build(s, 8, cfg) for cfg in CONFIGS}
            P6 = build(s, 6)
            for cfg in CONFIGS:                                                    # same layout in every configuration
                assert P[cfg]["S"] == P["N8"]["S"] and P[cfg]["src"] == P["N8"]["src"] and P[cfg]["rowsC"] == P["N8"]["rowsC"]
            sS, sE = P["N8"]["src"]
            def capq(p, headout=False, **kw):
                return dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=p["na"], opt=p["opt"], **({"headout": True} if headout else {})), **kw)
            def kof(cfg):
                cu = CUT.get(cfg)
                return None if cu is None else SourceCut.ko_for(sS, sE, cu["fa"], READ, 10)
            def cutentry(cfg):
                cu = CUT.get(cfg)
                return None if cu is None else dict(fa=set(cu["fa"]), dn=set(cu["dn"]))
            def withko(d, cfg):
                k_ = kof(cfg)
                if k_ is not None:
                    d["ko"] = k_
                return d
            # stage 1: pin donors START(3), START(4) for every 8-picture configuration (one batch of 24, instrument on)
            inps, qs, tag, cuts = [], [], [], []
            for cfg in CONFIGS:
                for q in PS:
                    inps.append(P[cfg]["enc"][f"START_q{q}"]); qs.append(withko(capq(P[cfg], grabq=set(EARLY), grabkb=set(LOOK), noop=set(READ)), cfg))
                    tag.append((cfg, q)); cuts.append(cutentry(cfg))
            lg, st = run1(inps, qs, dict(s=sS, e=sE, cuts=cuts))
            don = {}
            for b, (cfg, q) in enumerate(tag):
                put(rec, f"{cfg}_START_q{q}", measure(rec, lg[b], st[b])); nrun += 1
                don[(cfg, q)] = {li: st[b][("q", li)] for li in EARLY}
                if b > 0:                                                         # pictures 1-6 keys identical in every configuration
                    keyrel.append(max(float((st[b][("kb", li)][:6] - st[0][("kb", li)][:6]).abs().max() / st[0][("kb", li)][:6].abs().max()) for li in LOOK))
            # X6 START (2) and END N6 (4): natural kernels, no source
            lg, st = run1([P6["enc"][f"START_q{q}"] for q in PS], [capq(P6, grabq=set(EARLY), noop=set(READ)) for q in PS])
            don6 = {}
            for b, q in enumerate(PS):
                put(rec, f"N6_START_q{q}", measure(rec, lg[b], st[b])); nrun += 1
                don6[q] = {li: st[b][("q", li)] for li in EARLY}
            def pins(p, dq):
                return {li: [dict(qsub=dict(owner=p["owner6"], nb=6, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}
            inps, qs, tag = [], [], []
            for k in KS:
                for p_ in PS:
                    inps.append(P6["enc"][f"END_k{k}"]); qs.append(capq(P6, headout=True, edit=pins(P6, don6[p_]), noop=set(LATE))); tag.append(f"N6_k{k}p{p_}")
            lg, st = run1(inps, qs)
            for b, nm in enumerate(tag):
                put(rec, nm, measure(rec, lg[b], st[b]), st[b], keep_heads=True); nrun += 1
            # N8std: N8 without the cut instrument (natural kernels)
            inps, qs, tag = [], [], []
            for k in KS:
                for p_ in PS:
                    inps.append(P["N8"]["enc"][f"END_k{k}"]); qs.append(capq(P["N8"], edit=pins(P["N8"], don[("N8", p_)]), noop=set(LATE))); tag.append(f"N8std_k{k}p{p_}")
            lg, st = run1(inps, qs)
            for b, nm in enumerate(tag):
                put(rec, nm, measure(rec, lg[b], st[b])); nrun += 1
            # stage 2: X8 arms, one batch of 17 per (k, p), instrument on
            for k in KS:
                for p_ in PS:
                    idx = {arm: i for i, arm in enumerate(X8ARMS)}
                    inps, qs, cuts = [], [], []
                    for arm in X8ARMS:
                        cfg = BASE.get(arm, arm); p = P[cfg]
                        d = capq(p, headout=arm in HEADS_ARMS, edit=pins(p, don[(cfg, p_)]), noop=set(LATE))
                        d = withko(d, arm if arm in CUT else cfg)
                        if arm == "RW":
                            d["rpatch"] = {WL: dict(src=idx["N8"], rows=p["B14"])}
                        elif arm == "RT":
                            d["rpatch"] = {WL: dict(src=idx["N8"], rows=p["Q"])}
                        elif arm == "RQ":
                            d["qfrom"] = {li: idx["N8"] for li in LATE}
                        elif arm == "KQ":
                            d["qfrom"] = {li: idx["CA"] for li in LATE}
                        elif arm == "QSELF":
                            d["qfrom"] = {li: idx["QSELF"] for li in LATE}
                        inps.append(p["enc"][f"END_k{k}"]); qs.append(d); cuts.append(cutentry(arm if arm in CUT else cfg))
                    lg, st = run1(inps, qs, dict(s=sS, e=sE, cuts=cuts))
                    for b, arm in enumerate(X8ARMS):
                        put(rec, f"{arm}_k{k}p{p_}", measure(rec, lg[b], st[b]), st[b], keep_heads=arm in HEADS_ARMS); nrun += 1
            del don, don6
            if gi % 5 == 4 or gi == len(scenes) - 1:
                print(f"[{a.tag}] {gi + 1}/{len(scenes)} scenes · {nrun} runs · key diff max {max(keyrel):.2e} · calls {CUTTER.calls}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        TOOL.remove(); CUTTER.remove(); QF.remove()
    nm_ = "cut"
    json.dump(dict(tag=a.tag, model=a.model, ks=KS, ps=PS, arms=X8ARMS, configs=CONFIGS, base=BASE, read=READ, look=LOOK, early=EARLY, late=LATE,
                   kr=KR, wl=WL, windows=dict(E=[0, EARLY[0] - 1], M=[EARLY[0], LATE[0] - 1], L=[LATE[0], nl - 1]), fa_layers=FA_L, dn_layers=DN_L,
                   matcher_layers=sorted(MH), n_layer=nl, readout=os.environ.get("R3_READ"),
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun, calls=CUTTER.calls,
                   rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85900); ap.add_argument("--wl", type=int, default=46)
    main(ap.parse_args())
