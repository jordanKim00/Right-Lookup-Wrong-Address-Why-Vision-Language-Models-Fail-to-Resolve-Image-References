"""S858 (P3, user 10-18: scope frozen after S857; pre-registered in knowledge/DECISIONS.md before any S858 data).  Prior predictions of what
unperformed interventions do in new conditions, at three levels: the late queries (51-63), the late picture scores, the answer logits / answers.
New conditions (none used in S852-S857): N_a in {9, 7} pictures with the last two as the source, k {4, 6} at N 9 and {4, 5} at N 7, early lookup
pinned at p {2, 5} (address part U over the shared pictures 1..N-2 <- START(p) of the same configuration, layers 35-47), new scenes (seed 85980),
all nine single-token labels per scene (N 9 has no distractor; at N 7 the labels of pictures 8-9 are distractors).  Prompt form A with the
cell's count: 'There are N images in a row.'  Cut instrument exp/c_cut.py (full-attention masks, exact DeltaNet segmenting, convolution splice).
Per cell (N, k, p), one batch of 9 (instrument on):
  Nn     natural END(k)
  CA     the last two pictures cut from every later text row at every layer and route
  CAsw   CA with the last two pictures swapped (leak check: must equal CA bit for bit)
  CF     full-attention cut of the last two pictures in layers 35-50 only (DeltaNet and every other layer open) - never run before
  RQ     CA + the late queries (51-63, ten read rows, pre-RoPE) <- Nn's (same pass)
  KQ     Nn + the late queries <- CA's
  QSELF  Nn + its own late queries (bit-identity)
  S      START(k) pinned at p (forward reference)
  BANDF  Nn with the band 46-49 outputs (mixer and MLP, read rows) <- S's (same pass): no END-specific write
Other batches per scene and N: START(2), START(5) for Nn, CA, CAsw, CF (pin donors, 8, instrument on); the N-2 picture run (pictures 1..N-2,
same text): START(2), START(5) (2) and END(k) pinned p for the four cells (4); Nn without the instrument (Nstd, 4).
Readouts: answer, menu-word logits, block scores of every read row at 35-63, picture mass; pre-RoPE late queries of rows B1, B4, answer for Nn, CA,
CF and the N-2 run.
Usage: python s858_pred.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_j import QPathJ
from c_cut import SourceCut, QFrom

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s858_{}_{}.json"
NS = (9, 7); KN = {9: (4, 6), 7: (4, 5)}; PS = (2, 5)
ARMS = ("Nn", "CA", "CAsw", "CF", "RQ", "KQ", "QSELF", "S", "BANDF")
CONFIGS = ("Nn", "CA", "CAsw", "CF")
BASE = {"RQ": "CA", "KQ": "Nn", "QSELF": "Nn", "S": "Nn", "BANDF": "Nn"}
QARMS = ("Nn", "CA", "CF")


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S858_DEVICE", "cuda:0")
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
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE))      # smoke models: by relative depth
    assert band and all(b < min(LATE) for b in band)
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    W_M = set(range(EARLY[0], LATE[0]))
    FAs, DNs = set(FA_L), set(DN_L)
    CUT = {"CA": dict(fa=FAs, dn=DNs), "CAsw": dict(fa=FAs, dn=DNs), "CF": dict(fa=FAs & W_M, dn=set())}
    CUT["RQ"] = CUT["CA"]
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
    assert len(labels) >= 9, labels
    TOOL = QPathJ(v, READ, res_layers=(), comp_layers=(), patch_comps=band)
    CUTTER = SourceCut(v)
    QF = QFrom(v, LATE, TOOL)
    T1 = []

    def prompt(n_stated, na, q, frame, menu):
        return v.chat([("user", f"{v.IMG*na}\nThere are {n_stated} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
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

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · cut window M {EARLY[0]}-{LATE[0]-1} · FA {len(FA_L)} DN {len(DN_L)}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=list(labs), menu=menu, pics=pics))
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}) for s in scenes}

    def measure(rec, lgb, stb):
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))
        return dict(land=rec["labs"].index(w) + 1, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m, stb=None, keepq=False):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]
        if keepq and stb is not None:
            npz[f"s{rec['si']}_{nm}_q"] = np.stack([stb[("q", li)][KR].float().numpy() for li in LATE]).astype(np.float16)    # [L, 3, H, hd]

    def pics_for(s, cfg, na):
        p = list(s["pics"][:na])
        if cfg == "CAsw":
            p[na - 2], p[na - 1] = p[na - 1], p[na - 2]
        return p

    def build(s, N, na, cfg="Nn"):
        """N = the cell's full count (stated), na = pictures shown (N or N - 2)."""
        pics = pics_for(s, cfg, na)
        qs_ = sorted(set(PS) | set(KN[N]))
        nms = {**{f"END_k{k}": (k, "last") for k in KN[N]}, **{f"START_q{q}": (q, "first") for q in qs_}}
        enc = {nm: v.encode(prompt(N, na, q, fr, s["menu"]), pics) for nm, (q, fr) in nms.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = ids[f"END_k{KN[N][0]}"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        assert len(ve) == na and len(vs) == na
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        owner = [-1] * S_
        for b in range(na):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        owner_sh = [b if 0 <= b < N - 2 else -1 for b in owner]
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        src = (vs[N - 2], ve[N - 1]) if na == N else None
        return dict(enc=enc, S=S_, owner=owner, owner_sh=owner_sh, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, na=na, src=src)

    BA = list(range(10))
    try:
        for gi, s in enumerate(scenes):
            rec = recs[s["si"]]
            for N in NS:
                P = {cfg: build(s, N, N, cfg) for cfg in CONFIGS}
                Pm = build(s, N, N - 2)
                for cfg in CONFIGS:
                    assert P[cfg]["S"] == P["Nn"]["S"] and P[cfg]["src"] == P["Nn"]["src"] and P[cfg]["rowsC"] == P["Nn"]["rowsC"]
                sS, sE = P["Nn"]["src"]
                def capq(p, **kw):
                    return dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=p["na"], opt=p["opt"]), **kw)
                def cutentry(cfg):
                    cu = CUT.get(cfg)
                    return None if cu is None else dict(fa=set(cu["fa"]), dn=set(cu["dn"]))
                def withko(d, cfg):
                    cu = CUT.get(cfg)
                    k_ = None if cu is None else SourceCut.ko_for(sS, sE, cu["fa"], READ, 10)
                    if k_ is not None:
                        d["ko"] = k_
                    return d
                def pins(p, dq):
                    return {li: [dict(qsub=dict(owner=p["owner_sh"], nb=N - 2, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}
                # pin donors, instrument on (8)
                inps, qs, tag, cuts = [], [], [], []
                for cfg in CONFIGS:
                    for q in PS:
                        inps.append(P[cfg]["enc"][f"START_q{q}"]); qs.append(withko(capq(P[cfg], grabq=set(EARLY), grabkb=set(LOOK), noop=set(READ)), cfg))
                        tag.append((cfg, q)); cuts.append(cutentry(cfg))
                lg, st = run1(inps, qs, dict(s=sS, e=sE, cuts=cuts))
                don = {}
                for b, (cfg, q) in enumerate(tag):
                    put(rec, f"N{N}_{cfg}_START_q{q}", measure(rec, lg[b], st[b])); nrun += 1
                    don[(cfg, q)] = {li: st[b][("q", li)] for li in EARLY}
                    if b > 0:
                        keyrel.append(max(float((st[b][("kb", li)][:N - 2] - st[0][("kb", li)][:N - 2]).abs().max() / st[0][("kb", li)][:N - 2].abs().max()) for li in LOOK))
                # the N-2 picture run (natural kernels): donors (2) and END cells (4)
                lg, st = run1([Pm["enc"][f"START_q{q}"] for q in PS], [capq(Pm, grabq=set(EARLY), noop=set(READ)) for q in PS])
                donm = {}
                for b, q in enumerate(PS):
                    put(rec, f"N{N}_Nm2_START_q{q}", measure(rec, lg[b], st[b])); nrun += 1
                    donm[q] = {li: st[b][("q", li)] for li in EARLY}
                inps, qs, tag = [], [], []
                for k in KN[N]:
                    for p_ in PS:
                        inps.append(Pm["enc"][f"END_k{k}"]); qs.append(capq(Pm, edit=pins(Pm, donm[p_]), noop=set(LATE), grabq=set(LATE))); tag.append(f"N{N}_Nm2_k{k}p{p_}")
                lg, st = run1(inps, qs)
                for b, nm in enumerate(tag):
                    put(rec, nm, measure(rec, lg[b], st[b]), st[b], keepq=True); nrun += 1
                # Nn without the instrument (4)
                inps, qs, tag = [], [], []
                for k in KN[N]:
                    for p_ in PS:
                        inps.append(P["Nn"]["enc"][f"END_k{k}"]); qs.append(capq(P["Nn"], edit=pins(P["Nn"], don[("Nn", p_)]), noop=set(LATE))); tag.append(f"N{N}_Nstd_k{k}p{p_}")
                lg, st = run1(inps, qs)
                for b, nm in enumerate(tag):
                    put(rec, nm, measure(rec, lg[b], st[b])); nrun += 1
                # the cells, one batch of 9 each (instrument on)
                for k in KN[N]:
                    for p_ in PS:
                        idx = {arm: i for i, arm in enumerate(ARMS)}
                        inps, qs, cuts = [], [], []
                        for arm in ARMS:
                            cfg = BASE.get(arm, arm); p = P[cfg]
                            d = capq(p, edit=pins(p, don[(cfg, p_)]), noop=set(LATE), **({"grabq": set(LATE)} if arm in QARMS else {}))
                            d = withko(d, arm if arm in CUT else cfg)
                            if arm == "RQ":
                                d["qfrom"] = {li: idx["Nn"] for li in LATE}
                            elif arm == "KQ":
                                d["qfrom"] = {li: idx["CA"] for li in LATE}
                            elif arm == "QSELF":
                                d["qfrom"] = {li: idx["QSELF"] for li in LATE}
                            elif arm == "BANDF":
                                d["cpatch"] = dict(src=idx["S"], comps=COMPS)
                            e = p["enc"][f"START_q{k}"] if arm == "S" else p["enc"][f"END_k{k}"]
                            inps.append(e); qs.append(d); cuts.append(cutentry(arm if arm in CUT else cfg))
                        lg, st = run1(inps, qs, dict(s=sS, e=sE, cuts=cuts))
                        for b, arm in enumerate(ARMS):
                            put(rec, f"N{N}_{arm}_k{k}p{p_}", measure(rec, lg[b], st[b]), st[b], keepq=arm in QARMS); nrun += 1
                del don, donm
            if gi % 5 == 4 or gi == len(scenes) - 1:
                print(f"[{a.tag}] {gi + 1}/{len(scenes)} scenes · {nrun} runs · key diff max {max(keyrel):.2e} · calls {CUTTER.calls}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        TOOL.remove(); CUTTER.remove(); QF.remove()
    nm_ = "pred"
    json.dump(dict(tag=a.tag, model=a.model, ns=NS, kn={str(k_): v_ for k_, v_ in KN.items()}, ps=PS, arms=ARMS, configs=CONFIGS, base=BASE, band=band,
                   comps=COMPS, read=READ, look=LOOK, early=EARLY, late=LATE, kr=KR, window_M=[EARLY[0], LATE[0] - 1], fa_layers=FA_L, dn_layers=DN_L,
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
    ap.add_argument("--seed", type=int, default=85980); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
