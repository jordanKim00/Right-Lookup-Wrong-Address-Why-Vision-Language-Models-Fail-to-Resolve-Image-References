"""S860 (head-level causal test after S859; registered in knowledge/DECISIONS.md before its data) - which components of the writer suppress the early
address (un-anchoring), which favour the predecessor (selection), what sets the gain, and does the computation transfer across anchors?  Eight real
pictures, stated count 8, the early lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p)), as S844-S859.
One forward pass per (k, pins): k 5 with pins (3, 5) and (4, 7), k 3 with pins (3, 5) and (7); every pass holds, for each pin p,
  E   END(k) pinned at p            R   START(p) pinned at its own address (pure pointer; donor of the '_R' arms)
  EK  END(k') pinned at p, k' = the other ordinal (3 for k 5, 5 for k 3; donor of the gate test G16)
and the arms (edits at the ten read rows; donors in the same pass):
  SELF_FA / SELF_M / SELF_P   self donors through the FA head path / the component path / the attention-to-MLP path (bit-identity controls)
  BAND_R      all eight band components (46-49 mixers and MLPs) <- R (positive control)
  ANTI_R      FA47 anti-self heads {0, 2, 5, 13, 15} <- R          ANTI_Z   the same heads zeroed (their un-anchoring removed)
  M49_R       MLP49 <- R      ANTI_M49_R   both (the 2 x 2 with ANTI_R and M49_R)
  H16_R       FA47 h16 <- R   G16  h16 with EK's output gate on E's own read (does k open h16?)
  SELFADDR_R  FA47 self-address heads {1, 4, 12, 17, 20} <- R
  RAND5a/b_R  two random 5-head FA47 sets <- R;  RAND1a/b_R  two random single FA47 heads <- R (pool {3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23})
  M47_R / M48_R / M4748_R   MLP47 / MLP48 / both <- R
  PATH47      MLP47 reads R's FA47 output while the residual keeps E's (exp/c_headpatch.MLPInputPath; the FA47 -> MLP47 path)
  ALL47_R     FA47's whole output <- R      ANTI43_Z   FA43 anti-self heads {6, 8, 13, 15, 18} zeroed
  p-exchanges (two-pin passes; donor E(k, p'), p' = the other pin): M49_PX, M47_PX, M48_PX, H16_PX, BAND_PX (operator transfer: does the donor's
  predecessor follow the component?)
Head sets fixed before this run's data from S859 (FINDINGS §963), the co-investigator agents' analyses and the user's review (10-20).
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of B1, B4,
answer.  Usage: python s860_heads.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI
from c_headpatch import HeadPatch, FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s860_{}_{}.json"
N = 8
PASSES = [(5, (3, 5)), (5, (4, 7)), (3, (3, 5)), (3, (7,))]                                # (k, pins in one pass); the (3, 5) passes carry p-exchanges
CELLS = [(k, p) for k, ps in PASSES for p in ps]
KALT = {5: 3, 3: 5}                                                                         # the other ordinal of each pass (gate donor EK)
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]; ANTI43 = [6, 8, 13, 15, 18]   # FA47 / FA43 head sets (S859 §963, a co-analysis
#   A11, a co-analysis, user review 10-20)
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]
ARMS_R = ["SELF_FA", "SELF_M", "SELF_P", "BAND_R", "ANTI_R", "ANTI_Z", "M49_R", "ANTI_M49_R", "H16_R", "G16", "SELFADDR_R", "RAND5a_R",
          "RAND5b_R", "RAND1a_R", "RAND1b_R", "M47_R", "M48_R", "M4748_R", "PATH47", "ALL47_R", "ANTI43_Z"]
ARMS_PX = ["M49_PX", "M47_PX", "M48_PX", "H16_PX", "BAND_PX"]                               # donor = E(k, p') of the same pass (operator transfer)
ARMS = ["E", "R", "EK"] + ARMS_R + ARMS_PX


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S860_DEVICE", "cuda:0")
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
    FA_W = [l for l in band if l in FA_L]; DN_W = [l for l in band if l not in FA_L]
    F47 = FA_W[0]; D49 = DN_W[-1]; MLPS = [(l, "mlp") for l in band[1:]]                 # 27B: MLP 47, 48, 49
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    NHV = v.layers[D49].linear_attn.num_v_heads
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
    clip = lambda hs, n: [h for h in hs if h < n]                                          # smaller models (CPU smoke) have fewer heads
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
    TOOL = QPathI(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=())
    HP = HeadPatch(v, [D49])
    F43 = [l for l in EARLY if l < F47][-1]                                                # 27B: 43
    FHP = FAHeadPatchPinned(v, [F43, F47])                                                # after TOOL: corrects the rows its pins wrote
    MIP = MLPInputPath(v, [F47])                                                          # after FHP: the MLP of 47 may read a donor's attention
    T1 = []

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, hps, fhs, mps=None):
        TOOL.set(qs); HP.set(hps); FHP.set(fhs); MIP.set(mps)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); HP.set(None); FHP.set(None); MIP.set(None)
        return lg, st

    rng = random.Random(a.seed); nrun = 0
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · FA {F47} · DN {D49} · MLPs {MLPS} · RAND5 {RAND5} · RAND1 {RAND1}", flush=True)
    recs = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        ps_need = sorted({p for _, p in CELLS}); ks_need = sorted({k for k, _ in CELLS} | set(KALT.values()))
        enc = {**{f"END_k{k}": v.encode(prompt(k, "last", menu), pics) for k in ks_need},
               **{f"START_q{q}": v.encode(prompt(q, "first", menu), pics) for q in ps_need}}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = ids[f"END_k{ks_need[0]}"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in menu:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        rowsC = [qm] + list(range(uend, uend + 5)) + A4
        rec = dict(si=si, labs=labs, words=labs + extra, menu=menu, land={}, mlog={}, qk={}, zl={})
        capq = lambda **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=owner, nb=N, opt=opt), **kw)
        # stage 1: pin donors START(q) (natural)
        inps = [enc[f"START_q{q}"] for q in ps_need]
        lg, st = run1(inps, [capq(grabq=set(EARLY)) for _ in ps_need], None, None)
        don = {q: {li: st[b][("q", li)] for li in EARLY} for b, q in enumerate(ps_need)}
        for b, q in enumerate(ps_need):
            rec["land"][f"START_q{q}"] = (labs.index(menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())]) + 1
                                          if menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())] in labs else 0)
        def pin(p_):
            return {li: dict(qsub=dict(owner=owner, nb=N, part="U", donors=[(1.0, don[p_][li])])) for li in EARLY}
        # stage 2: one pass per (k, pins) holding, for every pin p, E(k, p), R(p) and the arms (p-exchanges between the two pins of a pass)
        for k, pins in PASSES:
            # symbolic arm specs (donors resolved inside each sub-pass); a sub-pass = the donors of every pin + SUBA arms, padded to one size
            specs = []
            for p in pins:
                for arm in ARMS_R:
                    specs.append((p, arm))
                if len(pins) == 2:
                    specs += [(p, arm) for arm in ARMS_PX]
            SUBA = 13 if len(pins) == 2 else 11
            chunks = [specs[i:i + SUBA] for i in range(0, len(specs), SUBA)]
            for sp, chunk in enumerate(chunks):
                chunk = chunk + [(pins[0], "PAD")] * (SUBA - len(chunk))
                inps, qs, hps, fhs, mps, tag = [], [], [], [], [], []
                def add(nm, p, which, cp=None, hp=None, fh=None, mp=None):
                    inps.append(enc[f"END_k{k}"] if which == "E" else enc[f"END_k{KALT[k]}"] if which == "EK" else enc[f"START_q{p}"])
                    q = capq(edit=pin(p))
                    if cp is not None:
                        q["cpatch"] = cp
                    qs.append(q); hps.append(hp); mps.append(mp)
                    fhs.append(dict(rows=rowsC, edit=q["edit"], prow=rowsC, hedits=fh or {}))
                    tag.append((p, nm))
                    return len(inps) - 1
                fa = lambda heads, op, src, L_=None: {(L_ or F47): [dict(heads=clip(heads, NH), op=op, src=src)]}
                iE, iR, iK = {}, {}, {}
                for p in pins:
                    iE[p] = add(f"E@{sp}", p, "E"); iR[p] = add(f"R@{sp}", p, "R"); iK[p] = add(f"EK@{sp}", p, "EK")
                for p, arm in chunk:
                    r_ = iR[p]
                    if arm == "PAD":
                        add(f"PAD@{sp}_{len(inps)}", p, "E")
                    elif arm == "SELF_FA":
                        add(arm, p, "E", fh={F47: [dict(heads=list(range(NH)), op="swap", src=len(inps))]})
                    elif arm == "SELF_M":
                        add(arm, p, "E", cp=dict(src=len(inps), comps=[MLPS[-1]]))
                    elif arm == "SELF_P":
                        add(arm, p, "E", mp=dict(rows=rowsC, src=len(inps)))
                    elif arm == "BAND_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=ALLC))
                    elif arm == "ANTI_R":
                        add(arm, p, "E", fh=fa(ANTI, "swap", r_))
                    elif arm == "ANTI_Z":
                        add(arm, p, "E", fh=fa(ANTI, "zero", None))
                    elif arm == "M49_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=[MLPS[-1]]))
                    elif arm == "ANTI_M49_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=[MLPS[-1]]), fh=fa(ANTI, "swap", r_))
                    elif arm == "H16_R":
                        add(arm, p, "E", fh=fa(H16, "swap", r_))
                    elif arm == "G16":
                        add(arm, p, "E", fh=fa(H16, "gate", iK[p]))
                    elif arm == "SELFADDR_R":
                        add(arm, p, "E", fh=fa(SELFADDR, "swap", r_))
                    elif arm.startswith("RAND5"):
                        add(arm, p, "E", fh=fa(RAND5["ab".index(arm[5])], "swap", r_))
                    elif arm.startswith("RAND1"):
                        add(arm, p, "E", fh=fa(RAND1["ab".index(arm[5])], "swap", r_))
                    elif arm == "M47_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=[MLPS[0]]))
                    elif arm == "M48_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=[MLPS[1]]))
                    elif arm == "M4748_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=MLPS[:2]))
                    elif arm == "PATH47":
                        add(arm, p, "E", mp=dict(rows=rowsC, src=r_))
                    elif arm == "ALL47_R":
                        add(arm, p, "E", cp=dict(src=r_, comps=[(F47, "mix")]))
                    elif arm == "ANTI43_Z":
                        add(arm, p, "E", fh=fa(ANTI43, "zero", None, F43))
                    else:                                                                            # p-exchanges: donor E(k, p') of this sub-pass
                        d_ = iE[pins[1] if p == pins[0] else pins[0]]
                        if arm == "M49_PX":
                            add(arm, p, "E", cp=dict(src=d_, comps=[MLPS[-1]]))
                        elif arm == "M47_PX":
                            add(arm, p, "E", cp=dict(src=d_, comps=[MLPS[0]]))
                        elif arm == "M48_PX":
                            add(arm, p, "E", cp=dict(src=d_, comps=[MLPS[1]]))
                        elif arm == "H16_PX":
                            add(arm, p, "E", fh=fa(H16, "swap", d_))
                        elif arm == "BAND_PX":
                            add(arm, p, "E", cp=dict(src=d_, comps=ALLC))
                lg, st = run1(inps, qs, hps, fhs, mps)
                for b, (p, nm) in enumerate(tag):
                    lgb, stb = lg[b], st[b]
                    T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                    w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                    keys = [f"k{k}p{p}_{nm}"] + ([f"k{k}p{p}_{nm.split('@')[0]}"] if nm.endswith("@0") else [])   # sub-pass 0 donors = canonical
                    for key in keys:
                        rec["land"][key] = labs.index(w) + 1 if w in labs else 0
                        rec["mlog"][key] = [float(sc(lgb, x)) for x in rec["words"]]
                        rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                        rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
                    rec.setdefault("subpass", {})[f"k{k}p{p}_{nm}"] = sp
                    nrun += 1
            print(f"[{a.tag}] scene {si + 1} k{k} pins {pins} · {len(chunks)} sub-passes of {len(inps)} runs", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · FA patch calls {FHP.calls} · path calls {MIP.calls}", flush=True)
    TOOL.remove(); HP.remove(); FHP.remove(); MIP.remove()
    json.dump(dict(tag=a.tag, model=a.model, passes=PASSES, cells=CELLS, arms=ARMS, arms_r=ARMS_R, arms_px=ARMS_PX, anti=ANTI, h16=H16, selfaddr=SELFADDR,
                   anti43=ANTI43, kalt=KALT, rand5=RAND5, rand1=RAND1,
                   band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs), open(OUT.format(a.tag, "heads"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'heads')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86000); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
