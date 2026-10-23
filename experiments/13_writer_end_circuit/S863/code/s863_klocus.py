"""S863 (registered in knowledge/DECISIONS.md before its data; the advisor's open gap 10-20 'where does k control the strength of leaving and
predecessor selection') - the k-locus decomposition of the writer's step in pinned runs (design revised after the co-analysis and mechanist critiques,
10-20 23:4x, before any data).  One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated =
shown), as S861 / S862.
  The step is off at k = 2 and on from k = 3 (S859, pinned).  Recipients E(k, p) = END(k) with the early lookup pinned at p in {4, 5} (U part
  at 35-47 <- START(p)); k in {2, 3, 5}.  Arms move one part between ordinals at the same pin, in the same sub-pass: OFF (E5 <- X of E2) and
  ON (E2 <- X of E5) for every X below; K3OFF (E3 <- BAND of E2), K3ON (E2 <- BAND of E3), K3ON_H16 (E2 <- H16 of E3); SELF_FA, SELF_TAIL and
  SELF_IN46 per pin (the two new edit types with an identical-input source: bit-identity).
  X (at the ten read rows unless named):
    BAND_TAIL  the band's eight component outputs at every position from the ordinal token on (exp/c_qpath_k 'cpatch_rows');
    BAND       the band's eight component outputs at the read rows;  L50RES / IN46  the read rows' residual at the input of 50 / 46 (rpatch);
    FA47, H16, SELFADDR {1, 4, 12, 17, 20}, ANTI {0, 2, 5, 13, 15}, REST47 (the other 13 heads); FA43 (all heads), ANTI43 {6, 8, 13, 15, 18}
    (pinned head patches); MLP47, MLP49, MLPS (46-49), DN49, DNS (46, 48, 49); PATH47 (MLP47 reads the donor's FA47 output);
    complements BAND-FA47, BAND-MLP47, BAND-MLP49 (the band without that component).
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of
B1, B4, answer, per-head L47 block scores of B1, B4, answer.  Usage: python s863_klocus.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import HeadPatch, FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s863_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]
REST47 = [h for h in range(24) if h not in ANTI + H16 + SELFADDR]; ANTI43 = [6, 8, 13, 15, 18]
XS = ["BAND_TAIL", "BAND", "L50RES", "IN46", "FA47", "H16", "SELFADDR", "ANTI", "REST47", "FA43", "ANTI43", "MLP47", "MLP49", "MLPS", "DN49", "DNS",
      "PATH47", "BAND-FA47", "BAND-MLP47", "BAND-MLP49"]
N = 8; PINS = (4, 5); KS = (2, 3, 5)


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S863_DEVICE", "cuda:0")
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
    band = a.band
    FA_W = [l for l in band if l in FA_L]
    F47 = FA_W[0]
    DNL = [l for l in band if l not in FA_L]
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    F43 = [l for l in EARLY if l < F47][-1]
    COMPS = {"BAND": ALLC, "FA47": [(F47, "mix")], "MLPS": [(l, "mlp") for l in band], "DNS": [(l, "mix") for l in DNL],
             **{f"MLP{l}": [(l, "mlp")] for l in band}, **{f"DN{l}": [(l, "mix")] for l in DNL},
             "BAND-FA47": [c for c in ALLC if c != (F47, "mix")], "BAND-MLP47": [c for c in ALLC if c != (F47, "mlp")],
             "BAND-MLP49": [c for c in ALLC if c != (band[-1], "mlp")]}
    HEADS = {"H16": (F47, H16), "SELFADDR": (F47, SELFADDR), "ANTI": (F47, ANTI), "REST47": (F47, REST47), "FA43": (F43, list(range(24))),
             "ANTI43": (F43, ANTI43)}
    RES = {"L50RES": band[-1] + 1, "IN46": band[0]}
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    assert NH == 24
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
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=sorted(RES.values()))
    FHP = FAHeadPatchPinned(v, [F43, F47])                                                # after TOOL: corrects the rows its pins wrote
    MIP = MLPInputPath(v, [F47])                                                          # after FHP: MLP47 may read a donor's attention
    T1 = []

    def prompt(form, q, menu, n):
        fr = "last" if form == "END" else "first"
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. What is in the {ORDW[q-1]} image from the {fr}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps):
        TOOL.set(qs); FHP.set(fhs); MIP.set(mps)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); FHP.set(None); MIP.set(None)
        return lg, st

    rng = random.Random(a.seed); nrun = 0; recs = []
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · FA {F43}/{F47} · DN {DNL} · REST47 {REST47} · res {RES}", flush=True)
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={}, qk={}, zl={}, subpass={})
        names = {**{f"END{k}": ("END", k) for k in KS}, **{f"START{p}": ("START", p) for p in PINS}}
        enc = {nm: v.encode(prompt(f, q, menu, N), pics[:N]) for nm, (f, q) in names.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "prompts must align"
        ref = next(iter(ids.values())); S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        for nm in ids:
            assert "?" in v.tok.decode([ids[nm][qm]])
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
        tail0 = min(i for i in range(S_) if ids["END2"][i] != ids["END5"][i])                # the ordinal token
        assert ve[-1] < tail0 < qm
        TAIL = list(range(tail0, S_))
        capq = lambda **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=owner, nb=N, opt=opt), **kw)

        # ---- pin donors: one START pass (natural); the batched QPath store holds per-sample queries
        inps = [enc[f"START{p}"] for p in PINS]
        qs = [capq(grabq=set(EARLY)) for _ in PINS]
        lg0, st0 = run1(inps, qs, [None] * len(PINS), [None] * len(PINS))
        qd = {}
        for b, p in enumerate(PINS):
            qd[p] = {li: st0[b][("q", li)] for li in EARLY}
            w = menu[int(torch.stack([sc(lg0[b], c) for c in menu]).argmax())]
            rec["land"][f"START{p}"] = labs.index(w) + 1
            T1.append(int(int(lg0[b].argmax()) in t1ids(menu)))
        nrun += len(PINS)
        del st0; gc.collect()

        def pin(p):
            return {li: dict(qsub=dict(owner=owner, nb=N, part="U", donors=[(1.0, qd[p][li])])) for li in EARLY}

        for p in PINS:
            pq = pin(p)
            refs = [f"E{k}" for k in KS]
            specs = [("SELF_FA", "E5", None, "self"), ("SELF_TAIL", "E5", "E5", "BAND_TAIL"), ("SELF_IN46", "E5", "E5", "IN46")]
            for X in XS:
                specs += [(f"{X}_OFF", "E5", "E2", X), (f"{X}_ON", "E2", "E5", X)]
            specs += [("BAND_K3OFF", "E3", "E2", "BAND"), ("BAND_K3ON", "E2", "E3", "BAND"), ("H16_K3ON", "E2", "E3", "H16")]
            size = a.size
            chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
            for sp, chunk in enumerate(chunks):
                chunk = chunk + [("PAD", "E5", None, "pad")] * max(0, size - len(chunk))
                inps, qs, fhs, mps, tag = [], [], [], [], []
                idx = {}
                def add(nm, rk, cp=None, fh=None, mp=None, cpr=None, rp=None):
                    inps.append(enc[f"END{rk[1:]}"])
                    q = capq(edit=pq)
                    if cp is not None:
                        q["cpatch"] = cp
                    if cpr is not None:
                        q["cpatch_rows"] = cpr
                    if rp is not None:
                        q["rpatch"] = rp
                    qs.append(q)
                    fhs.append(dict(rows=rowsC, edit=pq, prow=rowsC, hedits=fh or {}))
                    mps.append(mp)
                    tag.append(nm)
                    return len(inps) - 1
                for rk in refs:
                    idx[rk] = add(f"{rk}p{p}@{sp}", rk)
                for arm, rk, don, X in chunk:
                    nm = f"{rk}p{p}_{arm}"
                    if X == "pad":
                        add(f"PAD@{sp}_{len(inps)}", rk)
                    elif X == "self":
                        add(nm, rk, fh={F47: [dict(heads=list(range(NH)), op="swap", src=len(inps))]})
                    elif X == "BAND_TAIL":
                        add(nm, rk, cpr=dict(src=idx[don], comps=ALLC, rows=TAIL))
                    elif X in RES:
                        add(nm, rk, rp={RES[X]: dict(src=idx[don], rows=rowsC)})
                    elif X in COMPS:
                        add(nm, rk, cp=dict(src=idx[don], comps=COMPS[X]))
                    elif X in HEADS:
                        lay, hs_ = HEADS[X]
                        add(nm, rk, fh={lay: [dict(heads=hs_, op="swap", src=idx[don])]})
                    elif X == "PATH47":
                        add(nm, rk, mp=dict(rows=rowsC, src=idx[don]))
                    else:
                        raise ValueError(X)
                lg, st = run1(inps, qs, fhs, mps)
                for b, nm in enumerate(tag):
                    lgb, stb = lg[b], st[b]
                    T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                    w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                    keys = [nm] + ([nm.split("@")[0]] if nm.endswith("@0") else [])
                    for key in keys:
                        rec["land"][key] = labs.index(w) + 1
                        rec["mlog"][key] = [float(sc(lgb, x)) for x in labs[:N]]
                        rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                        rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
                        rec.setdefault("z47", {})[key] = stb[("cap", F47)][[2, 5, 9]].float().tolist() if ("cap", F47) in stb else None
                    rec["subpass"][nm] = sp
                    nrun += 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · pinned head-patch calls {FHP.calls} · path calls {MIP.calls}", flush=True)
    TOOL.remove(); FHP.remove(); MIP.remove()
    json.dump(dict(tag=a.tag, model=a.model, N=N, pins=list(PINS), ks=list(KS), xs=XS, anti=ANTI, h16=H16, selfaddr=SELFADDR, rest47=REST47, anti43=ANTI43, band=band,
                   read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "klocus"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'klocus')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86300); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--size", type=int, default=12)
    main(ap.parse_args())
