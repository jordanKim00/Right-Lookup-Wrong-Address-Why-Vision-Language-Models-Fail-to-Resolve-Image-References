"""S866 (registered in knowledge/DECISIONS.md before its data; user 10-21 'the complete algorithmic circuit of END is the top priority') -
sufficiency of the writer's END output: does the band's END computation alone make a pointer (START) run step back from its anchor?
Removal (S860-S862) showed necessity and exchange showed anchor-keying; this transplants END's parts into START at the same pinned anchor.
One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown), as S861-S865.
  Pinned references at p in {3, 4, 5} (U part at 35-47 <- the natural START(p)): E5 = END(5) (step on; j* = 4, p 4 its identity pin), E2 =
  END(2) (step off at k 2, S859), R = START(p) (the pointer at its own address).
  Arms on R (donor in the same sub-pass): BAND5 / BAND3 / BAND2 (the band's eight component outputs at the ten read rows <- E5 / E3 / E2: the k
  ladder), BAND5_L05 / BAND5_L2 (dose: R + lambda (E5 - R) on the band, lambda .5 / 2), TAIL5 (the band outputs at every position from the
  ordinal token on <- E5), FA47_5, H16_5, ANTI_5, SELFADDR_5, REST47_5 (pinned head patches <- E5), MLPS5 (the four MLPs <- E5), L50RES5 /
  L50RES2 (the read rows' residual at the input of 50 <- E5 / E2: everything before the late lookup), RAND1a_5 / RAND1b_5 / RAND5a_5 (the S860
  random heads <- E5; RAND5a count-matched for the five-head sets), SELF_FA, SELF_TAIL (identical-input source).
  Cross-anchor group (pins 3 and 5 in one pass; co-analysis critique 10-21): R(p') <- E5(p)'s band (BAND5X<p>) beside R(p) <- E5(p) - anchor-keyed
  (answer at the donor's p - 1) or a relative operation (p' - 1)?
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of
B1, B4, answer, per-head L47 block scores of B1, B4, answer.  Usage: python s866_suff.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import HeadPatch, FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s866_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]   # the S860 draws
REST47 = [h for h in range(24) if h not in ANTI + H16 + SELFADDR]
N = 8; PINS = (3, 4, 5)
ENC = {"E5": "END5", "E3": "END3", "E2": "END2"}
SPECS = [("SELF_FA", "R", None, "self"), ("SELF_TAIL", "R", "R", "TAIL"),
         ("BAND5", "R", "E5", "BAND"), ("BAND3", "R", "E3", "BAND"), ("BAND2", "R", "E2", "BAND"), ("TAIL5", "R", "E5", "TAIL"),
         ("FA47_5", "R", "E5", "FA47"), ("H16_5", "R", "E5", "H16"), ("ANTI_5", "R", "E5", "ANTI"), ("SELFADDR_5", "R", "E5", "SELFADDR"),
         ("REST47_5", "R", "E5", "REST47"), ("MLPS5", "R", "E5", "MLPS"), ("L50RES5", "R", "E5", "L50RES"), ("L50RES2", "R", "E2", "L50RES"),
         ("RAND1a_5", "R", "E5", "RAND1a"), ("RAND1b_5", "R", "E5", "RAND1b"), ("RAND5a_5", "R", "E5", "RAND5a"),
         ("BAND5_L05", "R", "E5", "LIN0.5"), ("BAND5_L2", "R", "E5", "LIN2")]
XPINS = (3, 5)                                                                     # cross-anchor transplant: R(p') <- the band of E5(p)


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
    COMPS = {"BAND": ALLC, "FA47": [(F47, "mix")], "MLPS": [(l, "mlp") for l in band]}
    HEADS = {"H16": (F47, H16), "SELFADDR": (F47, SELFADDR), "ANTI": (F47, ANTI), "REST47": (F47, REST47), "RAND1a": (F47, RAND1[0]),
             "RAND1b": (F47, RAND1[1]), "RAND5a": (F47, RAND5[0])}
    RES = {"L50RES": band[-1] + 1}
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
    FHP = FAHeadPatchPinned(v, [F47])                                                     # after TOOL: corrects the rows its pins wrote
    MIP = MLPInputPath(v, [F47])                                                          # after FHP: MLP47 may read a donor's attention
    T1 = []

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?",
             "BEF": lambda q: f"What is in the image just before the {ORDW[q-1]}?", "AFT": lambda q: f"What is in the image just after the {ORDW[q-1]}?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

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
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · FA {F47} · RAND1 {RAND1}", flush=True)
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={}, qk={}, zl={}, subpass={})
        names = {"END5": ("END", 5), "END3": ("END", 3), "END2": ("END", 2), **{f"START{p}": ("START", p) for p in PINS}}
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
        tail0 = min(i for i in range(S_) if len({ids[nm][i] for nm in ids}) > 1)            # the first question token that differs
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

        def group(refs, specs, size, gname):
            """refs = [(name, form key, pin)], specs = [(arm, recipient ref, donor ref, X)]; equal-size sub-passes, each holding every ref."""
            nonlocal nrun
            pinof = {rn: pp for rn, _, pp in refs}; encof = {rn: ek for rn, ek, _ in refs}
            chunks = [specs[i:i + size] for i in range(0, len(specs), size)]
            for sp, chunk in enumerate(chunks):
                chunk = chunk + [("PAD", refs[0][0], None, "pad")] * max(0, size - len(chunk))
                inps, qs, fhs, mps, tag = [], [], [], [], []
                idx = {}
                def add(nm, rk, cp=None, fh=None, cpr=None, rp=None, cpl=None):
                    inps.append(enc[encof[rk]])
                    pq = pin(pinof[rk])
                    q = capq(edit=pq)
                    if cp is not None:
                        q["cpatch"] = cp
                    if cpl is not None:
                        q["cpatch_lin"] = cpl
                    if cpr is not None:
                        q["cpatch_rows"] = cpr
                    if rp is not None:
                        q["rpatch"] = rp
                    qs.append(q)
                    fhs.append(dict(rows=rowsC, edit=pq, prow=rowsC, hedits=fh or {}))
                    mps.append(None)
                    tag.append(nm)
                    return len(inps) - 1
                for rn, _, _ in refs:
                    idx[rn] = add(f"{rn}@{sp}", rn)
                for arm, rk, don, X in chunk:
                    nm = f"{rk}_{arm}"
                    if X == "pad":
                        add(f"PAD{gname}@{sp}_{len(inps)}", rk)
                    elif X == "self":
                        add(nm, rk, fh={F47: [dict(heads=list(range(NH)), op="swap", src=len(inps))]})
                    elif X.startswith("LIN"):
                        lam = float(X[3:])
                        add(nm, rk, cpl=dict(terms=[(1.0, idx[rk]), (lam, idx[don]), (-lam, idx[rk])], comps=ALLC))
                    elif X == "TAIL":
                        add(nm, rk, cpr=dict(src=idx[don], comps=ALLC, rows=TAIL))
                    elif X in RES:
                        add(nm, rk, rp={RES[X]: dict(src=idx[don], rows=rowsC)})
                    elif X in COMPS:
                        add(nm, rk, cp=dict(src=idx[don], comps=COMPS[X]))
                    elif X in HEADS:
                        lay, hs_ = HEADS[X]
                        add(nm, rk, fh={lay: [dict(heads=hs_, op="swap", src=idx[don])]})
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

        for p in PINS:
            refs = [(f"E5p{p}", "END5", p), (f"E3p{p}", "END3", p), (f"E2p{p}", "END2", p), (f"Rp{p}", f"START{p}", p)]
            specs = [(arm, f"{rk}p{p}", (None if don is None else f"{don}p{p}"), X) for arm, rk, don, X in SPECS]
            group(refs, specs, a.size, f"p{p}")
        p_, q_ = XPINS                                                             # cross-anchor: R at one pin <- E5's band at the other
        refs = [(f"XE5p{p_}", "END5", p_), (f"XE5p{q_}", "END5", q_), (f"XRp{p_}", f"START{p_}", p_), (f"XRp{q_}", f"START{q_}", q_)]
        specs = [(f"BAND5X{q_}", f"XRp{p_}", f"XE5p{q_}", "BAND"), (f"BAND5X{p_}", f"XRp{q_}", f"XE5p{p_}", "BAND"),
                 (f"BAND5", f"XRp{p_}", f"XE5p{p_}", "BAND"), (f"BAND5", f"XRp{q_}", f"XE5p{q_}", "BAND")]
        group(refs, specs, len(specs), "X")
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · pinned head-patch calls {FHP.calls} · path calls {MIP.calls}", flush=True)
    TOOL.remove(); FHP.remove(); MIP.remove()
    json.dump(dict(tag=a.tag, model=a.model, N=N, pins=list(PINS), xpins=list(XPINS), rand5=RAND5, specs=[list(x) for x in SPECS], anti=ANTI, h16=H16, selfaddr=SELFADDR, rest47=REST47, rand1=RAND1, band=band,
                   read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "suff"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'suff')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86600); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--size", type=int, default=10)
    main(ap.parse_args())
