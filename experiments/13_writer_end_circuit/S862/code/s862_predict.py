"""S862 (registered in knowledge/DECISIONS.md before its data; user 10-20 'close the last chain and show that the identified algorithm predicts') -
the predictive test of the identified circuit on unseen conditions and interventions.  One scene = nine pictures with all nine names as the
menu; layouts show its first N pictures (N 6-9, stated = shown).
  G (natural grid): END(k) for every k = 1..N and START(q) for every q = 1..N, N in {6, 7, 8, 9}; the early anchor (L47 head-mean argmax) of
     every run is recorded - the frozen algorithm predicts each run's answer from it.
  I (natural interventions in cells never intervened on): N9 END(4), END(6), END(7); N7 END(5); N6 END(4), END(3); arms SELF, BAND_S (band 46-49
     <- START), ANTI_Z (FA47 anti-self {0, 2, 5, 13, 15} zeroed), SELFADDR_S (FA47 {1, 4, 12, 17, 20} <- START), H16_S (FA47 h16 <- START),
     RAND1a_S / RAND1b_S (the S860 random single heads <- START), RAND5a_Z (the S860 random five-head set zeroed: ANTI_Z's control);
     donors = the same scene's START of the same ordinal and layout.
  P (pinned at new anchors 2, 3 and 6; agents' critique 10-20): N9 END(5) and N8 END(5) - the same ordinal and pins in two layouts (the
     N-invariance test of the band's step) - with the early lookup pinned at p in {2, 3, 6} (U part at 35-47 <- START(p)), R(p) = START(p) at
     its own address; arms SELF_FA, H16_PX / BAND_PX (h16 / the whole band <- E(k, p') of another pin, every ordered pair of {2, 3, 6}),
     H16_R / BAND_R (<- R(p)).
Passes: every group runs in equal-size sub-passes, each holding every reference run of the group.
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of B1,
B4, answer.  Usage: python s862_predict.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI
from c_headpatch import HeadPatch, FAHeadPatchPinned

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s862_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]; ANTI43 = [6, 8, 13, 15, 18]
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]   # the S860 draws
END_ARMS = ["SELF", "BAND_S", "ALL47_S", "ANTI_Z", "ANTI43_Z", "SELFADDR_S", "H16_S", "RAND5a_S", "RAND5b_S", "RAND1a_S", "RAND1b_S"]
START_ARMS = ["ANTI_Z", "H16_Z", "SELFADDR_Z"]
C_ARMS = ["SELF_FA", "SELFADDR_G", "SELFADDR_RD", "H16_G", "H16_RD", "ANTI_G", "ANTI_RD"]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S862_DEVICE", "cuda:0")
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
    FA_W = [l for l in band if l in FA_L]
    F47 = FA_W[0]; F43 = [l for l in EARLY if l < F47][-1]
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
    clip = lambda hs: [h for h in hs if h < NH]
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
    TOOL = QPathI(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=())
    HP = HeadPatch(v, [F43, F47])                                                         # natural runs (o_proj input; exact)
    FHP = FAHeadPatchPinned(v, [F47])                                                     # pinned runs (part C)
    T1 = []

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?",
             "BEF": lambda q: f"What is in the image just before the {ORDW[q-1]}?", "AFT": lambda q: f"What is in the image just after the {ORDW[q-1]}?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, hps, fhs):
        TOOL.set(qs); HP.set(hps); FHP.set(fhs)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); HP.set(None); FHP.set(None)
        return lg, st

    rng = random.Random(a.seed); nrun = 0; recs = []
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · FA {F43}/{F47} · RAND5 {RAND5} · RAND1 {RAND1}", flush=True)

    def layout(n, names, menu, pics):
        enc = {nm: v.encode(prompt(f, q, menu, n), pics[:n]) for nm, (f, q) in names.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1, "prompts of a layout must align"
        ref = next(iter(ids.values())); S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        for nm in ids:
            assert len(ids[nm]) == S_ and "?" in v.tok.decode([ids[nm][qm]])
        owner = [-1] * S_
        for b in range(n):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in menu:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        rowsC = [qm] + list(range(uend, uend + 5)) + A4
        return dict(n=n, enc=enc, owner=owner, opt=opt, rowsC=rowsC)

    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        rec = dict(si=si, labs=labs, words=labs, menu=menu, land={}, mlog={}, qk={}, zl={})
        LAY = {}
        for n in (6, 7, 8, 9):
            names = {**{f"END{k}": ("END", k) for k in range(1, n + 1)}, **{f"START{q}": ("START", q) for q in range(1, n + 1)}}
            LAY[n] = layout(n, names, menu, pics)
        def group(n, refs, specs, tagp, pinp=None, size=None):
            """run reference runs + arms in equal-size sub-passes; specs = [(arm name, recipient ref, kind, payload)]."""
            nonlocal nrun
            L = LAY[n]
            capq = lambda **kw: dict(rows=L["rowsC"], cap=dict(rows=list(range(10)), owner=L["owner"], nb=n, opt=L["opt"]), **kw)
            size = 11 if size is None else size
            chunks = [specs[i:i + size] for i in range(0, len(specs), size)] if (specs and size) else [[]]
            got = {}
            for sp, chunk in enumerate(chunks):
                chunk = chunk + [("PAD", refs[0][0], "pad", None)] * max(0, size - len(chunk))
                inps, qs, hps, fhs, tag = [], [], [], [], []
                idx = {}
                def add(nm, encname, pin=None, cp=None, hp=None, fh=None):
                    inps.append(L["enc"][encname])
                    q = capq(edit=pin) if pin is not None else capq(grabq=set(EARLY))
                    if cp is not None:
                        q["cpatch"] = cp
                    qs.append(q); hps.append(hp)
                    fhs.append(dict(rows=L["rowsC"], edit=q.get("edit", {}), prow=L["rowsC"], hedits=fh or {}))
                    tag.append(nm)
                    return len(inps) - 1
                for rn, encname, pin in refs:
                    idx[rn] = add(f"{rn}@{sp}", encname, pin=pin)
                for arm, recp, kind, pay in chunk:
                    encname = dict((r[0], r[1]) for r in refs)[recp]; pin = dict((r[0], r[2]) for r in refs)[recp]
                    nm = f"{recp}_{arm}"
                    if kind == "pad":
                        add(f"PAD@{sp}_{len(inps)}", encname, pin=pin)
                    elif kind == "self_hp":
                        add(nm, encname, hp=dict(rows=L["rowsC"], edits={F47: [dict(heads=list(range(NH)), op="swap", src=len(inps))]}))
                    elif kind == "self_fh":
                        add(nm, encname, pin=pin, fh={F47: [dict(heads=list(range(NH)), op="swap", src=len(inps))]})
                    elif kind == "cp":
                        comps, src = pay
                        add(nm, encname, pin=pin, cp=dict(src=idx[src], comps=comps))
                    elif kind == "hp":
                        layer, hs_, op, src = pay
                        e = dict(heads=clip(hs_), op=op, src=(idx[src] if src is not None else None))
                        add(nm, encname, hp=dict(rows=L["rowsC"], edits={layer: [e]}))
                    elif kind == "fh":
                        hs_, op, src = pay
                        add(nm, encname, pin=pin, fh={F47: [dict(heads=clip(hs_), op=op, src=idx[src])]})
                lg, st = run1(inps, qs, hps, fhs)
                for b, nm in enumerate(tag):
                    lgb, stb = lg[b], st[b]
                    T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                    w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                    keys = [f"{tagp}_{nm}"] + ([f"{tagp}_{nm.split('@')[0]}"] if nm.endswith("@0") else [])
                    for key in keys:
                        rec["land"][key] = labs.index(w) + 1 if w in labs else 0
                        rec["mlog"][key] = [float(sc(lgb, x)) for x in labs[:n]]
                        rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                        rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
                    rec.setdefault("subpass", {})[f"{tagp}_{nm}"] = sp
                    if nm.endswith("@0") and ("q", EARLY[0]) in stb:
                        got[nm.split("@")[0]] = {li: stb[("q", li)] for li in EARLY}
                    nrun += 1
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return got

        GROUPS_DONE = {}
        # ---- G: natural grid (references only; START pin donors grabbed here)
        for n in (6, 7, 8, 9):
            refs = [(f"END{k}", f"END{k}", None) for k in range(1, n + 1)] + [(f"START{q}", f"START{q}", None) for q in range(1, n + 1)]
            GROUPS_DONE[n] = group(n, refs, [], f"G{n}", size=0)
        # ---- I: natural interventions in new cells
        def i_specs(recp, don):
            return [("SELF", recp, "self_hp", None), ("BAND_S", recp, "cp", (ALLC, don)), ("ANTI_Z", recp, "hp", (F47, ANTI, "zero", None)),
                    ("SELFADDR_S", recp, "hp", (F47, SELFADDR, "swap", don)), ("H16_S", recp, "hp", (F47, H16, "swap", don)),
                    ("RAND1a_S", recp, "hp", (F47, RAND1[0], "swap", don)), ("RAND1b_S", recp, "hp", (F47, RAND1[1], "swap", don)),
                    ("RAND5a_Z", recp, "hp", (F47, RAND5[0], "zero", None))]
        for n, ks in ((9, (4, 6, 7)), (7, (5,)), (6, (4, 3))):
            refs = [(f"END{k}", f"END{k}", None) for k in ks] + [(f"START{k}", f"START{k}", None) for k in ks]
            specs = sum([i_specs(f"END{k}", f"START{k}") for k in ks], [])
            group(n, refs, specs, f"I{n}", size=8)
        # ---- P: pinned at new anchors 2 and 6, h16 / band exchanged between the pins
        def pin(n, qd):
            return {li: dict(qsub=dict(owner=LAY[n]["owner"], nb=n, part="U", donors=[(1.0, qd[li])])) for li in EARLY}
        PINS = (2, 3, 6)
        for n, k in ((9, 5), (8, 5)):
            g_ = GROUPS_DONE[n]
            refsP = []
            for p in PINS:
                pq = pin(n, g_[f"START{p}"])
                refsP += [(f"E{k}p{p}", f"END{k}", pq), (f"Rp{p}", f"START{p}", pq)]
            specsP = [("SELF_FA", f"E{k}p3", "self_fh", None)]
            for p in PINS:
                for pp in PINS:
                    if pp != p:
                        specsP += [(f"H16_PX{pp}", f"E{k}p{p}", "fh", (H16, "swap", f"E{k}p{pp}")), (f"BAND_PX{pp}", f"E{k}p{p}", "cp", (ALLC, f"E{k}p{pp}"))]
                specsP += [("H16_R", f"E{k}p{p}", "fh", (H16, "swap", f"Rp{p}")), ("BAND_R", f"E{k}p{p}", "cp", (ALLC, f"Rp{p}"))]
            group(n, refsP, specsP, f"P{n}", size=7)
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · head-patch calls {HP.calls} · pinned head-patch calls {FHP.calls}", flush=True)
    TOOL.remove(); HP.remove(); FHP.remove()
    json.dump(dict(tag=a.tag, model=a.model, pins=[2, 3, 6], p_cells=[[9, 5], [8, 5]], anti=ANTI, h16=H16, selfaddr=SELFADDR, anti43=ANTI43, rand5=RAND5, rand1=RAND1, band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "predict"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'predict')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86200); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
