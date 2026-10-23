"""S867 (registered in knowledge/DECISIONS.md before its data; the adversary's alternative 10-21: in S862's pinned END5 runs the landing moves
end-ward at N9 vs N8 (+.44 picture, post hoc) although the one-back share does not - is the band N-aware, or does a late visible-end term move the
released answer?) - cross-layout transplant of the band output and of the late queries at a fixed pinned anchor.
One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (N 8 and 9, stated = shown).
  Pinned END(5) at p in {3, 6} (U part at 35-47 <- the same layout's natural START(p)).  Donor pass per layout: the pinned runs with the band's
  eight component outputs at the ten read rows grabbed (exp/c_qpath_l 'cgrab') and the late (51-63) queries grabbed.  Recipient pass per layout:
  the pinned references and arms per pin - BAND_SELF / BAND_XN (band outputs <- the same / the other layout's donor, 'cinj'), LQ_SELF / LQ_XN
  (the late queries, part 'all' <- the same / the other layout's donor), BOTH_XN (both from the other layout), CUT_LAST (the read rows may not
  read the last picture's positions at 51-63).  The SELF arms are the cross-pass numerical controls.
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of B1,
B4, answer.  Usage: python s867_xn.py MODEL TAG [--n 48] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_l import QPathL
from c_headpatch import HeadPatch, FAHeadPatchPinned

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s867_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]; ANTI43 = [6, 8, 13, 15, 18]
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]   # the S860 draws
LAYS = (8, 9); PINS = (3, 6); K = 5


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S867_DEVICE", "cuda:0")
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
    TOOL = QPathL(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=())
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
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · layouts {LAYS} · pins {PINS} · k {K}", flush=True)

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
        for n in LAYS:
            names = {"END5": ("END", K)} | {f"START{p}": ("START", p) for p in PINS}
            LAY[n] = layout(n, names, menu, pics)
        def run_refs(n, refs, tagp):
            """one pass; refs = [(run name, encoding name, cfg extras)]; returns the per-run store."""
            nonlocal nrun
            L = LAY[n]
            capq = lambda **kw: dict(rows=L["rowsC"], cap=dict(rows=list(range(10)), owner=L["owner"], nb=n, opt=L["opt"]), **kw)
            inps, qs, hps, fhs, tag = [], [], [], [], []
            for rn, encname, extra in refs:
                inps.append(L["enc"][encname]); qs.append(capq(**extra)); hps.append(None); fhs.append(None); tag.append(rn)
            lg, st = run1(inps, qs, hps, fhs)
            got = {}
            for b, nm in enumerate(tag):
                lgb, stb = lg[b], st[b]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                key = f"{tagp}_{nm}"
                rec["land"][key] = labs.index(w) + 1 if w in labs else 0
                rec["mlog"][key] = [float(sc(lgb, x)) for x in labs[:n]]
                rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
                got[nm] = stb
                nrun += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return got
        # ---- pin donors (natural START(p)) and the pinned donors of each layout
        PIN = {}
        for n in LAYS:
            g0 = run_refs(n, [(f"START{p}", f"START{p}", dict(grabq=set(EARLY))) for p in PINS], f"S{n}")
            PIN[n] = {p: {li: dict(qsub=dict(owner=LAY[n]["owner"], nb=n, part="U", donors=[(1.0, g0[f"START{p}"][("q", li)])])) for li in EARLY}
                      for p in PINS}
        DON = {}
        for n in LAYS:
            g1 = run_refs(n, [(f"E5p{p}", "END5", dict(edit=PIN[n][p], cgrab=ALLC, grabq=set(LATE))) for p in PINS], f"D{n}")
            DON[n] = {p: dict(band={c: g1[f"E5p{p}"][("cg", c[0], c[1])] for c in ALLC}, lq={li: g1[f"E5p{p}"][("q", li)] for li in LATE}) for p in PINS}
        # ---- recipients
        for n in LAYS:
            no = [x for x in LAYS if x != n][0]
            L = LAY[n]
            refs = []
            for p in PINS:
                base = PIN[n][p]
                def lq_edit(src):
                    e = dict(base)
                    for li in LATE:
                        e[li] = dict(qsub=dict(owner=L["owner"], nb=n, part="all", donors=[(1.0, DON[src][p]["lq"][li])]))
                    return e
                cut = dict(base)
                last = [i for i, o in enumerate(L["owner"]) if o == n - 1]
                ko = dict(layers=set(LATE), rows=list(range(10)), pos=last)
                refs += [(f"E5p{p}", "END5", dict(edit=base)),
                         (f"BAND_SELF_p{p}", "END5", dict(edit=base, cinj=dict(vals=DON[n][p]["band"]))),
                         (f"BAND_XN_p{p}", "END5", dict(edit=base, cinj=dict(vals=DON[no][p]["band"]))),
                         (f"LQ_SELF_p{p}", "END5", dict(edit=lq_edit(n))),
                         (f"LQ_XN_p{p}", "END5", dict(edit=lq_edit(no))),
                         (f"BOTH_XN_p{p}", "END5", dict(edit=lq_edit(no), cinj=dict(vals=DON[no][p]["band"]))),
                         (f"CUT_LAST_p{p}", "END5", dict(edit=cut, ko=ko))]
            run_refs(n, refs, f"X{n}")
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs", flush=True)
    TOOL.remove(); HP.remove(); FHP.remove()
    json.dump(dict(tag=a.tag, model=a.model, lays=list(LAYS), pins=list(PINS), k=K, band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "xn"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'xn')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86700); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
