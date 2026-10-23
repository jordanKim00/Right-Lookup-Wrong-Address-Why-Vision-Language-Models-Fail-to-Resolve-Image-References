"""S861 (registered in knowledge/DECISIONS.md before its data; user review 10-20 - three parts, judged separately) - does the S860 circuit act in
natural runs, is h16's predecessor signal the one language 'before' uses, and where does k act on the units?
One scene = nine pictures with all nine names as the menu; layouts show its first 7, 8 or 9 pictures (stated count = shown count).
  A (natural, no pins; patches at the ten read rows; donor = the same scene's START of the same ordinal, same layout):
     N7 END(4) and N9 END(5) (identity), N8 END(5) (target 4 = k - 1: the step helps), N8 END(3) (target 6), N8 END(2) (target 7: near END),
     controls START(4), START(5) (N8).
     Arms per END run: SELF (all FA47 heads <- itself: bit-identity), BAND_S (all eight band components 46-49 <- START), ALL47_S (FA47 output <-
     START), ANTI_Z (FA47 anti-self {0, 2, 5, 13, 15} zeroed), ANTI43_Z (FA43 {6, 8, 13, 15, 18} zeroed), SELFADDR_S (FA47 {1, 4, 12, 17, 20} <-
     START), H16_S (FA47 h16 <- START), RAND5a/b_S, RAND1a/b_S (random FA47 sets <- START; the S860 draws).  START controls: ANTI_Z, H16_Z,
     SELFADDR_Z (zeroed).
  B (natural, N8; operator identity): BEF5 = 'What is in the image just before the fifth?' (target 4), AFT5 = '... just after the fifth?' (target 6),
     with END5, START5, START4 of the same pass.  Arms: BEF5 with BAND_S5 / ALL47_S5 / H16_S5 / SELFADDR_S5 (<- START5), ANTI_Z, H16_X (h16 <- END5),
     RAND1a_S5; END5 with H16_X (h16 <- BEF5); AFT5 with H16_S5, BAND_S5.  (The four question forms have ten tokens each, so every read row aligns.)
  C (pinned as S860, N8; gate x read): E5 = END(5) and E3 = END(3) with the early lookup pinned at p (U part at 35-47 <- START(p)), p in {4, 5}, and
     R = START(p) at p; arms on E5: SELFADDR / H16 / ANTI with E3's output gate on E5's own read ('gate') or E3's read under E5's gate ('read'),
     and SELF_FA (bit-identity).
Passes: every group is split into equal-size sub-passes, each holding every reference run of the group (S860).
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer (the natural early anchor is read at
L43 / L47), per-head late block scores of B1, B4, answer.  Usage: python s861_verify.py MODEL TAG [--n 24] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI
from c_headpatch import HeadPatch, FAHeadPatchPinned

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s861_{}_{}.json"
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
    dev = os.environ.get("S861_DEVICE", "cuda:0")
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
        n8names = {"END5": ("END", 5), "END3": ("END", 3), "END2": ("END", 2), "START5": ("START", 5), "START3": ("START", 3), "START2": ("START", 2),
                   "START4": ("START", 4), "BEF5": ("BEF", 5), "AFT5": ("AFT", 5)}
        LAY = {8: layout(8, n8names, menu, pics), 7: layout(7, {"END4": ("END", 4), "START4": ("START", 4)}, menu, pics),
               9: layout(9, {"END5": ("END", 5), "START5": ("START", 5)}, menu, pics)}

        def group(n, refs, specs, tagp, pinp=None, size=None):
            """run reference runs + arms in equal-size sub-passes; specs = [(arm name, recipient ref, kind, payload)]."""
            nonlocal nrun
            L = LAY[n]
            capq = lambda **kw: dict(rows=L["rowsC"], cap=dict(rows=list(range(10)), owner=L["owner"], nb=n, opt=L["opt"]), **kw)
            size = size or 11
            chunks = [specs[i:i + size] for i in range(0, len(specs), size)] or [[]]
            got = {}
            for sp, chunk in enumerate(chunks):
                chunk = chunk + [("PAD", refs[0][0], "pad", None)] * (size - len(chunk))
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

        def end_specs(recp, don):
            return [("SELF", recp, "self_hp", None), ("BAND_S", recp, "cp", (ALLC, don)), ("ALL47_S", recp, "cp", ([(F47, "mix")], don)),
                    ("ANTI_Z", recp, "hp", (F47, ANTI, "zero", None)), ("ANTI43_Z", recp, "hp", (F43, ANTI43, "zero", None)),
                    ("SELFADDR_S", recp, "hp", (F47, SELFADDR, "swap", don)), ("H16_S", recp, "hp", (F47, H16, "swap", don)),
                    ("RAND5a_S", recp, "hp", (F47, RAND5[0], "swap", don)), ("RAND5b_S", recp, "hp", (F47, RAND5[1], "swap", don)),
                    ("RAND1a_S", recp, "hp", (F47, RAND1[0], "swap", don)), ("RAND1b_S", recp, "hp", (F47, RAND1[1], "swap", don))]
        # ---- N8: parts A and B (natural)
        refs8 = [(nm, nm, None) for nm in n8names]
        specs8 = end_specs("END5", "START5") + end_specs("END3", "START3") + end_specs("END2", "START2")
        for st_ in ("START4", "START5"):
            specs8 += [("ANTI_Z", st_, "hp", (F47, ANTI, "zero", None)), ("H16_Z", st_, "hp", (F47, H16, "zero", None)),
                       ("SELFADDR_Z", st_, "hp", (F47, SELFADDR, "zero", None))]
        specs8 += [("BAND_S5", "BEF5", "cp", (ALLC, "START5")), ("ALL47_S5", "BEF5", "cp", ([(F47, "mix")], "START5")),
                   ("H16_S5", "BEF5", "hp", (F47, H16, "swap", "START5")), ("SELFADDR_S5", "BEF5", "hp", (F47, SELFADDR, "swap", "START5")),
                   ("ANTI_Z", "BEF5", "hp", (F47, ANTI, "zero", None)), ("H16_X", "BEF5", "hp", (F47, H16, "swap", "END5")),
                   ("RAND1a_S5", "BEF5", "hp", (F47, RAND1[0], "swap", "START5")), ("H16_X", "END5", "hp", (F47, H16, "swap", "BEF5")),
                   ("H16_S5", "AFT5", "hp", (F47, H16, "swap", "START5")), ("BAND_S5", "AFT5", "cp", (ALLC, "START5"))]
        got8 = group(8, refs8, specs8, "N8", size=11)
        # ---- N7, N9 identity cells (natural)
        group(7, [("END4", "END4", None), ("START4", "START4", None)], end_specs("END4", "START4"), "N7", size=11)
        group(9, [("END5", "END5", None), ("START5", "START5", None)], end_specs("END5", "START5"), "N9", size=11)
        # ---- part C (pinned, N8): gate x read on E5 with E3's
        def pin(qd):
            return {li: dict(qsub=dict(owner=LAY[8]["owner"], nb=8, part="U", donors=[(1.0, qd[li])])) for li in EARLY}
        refsC, specsC = [], []
        for p in (4, 5):
            pq = pin(got8[f"START{p}"]) if f"START{p}" in got8 else None
            refsC += [(f"E5p{p}", "END5", pq), (f"E3p{p}", "END3", pq), (f"Rp{p}", f"START{p}", pq)]
            specsC += [("SELF_FA", f"E5p{p}", "self_fh", None)]
            for setn, hs_ in (("SELFADDR", SELFADDR), ("H16", H16), ("ANTI", ANTI)):
                specsC += [(f"{setn}_G", f"E5p{p}", "fh", (hs_, "gate", f"E3p{p}")), (f"{setn}_RD", f"E5p{p}", "fh", (hs_, "read", f"E3p{p}"))]
        group(8, refsC, specsC, "C8", size=14)
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · head-patch calls {HP.calls} · pinned head-patch calls {FHP.calls}", flush=True)
    TOOL.remove(); HP.remove(); FHP.remove()
    json.dump(dict(tag=a.tag, model=a.model, anti=ANTI, h16=H16, selfaddr=SELFADDR, anti43=ANTI43, rand5=RAND5, rand1=RAND1, end_arms=END_ARMS,
                   start_arms=START_ARMS, c_arms=C_ARMS, band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "verify"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'verify')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86100); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
