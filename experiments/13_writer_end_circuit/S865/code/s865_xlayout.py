"""S865 (registered in knowledge/DECISIONS.md before its data; S862 post hoc: the band's END part and the pinned one-back are N-invariant (D4)
while natural copy-anchor behaviour moves with the needed distance (A4) - does the N-dependent part of the answer live in the early address?) -
cross-layout early-address swap in natural runs.  One scene = nine pictures with all nine names as the menu; layouts show its first N pictures
(N 7-9, stated = shown).  Image keys of a block depend only on the tokens before it, so block b is the same picture with the same keys in
every layout.
  Per layout N_r: natural END(4), END(5) (references) and the same runs with the early address (U part of the read rows' queries at 35-47)
  replaced by the U part of: the same layout's END(k) grabbed in another pass (U_SELF, the cross-pass control), END(k) of each other layout
  N_d (U_FROM<N_d>), the same layout's START(k) (U_START, the pointer control); and the whole early query (address + reading-mode parts,
  part 'all') <- END(k) of each other layout (Q_FROM<N_d>; theorist critique 10-21: a U-only null cannot separate the late lookup from the
  early non-address parts).  Donor queries are pre-RoPE, rotated at the recipient's positions and (U) projected on the recipient's block
  span (exp/c_qpath.QPath._qsub).  Donor queries are grabbed in one natural pass per layout.
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of B1,
B4, answer.  Usage: python s865_xlayout.py MODEL TAG [--n 36] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_i import QPathI
from c_headpatch import HeadPatch, FAHeadPatchPinned

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s865_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]; ANTI43 = [6, 8, 13, 15, 18]
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]   # the S860 draws
LAYS = (7, 8, 9); KS = (4, 5)


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S865_DEVICE", "cuda:0")
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
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · layouts {LAYS} · k {KS}", flush=True)

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
            names = {f"END{k}": ("END", k) for k in KS} | {f"START{k}": ("START", k) for k in KS}
            LAY[n] = layout(n, names, menu, pics)
        def group(n, refs, tagp):
            """one pass of reference runs (each with its own pin or none); refs = [(run name, encoding name, pin)]."""
            nonlocal nrun
            L = LAY[n]
            capq = lambda **kw: dict(rows=L["rowsC"], cap=dict(rows=list(range(10)), owner=L["owner"], nb=n, opt=L["opt"]), **kw)
            inps, qs, hps, fhs, tag = [], [], [], [], []
            for rn, encname, pin in refs:
                inps.append(L["enc"][encname])
                q = capq(edit=pin) if pin is not None else capq(grabq=set(EARLY))
                qs.append(q); hps.append(None); fhs.append(None); tag.append(rn)
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
                if ("q", EARLY[0]) in stb:
                    got[nm] = {li: stb[("q", li)] for li in EARLY}
                nrun += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return got
        # ---- donors: one natural pass per layout (END(k), START(k)), queries grabbed
        DON = {n: group(n, [(f"END{k}", f"END{k}", None) for k in KS] + [(f"START{k}", f"START{k}", None) for k in KS], f"D{n}") for n in LAYS}
        def upin(n, qd, part="U"):
            return {li: dict(qsub=dict(owner=LAY[n]["owner"], nb=n, part=part, donors=[(1.0, qd[li])])) for li in EARLY}
        # ---- recipients: per layout, natural references and the address swaps, all in one pass
        for n in LAYS:
            refs = []
            for k in KS:
                refs += [(f"END{k}", f"END{k}", None), (f"U_SELF{k}", f"END{k}", upin(n, DON[n][f"END{k}"])),
                         (f"U_START{k}", f"END{k}", upin(n, DON[n][f"START{k}"]))]
                refs += [(f"U_FROM{nd}_{k}", f"END{k}", upin(n, DON[nd][f"END{k}"])) for nd in LAYS if nd != n]
                refs += [(f"Q_FROM{nd}_{k}", f"END{k}", upin(n, DON[nd][f"END{k}"], part="all")) for nd in LAYS if nd != n]
            group(n, refs, f"X{n}")
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs", flush=True)
    TOOL.remove(); HP.remove(); FHP.remove()
    json.dump(dict(tag=a.tag, model=a.model, lays=list(LAYS), ks=list(KS), band=band, read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE},
                   readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "xlayout"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'xlayout')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=36); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86500); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
