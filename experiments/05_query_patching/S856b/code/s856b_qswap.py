"""S856b (rerun of S856 after its reference check missed by one run; pre-registered in knowledge/DECISIONS.md before any S856b data).
Change from S856: every stage-2 and stage-3 batch of a form and picture count holds exactly 10 runs (S856 captured donor queries in 8-run
batches and injected them into 20-run batches; with 6 pictures the two batch shapes differed numerically), and both prompt forms carry all
five stage-3 arms.  Otherwise as S856:  Does the visual-prefix-conditioned late query carry the
actual-count difference of the lookup?  The same scene with 6 pictures (X6 = pictures 1-6) or 8 (X8 = 1-8), the same menu; k in {3, 5} (targets
for 6 / 8 pictures: k 3 -> 4 / 6, k 5 -> 2 / 4, all among the shared pictures 1-6), early lookup pinned to p in {3, 4}.  Two prompt forms: A states
'There are 8 images in a row.' (X6 is then a counterfactual probe), NC has no count sentence (no mismatch).
  stage 1  END(k), START(q) per form and picture count (pin donors)
  stage 2  E (natural, pinned) and EN (the same with the late full-attention layers 51-63 recomputed in float32 without an edit); EN stores the
           pre-RoPE late queries of the ten read rows
  stage 3  in each recipient (X6 or X8): EN copies (same-batch references) and late-query swaps at layers 51, 55, 59, 63 - the whole pre-RoPE query
           of the read rows <- the other count's EN query (same scene, form, k, p), rotated with the recipient's own positions (keys, values,
           gates and every other state stay the recipient's):
             QX    all ten read rows, whole query
             QXA   rows B1, B4, answer only, whole query
             QXU   all ten rows, only the address part U (span of the recipient's centred block-mean keys per head)
             SELF  the recipient's own EN query (must reproduce ENc bit for bit)
           (S856b: all five arms in both forms; stage 2 = one batch of 10 per form and count - E and EN for the four conditions + two padding
           EN copies PAD; stage 3 = two batches of 10 - two conditions x ENc, QX, QXA, QXU, SELF)
30 new scenes (seed 85800).  Readouts: answer, menu-word logits, positional scores of every read row, picture mass; per head at the late layers
for rows B1, B4, answer: block scores, picture mass and the gated output norms (total / picture part / mean gate).
Usage: python s856b_qswap.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_j import QPathJ

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s856_{}_{}.json"
KS = (3, 5); PS = (3, 4); FORMS = ("A", "NC")


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S856_DEVICE", "cuda:0")
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
    KR = [2, 5, 9]
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
    TOOL = QPathJ(v, READ, res_layers=(), comp_layers=(), patch_comps=())
    T1 = []

    def prompt(na, q, frame, menu, form):
        cnt = "There are 8 images in a row. " if form == "A" else ""
        return v.chat([("user", f"{v.IMG*na}\n{cnt}What is in the {ORDW[q-1]} image from the {frame}? "
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

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, 8); extra_w = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra_w; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra_w, menu=menu, pics=pics))
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}) for s in scenes}

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
            npz[f"s{rec['si']}_{nm}_capH"] = np.stack([stb[("cap", li)][KR].float().numpy() for li in LATE]).astype(np.float16)   # [L, 3, H, nb]
            npz[f"s{rec['si']}_{nm}_imgH"] = np.stack([stb[("img", li)][KR].float().numpy() for li in LATE]).astype(np.float16)   # [L, 3, H]
            if ("hout", LATE[0]) in stb:
                npz[f"s{rec['si']}_{nm}_hout"] = np.stack([stb[("hout", li)].float().numpy() for li in LATE]).astype(np.float16)  # [L, 3, r, H]

    def build_scene(s, na, form):
        nms = {**{f"END_k{k}": (k, "last") for k in KS}, **{f"START_q{q}": (q, "first") for q in (3, 4, 5)}}
        enc = {nm: v.encode(prompt(na, q, fr, s["menu"], form), s["pics"][:na]) for nm, (q, fr) in nms.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = ids["END_k5"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        assert len(ve) == na
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        for k in KS:
            Fw = [i for i, (x, y) in enumerate(zip(ids[f"START_q{k}"], ids[f"END_k{k}"])) if x != y]
            assert len(Fw) == 1 and Fw[0] < qm
        owner = [-1] * S_
        for b in range(na):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        return dict(enc=enc, S=S_, owner=owner, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, na=na)

    BA = list(range(10)); LQ = set(LATE)
    for gi, s in enumerate(scenes):
        rec = recs[s["si"]]
        for form in FORMS:
            P = {na: build_scene(s, na, form) for na in (6, 8)}
            don, qEN = {}, {}
            def capq(na, headout=False, **kw):
                p = P[na]
                return dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=na, opt=p["opt"], **({"headout": True} if headout else {})), **kw)
            def pins(na, p_):
                dq = don[(na, f"START_q{p_}")]["q"]
                return {li: [dict(qsub=dict(owner=P[na]["owner"], nb=na, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}
            def late_swap(na, dq_late, part, rows):
                return {li: [dict(qsub=dict(owner=P[na]["owner"], nb=na, part=part, donors=[(1.0, dq_late[li][rows])]), erows=rows)] for li in LATE}
            # stage 1
            for na in (6, 8):
                inps, qs, tag = [], [], []
                for nm in ("END_k3", "END_k5", "START_q3", "START_q4", "START_q5"):
                    inps.append(P[na]["enc"][nm]); qs.append(capq(na, grabq=set(EARLY), grabkb=set(LOOK))); tag.append(nm)
                lg, st = run1(inps, qs)
                for b, nm in enumerate(tag):
                    put(rec, f"{form}_{nm}_a{na}", measure(rec, lg[b], st[b])); nrun += 1
                    don[(na, nm)] = dict(q={li: st[b][("q", li)] for li in EARLY}, kb={li: st[b][("kb", li)] for li in LOOK})
                kr = max(max(float((don[(na, nm)]["kb"][li] - don[(na, "END_k5")]["kb"][li]).abs().max() / don[(na, "END_k5")]["kb"][li].abs().max())
                             for li in LOOK) for nm in tag)
                keyrel.append(kr)
            # stage 2: one batch of exactly 10 runs (E and EN for the four conditions + two EN padding copies) - the same batch shape as stage 3
            for na in (6, 8):
                inps, qs, tag = [], [], []
                for k in KS:
                    for p_ in PS:
                        inps.append(P[na]["enc"][f"END_k{k}"]); qs.append(capq(na, edit=pins(na, p_))); tag.append(f"E_k{k}p{p_}")
                        inps.append(P[na]["enc"][f"END_k{k}"]); qs.append(capq(na, edit=pins(na, p_), noop=set(LATE), grabq=LQ)); tag.append(f"EN_k{k}p{p_}")
                for k, p_ in ((3, 3), (5, 4)):
                    inps.append(P[na]["enc"][f"END_k{k}"]); qs.append(capq(na, edit=pins(na, p_), noop=set(LATE))); tag.append(f"PAD_k{k}p{p_}")
                assert len(inps) == 10
                lg, st = run1(inps, qs)
                for b, nm in enumerate(tag):
                    put(rec, f"{form}_{nm}_a{na}", measure(rec, lg[b], st[b])); nrun += 1
                    if nm.startswith("EN_"):
                        qEN[(na, nm[3:])] = {li: st[b][("q", li)] for li in LATE}
            # stage 3: two batches of exactly 10 runs (two conditions x five arms each), both forms
            for na in (6, 8):
                other = 14 - na
                for half in (((3, 3), (3, 4)), ((5, 3), (5, 4))):
                    inps, qs, tag = [], [], []
                    for k, p_ in half:
                        c = f"k{k}p{p_}"; e = P[na]["enc"][f"END_k{k}"]
                        inps.append(e); qs.append(capq(na, headout=True, edit=pins(na, p_), noop=set(LATE))); tag.append(f"ENc_{c}")
                        inps.append(e); qs.append(capq(na, headout=True, edit={**pins(na, p_), **late_swap(na, qEN[(other, c)], "all", BA)})); tag.append(f"QX_{c}")
                        inps.append(e); qs.append(capq(na, headout=True, edit={**pins(na, p_), **late_swap(na, qEN[(other, c)], "all", KR)})); tag.append(f"QXA_{c}")
                        inps.append(e); qs.append(capq(na, headout=True, edit={**pins(na, p_), **late_swap(na, qEN[(other, c)], "U", BA)})); tag.append(f"QXU_{c}")
                        inps.append(e); qs.append(capq(na, headout=True, edit={**pins(na, p_), **late_swap(na, qEN[(na, c)], "all", BA)})); tag.append(f"SELF_{c}")
                    assert len(inps) == 10
                    lg, st = run1(inps, qs)
                    for b, nm in enumerate(tag):
                        put(rec, f"{form}_{nm}_a{na}", measure(rec, lg[b], st[b]), st[b], keep_heads=True); nrun += 1
            del don, qEN
        if gi % 5 == 4 or gi == len(scenes) - 1:
            print(f"[{a.tag}] {gi + 1}/{len(scenes)} scenes · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    TOOL.remove()
    nm_ = "qswap"
    json.dump(dict(tag=a.tag, model=a.model, ks=KS, ps=PS, forms=FORMS, read=READ, look=LOOK, early=EARLY, late=LATE, kr=KR, matcher_layers=sorted(MH),
                   n_layer=nl, readout=os.environ.get("R3_READ"), qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun,
                   rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85800)
    main(ap.parse_args())
