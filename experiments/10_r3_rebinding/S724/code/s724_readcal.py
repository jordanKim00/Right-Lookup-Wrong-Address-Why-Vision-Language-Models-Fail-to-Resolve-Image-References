"""S724: per-model calibration measurements for R³'s Read components -- no end-relative answer, no benchmark item.
Calibration items are generated with known structure: images first (N 2-10; procedural images, or natural images from caltech256 / food101
rows that R³-Ref does not use, resized to 448 px), then a question, a look-at statement, or an option list (lower- and upper-case "The"),
each reference "the k-th image" followed by no frame, a START phrase or an END phrase.  Ground truth per item: the reference ordinal tokens
and their k, whether each is end-bound, and N.  No answer is read.
For every item the capture forward of exp/r3_min_md.R3MinMD stores, for every question token, the chart and full-state distances to the
nearest / second-nearest ordinal mean, the distances to the end / last frame means, and the image-block codes (as exp/s715_detdiag.py).
For the four smaller models, the learned module of the earlier version (exp/r3_repair.R3Repair) also plans each item, and its per-source
frame gate g is stored.  Selection of the detector, frame binder and count mode per model is done offline (exp/s724_select.py).
Amendment (before any S724 data, user direction 2026-10-10): plan-level ground truth and mixed-frame items.  Kinds none / start / end / mixed
(1 : 1 : 2 : 2); a mixed item carries references with different frames in one input (two references in a question or look-at statement,
or per-option frames with at least one END and one non-END option).  Every reference stores its gold plan: ordinal token, k, frame, and
the frame word's token (rec["gold"]).  The earlier module plans twice: its own plan (pointer confidence tau of the checkpoint = R³-v1, with
its learned target t and its writes) and with tau = 0 (= the frame positions exp/r3_v1c.R3V1C writes).  Two splits with disjoint images
and seeds: --split sel (selection) and --split val (plan-level confirmation, exp/s724_plan.py).
Usage: python s724_readcal.py MODEL TAG --base B [--split sel|val] [--v1ckpt PATH --v1reader R] [--n 300] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3ref import ORD, load_image
from r3_min_md import R3MinMD
from s588c_reader import probs as reader_probs

FR_START = (" from the start", " from the beginning")
FR_END = (" from the end", " from the last")


def nat_keys(rng, n):
    import glob, pyarrow.parquet as pq
    used = set()
    for f in glob.glob("/vlm/assets/r3ref_v1/*.jsonl"):
        for l in open(f):
            used.update(json.loads(l).get("images", []))
    keys = []
    for src, pat in (("caltech256", "/vlm/assets/r3_images/caltech256/data/*.parquet"), ("food101", "/vlm/assets/r3_images/food101/data/*.parquet")):
        for f in sorted(glob.glob(pat)):
            rows = pq.ParquetFile(f).metadata.num_rows; base = f.split("/")[-1]
            keys += [k for k in (f"{src}:{base}:{i}" for i in range(rows)) if k not in used]
    rng.shuffle(keys)
    return keys[:n]


def frame_tok(toks, p, stop, word):
    for q in range(p + 1, stop):
        if toks[q].strip().lower() == word:
            return q
    return None


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rep = R3MinMD(v, f"/vlm/results/r3/s709_{a.base}_cal.pt", v2=True, det_rule="margin", det_param=.5)
    v1 = None
    if a.v1ckpt:
        from r3_repair import R3Repair
        v1 = R3Repair(v, a.v1ckpt, a.v1reader)
    tau1 = v1.tau if v1 is not None else None
    nat_all = nat_keys(random.Random(72400), 8000)                                   # one fixed shuffle; sel and val take disjoint halves
    rng = random.Random(72400 if a.split == "sel" else 72410)
    nat = nat_all[:4000] if a.split == "sel" else nat_all[4000:]
    # S738b: a calibration image found to duplicate an evaluation image trades places with the head of its list, which the items never reach
    BL = "/vlm/research/ces/outputs/r3_lure_latest/s738_duplicates.json"
    blocked = set(json.load(open(BL))["calibration_keys"]) if os.path.exists(BL) else set()
    for bi in [i for i, k in enumerate(nat) if k in blocked]:
        hj = next(j for j in range(len(nat)) if nat[j] not in blocked)
        nat[bi], nat[hj] = nat[hj], nat[bi]
    nat_used = []

    def take_nat():
        k = nat.pop()
        assert k not in blocked, k
        nat_used.append(k)
        return k
    lo, hi = (0, 29_999) if a.split == "sel" else (30_000, 59_999)
    proc = [f"proc:{rng.randint(lo, hi)}" for _ in range(4000)]
    cal, fr = rep.cal, rep.cal["frame"]
    FR = {"start": FR_START, "end": FR_END}
    out = []
    for i in range(a.n):
        natural = i % 2 == 1
        N = rng.randint(2, 10)
        pics = [load_image(take_nat() if natural else proc.pop(), size=448 if natural else 224) for _ in range(N)]
        fmt = rng.choice(["question", "statement", "options", "options_cap"])
        kind = rng.choice(["none", "start", "end", "end", "mixed", "mixed"])
        if fmt in ("options", "options_cap"):
            ks = rng.sample(range(1, N + 1), min(4, N))
            if kind == "mixed":
                while True:
                    frs = [rng.choice(["none", "start", "end"]) for _ in ks]
                    if "end" in frs and any(f != "end" for f in frs):
                        break
            else:
                frs = [kind] * len(ks)
            phs = ["" if f == "none" else rng.choice(FR[f]) for f in frs]
            the = "The" if fmt == "options_cap" else "the"
            body = "Which image shows a circle?\nSelect from the following choices.\n" + "\n".join(
                f"{chr(65 + j)}: {the} {ORD[k - 1]} image{ph}" for j, (k, ph) in enumerate(zip(ks, phs))) + "\nAnswer with the option's letter from the given choices directly."
        elif kind == "mixed":
            ks = [rng.randint(1, N), rng.randint(1, N)]
            frs = list(rng.choice([("end", "start"), ("start", "end"), ("end", "none"), ("none", "end")]))
            phs = ["" if f == "none" else rng.choice(FR[f]) for f in frs]
            o1, o2 = ORD[ks[0] - 1], ORD[ks[1] - 1]
            body = (f"What is in the {o1} image{phs[0]}, and what is in the {o2} image{phs[1]}? Answer with two words from: circle, star, ring." if fmt == "question" else
                    f"Look at the {o1} image{phs[0]} and the {o2} image{phs[1]}. What is in them? Answer with two words from: circle, star, ring.")
        else:
            ks = [rng.randint(1, N)]; frs = [kind]
            phs = ["" if kind == "none" else rng.choice(FR[kind])]
            body = (f"What is in the {ORD[ks[0] - 1]} image{phs[0]}? Answer with only one word from: circle, star, ring." if fmt == "question" else
                    f"Look at the {ORD[ks[0] - 1]} image{phs[0]}. What is in it? Answer with only one word from: circle, star, ring.")
        text = v.chat([("user", f"{v.IMG * N}\n{body}")], prefix_assistant=THINK)
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b) for b in v.image_blocks(inp["input_ids"], N)]
        turn = max(j for j, x in enumerate(ids) if x == rep.im_start); s0 = int(B[-1][-1]) + 2
        rep.ST = dict(cap=True, span=torch.arange(s0, turn), B=B)
        with torch.no_grad():
            v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        H, bf = rep.ST["H"], rep.ST.get("bf", {}); rep.ST = {}
        Hb = H[rep.Lb]
        Dc = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]); Df = torch.cdist(Hb, cal["Mb"])
        dc2 = Dc.topk(2, largest=False); df2 = Df.topk(2, largest=False)
        X = torch.cat([bf[l] for l in rep.Rlayers], 1)
        codes = (reader_probs(rep.RP, X).argmax(1) + 1).long().tolist()
        r_ = lambda x: [round(float(y), 3) for y in x]
        toks = [v.tok.decode([t]) for t in ids[s0:turn]]
        rpos = [j for j, w in enumerate(toks) if w.strip().lower() in ORD and j + 1 < len(toks) and toks[j + 1].strip().lower() == "image"]
        assert len(rpos) == len(ks) and all(ORD.index(toks[p].strip().lower()) + 1 == k for p, k in zip(rpos, ks)), (toks, ks)
        gold = []
        for r, (p, k, f, ph) in enumerate(zip(rpos, ks, frs, phs)):
            ft = frame_tok(toks, p, rpos[r + 1] if r + 1 < len(rpos) else len(toks), ph.split()[-1]) if f != "none" else None
            assert f == "none" or ft is not None, (toks, p, ph)
            gold.append(dict(pos=p, k=k, frame=f, ftok=ft))
        rec = dict(i=i, split=a.split, natural=natural, N=N, fmt=fmt, kind=kind, ks=ks, frames=frs, gold=gold, codes=codes, toks=toks,
                   kh_c=(dc2.indices[:, 0] + 1).tolist(), dc=r_(dc2.values[:, 0]), dc2=r_(dc2.values[:, 1]),
                   kh_f=(df2.indices[:, 0] + 1).tolist(), df=r_(df2.values[:, 0]), df2=r_(df2.values[:, 1]),
                   d_end=r_((Hb - fr["mu"]["end"]).norm(dim=1)), d_last=r_((Hb - fr["mu"]["last"]).norm(dim=1)),
                   th_chart=cal["th_chart"], th_full=cal["th_full"], det_space=cal["det_space"], thr=fr["thr"])
        if v1 is not None:
            rel = lambda x: (int(x) - s0) if x is not None else None
            for key, tau in (("v1", tau1), ("v1c", 0.0)):
                v1.tau = tau
                with torch.no_grad():
                    v1.plan(text, pics)
                rec[key] = [dict(pos=int(d["pos"]) - s0, kh=int(d["kh"]), g=float(d["map"][1]) if d["map"][0] == "scan" else None,
                                 t=int(d["t"]), act=bool(d["act"]), conf=round(float(d["wmax"]), 3), fpos=rel(d.get("fpos"))) for d in v1.last_detail]
            v1.tau = tau1
            rec["v1_codes"] = getattr(v1, "last_codes", None)
        out.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[{a.tag}] {i + 1}/{a.n}", flush=True)
        torch.cuda.empty_cache()
    json.dump(dict(tag=a.tag, model=a.model, base=a.base, split=a.split, v1ckpt=a.v1ckpt, tau1=tau1, blocked=sorted(blocked), nat_used=nat_used, rec=out),
              open(f"/vlm/results/r3/s724_{a.tag}_readcal.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s724_{a.tag}_readcal.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True)
    ap.add_argument("--v1ckpt", default=None); ap.add_argument("--v1reader", default=None)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--device_map", default=None)
    ap.add_argument("--split", default="sel", choices=["sel", "val"])
    main(ap.parse_args())
