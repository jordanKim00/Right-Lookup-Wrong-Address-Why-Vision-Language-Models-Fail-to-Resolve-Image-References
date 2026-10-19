"""S588p: synthetic evaluation suite (S588g's sets: 3 formats x forward / reverse train phrasings / held-out phrasings / N9,
dev paraphrases, N 7-10, content 2AFC) for any saved NRR module through nrr_eval.NRRMLRepair (the same code path as the
benchmarks). Reports base vs repaired and how many sources were written / blocked."""
from __future__ import annotations
import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni
from nrr_eval import NRRMLRepair
from s588b_nrr import REV_TRAIN, REV_HELD
ORD = list(_ORD) + ["eleventh", "twelfth"]
CIFAR = "/vlm/assets/cifar10"
DEV = [" from the back", " in reverse order", " starting from the right"]


SEAL_ANCHOR = [" from the very end", " counting back from the last one", " from the far end"]    # S601: fixed before any result
SEAL_NOANCHOR = [" counting from the rear", " in backward order", " going backwards"]


def load_cifar(pixels=224, sealed=False):
    with open(f"{CIFAR}/batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]
    with open(f"{CIFAR}/test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    by = {i: [] for i in range(len(names))}
    seen = {i: 0 for i in range(len(names))}
    for arr, y in zip(data, d[b"labels"]):
        seen[y] += 1
        if sealed:                                                                # S601: pictures 401-1000 of each class, never used
            if seen[y] > 400:
                by[y].append(arr)
        elif len(by[y]) < 400:
            by[y].append(arr)
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC) for a in v_] for i, v_ in by.items()}


def main(v, ckpt, tag, neval=40, seed=58800, tau=None, diag=False, sealed=False, kind="nrr", reader=None, seal_seed=60110):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(sealed=sealed)
    NMAX = 10 if sealed else 8
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    if kind == "r3":                                                              # R³-Cal / R³-Ref modules (r3_build.py)
        from r3_repair import R3Repair
        rep = R3Repair(v, ckpt, reader, tau=tau)
    else:
        rep = NRRMLRepair(v, ckpt, tau=tau)

    def make(rng, N=None, fmt=None, rev=None, phr=None):
        N = N or rng.randint(3, NMAX)
        ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        rev = rng.random() < .5 if rev is None else rev
        phr = phr if phr is not None else (rng.choice(REV_TRAIN) if rev else "")
        if fmt == "options":
            m = min(4, N); ks = rng.sample(range(1, N + 1), m); target = rng.randrange(m)
            tgt = (N + 1 - ks[target]) if rev else ks[target]
            lines = [f"{chr(65 + i)}: The {ORD[kk-1]} image{phr}" for i, kk in enumerate(ks)]
            text = (f"{v.IMG*N}\nWhich image shows a {cl[tgt - 1]}?\nSelect from the following choices.\n" + "\n".join(lines) +
                    "\nAnswer with the option's letter from the given choices directly.")
            cand = [tok1(chr(65 + i)) for i in range(m)]; gold = target; jsrc = tgt
        elif fmt == "content":
            N2 = rng.randint(2, 8); ci = rng.sample(range(len(names)), N2)
            cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
            absent = rng.choice([names[x] for x in range(len(names)) if x not in ci])
            ch = [rng.choice(cl), absent]; rng.shuffle(ch)
            text = f"{v.IMG*N2}\nWhich of these is shown in one of the images? Answer with only one word from: {', '.join(ch)}."
            cand = [tok1(" " + c) for c in ch]; gold = ch.index(next(c for c in ch if c in cl))
        else:
            k = rng.randint(1, N); tgt = (N + 1 - k) if rev else k
            opts = cl[:]; rng.shuffle(opts)
            if fmt == "question":
                text = f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{phr}? Answer with only one word from: {', '.join(opts)}."
            else:
                text = f"{v.IMG*N}\nLook at the {ORD[k-1]} image{phr}. What is in it? Answer with only one word from: {', '.join(opts)}."
            cand = [tok1(" " + c) for c in cl]; gold = tgt - 1; jsrc = tgt
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=cand, gold=gold,
                    j_src=jsrc if fmt != "content" else None)

    erng = random.Random(seed + 2); drng = random.Random(seed + 3); orng = random.Random(seed + 4)
    sets = {}
    if sealed:                                                                    # S601 sealed synthetic set
        srng = random.Random(seal_seed)                                           # S729: a fresh draw of the sealed design (default = S601)
        for fmt in ("question", "statement", "options"):
            for rev in (False, True):
                sets[f"{fmt}|{'R' if rev else 'F'}|train-phr"] = [make(srng, fmt=fmt, rev=rev) for _ in range(neval)]
            for ph in SEAL_ANCHOR:
                sets[f"{fmt}|R|seal-anchor|{ph.strip()}"] = [make(srng, fmt=fmt, rev=True, phr=ph) for _ in range(20)]
            for ph in SEAL_NOANCHOR:
                sets[f"{fmt}|R|seal-noanchor|{ph.strip()}"] = [make(srng, fmt=fmt, rev=True, phr=ph) for _ in range(20)]
            for nn_ in (9, 10):
                sets[f"{fmt}|R|N{nn_}"] = [make(srng, N=nn_, fmt=fmt, rev=True) for _ in range(20)]
                sets[f"{fmt}|F|N{nn_}"] = [make(srng, N=nn_, fmt=fmt, rev=False) for _ in range(20)]
        sets["content|2AFC"] = [make(srng, fmt="content") for _ in range(neval)]
    for fmt in (() if sealed else ("question", "statement", "options")):
        for rev in (False, True):
            sets[f"{fmt}|{'R' if rev else 'F'}|train-phr"] = [make(erng, fmt=fmt, rev=rev) for _ in range(neval)]
        sets[f"{fmt}|R|held-phr"] = [make(erng, fmt=fmt, rev=True, phr=erng.choice(REV_HELD)) for _ in range(neval)]
        sets[f"{fmt}|R|N9"] = [make(erng, N=9, fmt=fmt, rev=True) for _ in range(neval // 2)]
        sets[f"{fmt}|F|N9"] = [make(erng, N=9, fmt=fmt, rev=False) for _ in range(neval // 2)]
        sets[f"{fmt}|R|dev-phr"] = [make(drng, fmt=fmt, rev=True, phr=drng.choice(DEV)) for _ in range(neval)]
        for nn_ in (7, 8, 9, 10):
            sets[f"{fmt}|R|N{nn_}"] = [make(orng, N=nn_, fmt=fmt, rev=True) for _ in range(neval // 2)]
    sets["content|2AFC"] = [make(erng, fmt="content") for _ in range(neval)]
    res = {}
    if diag and not sealed:                                                        # per-format target hit / act rate on reverse items
        for fmt in ("question", "statement", "options"):
            hit, acted, n = 0, 0, 0
            for ex in sets[f"{fmt}|R|train-phr"]:
                rep.set(True); rep.logits(ex["text"], ex["pics"]); rep.set(False)
                for dd in rep.last_detail:
                    n += 1; acted += bool(dd.get("act")); hit += (dd["t"] == ex.get("j_src", -1))
            print(f"[{tag}] 진단 {fmt}: 원천 {n} · 쓰기 {acted} · 사상 목표 = 정답 원천 표적 {hit}", flush=True)
    for nm, exs in sets.items():
        ab = [rep.both(ex["text"], ex["pics"], ex["cand"]) for ex in exs]
        res[nm] = dict(base=float(np.mean([a == ex["gold"] for (a, _), ex in zip(ab, exs)])),
                       rep=float(np.mean([b == ex["gold"] for (_, b), ex in zip(ab, exs)])))
        print(f"[{tag}] {nm:<26s} {res[nm]['base']:.2f} → {res[nm]['rep']:.2f}", flush=True)
    F_ = ("question", "statement", "options")
    if sealed:
        mr = lambda pre: float(np.mean([v_["rep"] for k_, v_ in res.items() if k_.startswith(pre) or any(k_.startswith(f"{f}|{pre}") for f in F_)]))
        msets = lambda tagp: float(np.mean([v_["rep"] for k_, v_ in res.items() if f"|R|{tagp}" in k_]))
        dF = max(abs(res[f"{f}|F|{x}"]["rep"] - res[f"{f}|F|{x}"]["base"]) for f in F_ for x in ("train-phr", "N9", "N10"))
        summ = dict(R=float(np.mean([res[f"{f}|R|train-phr"]["rep"] for f in F_])), seal_anchor=msets("seal-anchor"),
                    seal_noanchor=msets("seal-noanchor"), held=float(np.mean([msets("seal-anchor"), msets("seal-noanchor")])),
                    N9=msets("N9"), N10=msets("N10"), dF=dF, content=(res["content|2AFC"]["base"], res["content|2AFC"]["rep"]), stats=rep.stats)
        print(f"\n[{tag}] 봉인 요약: 역순 {summ['R']:.3f} · 새 표현(앵커) {summ['seal_anchor']:.3f} · 새 표현(앵커 없음) {summ['seal_noanchor']:.3f}"
              f" · N9 {summ['N9']:.3f} · N10 {summ['N10']:.3f} · 전방 최대 변화 {dF:.3f} · 내용 {summ['content'][0]:.2f} → {summ['content'][1]:.2f}", flush=True)
        json.dump(dict(res=res, summary=summ), open(f"/vlm/results/s588p_syneval_{tag}.json", "w"), indent=1, default=str)
        return
    mean = lambda key: float(np.mean([res[f"{f}|{key}"]["rep"] for f in F_]))
    dF = max(abs(res[f"{f}|F|train-phr"]["rep"] - res[f"{f}|F|train-phr"]["base"]) for f in F_)
    summ = dict(R=mean("R|train-phr"), held=mean("R|held-phr"), N9=mean("R|N9"), dev=mean("R|dev-phr"),
                ood={n: mean(f"R|N{n}") for n in (7, 8, 9, 10)}, dF=dF,
                content=(res["content|2AFC"]["base"], res["content|2AFC"]["rep"]), stats=rep.stats)
    print(f"\n[{tag}] 요약: 역순 {summ['R']:.3f} · 보류 {summ['held']:.3f} · N9 {summ['N9']:.3f} · 개발 {summ['dev']:.3f} · N7-10 "
          + " ".join(f"{v_:.2f}" for v_ in summ["ood"].values()) + f" · 전방 최대 변화 {dF:.3f} · 내용 {summ['content'][0]:.2f} → {summ['content'][1]:.2f} · 통계 {rep.stats}", flush=True)
    json.dump(dict(res=res, summary=summ), open(f"/vlm/results/s588p_syneval_{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--kind", default="nrr", choices=["nrr", "r3"])
    ap.add_argument("--reader", default=None)
    ap.add_argument("--seal_seed", type=int, default=60110)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    main(v, a.ckpt, a.tag, tau=a.tau, diag=a.diag, sealed=a.sealed, kind=a.kind, reader=a.reader, seal_seed=a.seal_seed)
    print(f"[{a.tag}] 끝 -> /vlm/results/s588p_syneval_{a.tag}.json", flush=True)
