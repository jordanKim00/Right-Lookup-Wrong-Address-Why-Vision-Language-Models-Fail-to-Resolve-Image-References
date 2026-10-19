"""S545: how does 27B know where the end is? Shift every image block's own index by +2 without changing content or order.

An image block cannot know that it is the last one (causal mask), so the end must be inferred from the blocks by the
text that follows. Candidates: (i) the largest index code among the blocks ("the last block says it is the N-th"),
(ii) the number of blocks, (iii) no index at all (a relation). Donor scene: two new pictures followed by the recipient's
N pictures, so donor blocks 3..N+2 hold the recipient's pictures 1..N but believe they are the 3rd..(N+2)-th. Every
image-block token (with its vision start/end markers) of the recipient takes the donor's residual state at every layer
(block b <- donor block b+2), so the recipient's images carry index codes 3..N+2, unchanged content and order.
    forward "k-th"           absolute-index lookup -> lands on block k-2
    reverse "k-th from last" (i) largest index -> still j (N+2+1-k on codes = block j);  (ii) counting -> j-2
Control: a donor with the two extra pictures appended AFTER the recipient's (causally identical blocks 1..N) must change
nothing.
"""
from __future__ import annotations

import os
import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s545_indexshift_{}.json"
CIFAR = "/vlm/assets/cifar10"


def load_cifar(pixels):
    with open(f"{CIFAR}/batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]
    with open(f"{CIFAR}/test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    by = {i: [] for i in range(len(names))}
    for arr, y in zip(data, d[b"labels"]):
        if len(by[y]) < 200:
            by[y].append(arr)
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC)
                       for a in v_] for i, v_ in by.items()}


def run(v, tag="job", ntest=8, pixels=224, Ns=(5, 6), ks=(2, 3, 4), seed=54500, shift=2):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    def aid(s):                                          # answer readout (S787): R3_READ=bare reads the first bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return tid1(s)
        return v.tok(s, add_special_tokens=False)["input_ids"][0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    Lh = v.n_layer
    ST = {"mode": None}

    def mk(l):
        def hook(m, i, o):
            if ST["mode"] is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST["mode"] == "cap":
                ST["cap"][l] = t[0, ST["span"]].detach().clone()                       # [len(span), d]
                return o
            if ST["mode"] == "put":
                t = t.clone()
                D = ST["don"][l]
                t[0, ST["dst"].to(t.device)] = D[ST["src"].to(D.device)].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return hook
    hks = [v.layers[l].register_forward_hook(mk(l)) for l in range(Lh)]

    def prompt(rev, k, N, opts):
        tail = " from the last" if rev else ""
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORD[k-1]} image{tail}? "
                                f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)

    def blocks(inp, N):
        B = v.image_blocks(inp["input_ids"], N)
        return [torch.arange(int(b[0]) - 1, int(b[-1]) + 2) for b in B]                 # with vision start / end markers

    @torch.no_grad()
    def forward(rev, k, N, cl, pics, opts, mode=None, **kw):
        inp = v.encode(prompt(rev, k, N, opts), pics)
        ST.update(mode=mode, cap={}, **kw)
        if mode == "cap":
            BB = blocks(inp, N); ST["span"] = torch.cat(BB); ST["blocks"] = BB
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["mode"] = None
        return int(np.argmax([sc(out.logits[0, -1], c).item() for c in cl])) + 1, inp

    rng = random.Random(seed)
    rows = []
    for t_ in range(ntest):
        for N in Ns:
            ci = rng.sample(range(len(names)), N + shift)
            cl = [names[x] for x in ci[:N]]; pics = [rng.choice(tiles[x]) for x in ci[:N]]
            extra = [rng.choice(tiles[x]) for x in ci[N:]]
            opts = cl[:]; rng.shuffle(opts)
            # donors: prepended (index shift) and appended (control); capture image-block residuals at every layer
            dons = {}
            for nm, pp in (("pre", extra + pics), ("app", pics + extra)):
                _, inp_d = forward(False, 1, N + shift, cl, pp, opts, mode="cap")
                off = np.cumsum([0] + [len(b) for b in ST["blocks"]])
                dons[nm] = (ST["cap"], ST["blocks"], off)
            for k in ks:
                j = N + 1 - k
                if j == k:
                    continue
                r = dict(t=t_, N=N, k=k, j=j)
                for rev in (False, True):
                    nm_ = "R" if rev else "F"
                    a0, inp = forward(rev, k, N, cl, pics, opts)
                    r[f"{nm_}|none"] = a0
                    Brec = blocks(inp, N)
                    for dn, first in (("pre", shift), ("app", 0)):
                        cap, Bd, off = dons[dn]
                        dst = torch.cat(Brec)
                        src = torch.cat([torch.arange(off[first + b], off[first + b + 1]) for b in range(N)])
                        if len(dst) != len(src):
                            r[f"{nm_}|{dn}"] = -1; continue
                        r[f"{nm_}|{dn}"] = forward(rev, k, N, cl, pics, opts, mode="put", don=cap, dst=dst, src=src)[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()

    res = {}
    print(f"\n  팔            전방: k / k−{shift}            역순: j / j−{shift} / k")
    for arm in ("none", "pre", "app"):
        F = [r[f"F|{arm}"] for r in rows]; R = [r[f"R|{arm}"] for r in rows]
        d = dict(F_k=float(np.mean([a == r["k"] for a, r in zip(F, rows)])), F_km=float(np.mean([a == r["k"] - shift for a, r in zip(F, rows)])),
                 R_j=float(np.mean([a == r["j"] for a, r in zip(R, rows)])), R_jm=float(np.mean([a == r["j"] - shift for a, r in zip(R, rows)])),
                 R_k=float(np.mean([a == r["k"] for a, r in zip(R, rows)])))
        res[arm] = d
        print(f"  {arm:>5s}      {d['F_k']:.3f} / {d['F_km']:.3f}            {d['R_j']:.3f} / {d['R_jm']:.3f} / {d['R_k']:.3f}", flush=True)
    Fk3 = [r for r in rows if r["k"] - shift >= 1]
    res["pre_F_km_k3"] = float(np.mean([r["F|pre"] == r["k"] - shift for r in Fk3])) if Fk3 else float("nan")
    print(f"  (전방 k >= {shift+1} 문항만: pre 의 k−{shift} 착지 {res['pre_F_km_k3']:.3f}, n {len(Fk3)})")
    print(f"\n  가름 (사전 등록 S545) (x0) app(뒤에 붙인 공여)는 전방·역순 모두 없음과 .05 안. (x1) 전방 pre: k−2 착지 >= .50(k >= 3 문항) — 절대 색인 조회.")
    print(f"    (x2) 역순 pre: j 유지(>= 없음 − .10)면 '가장 큰 색인' 으로 끝을 앎; j−2 착지 >= .30 이면 블록 개수로 앎. (x3, 보고) 그 밖의 착지.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=8)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, ntest=a.ntest)
