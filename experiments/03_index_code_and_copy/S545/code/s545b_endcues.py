"""S545b: which cues set "the end" when the blocks' index codes disagree with their number? S545's +2 index shift with
the count sentence varied: C1 "There are N images" (true count; S545), C2 no count sentence, C3 "There are N+2 images"
(agrees with the shifted codes). Reverse questions only (forward is an exact -2 shift, §571); no-shift baseline per
condition. Offsets from j: 0 = the largest index code, -2 = the block count / a stated N.

S545 docstring follows.
S545: how does 27B know where the end is? Shift every image block's own index by +2 without changing content or order.

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

import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s545b_endcues_{}.json"
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


def run(v, tag="job", ntest=8, pixels=224, Ns=(5, 6), ks=(2, 3, 4), seed=54510, shift=2):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
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

    CNT = {"C1": 0, "C2": None, "C3": shift}

    def prompt(rev, k, N, opts, cond="C1"):
        tail = " from the last" if rev else ""
        cnt = "" if CNT[cond] is None else f"There are {N + CNT[cond]} images in a row. "
        return v.chat([("user", f"{v.IMG*N}\n{cnt}What is in the {ORD[k-1]} image{tail}? "
                                f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)

    def blocks(inp, N):
        B = v.image_blocks(inp["input_ids"], N)
        return [torch.arange(int(b[0]) - 1, int(b[-1]) + 2) for b in B]                 # with vision start / end markers

    @torch.no_grad()
    def forward(rev, k, N, cl, pics, opts, mode=None, cond="C1", **kw):
        inp = v.encode(prompt(rev, k, N, opts, cond), pics)
        ST.update(mode=mode, cap={}, **kw)
        if mode == "cap":
            BB = blocks(inp, N); ST["span"] = torch.cat(BB); ST["blocks"] = BB
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["mode"] = None
        return int(np.argmax([out.logits[0, -1, tid1(c)].item() for c in cl])) + 1, inp

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
                for cond in ("C1", "C2", "C3"):
                    nm_ = cond
                    a0, inp = forward(True, k, N, cl, pics, opts, cond=cond)
                    r[f"{nm_}|none"] = a0
                    Brec = blocks(inp, N)
                    for dn, first in (("pre", shift),):
                        cap, Bd, off = dons[dn]
                        dst = torch.cat(Brec)
                        src = torch.cat([torch.arange(off[first + b], off[first + b + 1]) for b in range(N)])
                        if len(dst) != len(src):
                            r[f"{nm_}|{dn}"] = -1; continue
                        r[f"{nm_}|{dn}"] = forward(True, k, N, cl, pics, opts, mode="put", cond=cond, don=cap, dst=dst, src=src)[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()

    res = {}
    rng_ = np.random.default_rng(0)
    print(f"\n  조건   팔     역순 j / j−1 / j−2 / k      평균 어긋남(목표 기준) [95%]")
    for cond in ("C1", "C2", "C3"):
        for arm in ("none", "pre"):
            x = [r[f"{cond}|{arm}"] for r in rows]
            off = np.array([a - r["j"] for a, r in zip(x, rows)], float)
            offc = np.clip(off, -3, 3)
            bs = [offc[rng_.integers(0, len(offc), len(offc))].mean() for _ in range(4000)]
            d = dict(j=float(np.mean(off == 0)), jm1=float(np.mean(off == -1)), jm2=float(np.mean(off == -2)),
                     k=float(np.mean([a == r["k"] for a, r in zip(x, rows)])), mean_off=[float(offc.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            res[f"{cond}|{arm}"] = d
            print(f"  {cond}    {arm:>4s}   {d['j']:.3f} / {d['jm1']:.3f} / {d['jm2']:.3f} / {d['k']:.3f}      {d['mean_off'][0]:+.2f} [{d['mean_off'][1]:+.2f}, {d['mean_off'][2]:+.2f}]", flush=True)
    print(f"\n  가름 (사전 등록 S545b) (y1) C2(개수 문장 없음)의 pre: j 유지(>= none − .10)면 끝 = 가장 큰 색인; 평균 어긋남이 −.5 보다 작으면(더 음수) 블록 개수도 끝에 들어간다.")
    print(f"    (y2) C3(말한 개수 = 옮긴 색인과 일치)의 pre 평균 어긋남이 C1 보다 0 쪽으로 .3 이상 → 말한 개수가 끝 추정에 들어간다(내용 질문에서도).")
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
