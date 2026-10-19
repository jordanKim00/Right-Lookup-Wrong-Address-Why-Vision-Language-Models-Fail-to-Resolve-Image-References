"""S741: is the index code a low-dimensional component of the image-block states, separable from content?  (reply to the mock review
W2, 2026-10-11; pre-registration in knowledge/DECISIONS.md before any S741 data; analysis exp/s741_analyze.py, frozen with it.)

The S545 code shift is a full-state exchange: every image-block token (with its vision markers) of the recipient takes, at every layer,
the residual state of the same picture in a donor sequence with two new pictures in front, so blocks 1..N carry codes 3..N+2 together
with whatever else the donor context changed.  Here the same exchange is split into two orthogonal parts.  The code subspace U_l of
layer l is the span of the centred position means of block states (mean over a block's tokens), estimated on separate calibration
scenes of eight CIFAR-10 pictures disjoint from the recipients' (test pictures 200-399 of each class; recipients use the first 200,
as S545), d = 7 dimensions per layer.
    NONE   no edit
    FULL   h <- h_donor                                   (S545)
    CODE   h <- h + U U^T (h_donor - h)                   only the code-subspace part of the exchange
    REST   h <- h + (I - U U^T)(h_donor - h)              everything but the code-subspace part
    RAND   h <- h + Q Q^T (h_donor - h)                   a random d-dimensional subspace (control)
    APP    h <- h_app                                     donor with the two new pictures appended after the scene (control)
Questions: forward "What is in the k-th image?", N in {5, 6}, k in {3, ..., N}; readout = restricted argmax over the scene's classes.
Usage: python s741_codesub.py MODEL TAG [--ntest 20] [--ncal 32] [--d 7]
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
OUT = "/vlm/results/r3/s741_{}_codesub.json"
CIFAR = "/vlm/assets/cifar10"
SHIFT = 2


def load_cifar(pixels, lo=0, hi=200):
    """CIFAR-10 test pictures lo..hi-1 of each class (recipients: the first 200, as S545; calibration: 200-399, disjoint)."""
    with open(f"{CIFAR}/batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]
    with open(f"{CIFAR}/test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    by = {i: [] for i in range(len(names))}; seen = {i: 0 for i in range(len(names))}
    for arr, y in zip(data, d[b"labels"]):
        if lo <= seen[y] < hi:
            by[y].append(arr)
        seen[y] += 1
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC) for a in v_] for i, v_ in by.items()}


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(a.pixels, 0, 200)
    _, ctiles = load_cifar(a.pixels, 200, 400)                                          # calibration pictures, disjoint from the recipients
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    Lh = v.n_layer
    ST = {"mode": None}

    def mk(l):
        def hook(m, i, o):
            if ST["mode"] is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST["mode"] == "cap":
                ST["cap"][l] = t[0, ST["span"]].detach().float().clone()
                return o
            if ST["mode"] == "put":
                t = t.clone()
                dst = ST["dst"].to(t.device)
                h = t[0, dst].float(); hd = ST["don"][l][ST["src"]].to(t.device).float()
                arm = ST["arm"]
                if arm == "FULL":
                    new = hd
                else:
                    B = ST["basis"][l].to(t.device)                                     # [D, d] orthonormal
                    delta = hd - h; proj = (delta @ B) @ B.T
                    new = h + proj if arm in ("CODE", "RAND") else h + (delta - proj)
                t[0, dst] = new.to(t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return hook
    hks = [v.layers[l].register_forward_hook(mk(l)) for l in range(Lh)]

    def prompt(k, N, opts):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORD[k-1]} image? "
                                f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)

    def blocks(inp, N):
        B = v.image_blocks(inp["input_ids"], N)
        return [torch.arange(int(b[0]) - 1, int(b[-1]) + 2) for b in B]                 # with vision start / end markers

    @torch.no_grad()
    def forward(k, N, cl, pics, opts, mode=None, **kw):
        inp = v.encode(prompt(k, N, opts), pics)
        ST.update(mode=mode, cap={}, **kw)
        if mode == "cap":
            BB = blocks(inp, N); ST["span"] = torch.cat(BB); ST["blocks"] = BB
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["mode"] = None
        return int(np.argmax([out.logits[0, -1, tid1(c)].item() for c in cl])) + 1, inp

    # 1. code subspace per layer: centred position means of block states on calibration scenes (8 training pictures each)
    crng = random.Random(74101); NC = 8
    sums = None; cnt = 0
    for _ in range(a.ncal):
        ci = crng.sample(range(len(names)), NC)
        cl = [names[x] for x in ci]; pics = [crng.choice(ctiles[x]) for x in ci]
        forward(1, NC, cl, pics, cl, mode="cap")
        off = np.cumsum([0] + [len(b) for b in ST["blocks"]])
        bm = {l: torch.stack([ST["cap"][l][off[b]:off[b + 1]].mean(0) for b in range(NC)]).cpu().numpy() for l in range(Lh)}
        sums = bm if sums is None else {l: sums[l] + bm[l] for l in range(Lh)}; cnt += 1
    basis, rbasis, var = {}, {}, {}
    g = torch.Generator().manual_seed(74102)
    for l in range(Lh):
        mu = sums[l] / cnt; C = mu - mu.mean(0, keepdims=True)                         # [NC, D]
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        basis[l] = torch.tensor(Vt[:a.d].T, dtype=torch.float32)                         # [D, d]
        q, _ = torch.linalg.qr(torch.randn(C.shape[1], a.d, generator=g))
        rbasis[l] = q
        var[l] = float((S[:a.d] ** 2).sum() / max((S ** 2).sum(), 1e-12))
    D = int(basis[0].shape[0])
    print(f"[{a.tag}] code subspace: d = {a.d} of D = {D} per layer ({a.ncal} calibration scenes)", flush=True)

    # 2. recipients
    rng = random.Random(74100)
    rows = []
    for t_ in range(a.ntest):
        for N in (5, 6):
            ci = rng.sample(range(len(names)), N + SHIFT)
            cl = [names[x] for x in ci[:N]]; pics = [rng.choice(tiles[x]) for x in ci[:N]]
            extra = [rng.choice(tiles[x]) for x in ci[N:]]
            opts = cl[:]; rng.shuffle(opts)
            dons = {}
            for nm, pp in (("pre", extra + pics), ("app", pics + extra)):
                forward(1, N + SHIFT, cl, pp, opts, mode="cap")
                off = np.cumsum([0] + [len(b) for b in ST["blocks"]])
                dons[nm] = (ST["cap"], off)
            for k in range(3, N + 1):
                r = dict(t=t_, N=N, k=k)
                r["NONE"], inp = forward(k, N, cl, pics, opts)
                dst = torch.cat(blocks(inp, N))
                for arm, dn, first in (("FULL", "pre", SHIFT), ("CODE", "pre", SHIFT), ("REST", "pre", SHIFT), ("RAND", "pre", SHIFT), ("APP", "app", 0)):
                    cap, off = dons[dn]
                    src = torch.cat([torch.arange(off[first + b], off[first + b + 1]) for b in range(N)])
                    if len(dst) != len(src):
                        r[arm] = -1; continue
                    r[arm] = forward(k, N, cl, pics, opts, mode="put", don=cap, dst=dst, src=src, arm=("FULL" if arm == "APP" else arm),
                                     basis=(rbasis if arm == "RAND" else basis))[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{a.tag}] {t_ + 1}/{a.ntest} rows {len(rows)}", flush=True)
    for h in hks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, d=a.d, D=D, ncal=a.ncal, var_explained=var, rows=rows), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=20); ap.add_argument("--ncal", type=int, default=32); ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--pixels", type=int, default=224)
    main(ap.parse_args())
