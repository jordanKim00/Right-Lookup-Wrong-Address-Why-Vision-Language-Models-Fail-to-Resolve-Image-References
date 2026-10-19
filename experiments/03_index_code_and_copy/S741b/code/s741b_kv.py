"""S741b: does the lookup read the index code from the image keys?  (follow-up to S741, reply to the mock review W2, 2026-10-11;
pre-registration in knowledge/DECISIONS.md before any S741b data; analysis exp/s741b_analyze.py, frozen with it.)
S741 showed that the S545 code shift is not carried by the seven-dimensional span of the block position means.  Here the same donor
(two new pictures in front, so blocks 1..N hold codes 3..N+2) is split by what attention reads from the image tokens instead:
    NONE   no edit
    FULL   every image-block token's residual state <- the donor's, at every layer                     (S545)
    KEY    the key projection (k_proj output, before key norm and rotary) of every image-block token <- the donor's, in every
           full-attention layer; residual states and values stay the recipient's
    VAL    the value projection (v_proj output) likewise; keys stay the recipient's
    KV     both
    AKEY   KEY with the donor whose two new pictures are appended after the scene (control: codes unchanged)
Questions, scenes and readout as S741 (forward "What is in the k-th image?", N in {5, 6}, k in {3, ..., N}, seed 74100).
Usage: python s741b_kv.py MODEL TAG [--ntest 20]
"""

from __future__ import annotations

import argparse, os, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/r3/s741b_{}_kv.json"
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
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    # answer readout (S789): R3_READ=bare reads the first bare token of the class word, the form the model generates (FINDINGS §846)
    aid1 = (lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]) if os.environ.get("R3_READ") == "bare" else tid1
    def sc1(lg, s):                                      # S794: R3_READ=max scores a class by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid1(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    Lh = v.n_layer
    FA = [l for l in range(Lh) if getattr(v.layers[l], "self_attn", None) is not None]
    ST = {"mode": None}

    def mk_res(l):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if ST["mode"] == "cap":
                ST["cap"][("res", l)] = t[0, ST["span"]].detach().clone(); return o
            if ST["mode"] == "put" and "res" in ST["what"]:
                t = t.clone(); D = ST["don"][("res", l)]
                t[0, ST["dst"].to(t.device)] = D[ST["src"].to(D.device)].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return hook

    def mk_proj(l, kind):
        def hook(m, i, o):
            if ST["mode"] == "cap":
                ST["cap"][(kind, l)] = o[0, ST["span"]].detach().clone(); return o
            if ST["mode"] == "put" and kind in ST["what"]:
                o = o.clone(); D = ST["don"][(kind, l)]
                o[0, ST["dst"].to(o.device)] = D[ST["src"].to(D.device)].to(o.device, o.dtype)
                return o
            return o
        return hook
    hks = [v.layers[l].register_forward_hook(mk_res(l)) for l in range(Lh)]
    hks += [v.layers[l].self_attn.k_proj.register_forward_hook(mk_proj(l, "k")) for l in FA]
    hks += [v.layers[l].self_attn.v_proj.register_forward_hook(mk_proj(l, "v")) for l in FA]

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
        return int(np.argmax([sc1(out.logits[0, -1], c).item() for c in cl])) + 1, inp

    rng = random.Random(74100)
    rows = []
    ARMS = (("FULL", "pre", ("res",)), ("KEY", "pre", ("k",)), ("VAL", "pre", ("v",)), ("KV", "pre", ("k", "v")), ("AKEY", "app", ("k",)))
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
                for arm, dn, what in ARMS:
                    cap, off = dons[dn]; first = SHIFT if dn == "pre" else 0
                    src = torch.cat([torch.arange(off[first + b], off[first + b + 1]) for b in range(N)])
                    if len(dst) != len(src):
                        r[arm] = -1; continue
                    r[arm] = forward(k, N, cl, pics, opts, mode="put", don=cap, dst=dst, src=src, what=what)[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{a.tag}] {t_ + 1}/{a.ntest} rows {len(rows)}", flush=True)
    for h in hks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, fa_layers=FA, rows=rows), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=20); ap.add_argument("--pixels", type=int, default=224)
    main(ap.parse_args())
