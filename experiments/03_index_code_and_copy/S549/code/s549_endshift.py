"""S549: is the reverse target computed as  end + 1 - k  with end = the largest block index? Move ONLY the last block's own
index code (-1 or +1); every other block keeps its codes.

§574: with no count sentence the end is the largest index code carried by the blocks. Rule under test: target index =
end + 1 - k, then an absolute-index lookup (§571). Recipient pictures P1..PN. Donors for the last block (residuals at
every layer, block with its vision markers):
    m1  [P1 .. P(N-2), PN]        PN sits at index N-1  -> end N-1 -> target j-1 for every k   (prediction: offset -1)
    p1  [P1 .. P(N-1), X, PN]     PN sits at index N+1  -> end N+1 -> target j+1 (exists when j+1 <= N-1, i.e. k >= 3)
    id  [P1 .. PN]                identity control (no change)
Reverse questions without a count sentence (the codes alone set the end, §574); forward questions as a control (only
questions about the last image can move).

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

import os
import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s549_endshift_{}.json"
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


def run(v, tag="job", ntest=8, pixels=224, Ns=(6, 7), ks=(2, 3, 4), seed=54900, shift=2):
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
        return v.chat([("user", f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{tail}? "
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
            ci = rng.sample(range(len(names)), N + 1)
            cl = [names[x] for x in ci[:N]]; pics = [rng.choice(tiles[x]) for x in ci[:N]]
            X = rng.choice(tiles[ci[N]])
            opts = cl[:]; rng.shuffle(opts)
            donors = {"m1": pics[:N - 2] + [pics[N - 1]], "p1": pics[:N - 1] + [X, pics[N - 1]], "id": pics[:]}
            dons = {}
            for nm, pp in donors.items():
                _, inp_d = forward(False, 1, len(pp), cl, pp, opts, mode="cap")
                off = np.cumsum([0] + [len(b) for b in ST["blocks"]])
                dons[nm] = (ST["cap"], off, len(pp))
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
                    for dn, (cap, off, Nd) in dons.items():
                        dst = Brec[N - 1]
                        src = torch.arange(off[Nd - 1], off[Nd])                  # the donor's last block (PN)
                        if len(dst) != len(src):
                            r[f"{nm_}|{dn}"] = -1; continue
                        r[f"{nm_}|{dn}"] = forward(rev, k, N, cl, pics, opts, mode="put", don=cap, dst=dst, src=src)[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()

    res = {}
    rng_ = np.random.default_rng(0)
    print(f"\n  팔     역순: j / j−1 / j+1 / k     짝지은 평균 어긋남 이동 [95%]       전방 k")
    for arm in ("none", "id", "m1", "p1"):
        x = [r[f"R|{arm}"] for r in rows]
        d = dict(j=float(np.mean([a == r["j"] for a, r in zip(x, rows)])), jm1=float(np.mean([a == r["j"] - 1 for a, r in zip(x, rows)])),
                 jp1=float(np.mean([a == r["j"] + 1 for a, r in zip(x, rows)])), k=float(np.mean([a == r["k"] for a, r in zip(x, rows)])),
                 F=float(np.mean([r[f"F|{arm}"] == r["k"] for r in rows])))
        sh = np.array([np.clip(r[f"R|{arm}"] - r["j"], -3, 3) - np.clip(r["R|none"] - r["j"], -3, 3) for r in rows], float)
        bs = [sh[rng_.integers(0, len(sh), len(sh))].mean() for _ in range(4000)]
        d["shift"] = [float(sh.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[arm] = d
        print(f"  {arm:>4s}   {d['j']:.3f} / {d['jm1']:.3f} / {d['jp1']:.3f} / {d['k']:.3f}     {d['shift'][0]:+.2f} [{d['shift'][1]:+.2f}, {d['shift'][2]:+.2f}]       {d['F']:.3f}", flush=True)
    K3 = [r for r in rows if r["k"] >= 3]
    res["p1_k3_jp1"] = float(np.mean([r["R|p1"] == r["j"] + 1 for r in K3])) if K3 else float("nan")
    print(f"  (k >= 3 문항: p1 의 j+1 착지 {res['p1_k3_jp1']:.3f}, n {len(K3)})")
    print(f"\n  가름 (사전 등록 S549) (e0) id 는 없음과 .05 안. (e1) m1: 짝지은 이동 < 0(구간 상한 < 0)이고 j−1 착지 >= 없음의 j 착지 − .15 → 끝 = 가장 큰 색인, 목표 = 끝+1−k.")
    print(f"    (e2) p1: k >= 3 문항의 j+1 착지 >= .30. (e3) 전방은 마지막 블록을 묻지 않으므로 .05 안.")
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
