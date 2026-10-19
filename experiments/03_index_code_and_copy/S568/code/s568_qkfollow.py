"""S568: does the retrieval query's attention follow the block that carries the computed target code? A QK readout
under the code manipulations whose behavioural effects are known (27B, reverse questions without a count sentence).
Arms: none; M2 (the last block claims N-1 -> the target code becomes j-1; §593 k = 2: 27/36 land on j-1);
res_frozen (the target block j claims x, blocks after it frozen -> code j is missing; §591); zclaim (another block z,
1 <= z <= j-2, claims code j, blocks after z frozen -> two blocks carry code j). Readout: at every full-attention layer,
the answer token's attention mass on each image block (post-RoPE q, k; head mean) and on the blocks that matter
(j, j-1, z, k), plus the answer.

S564 docstring follows.
S564: which rule reads the END from the block codes? Predictions of three rules for chain-free code changes near the end.

§574 §576 §585: the reverse target is end + 1 - k with the end read from the image blocks' index codes. Lowering the
last block's code moves the target down (S549 m1: j-1, .65), the chain shift of S557 mid gives j-2, but raising the
last block's code alone does NOT give j+1 (S549 p1: mostly j-1 / j). Rules for the end, given the codes c_1..c_N:
    (a) max      the largest code
    (b) cmax     the largest code among blocks that continue their predecessor (c_b = c_{b-1} + 1; block 1 if c_1 = 1)
    (c) last     the last block's code
After the fact, (b) agrees with every earlier manipulation (S545 +2, S549 m1/p1, S557 mid/out); (a) fails p1. Here the
manipulations are chosen so that the rules predict different blocks. Target code T = end + 1 - k; the predicted landing
is the unique block carrying T (no prediction when no block or two blocks carry it). Manipulations (residual at every
layer from a donor showing the same picture at another index; the block after a manipulated block is frozen to its
clean residual so its code stays):
    M2  last block claims N-1            codes [1..N-1, N-1]        all rules -> j-1 (control)
    M1  last block claims N+1            codes [1..N-1, N+1]        (a)(c) j+1   (b) j-1
    M3  block N-1 claims N-3, N frozen   codes [1..N-2, N-3, N]     (a)(c) j     (b) j-2
    M4  block N-1 claims N+1, N frozen   codes [1..N-2, N+1, N]     (a) j+1  (b) j-2  (c) j     (where unique)
Reverse questions without a count sentence (k = 2, 3, 4), N = 6-8.
"""
from __future__ import annotations

import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni
import transformers.models.qwen3_5.modeling_qwen3_5 as Mq

RP = {"layer": None, "on": False, "att": {}}
_orig_rope = Mq.apply_rotary_pos_emb


def _wrap_rope(q, k, cos, sin, unsqueeze_dim=1):
    qe, ke = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    if RP["on"] and RP["layer"] is not None:
        q1 = qe[0, :, -1].detach().float(); k1 = ke[0].detach().float()
        H, Hk = q1.shape[0], k1.shape[0]
        s_ = torch.einsum("hd,htd->ht", q1, k1.repeat_interleave(H // Hk, 0)) / q1.shape[-1] ** 0.5
        RP["att"][RP["layer"]] = torch.softmax(s_, -1).mean(0).cpu()
    return qe, ke


Mq.apply_rotary_pos_emb = _wrap_rope

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s568_qkfollow_{}.json"
CIFAR = "/vlm/assets/cifar10"
MANS = ("M2", "M1", "M3", "M4")
RULES = ("max", "cmax", "last")


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


def codes_of(man, N):
    c = list(range(1, N + 1))
    if man == "M2":
        c[N - 1] = N - 1
    elif man == "M1":
        c[N - 1] = N + 1
    elif man == "M3":
        c[N - 2] = N - 3
    elif man == "M4":
        c[N - 2] = N + 1
    return c


def end_of(rule, c):
    if rule == "max":
        return max(c)
    if rule == "last":
        return c[-1]
    ok = [c[0]] if c[0] == 1 else []
    ok += [c[b] for b in range(1, len(c)) if c[b] == c[b - 1] + 1]
    return max(ok)


def predict(rule, c, k):
    T = end_of(rule, c) + 1 - k
    hits = [b + 1 for b, x in enumerate(c) if x == T]
    return hits[0] if len(hits) == 1 else None


def run(v, tag="job", ntest=12, pixels=224, Ns=(6, 7, 8), ks=(2, 3), seed=56800):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    Lh = v.n_layer
    ST = {"op": None}

    def mk(l):
        def h(m, i, o):
            if ST["op"] is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST["op"] == "cap":
                ST["store"][l] = t[0].detach().clone()
                return o
            t = t.clone()
            for rows, src in ST["puts"]:
                t[0, rows.to(t.device)] = src[l].to(t.device, t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return h
    hks = [v.layers[l].register_forward_hook(mk(l)) for l in range(Lh)]
    full = [i for i, L in enumerate(v.layers) if not hasattr(L, "linear_attn")]

    def on(l):
        def pre(m, args, kwargs):
            RP["layer"] = l
            return None
        return pre

    def off(m, i, o):
        RP["layer"] = None
        return o
    for l in full:
        hks += [v.layers[l].self_attn.register_forward_pre_hook(on(l), with_kwargs=True), v.layers[l].self_attn.register_forward_hook(off)]

    def prompt(k, opts):
        return v.chat([("user", f"{v.IMG*len_[0]}\nWhat is in the {ORD[k-1]} image from the last? "
                                f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)
    len_ = [0]

    def blocks(inp, N):
        return [torch.arange(int(b[0]) - 1, int(b[-1]) + 2) for b in v.image_blocks(inp["input_ids"], N)]

    @torch.no_grad()
    def capture(pics, opts):
        len_[0] = len(pics)
        inp = v.encode(prompt(1, opts), pics)
        ST.update(op="cap", store={})
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["op"] = None
        return ST["store"], blocks(inp, len(pics))

    @torch.no_grad()
    def ask(k, cl, pics, opts, puts=None):
        len_[0] = len(pics)
        inp = v.encode(prompt(k, opts), pics)
        ST.update(op=("put" if puts else None), puts=puts or [])
        RP.update(on=True, att={})
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["op"] = None; RP["on"] = False
        B = blocks(inp, len(pics))
        mass = {l: [float(a[b].sum()) for b in B] for l, a in RP["att"].items()}
        return int(np.argmax([out.logits[0, -1, tid1(c)].item() for c in cl])) + 1, mass

    def sl(store, rows):
        return {l: store[l][rows.to(store[l].device)] for l in store}

    rng = random.Random(seed)
    rows = []
    for t_ in range(ntest):
        for N in Ns:
            ci = rng.sample(range(len(names)), N)
            cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
            extra = [rng.choice(tiles[rng.randrange(len(names))]) for _ in range(3)]
            opts = cl[:]; rng.shuffle(opts)
            clean, B = capture(pics, opts)
            for k in ks:
                j = N + 1 - k
                z = rng.choice(list(range(1, j - 1)))
                x = rng.choice([y for y in range(1, N + 1) if y not in (j, z, j - 1, j + 1)] or [y for y in range(1, N + 1) if y != j])
                donors = {"M2": (pics[:N - 2] + [pics[N - 1]], N - 1, N),
                          "res_frozen": (None, x, j), "zclaim": (None, j, z)}
                puts = {}
                for man, (dp, db, rb) in donors.items():
                    if dp is None:                                               # swap scene: picture rb at index db
                        dp = pics[:]; dp[rb - 1], dp[db - 1] = pics[db - 1], pics[rb - 1]
                    st, Bd = capture(dp, opts)
                    if len(Bd[db - 1]) != len(B[rb - 1]):
                        puts[man] = None; continue
                    pl = [(B[rb - 1], sl(st, Bd[db - 1]))]
                    if rb < N:
                        after = torch.cat(B[rb:])
                        pl.append((after, sl(clean, after)))
                    puts[man] = pl
                    del st
                r = dict(t=t_, N=N, k=k, j=j, z=z, x=x)
                r["none"], r["mass|none"] = ask(k, cl, pics, opts)
                for man in ("M2", "res_frozen", "zclaim"):
                    if puts[man] is None:
                        continue
                    r[man], r[f"mass|{man}"] = ask(k, cl, pics, opts, puts[man])
                rows.append(r)
            del clean
            torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()
    Mq.apply_rotary_pos_emb = _orig_rope

    res = {}
    show = [l for l in full if l >= 31]
    roles = {"none": ("j", "j−1"), "M2": ("j", "j−1"), "res_frozen": ("j", "j−1"), "zclaim": ("j", "z")}
    print(f"\n  역순 정답률(없음, 개수 문장 없음): " + " · ".join(f"k{k} {np.mean([r['none'] == r['j'] for r in rows if r['k'] == k]):.3f}" for k in ks))
    for k in ks:
        R = [r for r in rows if r["k"] == k]
        print(f"\n  k = {k} (n {len(R)})   답: j / j−1 / z     |  층별 답 토큰 주의: 블록 j · 블록 j−1 · 블록 z · 표면 블록 k")
        for arm in ("none", "M2", "res_frozen", "zclaim"):
            A = [r for r in R if arm in r]
            if not A:
                continue
            d = dict(n=len(A), j=float(np.mean([r[arm] == r["j"] for r in A])), jm1=float(np.mean([r[arm] == r["j"] - 1 for r in A])),
                     z=float(np.mean([r[arm] == r["z"] for r in A])))
            d["att"] = {}
            for l in show:
                g = lambda key: float(np.mean([r[f"mass|{arm}"][l][key - 1] for r in A]))
                d["att"][l] = dict(j=float(np.mean([r[f"mass|{arm}"][l][r["j"] - 1] for r in A])),
                                   jm1=float(np.mean([r[f"mass|{arm}"][l][r["j"] - 2] for r in A])),
                                   z=float(np.mean([r[f"mass|{arm}"][l][r["z"] - 1] for r in A])),
                                   k=float(np.mean([r[f"mass|{arm}"][l][r["k"] - 1] for r in A])))
            res[f"k{k}|{arm}"] = d
            print(f"   [{arm}]  답 {d['j']:.3f} / {d['jm1']:.3f} / {d['z']:.3f}", flush=True)
            for l in show:
                a = d["att"][l]
                print(f"      L{l:<2d}  {a['j']:.3f} · {a['jm1']:.3f} · {a['z']:.3f} · {a['k']:.3f}", flush=True)
    print(f"\n  가름 (사전 등록 S568) (w1) k = 2, M2: L51 답 토큰 주의에서 블록 j−1 이 none 대비 +.05 이상 늘고 블록 j 가 줄어든다(질의가 계산된 목표 색인을 따라감).")
    print(f"    (w2) zclaim: L51 의 블록 z 주의가 none 대비 +.03 이상. (w3) res_frozen: L51 의 블록 j 주의가 none 의 .7 배 이하. (w4, 보고) 층별 궤적, k = 3.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=12)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, ntest=a.ntest)
