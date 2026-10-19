"""S474b: forward-donor full-state patching along the route. Where is the reverse question's address still a bare
ordinal, where is it bound to its direction, and where has retrieved content replaced it?
Parameter-free replacement for S474 (DAS back-propagation through 27B ran out of memory on three cards).

Receiver: "There are N images in a row. What is in the k-th image from the last?" (N 7-9, k 2-4; the S463a/S471b
count condition). Donor: the FORWARD question "What is in the x-th image?" over a permutation of the receiver's
classes (other photos), built so the donor's retrieved class sits at a known receiver position y. Option lists are
shuffled independently of image order (§152, §359). The donor state at one site (ord / "?" / answer token) and
layer replaces the receiver's. Landing readings (exclusive; x, N+1-x, j, k, y are distinct by construction and x is
never the middle position, so N+1-x is never x+-1):
    x        the site holds an address already bound to its direction (the forward donor's finished address)
    N+1-x    the site holds a bare ordinal; the receiver applies its own reverse downstream
    j        no effect (the receiver's own correct answer, j = N+1-k)
    k        surface
    y        content: the site already holds the donor's retrieved class
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
OUT = "/vlm/results/s474b_fwddonor_{}.json"
CIFAR = "/vlm/assets/cifar10"
SITES = ("ord", "qm", "last")
CATS = ("x", "rev_x", "own_j", "surf_k", "cont_y", "near_x", "near_rev", "other")


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


def classify(a, r):
    """Exclusive reading of one landing a for pair r."""
    prim = dict(x=r["x"], rev_x=r["N"] + 1 - r["x"], own_j=r["j"], surf_k=r["k"], cont_y=r["y"])
    for c, p in prim.items():
        if a == p:
            return c
    if abs(a - r["x"]) == 1:
        return "near_x"
    if abs(a - (r["N"] + 1 - r["x"])) == 1:
        return "near_rev"
    return "other"


def run(model, tag="job", npair=40, pixels=224, Ns=(7, 8, 9), ks=(2, 3, 4), seed=47450, rels=None):
    v = model
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    cidx = {n: i for i, n in enumerate(names)}
    nl = v.n_layer
    RELS = rels or [0.05, 0.13, 0.2, 0.3, 0.37, 0.46, 0.55, 0.63, 0.72, 0.8, 0.9, 1.0]
    LS = sorted({int(round(r * (nl - 1))) for r in RELS})
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
    im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
    print(f"  결과 {OUT.format(tag)}  층 {LS}", flush=True)
    CAP, PATCH, GOT = {}, {}, {}

    def mk(li):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if li in PATCH:
                p, vec = PATCH[li]
                t = t.clone(); t[0, p] = vec.to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            if li in CAP:
                GOT[li] = t[0, CAP[li]].detach().float().cpu().clone()
            return o
        return h
    hooks = [v.layers[li].register_forward_hook(mk(li)) for li in LS]

    def build(q, N, cl, pics, rev):
        opts = cl[:]; rng.shuffle(opts)                      # option order independent of image order
        tail = " from the last" if rev else ""
        txt = v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORD[q-1]} image{tail}? "
                               f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)
        inp = v.encode(txt, pics)
        ids = inp["input_ids"][0].tolist()
        turn = max(i for i, t in enumerate(ids) if t == im_start)
        o = max(i for i, t in enumerate(ids[:turn]) if t == tid1(ORD[q - 1]))
        qm = next(i for i in range(o + 1, turn) if "?" in v.tok.decode([ids[i]]))
        return inp, dict(ord=o, qm=qm, last=len(ids) - 1)

    @torch.no_grad()
    def forward(inp, pos, cl, capture=False, patch=None):
        CAP.clear(); PATCH.clear(); GOT.clear()
        if capture:
            for li in LS:
                CAP[li] = [pos[s] for s in SITES]
        if patch:
            li, s, vec = patch
            PATCH[li] = (pos[s], vec)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        a = int(np.argmax([sc(out.logits[0, -1], c).item() for c in cl])) + 1   # landing = image position
        st = ({li: {s: GOT[li][n_] for n_, s in enumerate(SITES)} for li in LS} if capture else None)
        CAP.clear(); PATCH.clear()
        return a, st

    rng = random.Random(seed)
    rows = []
    while len(rows) < npair:
        N = rng.choice(Ns); k = rng.choice(ks); j = N + 1 - k
        xs = [p for p in range(2, N) if p not in (k, j) and abs(N + 1 - 2 * p) >= 2]
        if not xs:
            continue
        x = rng.choice(xs); rx = N + 1 - x
        ys = [p for p in range(1, N + 1) if p not in (x, rx, j, k)]
        far = [p for p in ys if min(abs(p - x), abs(p - rx), abs(p - j)) >= 2]
        y = rng.choice(far or ys)
        ci = rng.sample(range(len(names)), N)
        cl = [names[c] for c in ci]
        pics = [rng.choice(tiles[c]) for c in ci]
        rest = [c for c in cl if c != cl[y - 1]]; rng.shuffle(rest)
        cld = rest[:x - 1] + [cl[y - 1]] + rest[x - 1:]            # donor's x-th class = receiver's y-th class
        picd = [rng.choice(tiles[cidx[c]]) for c in cld]            # other photos of the same classes
        inp_r, pos_r = build(k, N, cl, pics, True)
        base, _ = forward(inp_r, pos_r, cl)
        inp_d, pos_d = build(x, N, cld, picd, False)
        dans, st = forward(inp_d, pos_d, cld, capture=True)
        rec = dict(N=N, k=k, j=j, x=x, y=y, base=base, donor_ok=int(dans == x), grid={})
        for li in LS:
            for s in SITES:
                rec["grid"][f"{s}|{li}"] = forward(inp_r, pos_r, cl, patch=(li, s, st[li][s]))[0]
        rows.append(rec)
        torch.cuda.empty_cache()
        print(f"[{tag}] 쌍 {len(rows)}/{npair}  무개입 {base} (j {j}, k {k})  기증 {dans} (x {x})", flush=True)
    for h in hooks:
        h.remove()
    res = {"layers": LS, "base_acc": float(np.mean([r["base"] == r["j"] for r in rows])),
           "base_surf": float(np.mean([r["base"] == r["k"] for r in rows])),
           "donor_acc": float(np.mean([r["donor_ok"] for r in rows]))}
    clean = [r for r in rows if r["donor_ok"]]
    print(f"\n  무개입 역순 정답 {res['base_acc']:.3f}  표면 {res['base_surf']:.3f}  기증(전방) 정답 {res['donor_acc']:.3f}")
    print(f"  읽기: x=방향이 묶인 주소  N+1−x=맨 서수(하류 역변환)  j=효과 없음  k=표면  y=내용  (배타, 기증 정답 {len(clean)} 쌍)")
    for s in SITES:
        print(f"   [{s}]")
        for li in LS:
            c = {cat: 0 for cat in CATS}
            for r in clean:
                c[classify(r["grid"][f"{s}|{li}"], r)] += 1
            fr = {cat: c[cat] / max(len(clean), 1) for cat in CATS}
            res[f"{s}/L{li}"] = fr
            print(f"     L{li:<3d} " + "  ".join(f"{cat} {fr[cat]:.2f}" for cat in CATS), flush=True)
    print(f"\n  가름 (사전 등록 S474b) (w0) 마지막 층 답 토큰 → y >= .8, 가장 이른 층 서수 → 27B N+1−x >= .5 / 4B x >= .5.")
    print(f"    (w1) 27B '?' (상대 .45-.63): N+1−x > x 이고 >= .3 → 맨 서수 중계, 방향은 하류. x > N+1−x 이고 >= .3 → '?' 에서 방향이 묶인다.")
    print(f"    (w2) 답 토큰: y 전에 x(또는 N+1−x) >= .3 인 층이 있으면 검색 전 주소 구간이 답 토큰에 있다. j → y 로 바로 가면 없다.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--npair", type=int, default=40)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--Ns", type=int, nargs="+", default=[7, 8, 9])
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, npair=a.npair, Ns=tuple(a.Ns))
