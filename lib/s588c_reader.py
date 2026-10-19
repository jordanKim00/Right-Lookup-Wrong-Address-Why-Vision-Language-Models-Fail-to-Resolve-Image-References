"""S588c: where should the end of the image sequence be read from? Calibration of the block-index-code reader
(pre-registered in exp/QUEUE.md; no answers and no reverse data -- forward scenes and block positions only).

§612: every error of the rebinding came from reading the end (correct end -> .85 = the prototype ceiling), and the
block-mean probe decodes the compressed high codes (8-10) only about half of the time. Candidates: layer x site
(block-mean of image tokens / vision-start marker / vision-end marker / last image token), combinations
(mean + end marker, two layers), PCA 64 / 128, categorical vs cumulative (ordinal) logistic. The end of a scene is the
code of the block with the largest expected code. Selection (fixed in advance): best held-out end accuracy on N 6-10.
"""
from __future__ import annotations

import argparse, itertools, json, pickle, random, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s588c_reader_{}.json"
READER = "/vlm/results/s588c_reader_{}.pt"
CIFAR = "/vlm/assets/cifar10"
C = 10
SITES = ("mean", "vstart", "vend", "lastimg")


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
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC) for a in v_] for i, v_ in by.items()}


def fit(F, ytr, nc, kind, m_, s_, V, steps=800):
    """F: training features already standardised and projected on the top principal axes (GPU); V: [dim, >= nc]."""
    dev = F.device
    F = F[:, :nc]; V = V[:, :nc]
    ytr = ytr.to(dev)
    cnt = torch.bincount(ytr - 1, minlength=C).float().clamp(min=1)
    wcls = (cnt.sum() / (C * cnt))
    if kind == "cat":
        W = torch.zeros(nc, C, device=dev, requires_grad=True); b = torch.zeros(C, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            loss = torch.nn.functional.cross_entropy(F @ W + b, ytr - 1, weight=wcls)
            opt.zero_grad(); loss.backward(); opt.step()
        P = dict(kind=kind, m=m_.cpu(), s=s_.cpu(), V=V.cpu(), W=W.detach().cpu(), b=b.detach().cpu())
    else:                                                                         # cumulative link: P(code > c) = sig(w.x - th_c)
        w = torch.zeros(nc, device=dev, requires_grad=True); th = torch.linspace(-2, 2, C - 1, device=dev).clone().requires_grad_(True)
        opt = torch.optim.Adam([w, th], lr=1e-2, weight_decay=1e-4)
        tgt = (ytr[:, None] > torch.arange(1, C, device=dev)[None]).float()     # [n, C-1]
        sw = wcls[ytr - 1][:, None]
        for _ in range(steps):
            z = (F @ w)[:, None] - th[None]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, tgt, reduction="none") * sw).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        P = dict(kind=kind, m=m_.cpu(), s=s_.cpu(), V=V.cpu(), w=w.detach().cpu(), th=th.detach().cpu())
    return P


def probs(P, X):
    F = ((X - P["m"]) / P["s"]) @ P["V"]
    if P["kind"] == "cat":
        return torch.softmax(F @ P["W"] + P["b"], -1)
    g = torch.sigmoid((F @ P["w"])[:, None] - P["th"][None])                     # P(code > c), c = 1..C-1
    g = torch.cat([torch.ones(len(F), 1), g, torch.zeros(len(F), 1)], 1)
    return (g[:, :-1] - g[:, 1:]).clamp(min=1e-6)


def run(v, tag, Ls, ntr=500, nte=200, pixels=224, seed=58830, varsize=False, fixed_reader=None, rope_aug=False, natural=False):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    ST = {}

    def hook(l):
        def h(m, i, o):
            if l in ST.get("Ls", ()):
                t = o[0] if isinstance(o, tuple) else o
                f = []
                for b in ST["B"]:
                    f.append(torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                          t[0, int(b[-1])].float()]))
                ST["F"][l] = torch.stack(f).half().cpu()                         # [N, 4, d]
            return o
        return h
    hooks = [v.layers[l].register_forward_hook(hook(l)) for l in Ls]
    rng = random.Random(seed)
    INET_FILES = []
    if natural:                                                                   # S588c4: natural photos, random size and aspect
        import glob
        INET_FILES = sorted(glob.glob("/vlm/assets/imagenette2-160/val/*/*.JPEG"))
    # RoPE position perturbation (S588c3): swap the position ids of two random equal-length blocks
    inner = v.model.model
    orig_rope_idx = type(inner).get_rope_index
    POS = {"swap": None}

    def rope_idx(self_, *a, **kw):
        out = orig_rope_idx(self_, *a, **kw)
        if POS["swap"] is None:
            return out
        pid, deltas = out
        ia, ib = (x.to(pid.device) for x in POS["swap"])
        pid = pid.clone(); tmp = pid[..., ia].clone(); pid[..., ia] = pid[..., ib]; pid[..., ib] = tmp
        return pid, deltas
    if rope_aug:
        type(inner).get_rope_index = rope_idx
    SW = {"flag": []}
    t0 = time.time()

    @torch.no_grad()
    def data(n):
        X = {l: [] for l in Ls}; y = []; sid = []
        for i in range(n):
            N = rng.randint(3, 10)
            ci = rng.sample(range(len(names)), N)
            cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
            if varsize:
                pics = [p_.resize((sz, sz), Image.BICUBIC) for p_, sz in zip(pics, [rng.choice(range(128, 449, 32)) for _ in pics])]
            if natural and INET_FILES:
                for ii in range(N):
                    if rng.random() < .7:                                        # 70% natural photos, the rest CIFAR tiles
                        im = Image.open(rng.choice(INET_FILES)).convert("RGB")
                        pics[ii] = im.resize((rng.choice(range(128, 449, 32)), rng.choice(range(128, 449, 32))), Image.BICUBIC)
            opts = cl[:]; rng.shuffle(opts)
            k = rng.randint(1, N)
            text = v.chat([("user", f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(opts)}.")],
                          prefix_assistant=THINK)
            inp = v.encode(text, pics)
            Bl = v.image_blocks(inp["input_ids"], N)
            ST.update(Ls=set(Ls), B=[torch.tensor(b) for b in Bl], F={})
            POS["swap"] = None
            if rope_aug and rng.random() < .5:
                a_, b_ = rng.sample(range(N), 2)
                ra = torch.arange(int(Bl[a_][0]) - 1, int(Bl[a_][-1]) + 2); rb = torch.arange(int(Bl[b_][0]) - 1, int(Bl[b_][-1]) + 2)
                if len(ra) == len(rb):
                    POS["swap"] = (ra, rb)
            SW["flag"] += [POS["swap"] is not None] * N
            v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
            POS["swap"] = None
            for l in Ls:
                X[l].append(ST["F"][l])
            y += list(range(1, N + 1)); sid += [i] * N
            ST.clear()
            if (i + 1) % 100 == 0:
                print(f"[{tag}] 장면 {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        return {l: torch.cat(X[l]).float() for l in Ls}, torch.tensor(y), torch.tensor(sid)
    Xtr, ytr, _ = data(ntr); SW["flag"] = []; Xte, yte, ste = data(nte)
    swf = torch.tensor(SW["flag"]) if rope_aug else torch.zeros(len(yte), dtype=torch.bool)
    fixed_res = None
    if fixed_reader:                                                              # the fixed-size reader applied to these scenes
        FR = torch.load(f"/vlm/results/s588c_reader_{fixed_reader}.pt", weights_only=False)
        Xf = torch.cat([torch.cat([Xte[l][:, SITES.index(s_)] for s_ in FR["sites"]], 1) for l in FR["layers"]], 1)
        prf = probs(FR["P"], Xf); codef = prf.argmax(1) + 1; Ef = prf @ torch.arange(1, C + 1).float(); endsf, endsf_sw = [], []
        for sc in ste.unique().tolist():
            idx = (ste == sc).nonzero().flatten()
            if len(idx) >= 6:
                (endsf_sw if bool(swf[idx[0]]) else endsf).append(int(codef[idx[int(Ef[idx].argmax())]]) == len(idx))
        fixed_res = dict(exact=float((codef == yte).float().mean()), end_acc=float(np.mean(endsf)),
                         end_acc_swapped=float(np.mean(endsf_sw)) if endsf_sw else None)
        print(f"[{tag}] 기존 읽기({fixed_reader})를 이 장면에: 부호 정확 {fixed_res['exact']:.3f} · 끝(자연) {fixed_res['end_acc']:.3f}"
              f" · 끝(위치 교란) {fixed_res['end_acc_swapped']}", flush=True)
    for h in hooks:
        h.remove()

    def feat(X, l, sites):
        return torch.cat([X[l][:, SITES.index(s_)] for s_ in sites], 1)
    cands = []
    for l in Ls:
        for s_ in SITES:
            cands.append(((l,), (s_,)))
        cands.append(((l,), ("mean", "vend")))
    for la, lb in itertools.combinations(Ls, 2):
        cands.append(((la, lb), ("mean",)))
        cands.append(((la, lb), ("mean", "vend")))
    res = []
    best = None
    gdev = v.device
    for ci_, (layers, sites) in enumerate(cands):
        Ftr = torch.cat([feat(Xtr, l, sites) for l in layers], 1).to(gdev); Fte = torch.cat([feat(Xte, l, sites) for l in layers], 1)
        m_, s_ = Ftr.mean(0), Ftr.std(0) + 1e-4
        Z = (Ftr - m_) / s_
        V = torch.linalg.svd(Z, full_matrices=False)[2][:128].T                  # one decomposition per feature set
        FZ = Z @ V
        del Z, Ftr
        for nc in (64, 128):
            for kind in ("cat", "ord"):
                P = fit(FZ, ytr, nc, kind, m_, s_, V)
                pr = probs(P, Fte)
                code = pr.argmax(1) + 1
                ex = float((code == yte).float().mean())
                per = [float((code[yte == c] == c).float().mean()) for c in range(1, C + 1)]
                E = pr @ torch.arange(1, C + 1).float()
                ends, ends_sw = [], []
                for sc in ste.unique().tolist():
                    idx = (ste == sc).nonzero().flatten()
                    N = len(idx)
                    if N < 6:
                        continue
                    bstar = idx[int(E[idx].argmax())]
                    (ends_sw if bool(swf[idx[0]]) else ends).append(int(code[bstar]) == N)
                r = dict(layers=list(layers), sites=list(sites), nc=nc, kind=kind, exact=ex, per_code=per,
                         end_acc=float(np.mean(ends)), n_end=len(ends),
                         end_acc_swapped=float(np.mean(ends_sw)) if ends_sw else None, n_swapped=len(ends_sw))
                if rope_aug:
                    r["select"] = r["end_acc_swapped"] if r["end_acc"] >= .99 else -1.0
                else:
                    r["select"] = r["end_acc"]
                res.append(r)
                if best is None or r["select"] > best[0]["select"] + 1e-9:
                    best = (r, P)
        del FZ, V
        torch.cuda.empty_cache()
        print(f"[{tag}] 후보 {ci_+1}/{len(cands)}  ({time.time()-t0:.0f}s)", flush=True)
    res.sort(key=lambda r: -r["select"])
    print(f"\n[{tag}] 끝 읽기 후보 상위 10 (보류 장면 N 6-10 의 ê = N):")
    for r in res[:10]:
        print(f"  L{'+'.join(map(str, r['layers'])):<6s} {'+'.join(r['sites']):<12s} PCA{r['nc']:<4d} {r['kind']}  끝 {r['end_acc']:.3f} (교란 {r['end_acc_swapped']})  부호 정확 {r['exact']:.3f}"
              f"  부호 8/9/10 {r['per_code'][7]:.2f}/{r['per_code'][8]:.2f}/{r['per_code'][9]:.2f}", flush=True)
    base = [r for r in res if r["layers"] == [16] and r["sites"] == ["mean"] and r["nc"] == 64 and r["kind"] == "cat"]
    if base:
        print(f"  (S588a 와 같은 꼴 L16 mean PCA64 cat: 끝 {base[0]['end_acc']:.3f})")
    r, P = best
    print(f"\n  선택: L{'+'.join(map(str, r['layers']))} {'+'.join(r['sites'])} PCA{r['nc']} {r['kind']} — 끝 {r['end_acc']:.3f} (교란 {r['end_acc_swapped']})", flush=True)
    torch.save(dict(P=P, layers=r["layers"], sites=r["sites"], SITES=SITES), READER.format(tag))
    json.dump(dict(tag=tag, Ls=Ls, res=res, chosen=r, varsize=varsize, fixed_reader=fixed_reader, fixed_res=fixed_res), open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}  {READER.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--Ls", type=int, nargs="+", required=True)
    ap.add_argument("--ntr", type=int, default=500)
    ap.add_argument("--nte", type=int, default=200)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--varsize", action="store_true")
    ap.add_argument("--fixed_reader", default=None)
    ap.add_argument("--rope_aug", action="store_true")
    ap.add_argument("--natural", action="store_true")
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, a.tag, a.Ls, ntr=a.ntr, nte=a.nte, varsize=a.varsize, fixed_reader=a.fixed_reader, rope_aug=a.rope_aug, natural=a.natural)
