"""S608: where does the model carry the image index across prompt layouts? (pre-registered, exp/QUEUE.md S608)
A linear probe of the block index is fitted on images-first scenes at every (layer, site) and tested on held-out scenes
in four layouts: images first / "Image k is <image>." / "Picture k: <image>" / images separated by newlines only
(the same scenes in every layout). The module's fixed reader is applied to the same scenes. Images: procedural, training pool.
Usage: python s608_layout.py MODEL TAG --reader R [--ntr 300] [--nte 80] [--step 2]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from s588c_reader import fit, probs, SITES, C
from r3_proc import proc_image
from r3ref import ORD

LAYOUTS = ("first", "imgis", "picture", "newline")


def body(v, lay, N, q):
    if lay == "first":
        return v.IMG * N + "\n" + q
    if lay == "imgis":
        return " ".join(f"Image {i+1} is {v.IMG}." for i in range(N)) + "\n" + q
    if lay == "picture":
        return "\n".join(f"Picture {i+1}: {v.IMG}" for i in range(N)) + "\n" + q
    return "\n".join(v.IMG for _ in range(N)) + "\n" + q


def main(v, tag, reader, ntr, nte, step):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    R = torch.load(f"/vlm/results/s588c_reader_{reader}.pt", weights_only=False) if reader else None
    Ls = sorted(set(range(0, len(v.layers), step)) | (set(R["layers"]) if R else set()))
    ST = {}

    def hook(l):
        def h(m, i, o):
            if ST.get("B") is not None:
                t = o[0] if isinstance(o, tuple) else o
                ST["F"][l] = torch.stack([torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                                       t[0, int(b[-1])].float()]) for b in ST["B"]]).half().cpu()
            return o
        return h
    hooks = [v.layers[l].register_forward_hook(hook(l)) for l in Ls]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(60800)
    t0 = time.time()

    @torch.no_grad()
    def scenes(n, lays):
        X = {(lay, l): [] for lay in lays for l in Ls}; y = []
        for i in range(n):
            N = rng.randint(2, 8)
            ps = rng.sample(pids, N)
            rs = [proc_image(p) for p in ps]; pics, labs = [r_[0] for r_ in rs], [r_[1] for r_ in rs]
            k = rng.randint(1, N); opts = list(labs); rng.shuffle(opts)
            q = f"What is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(opts)}."
            for lay in lays:
                inp = v.encode(v.chat([("user", body(v, lay, N, q))], prefix_assistant=THINK), pics)
                ST.update(B=[torch.tensor(b) for b in v.image_blocks(inp["input_ids"], N)], F={})
                v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
                for l in Ls:
                    X[(lay, l)].append(ST["F"][l])
                ST["B"] = None
            y += list(range(1, N + 1))
            if (i + 1) % 50 == 0:
                print(f"[{tag}] 장면 {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        return {k_: torch.cat(v_).float() for k_, v_ in X.items()}, torch.tensor(y)
    Xtr, ytr = scenes(ntr, ("first",))
    Xte, yte = scenes(nte, LAYOUTS)
    for h in hooks:
        h.remove()
    out = {}
    # the module's fixed reader on every layout
    fixed = {}
    for lay in (LAYOUTS if R else ()):
        Xf = torch.cat([torch.cat([Xte[(lay, l)][:, SITES.index(s_)] for s_ in R["sites"]], 1) for l in R["layers"]], 1)
        fixed[lay] = float(((probs(R["P"], Xf).argmax(1) + 1) == yte).float().mean())
    if R:
        print(f"[{tag}] 모듈의 끝 읽기(L{R['layers']} {R['sites']}) 부호 정확: " + " · ".join(f"{k_} {x:.3f}" for k_, x in fixed.items()), flush=True)
    gdev = v.device
    for l in Ls:
        for s_ in SITES:
            Ftr = Xtr[("first", l)][:, SITES.index(s_)].to(gdev)
            m_, sd_ = Ftr.mean(0), Ftr.std(0) + 1e-4
            Z = (Ftr - m_) / sd_
            V = torch.linalg.svd(Z, full_matrices=False)[2][:64].T
            P = fit(Z @ V, ytr, 64, "cat", m_, sd_, V)
            acc = {lay: float(((probs(P, Xte[(lay, l)][:, SITES.index(s_)]).argmax(1) + 1) == yte).float().mean()) for lay in LAYOUTS}
            out[f"L{l}|{s_}"] = acc
        print(f"[{tag}] L{l:>2d} " + " | ".join(f"{s_} " + "/".join(f"{out[f'L{l}|{s_}'][lay]:.2f}" for lay in LAYOUTS) for s_ in SITES), flush=True)
    inv = sorted(out.items(), key=lambda kv: -min(kv[1][lay] for lay in LAYOUTS))
    print(f"\n[{tag}] 배치 불변 상위(네 배치 중 최소 정확): " + " · ".join(f"{k_} {min(a.values()):.3f}" for k_, a in inv[:8]), flush=True)
    json.dump(dict(tag=tag, reader=reader, fixed=fixed, probes=out, layouts=LAYOUTS, ntr=ntr, nte=nte), open(f"/vlm/results/r3/{tag}_s608.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--reader", default=None)
    ap.add_argument("--ntr", type=int, default=300); ap.add_argument("--nte", type=int, default=80); ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    main(v, a.tag, a.reader, a.ntr, a.nte, a.step)
