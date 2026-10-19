"""S709-cal: model-internal R³ readers for Qwen3.5-27B under the existing R³ calibration protocol (no end-relative answers anywhere).
(1) End reader (S609 protocol, exp/s609_reader.py): block index codes at the vision_end site, fitted on images-first procedural scenes
    (N 2-10, forward questions "What is in the k-th image?"), checked on held-out scenes in the four S608 layouts; candidate layers
    L35 / 39 / 43 / 47 / 51 (27B; relative .56-.81) -- the layer with the best mean held-out accuracy is kept.  Saved as
    /vlm/results/s588c_reader_s709_{base}.pt (the format r3_build / r3_repair load).
(2) R³ calibration (exp/r3_build.calibrate, unchanged): ordinal prototypes Mb at L_b, detector rule, frame anchors (start / end / last)
    and their thresholds, from forward texts and "from the start / beginning / end / last" phrases with procedural images.
    L_b = L_w = 3 (27B early ordinal interface, S708).  Saved as /vlm/results/r3/s709_{base}_cal.pt.
Usage: python s709_cal.py MODEL BASE [--ntr 500] [--nte 60] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from s588c_reader import fit, probs, SITES
from r3_proc import proc_image
from r3ref import ORD
from s608_layout import LAYOUTS, body
import r3_build
from r3_build_md import ModelMD


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    CAND = sorted({int(round(r * (nl - 1))) for r in (35 / 63, 39 / 63, 43 / 63, 47 / 63, 51 / 63)})
    ST = {}

    def mk(L):
        def hook(m, i, o):
            if ST.get("B") is not None:
                t = o[0] if isinstance(o, tuple) else o
                ST["F"][L] = torch.stack([t[0, int(b[-1]) + 1].float() for b in ST["B"]]).cpu()   # vision_end of each block
            return o
        return hook
    hks = [v.layers[L].register_forward_hook(mk(L)) for L in CAND]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(70910)
    t0 = time.time()

    @torch.no_grad()
    def scenes(n, lays):
        X = {(lay, L): [] for lay in lays for L in CAND}; y = []
        for i in range(n):
            N = rng.randint(2, 10)
            rs = [proc_image(p) for p in rng.sample(pids, N)]
            pics, labs = [r_[0] for r_ in rs], [r_[1] for r_ in rs]
            k = rng.randint(1, N); opts = list(labs); rng.shuffle(opts)
            q = f"What is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(opts)}."
            for lay in lays:
                inp = v.encode(v.chat([("user", body(v, lay, N, q))], prefix_assistant=THINK), pics)
                ST.update(B=[torch.tensor(b) for b in v.image_blocks(inp["input_ids"], N)], F={})
                v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
                for L in CAND:
                    X[(lay, L)].append(ST["F"][L])
                ST["B"] = None
            y += list(range(1, N + 1))
            if (i + 1) % 100 == 0:
                print(f"[s709cal] 장면 {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        return {k_: torch.cat(x) for k_, x in X.items()}, torch.tensor(y)
    rfile = f"/vlm/results/s588c_reader_s709_{a.base}.pt"
    if a.reuse_reader and __import__("os").path.exists(rfile):                    # reader already fitted (same protocol, same seed)
        R_ = torch.load(rfile, weights_only=False); Lr_, res = R_["layers"][0], R_["s709"]["acc"]
        for h in hks:
            h.remove()
        print(f"[s709cal] 판독기 재사용 L{Lr_} (보류 정확 {res[Lr_] if Lr_ in res else res})", flush=True)
        return calib_only(v, a, nl, Lr_, res)
    Xtr, ytr = scenes(a.ntr, ("first",))
    Xte, yte = scenes(a.nte, LAYOUTS)
    for h in hks:
        h.remove()
    best, res = None, {}
    for L in CAND:
        F = Xtr[("first", L)].to(v.device); m_, sd_ = F.mean(0), F.std(0) + 1e-4
        Z = (F - m_) / sd_; V = torch.linalg.svd(Z, full_matrices=False)[2][:64].T
        P = fit(Z @ V, ytr, 64, "cat", m_, sd_, V)
        acc = {lay: float(((probs(P, Xte[(lay, L)]).argmax(1) + 1) == yte).float().mean()) for lay in LAYOUTS}
        res[L] = acc; mean = sum(acc.values()) / len(acc)
        print(f"[s709cal] L{L} vend 부호 정확(보류): " + " · ".join(f"{k_} {x:.3f}" for k_, x in acc.items()) + f" | 평균 {mean:.3f}", flush=True)
        if best is None or mean > best[0]:
            best = (mean, L, P)
    _, Lr_, P = best
    torch.save(dict(P=P, layers=[Lr_], sites=["vend"], SITES=SITES, s709=dict(acc=res, chosen=Lr_, ntr=a.ntr, nte=a.nte)),
               f"/vlm/results/s588c_reader_s709_{a.base}.pt")
    print(f"[s709cal] 판독기 L{Lr_} -> /vlm/results/s588c_reader_s709_{a.base}.pt", flush=True)
    return calib_only(v, a, nl, Lr_, res)


def calib_only(v, a, nl, Lr_, res):
    Lb = max(1, int(round(3 / 63 * (nl - 1))))
    M = ModelMD(v, Lb, Lb, [Lb], f"s709_{a.base}")
    cal = r3_build.calibrate(M, nproto=16, seed=60400)
    torch.save(dict(cal=cal, Lb=Lb, Lw=Lb, reader=f"s709_{a.base}", reader_layer=Lr_, reader_acc=res),
               f"/vlm/results/r3/s709_{a.base}_cal.pt")
    print(f"[s709cal] 끝 -> /vlm/results/r3/s709_{a.base}_cal.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("base")
    ap.add_argument("--ntr", type=int, default=500); ap.add_argument("--nte", type=int, default=60)
    ap.add_argument("--device_map", default=None); ap.add_argument("--reuse_reader", action="store_true")
    main(ap.parse_args())
