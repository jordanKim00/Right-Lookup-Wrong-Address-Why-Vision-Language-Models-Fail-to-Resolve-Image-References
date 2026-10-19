"""S609 (3): the end reader at the site where the model carries the image index independently of the prompt layout (S608).
Fitted on images-first scenes only (procedural training-pool images, N 2-10); checked on held-out scenes in all four
layouts. Saved in the format the repair modules load: /vlm/results/s588c_reader_s609_{base}.pt
Usage: python s609_reader.py MODEL TAG --base B --layer L --site vend [--ntr 500] [--nte 60]
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


def main(v, tag, base, L, site, ntr, nte):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ST = {}

    def hook(m, i, o):
        if ST.get("B") is not None:
            t = o[0] if isinstance(o, tuple) else o
            ST["F"] = torch.stack([torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                                t[0, int(b[-1])].float()])[SITES.index(site)] for b in ST["B"]]).cpu()
        return o
    hk = v.layers[L].register_forward_hook(hook)
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(60910)
    t0 = time.time()

    @torch.no_grad()
    def scenes(n, lays):
        X = {lay: [] for lay in lays}; y = []
        for i in range(n):
            N = rng.randint(2, 10)
            rs = [proc_image(p) for p in rng.sample(pids, N)]
            pics, labs = [r_[0] for r_ in rs], [r_[1] for r_ in rs]
            k = rng.randint(1, N); opts = list(labs); rng.shuffle(opts)
            q = f"What is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(opts)}."
            for lay in lays:
                inp = v.encode(v.chat([("user", body(v, lay, N, q))], prefix_assistant=THINK), pics)
                ST.update(B=[torch.tensor(b) for b in v.image_blocks(inp["input_ids"], N)])
                v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
                X[lay].append(ST["F"]); ST["B"] = None
            y += list(range(1, N + 1))
            if (i + 1) % 100 == 0:
                print(f"[{tag}] 장면 {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
        return {k_: torch.cat(x) for k_, x in X.items()}, torch.tensor(y)
    Xtr, ytr = scenes(ntr, ("first",))
    Xte, yte = scenes(nte, LAYOUTS)
    hk.remove()
    F = Xtr["first"].to(v.device); m_, sd_ = F.mean(0), F.std(0) + 1e-4
    Z = (F - m_) / sd_; V = torch.linalg.svd(Z, full_matrices=False)[2][:64].T
    P = fit(Z @ V, ytr, 64, "cat", m_, sd_, V)
    acc = {lay: float(((probs(P, Xte[lay]).argmax(1) + 1) == yte).float().mean()) for lay in LAYOUTS}
    print(f"[{tag}] L{L} {site} 부호 정확(보류 장면): " + " · ".join(f"{k_} {x:.3f}" for k_, x in acc.items()), flush=True)
    torch.save(dict(P=P, layers=[L], sites=[site], SITES=SITES, s609=dict(acc=acc, ntr=ntr, nte=nte)),
               f"/vlm/results/s588c_reader_s609_{base}.pt")
    print(f"[{tag}] -> /vlm/results/s588c_reader_s609_{base}.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--base", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--site", default="vend")
    ap.add_argument("--ntr", type=int, default=500); ap.add_argument("--nte", type=int, default=60)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    main(v, a.tag, a.base, a.layer, a.site, a.ntr, a.nte)
