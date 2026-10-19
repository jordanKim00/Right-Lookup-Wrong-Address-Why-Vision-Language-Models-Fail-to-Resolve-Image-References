"""S729: re-calibrate the end reader (READ's count part) over image resolutions -- block-position labels only, no answer, no CIFAR.
The S609 / S709 readers were fitted on procedural images only; on blurred low-resolution tiles they misread the last block at N = 9 / 10
(§756: 9B reads block 10 as 9, Qwen3-VL-4B reads block 9 as 10).  Same layer and site as the model's current reader, images-first scenes,
N 2-10, three image kinds in equal shares: procedural (224 px), natural (caltech256 / food101 rows unused by R³-Ref, 448 px) and the same
natural pool degraded (downscaled to 32 / 48 / 64 px, upscaled to 224 px).  Held-out scenes per kind score both readers (N_hat = code of
the last block = N), and the new reader is saved only as a candidate; admission is exp/s729_select (pre-registered rule).
Usage: python s729_reader.py MODEL TAG --base B --old READERTAG [--ntr 600] [--nte 90]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from s588c_reader import fit, probs, SITES
from r3_proc import proc_image
from r3ref import ORD, load_image
from s724_readcal import nat_keys


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    old = torch.load(f"/vlm/results/s588c_reader_{a.old}.pt", weights_only=False)
    L, site = old["layers"][0], old["sites"][0]
    ST = {}

    def hook(m, i, o):
        if ST.get("B") is not None:
            t = o[0] if isinstance(o, tuple) else o
            ST["F"] = torch.stack([torch.stack([t[0, b.to(t.device)].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                                t[0, int(b[-1])].float()])[SITES.index(site)] for b in ST["B"]]).cpu()
        return o
    hk = v.layers[L].register_forward_hook(hook)
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    nat = nat_keys(random.Random(72400), 12000)[8000:]                              # same shuffle as S724, past its 8,000 (disjoint)
    rng = random.Random(72901)
    t0 = time.time()

    def picture(kind):
        if kind == "proc":
            return proc_image(rng.choice(pids))[0]
        im = load_image(nat.pop(), size=448)
        if kind == "low":
            s = rng.choice([32, 48, 64])
            im = im.resize((s, max(8, int(s * im.height / im.width))), Image.BICUBIC).resize((224, max(28, int(224 * im.height / im.width))), Image.BICUBIC)
        return im

    @torch.no_grad()
    def scenes(n):
        X, y, kinds, Ns = [], [], [], []
        for i in range(n):
            kind = ["proc", "nat", "low"][i % 3]; N = rng.randint(2, 10)
            pics = [picture(kind) for _ in range(N)]
            q = f"What is in the {ORD[rng.randint(1, N) - 1]} image? Answer briefly."
            inp = v.encode(v.chat([("user", f"{v.IMG * N}\n{q}")], prefix_assistant=THINK), pics)
            ST.update(B=[torch.tensor(b) for b in v.image_blocks(inp["input_ids"], N)])
            v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
            X.append(ST["F"]); ST["B"] = None
            y += list(range(1, N + 1)); kinds.append(kind); Ns.append(N)
            if (i + 1) % 100 == 0:
                print(f"[{a.tag}] 장면 {i + 1}/{n}  ({time.time() - t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
        return X, torch.tensor(y), kinds, Ns
    Xtr, ytr, _, _ = scenes(a.ntr)
    Xte, yte, kte, Nte = scenes(a.nte)
    hk.remove()
    F = torch.cat(Xtr).to(v.device); m_, sd_ = F.mean(0), F.std(0) + 1e-4
    Z = (F - m_) / sd_; V = torch.linalg.svd(Z, full_matrices=False)[2][:64].T
    P = fit(Z @ V, ytr, 64, "cat", m_, sd_, V)
    res = {}
    for nm, R in (("new", P), ("old", old["P"])):
        last = [int(probs(R, x).argmax(1)[-1]) + 1 == n for x, n in zip(Xte, Nte)]
        blk = [float(((probs(R, x).argmax(1) + 1) == torch.arange(1, n + 1)).float().mean()) for x, n in zip(Xte, Nte)]
        res[nm] = {k: dict(count=float(sum(l for l, kk in zip(last, kte) if kk == k) / max(1, kte.count(k))),
                           block=float(sum(b for b, kk in zip(blk, kte) if kk == k) / max(1, kte.count(k)))) for k in ("proc", "nat", "low")}
        res[nm]["all"] = dict(count=float(sum(last) / len(last)), block=float(sum(blk) / len(blk)))
        res[nm]["N10"] = float(sum(l for l, n in zip(last, Nte) if n >= 9) / max(1, sum(n >= 9 for n in Nte)))
        print(f"[{a.tag}] {nm}: 개수(마지막 코드 = N) " + " · ".join(f"{k} {res[nm][k]['count']:.3f}" for k in ("proc", "nat", "low", "all")) +
              f" · N≥9 {res[nm]['N10']:.3f} · 블록 정확 " + " · ".join(f"{k} {res[nm][k]['block']:.3f}" for k in ("proc", "nat", "low")), flush=True)
    torch.save(dict(P=P, layers=[L], sites=[site], SITES=SITES, s729=dict(res=res, ntr=a.ntr, nte=a.nte, old=a.old)),
               f"/vlm/results/s588c_reader_s729_{a.base}.pt")
    json.dump(dict(tag=a.tag, base=a.base, old=a.old, layer=L, site=site, res=res), open(f"/vlm/results/r3/s729_{a.tag}_reader.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/s588c_reader_s729_{a.base}.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--old", required=True)
    ap.add_argument("--ntr", type=int, default=600); ap.add_argument("--nte", type=int, default=90); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
