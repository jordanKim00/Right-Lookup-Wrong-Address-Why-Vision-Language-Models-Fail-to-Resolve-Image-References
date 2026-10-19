"""S660: the two skills and their composition on the SAME scene (review W2, 2026-10-07).
Procedural scenes as S643 (one-token shapes, N 5-7, restricted content answers over the scene's shapes + 2 distractors).
Per scene, with k in 1..min(N,5), j = N+1-k != k:
  idx   "Counting from the start, which image is the {k} image from the end? Answer with a number only."  (greedy, first integer)
  ret   "What is in the {j} image?"                                                                       (explicit forward retrieval of j)
  comp  "What is in the {k} image from the end?"                                                          (the composition)
Reported: P(comp correct | idx correct and ret correct) with its subset size, and the share of comp answers on the surface image k.
Usage: python s660_samescene.py MODEL TAG [--n 80]
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(66000)
    rows = []; t0 = time.time()

    def content(q, pics, cands, N):
        text = v.chat([("user", f"{v.IMG*N}\n{q} Answer with only one word from: {', '.join(cands)}.")], prefix_assistant=THINK)
        inp_ = v.encode(text, pics)
        with torch.no_grad():
            lg = v.model(**inp_, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        if os.environ.get("R3_READ") == "bare":                                 # S784: bare-token readout (= greedy first word); space readout kept
            bare = [v.tok(c, add_special_tokens=False)["input_ids"] for c in cands]
            assert all(len(b) == 1 for b in bare), cands
            content.space.append(cands[int(torch.stack([lg[ids1(c)[0]] for c in cands]).argmax())])
            with torch.no_grad():
                g = v.model.generate(**inp_, max_new_tokens=6, do_sample=False)
            fw = re.findall(r"[A-Za-z]+", v.tok.decode(g[0, inp_["input_ids"].shape[1]:], skip_special_tokens=True))
            content.free.append(fw[0].lower() if fw else "")
            return cands[int(torch.stack([lg[b[0]] for b in bare]).argmax())]
        return cands[int(torch.stack([lg[ids1(c)[0]] for c in cands]).argmax())]
    content.space, content.free = [], []

    while len(rows) < a.n:
        N = rng.randint(5, min(7, len(labels) - 2))
        labs = rng.sample(labels, N + 2); scene = labs[:N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene]
        cands = list(labs); rng.shuffle(cands)
        k = rng.randint(1, min(N, 5)); j = N + 1 - k
        if j == k:
            continue
        q_idx = f"Counting from the start, which image is the {ORDW[k-1]} image from the end? Answer with a number only."
        inp = v.encode(v.chat([("user", f"{v.IMG*N}\n{q_idx}")], prefix_assistant=THINK), pics)
        with torch.no_grad():
            out = v.model.generate(**inp, max_new_tokens=6, do_sample=False)
        s = v.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        mm = re.search(r"\d+", s); idx = int(mm.group()) if mm else None
        with torch.no_grad():                                                   # secondary (amendment 2026-10-07 15:40): restricted digit argmax
            lg0 = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        dig = [v.tok(str(i), add_special_tokens=False)["input_ids"][0] for i in range(1, N + 1)]
        idx_r = int(torch.stack([lg0[t] for t in dig]).argmax()) + 1
        ret = content(f"What is in the {ORDW[j-1]} image?", pics, cands, N)
        comp = content(f"What is in the {ORDW[k-1]} image from the end?", pics, cands, N)
        rows.append(dict(N=N, k=k, j=j, idx=idx, idx_raw=s[:12], idx_ok=int(idx == j), idx_surface=int(idx == k),
                         idx_r=idx_r, idx_r_ok=int(idx_r == j),
                         ret_ok=int(ret == scene[j - 1]), comp_ok=int(comp == scene[j - 1]), comp_surface=int(comp == scene[k - 1]),
                         **(dict(ret_space=content.space[-2], comp_space=content.space[-1], ret_free=content.free[-2], comp_free=content.free[-1],
                                 ret=ret, comp=comp) if os.environ.get("R3_READ") == "bare" else {})))
        if len(rows) % 20 == 0:
            r_ = rows
            both = [r for r in r_ if r["idx_ok"] and r["ret_ok"]]
            print(f"[{a.tag}] {len(r_)}: idx {sum(r['idx_ok'] for r in r_)/len(r_):.2f} ret {sum(r['ret_ok'] for r in r_)/len(r_):.2f} "
                  f"comp {sum(r['comp_ok'] for r in r_)/len(r_):.2f} | both {len(both)} comp|both "
                  f"{(sum(r['comp_ok'] for r in both)/len(both)) if both else float('nan'):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    both = [r for r in rows if r["idx_ok"] and r["ret_ok"]]
    both_r = [r for r in rows if r["idx_r_ok"] and r["ret_ok"]]
    summ = dict(idx_r=sum(r["idx_r_ok"] for r in rows) / len(rows), both_r_n=len(both_r),
                comp_given_both_r=(sum(r["comp_ok"] for r in both_r) / len(both_r)) if both_r else None,
                invalid_idx=sum(r["idx"] is None for r in rows),n=len(rows), idx=sum(r["idx_ok"] for r in rows) / len(rows), ret=sum(r["ret_ok"] for r in rows) / len(rows),
                comp=sum(r["comp_ok"] for r in rows) / len(rows), both_n=len(both),
                comp_given_both=(sum(r["comp_ok"] for r in both) / len(both)) if both else None,
                surface_given_both=(sum(r["comp_surface"] for r in both) / len(both)) if both else None)
    print(f"[{a.tag}] " + json.dumps(summ), flush=True)
    out = f"/vlm/results/r3/s660_{a.tag}_samescene.json"
    json.dump(dict(tag=a.tag, model=a.model, summary=summ, rows=rows), open(out, "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
