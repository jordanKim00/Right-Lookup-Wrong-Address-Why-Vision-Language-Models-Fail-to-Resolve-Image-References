"""S648: question-form vs option-form end-relative references on the same procedural scenes.
27B composes end-relative references in question form (S626/§576/§597) but fails MMIU/MUIR option lines (base .262/.329, §535 §540).
  QF  "What is in the {t}th image?"                                  content answer (scene shapes + 2 distractors)
  QR  "What is in the {N+1-t}th image from the end?"                 content answer
  OF  "Which image shows the {X}?  A: The {p}th image  B: ..."       letter answer, 4 options, X = image t
  OR  same, options "The {N+1-p}th image from the end"               letter answer
Options cover four distinct physical positions: the target t, the lure N+1-t (its option in OR reads "the t-th image from the end", i.e.
the target's own forward ordinal -- the pointer-copy choice), and two others. Rows with N+1-t == t are skipped. Letters are shuffled.
Metric: accuracy; OF/OR lure rate (choosing the option whose ordinal word equals the target's forward index).
Usage: python s648_format.py MODEL TAG [--n 80]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
LET = "ABCD"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    lid = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in LET]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(64800)
    rows = []; t0 = time.time()
    while len(rows) < a.n:
        N = rng.randint(5, min(7, len(labels) - 2))
        labs = rng.sample(labels, N + 2); scene = labs[:N]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene]
        cands = list(labs); rng.shuffle(cands)
        t = rng.randint(1, N); lure = N + 1 - t
        if lure == t:
            continue
        others = rng.sample([p for p in range(1, N + 1) if p not in (t, lure)], 2)
        pos = [t, lure] + others; rng.shuffle(pos)
        X = scene[t - 1]
        r = dict(N=N, t=t, lure=lure, pos=pos)
        for fmt in ("QF", "QR", "OF", "OR"):
            if fmt in ("QF", "QR"):
                q = f"What is in the {ORDW[t-1]} image?" if fmt == "QF" else f"What is in the {ORDW[N-t]} image from the end?"
                text = v.chat([("user", f"{v.IMG*N}\n{q} Answer with only one word from: {', '.join(cands)}.")], prefix_assistant=THINK)
                with torch.no_grad():
                    lg = v.model(**v.encode(text, pics), use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
                ans = cands[int(torch.stack([lg[ids1(c)[0]] for c in cands]).argmax())]
                r[fmt] = dict(ok=ans == X, surf=(ans == scene[lure - 1]), ans=ans)
            else:
                ref = (lambda p: f"The {ORDW[p-1]} image") if fmt == "OF" else (lambda p: f"The {ORDW[N-p]} image from the end")
                opts = "\n".join(f"{L}: {ref(p)}" for L, p in zip(LET, pos))
                text = v.chat([("user", f"{v.IMG*N}\nWhich image shows the {X}?\n{opts}\nAnswer with the option's letter from the given choices directly.")],
                              prefix_assistant=THINK)
                with torch.no_grad():
                    lg = v.model(**v.encode(text, pics), use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
                ch = int(torch.stack([lg[i] for i in lid]).argmax())
                r[fmt] = dict(ok=pos[ch] == t, lure=pos[ch] == lure, pick=pos[ch])
        rows.append(r)
        if len(rows) % 20 == 0:
            s = " · ".join(f"{f} {sum(x[f]['ok'] for x in rows)/len(rows):.2f}" for f in ("QF", "QR", "OF", "OR"))
            print(f"[{a.tag}] {len(rows)}: {s}  ({time.time()-t0:.0f}s)", flush=True)
    summ = {f: dict(acc=sum(x[f]["ok"] for x in rows) / len(rows)) for f in ("QF", "QR", "OF", "OR")}
    for f in ("OF", "OR"):
        summ[f]["lure"] = sum(x[f]["lure"] for x in rows) / len(rows)
    summ["QR"]["surf"] = sum(x["QR"]["surf"] for x in rows) / len(rows)
    print(f"[{a.tag}] " + json.dumps(summ), flush=True)
    json.dump(dict(tag=a.tag, model=a.model, summary=summ, rows=rows), open(f"/vlm/results/r3/s648_{a.tag}_format.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s648_{a.tag}_format.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
