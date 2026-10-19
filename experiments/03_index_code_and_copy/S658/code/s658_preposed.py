"""S658: does a frame placed before the ordinal rescue end-relative reference?  (review point: the ordinal token of a postposed
question "the third image from the end" cannot see "from the end" under the causal mask, so a literal early state is expected.)
Same procedural scenes as S648 (same seed and draw order, so QR and OR reproduce S648 row by row).
  QF    "What is in the {t}th image?"                                              forward ceiling
  QFP   "Counting from the start, what is in the {t}th image?"                     preposed-construction ceiling
  QR    "What is in the {N+1-t}th image from the end?"                             postposed (S648 QR)
  QRP1  "Counting from the end, what is in the {N+1-t}th image?"                   frame visible to the ordinal
  QRP2  "Starting from the last image and counting backwards, what is in the {N+1-t}th image?"   second preposed wording
  OR    "Which image shows the X?  A: The {N+1-p}th image from the end ..."        postposed option lines (S648 OR)
  ORP   "Counting the images from the end, which image shows the X?  A: The {N+1-p}th image ..."  frame stated once, before every option
Content answers are restricted to the scene's shapes + 2 distractors (as S648); surf = the start-relative image N+1-t (the ordinal read forward).
Option answers: 4 options (target, lure, two others); lure = the option whose ordinal word equals the target's forward index.
Usage: python s658_preposed.py MODEL TAG [--n 80]
"""
from __future__ import annotations

import os
import argparse, json, random, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
LET = "ABCD"
QFMT = ("QF", "QFP", "QR", "QRP1", "QRP2")
OFMT = ("OR", "ORP")


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(s):                                          # answer readout (S787): R3_READ=bare reads the first bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return ids1(s)[0]
        return v.tok(s, add_special_tokens=False)["input_ids"][0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    lid = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in LET]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(getattr(a, "seed", None) or 64800)                         # S648's seed and draw order (S787: a new seed)
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
        X = scene[t - 1]; m = N + 1 - t
        r = dict(N=N, t=t, lure=lure, pos=pos)
        qs = {"QF": f"What is in the {ORDW[t-1]} image?",
              "QFP": f"Counting from the start, what is in the {ORDW[t-1]} image?",
              "QR": f"What is in the {ORDW[m-1]} image from the end?",
              "QRP1": f"Counting from the end, what is in the {ORDW[m-1]} image?",
              "QRP2": f"Starting from the last image and counting backwards, what is in the {ORDW[m-1]} image?"}
        for fmt in QFMT:
            text = v.chat([("user", f"{v.IMG*N}\n{qs[fmt]} Answer with only one word from: {', '.join(cands)}.")], prefix_assistant=THINK)
            with torch.no_grad():
                lg = v.model(**v.encode(text, pics), use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
            ans = cands[int(torch.stack([sc(lg, c) for c in cands]).argmax())]
            r[fmt] = dict(ok=ans == X, surf=(ans == scene[lure - 1]), ans=ans)
        for fmt in OFMT:
            if fmt == "OR":
                stem, ref = f"Which image shows the {X}?", (lambda p: f"The {ORDW[N-p]} image from the end")
            else:
                stem, ref = f"Counting the images from the end, which image shows the {X}?", (lambda p: f"The {ORDW[N-p]} image")
            opts = "\n".join(f"{L}: {ref(p)}" for L, p in zip(LET, pos))
            text = v.chat([("user", f"{v.IMG*N}\n{stem}\n{opts}\nAnswer with the option's letter from the given choices directly.")],
                          prefix_assistant=THINK)
            with torch.no_grad():
                lg = v.model(**v.encode(text, pics), use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
            ch = int(torch.stack([lg[i] for i in lid]).argmax())
            r[fmt] = dict(ok=pos[ch] == t, lure=pos[ch] == lure, pick=pos[ch])
        rows.append(r)
        if len(rows) % 20 == 0:
            s = " · ".join(f"{f} {sum(x[f]['ok'] for x in rows)/len(rows):.2f}" for f in QFMT + OFMT)
            print(f"[{a.tag}] {len(rows)}: {s}  ({time.time()-t0:.0f}s)", flush=True)
    summ = {f: dict(acc=sum(x[f]["ok"] for x in rows) / len(rows)) for f in QFMT + OFMT}
    for f in QFMT:
        summ[f]["surf"] = sum(x[f]["surf"] for x in rows) / len(rows)
    for f in OFMT:
        summ[f]["lure"] = sum(x[f]["lure"] for x in rows) / len(rows)
    print(f"[{a.tag}] " + json.dumps(summ), flush=True)
    out = f"/vlm/results/r3/s658_{a.tag}_preposed.json"
    json.dump(dict(tag=a.tag, model=a.model, summary=summ, rows=rows), open(out, "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
