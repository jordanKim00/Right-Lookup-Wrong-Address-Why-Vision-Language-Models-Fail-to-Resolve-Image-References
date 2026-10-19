"""S643: moved anchors in multi-image references -- pointer form (ordinal word vs cardinal offset) x anchor (start / end / an item).
Procedural scenes as in exp/s626_patch.py (one-token shape words, distinct shapes per scene), restricted choice over
the scene's shape words plus two distractors (logit argmax, as S626); N in 5..7 (only 9 one-token shape words). Ops (m in 1..3):
  Start   "What is in the {ord} image?"                                   t = k
  End     "What is in the {ord} image from the end?"                      t = N+1-k
  RelO+/- "What is in the {ord(m)} image after/before the {X}?"            t = idx(X) +/- m      (ordinal pointer word, item anchor)
  RelC+/- "What is in the image {m} image(s) after/before the {X}?"       t = idx(X) +/- m      (cardinal offset, item anchor)
  EndC    "What is in the image {m} image(s) before the last image?"      t = N-m               (cardinal offset, end anchor)
  IdxX    "Counting from the start, which image shows the {X}? Answer with a number only."            (anchor localisation, greedy)
  IdxEnd  "Counting from the start, which image is the {ord} image from the end? Answer with a number only."  (count-back when asked)
  RelOx+/- "Start at the image showing the {X} and, without counting it, count forward/backward. What is in the {ord(m)} image you reach?"  (S661)
Error labels: copy = the image at the query's ordinal/number read as a start-relative address (End: k; Rel*/EndC: m),
anchor = X itself, wrongdir = idx(X) -/+ m, neighbour = t +/- 1.
Usage: python s643_relimg.py MODEL TAG [--n 80]
"""
from __future__ import annotations

import argparse, json, os, random, re, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
NUMW = ["one", "two", "three"]
OPS = ("Start", "End", "RelO+", "RelO-", "RelC+", "RelC-", "EndC", "IdxX", "IdxEnd")
XOPS = ("RelOx+", "RelOx-")   # S661 (2026-10-07): anchor and direction stated, anchor excluded from the count


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
    print(f"[{a.tag}] 도형 낱말 {len(labels)}", flush=True)
    rng = random.Random(64300)
    rows = []; t0 = time.time()
    for op in a.ops.split(","):
        for _ in range(a.n):
            N = rng.randint(5, min(7, len(labels) - 2))                            # 9 one-token shape words: N + 2 distractors <= 9
            labs = rng.sample(labels, N + 2); scene = labs[:N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene]
            cands = list(labs); rng.shuffle(cands)
            lab = {}
            if op in ("Start", "End", "IdxEnd"):
                k = rng.randint(1, min(N, 5))
                t = k if op == "Start" else N + 1 - k
                q = {"Start": f"What is in the {ORDW[k-1]} image?", "End": f"What is in the {ORDW[k-1]} image from the end?",
                     "IdxEnd": f"Counting from the start, which image is the {ORDW[k-1]} image from the end? Answer with a number only."}[op]
                lab = dict(copy=k, neighbour=[t - 1, t + 1])
            elif op in ("RelO+", "RelO-", "RelC+", "RelC-") + XOPS:
                m = rng.randint(1, 3); fwd = op.endswith("+")
                x = rng.randint(1, N - m) if fwd else rng.randint(m + 1, N)
                t = x + m if fwd else x - m; X = scene[x - 1]; d = "after" if fwd else "before"
                q = (f"Start at the image showing the {X} and, without counting it, count {'forward' if fwd else 'backward'}. "
                     f"What is in the {ORDW[m-1]} image you reach?" if op in XOPS else
                     f"What is in the {ORDW[m-1]} image {d} the {X}?" if op.startswith("RelO") else
                     f"What is in the image {NUMW[m-1]} image{'s' if m > 1 else ''} {d} the {X}?")
                lab = dict(copy=m, neighbour=[t - 1, t + 1], anchor=x, wrongdir=(x - m if fwd else x + m))
            elif op == "EndC":
                m = rng.randint(1, 3); t = N - m
                q = f"What is in the image {NUMW[m-1]} image{'s' if m > 1 else ''} before the last image?"
                lab = dict(copy=m, neighbour=[t - 1, t + 1], anchor=N)
            else:                                                                  # IdxX
                x = rng.randint(1, N); t = x
                q = f"Counting from the start, which image shows the {scene[x-1]}? Answer with a number only."
            if op.startswith("Idx"):
                inp = v.encode(v.chat([("user", f"{v.IMG*N}\n{q}")], prefix_assistant=THINK), pics)
                with torch.no_grad():
                    out = v.model.generate(**inp, max_new_tokens=6, do_sample=False)
                ans = v.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                mm = re.search(r"\d+", ans); pos = int(mm.group()) if mm else None
                kind = "correct" if pos == t else ("copy" if pos == lab.get("copy") else "neighbour" if pos in lab.get("neighbour", []) else
                                                   "other" if pos is not None else "invalid")
            else:
                text = v.chat([("user", f"{v.IMG*N}\n{q} Answer with only one word from: {', '.join(cands)}.")], prefix_assistant=THINK)
                inp = v.encode(text, pics)
                with torch.no_grad():
                    lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
                ans_sp = cands[int(torch.stack([lg[ids1(c)[0]] for c in cands]).argmax())]
                if os.environ.get("R3_READ") == "bare":                         # S784: bare-token readout + free generation's first word
                    bare = [v.tok(c, add_special_tokens=False)["input_ids"] for c in cands]
                    assert all(len(b) == 1 for b in bare), cands
                    ans = cands[int(torch.stack([lg[b[0]] for b in bare]).argmax())]
                    with torch.no_grad():
                        g = v.model.generate(**inp, max_new_tokens=6, do_sample=False)
                    ftxt = v.tok.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                    fw = re.findall(r"[A-Za-z]+", ftxt); free = fw[0].lower() if fw else ""
                    xread = dict(ans_space=ans_sp, ans_free=free, free_text=ftxt[:30])
                else:
                    ans = ans_sp; xread = {}
                pos = scene.index(ans) + 1 if ans in scene else None
                kind = ("correct" if pos == t else "copy" if pos == lab.get("copy") else "anchor" if "anchor" in lab and pos == lab["anchor"] else
                        "wrongdir" if pos is not None and pos == lab.get("wrongdir") else "neighbour" if pos in lab.get("neighbour", []) else
                        "other" if pos is not None else "distractor")
            rows.append(dict(op=op, N=N, target=t, pos=pos, kind=kind, q=q, ans=str(ans)[:20], **{f"lab_{k_}": v_ for k_, v_ in lab.items()},
                             **(xread if not op.startswith("Idx") else {})))
        acc = [r for r in rows if r["op"] == op]
        c = {k_: sum(r["kind"] == k_ for r in acc) / len(acc) for k_ in ("correct", "copy", "anchor", "wrongdir", "neighbour", "other", "distractor", "invalid")}
        print(f"[{a.tag}] {op:>6s}: " + " · ".join(f"{k_} {x:.2f}" for k_, x in c.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, rows=rows), open(f"/vlm/results/r3/s643_{a.tag}_relimg.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s643_{a.tag}_relimg.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--ops", default=",".join(OPS)); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
