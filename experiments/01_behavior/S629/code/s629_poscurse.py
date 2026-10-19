"""S629: the Position Curse protocol (Zhang et al. 2026, arXiv 2605.07127, Sec. 3) on our models, text only, item retrieval.
A list of L distinct items (uppercase letters, or one-token animal names) and one query, with the four operations of the paper:
  End+  "What item is at the {k}th position from the beginning?"      target = item k
  End-  "What item is at the {k}th position from the end?"            target = item L+1-k
  Rel+  "What item is {m} positions after {X}?"                        target = item idx(X)+m
  Rel-  "What item is {m} positions before {X}?"                       target = item idx(X)-m
Greedy free generation (thinking off), first item-like token parsed, exact match. Their exact templates are not public; deviations:
comma-separated list in one line, numeric ordinals ("3rd"), zero-shot, greedy (they use T 0.7). Error labels (the mechanism's
predictions): copy = the item at the query's number read as a forward address (End-: item k; Rel+/Rel-: item m), neighbour = target±1,
anchor = X itself (Rel), wrongdir = idx(X)-m for Rel+ / idx(X)+m for Rel-.
S643b (--ops): RelO+/RelO- "What is the {m}th item after/before {X}?" (ordinal pointer word + item anchor) and
EndC "What item is {m} positions before the last item?" (cardinal offset + end anchor) -- the 2x2 of pointer form x anchor.
Usage: python s629_poscurse.py MODEL TAG [--n 100] [--lens 10,20]
"""
from __future__ import annotations

import argparse, json, random, re, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni

ANIMALS = ["cat", "dog", "cow", "pig", "fox", "owl", "bee", "ant", "rat", "bat", "hen", "eel", "elk", "yak", "ape", "emu",
           "horse", "sheep", "goat", "duck", "frog", "bear", "deer", "wolf", "lion", "tiger", "mouse", "snake", "whale", "shark"]
LETTERS = [chr(65 + i) for i in range(26)]


def ordn(k):
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    one = lambda w: len(v.tok(" " + w, add_special_tokens=False)["input_ids"]) == 1
    pools = {"letters": LETTERS, "animals": [w for w in ANIMALS if one(w)]}
    print(f"[{a.tag}] 한 토큰 동물 {len(pools['animals'])}", flush=True)
    rng = random.Random(a.seed)
    rows = []; t0 = time.time()
    for L in [int(x) for x in a.lens.split(",")]:
        for pname, pool in pools.items():
            if len(pool) < L:
                continue
            for op in a.ops.split(","):
                for _ in range(a.n):
                    items = rng.sample(pool, L)
                    if op in ("RelO+", "RelO-"):                                      # S643b: ordinal pointer word + item anchor
                        m = rng.randint(1, 3)
                        x = rng.randint(1, L - m) if op == "RelO+" else rng.randint(m + 1, L)
                        q = f"What is the {ordn(m)} item {'after' if op == 'RelO+' else 'before'} {items[x - 1]}?"
                        t = x + m if op == "RelO+" else x - m
                        lab = dict(copy=m, neighbour=[t - 1, t + 1], anchor=x, wrongdir=(x - m if op == "RelO+" else x + m))
                    elif op == "EndC":                                                 # S643b: cardinal offset + end anchor
                        m = rng.randint(1, 3)
                        q = f"What item is {m} position{'s' if m > 1 else ''} before the last item?"
                        t = L - m
                        lab = dict(copy=m, neighbour=[t - 1, t + 1], anchor=L)
                    elif op in ("End+", "End-"):
                        k = rng.randint(1, L)
                        q = f"What item is at the {ordn(k)} position from the {'beginning' if op == 'End+' else 'end'}?"
                        t = k if op == "End+" else L + 1 - k
                        lab = dict(copy=k, neighbour=[t - 1, t + 1])
                    else:
                        m = rng.randint(1, 3)
                        x = rng.randint(1, L - m) if op == "Rel+" else rng.randint(m + 1, L)
                        q = f"What item is {m} position{'s' if m > 1 else ''} {'after' if op == 'Rel+' else 'before'} {items[x - 1]}?"
                        t = x + m if op == "Rel+" else x - m
                        lab = dict(copy=m, neighbour=[t - 1, t + 1], anchor=x, wrongdir=(x - m if op == "Rel+" else x + m))
                    body = f"Here is a list of {L} items: {', '.join(items)}.\n{q} Answer with only the item."
                    inp = v.encode(v.chat([("user", body)], prefix_assistant=THINK), [])
                    with torch.no_grad():
                        out = v.model.generate(**inp, max_new_tokens=6, do_sample=False)
                    ans = v.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    w = re.findall(r"[A-Za-z]+", ans)
                    pred = (w[0].upper() if pname == "letters" else w[0].lower()) if w else ""
                    pos = items.index(pred) + 1 if pred in items else None
                    kind = ("correct" if pos == t else "copy" if pos == lab["copy"] else "anchor" if "anchor" in lab and pos == lab["anchor"] else
                            "wrongdir" if pos is not None and pos == lab.get("wrongdir") else "neighbour" if pos in lab["neighbour"] else
                            "other" if pos is not None else "invalid")
                    rows.append(dict(L=L, pool=pname, op=op, target=t, pos=pos, kind=kind, ans=ans[:20]))
                acc = [r for r in rows if r["L"] == L and r["pool"] == pname and r["op"] == op]
                c = {k_: sum(r["kind"] == k_ for r in acc) / len(acc) for k_ in ("correct", "copy", "neighbour", "anchor", "wrongdir", "other", "invalid")}
                print(f"[{a.tag}] L{L} {pname:>7s} {op}: 정답 {c['correct']:.2f} · 복사 {c['copy']:.2f} · 이웃 {c['neighbour']:.2f} · 앵커 {c['anchor']:.2f} · "
                      f"반대방향 {c['wrongdir']:.2f} · 기타 {c['other']:.2f} · 무효 {c['invalid']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, rows=rows), open(f"/vlm/results/r3/s629_{a.tag}_poscurse.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s629_{a.tag}_poscurse.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--lens", default="10,20")
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--ops", default="End+,End-,Rel+,Rel-")                               # S643b adds RelO+,RelO-,EndC
    ap.add_argument("--seed", type=int, default=62900)
    main(ap.parse_args())
