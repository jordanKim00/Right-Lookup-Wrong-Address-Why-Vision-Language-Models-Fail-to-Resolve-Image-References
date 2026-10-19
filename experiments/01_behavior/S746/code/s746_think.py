"""S746: does reasoning resolve end-relative references, and does the answer follow the address the reasoning states?
(VisualSwap's Instruct-vs-Thinking axis; pre-registration in knowledge/DECISIONS.md before any S746 data; analysis exp/s746_analyze.py.)
Qwen3-VL Thinking checkpoints, the S730 question form and menu, seven procedural pictures (160 px), seed 74600.
    OFF  reasoning skipped ("<think>\\n\\n</think>\\n\\n" prefix), restricted argmax over the menu (as every other S7xx run)
    ON   reasoning on ("<think>\\n" prefix, the model's own template), greedy generation up to --max_new tokens; the answer is the first
         menu word after "</think>"; the forward position the trace settles on is the last "image N" / "N-th image" / "position N"
         mention (digits or ordinal words) that is not followed by "from the end / last" (a heuristic readout; traces are saved)
Questions per scene: END(k) for k in {2, 3, 5, 6} (target j = 8 - k, literal image k) and START(j).
Usage: python s746_think.py MODEL TAG [--n 20] [--batch 8] [--max_new 2048]
"""
from __future__ import annotations

import argparse, json, os, random, re, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
ORDRE = "|".join(ORDW)
POS_RE = re.compile(rf"\b(?:image|picture|position|photo)\s*(?:#|no\.?|number)?\s*(\d)\b|\b(\d)(?:st|nd|rd|th)\s+(?:image|picture|position|one|photo)\b|\b({ORDRE})\s+(?:image|picture|position|one|photo)\b", re.I)
REV_RE = re.compile(r"^\s*(?:image|picture|one|position|photo)?\s*(?:from|counting from|to)\s+(?:the\s+)?(?:end|last|right|back)", re.I)


def trace_position(trace):
    pos = None
    for m in POS_RE.finditer(trace):
        tail = trace[m.end():m.end() + 40]
        if REV_RE.match(tail):
            continue                                                                    # "the third image from the end" restates the question
        d = m.group(1) or m.group(2)
        p = int(d) if d else ORDW.index(m.group(3).lower()) + 1
        if 1 <= p <= N:
            pos = p
    return pos


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    OFF, ON = "<think>\n\n</think>\n\n", "<think>\n"

    def prompt(o, frame, menu, prefix):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {o} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=prefix)

    def aid(c):                                          # answer readout (S788): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return ids1(c)[0]
        t = v.tok(c, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, c
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    T1 = []                                              # is the vocabulary-wide top-1 a bare menu token?

    @torch.no_grad()
    def off(o, frame, pics, menu, labs):
        inp = v.encode(prompt(o, frame, menu, OFF), pics)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        return labs.index(lab) + 1 if lab in labs else 0

    tok = v.processor.tokenizer
    tok.padding_side = "left"

    @torch.no_grad()
    def on(batch):                                                                       # batch: list of (o, frame, pics, menu)
        texts = [prompt(o, fr, menu, ON) for o, fr, _, menu in batch]
        imgs = [im for _, _, pics, _ in batch for im in pics]
        inp = v.processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(v.model.device)
        out = v.model.generate(**inp, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        return [tok.decode(o_[inp["input_ids"].shape[1]:], skip_special_tokens=True) for o_ in out]

    def parse(text, menu, labs):
        if "</think>" not in text:
            return None, text, None
        trace, ans = text.split("</think>", 1)
        words = re.findall(r"[A-Za-z]+", ans.lower())
        lab = next((w for w in words if w in menu), None)
        return (labs.index(lab) + 1 if lab in labs else 0) if lab else 0, trace, trace_position(trace)

    rng = random.Random(74600); rows = []; queue = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        for k in (2, 3, 5, 6):
            j = N + 1 - k
            for form, o, fr in (("END", ORDW[k - 1], "last"), ("START", ORDW[j - 1], "first")):
                r = dict(si=si, k=k, j=j, form=form, off=off(o, fr, pics, menu, labs))
                rows.append(r); queue.append((r, (o, fr, pics, menu), labs, menu))
    for b0 in (range(0, len(queue), a.batch) if not a.off_only else ()):    # S788: --off_only re-measures the OFF arm alone
        chunk = queue[b0:b0 + a.batch]
        outs = on([c[1] for c in chunk])
        for (r, _, labs, menu), text in zip(chunk, outs):
            land, trace, tp = parse(text, menu, labs)
            r.update(on=land, trace_pos=tp, n_chars=len(text), text=text[-4000:])
        torch.cuda.empty_cache()
        print(f"[{a.tag}] {min(b0 + a.batch, len(queue))}/{len(queue)}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, max_new=a.max_new, off_only=a.off_only, readout=os.environ.get("R3_READ", "space"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=rows), open(f"/vlm/results/r3/s746_{a.tag}_think.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s746_{a.tag}_think.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--off_only", action="store_true")
    main(ap.parse_args())
