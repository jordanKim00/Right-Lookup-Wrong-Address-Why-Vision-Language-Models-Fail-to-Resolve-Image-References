"""S782: measurement validity - does the menu-restricted argmax at the answer position agree with what the model generates freely?  Same scenes as
S780 (seeds 78000 / 78001, N = 7 / 8, wording "last").  Per scene END(5), START(N - 4), END(2) and R3 on END(5) (exp/r3_select.build: model-specific
READ, write layer L3): greedy generation (8 new tokens) with the first step's scores; restricted answer = argmax of the first step over the menu words'
first tokens; free answer = the first word of the generated text (lowercased, punctuation stripped).
Pre-registration in knowledge/DECISIONS.md before any S782 data; analysis exp/s782_analyze.py.
Usage: python s782_freegen.py MODEL TAG --N 7 [--n 80] [--device_map balanced]
"""
from __future__ import annotations

import argparse, gc, json, random, re, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
import r3_select

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s782_{}_freegen.json"


def main(a, v=None):
    v = v or Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    N = a.N
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rep = r3_select.build(v, a.r3base, a.r3rc)

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def gen(inp, menu, writes=None):
        rep.ST = dict(writes=writes, wmax=max([p_ for p_, _ in writes], default=-1)) if writes else {}
        out = v.model.generate(**inp, max_new_tokens=8, do_sample=False, output_scores=True, return_dict_in_generate=True)
        rep.ST = {}
        s0 = out.scores[0][0].float()
        restricted = menu[int(torch.stack([s0[tid(c)] for c in menu]).argmax())]
        text = v.tok.decode(out.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        w = re.findall(r"[A-Za-z]+", text)
        return dict(restricted=restricted, text=text, free=(w[0].lower() if w else ""))

    ta = N - 4
    rng = random.Random(a.seed)
    meta = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        ims = [pic[l] for l in labs]
        rec = dict(si=si, labs=labs, extra=extra, menu=menu, out={})
        for nm, (fr, k) in (("E5", ("E", 5)), ("S", ("S", ta)), ("E2", ("E", 2))):
            rec["out"][nm] = gen(v.encode(prompt(fr, k, menu), ims), menu)
        inR, wR = rep.plan(prompt("E", 5, menu), ims)
        rec["out"]["R3"] = gen(inR, menu, wR)
        meta.append(rec)
        gc.collect(); torch.cuda.empty_cache()
        if (si + 1) % 10 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, N=N, ta=ta, meta=meta), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None); ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--r3base", default="r327b"); ap.add_argument("--r3rc", default="rc_27b")
    a = ap.parse_args(); a.seed = a.seed if a.seed is not None else 78000 + (a.N - 7)
    main(a)
