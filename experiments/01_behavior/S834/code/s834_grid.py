"""S834 (descriptive, registered in knowledge/DECISIONS.md before any S834 data; no hypothesis): complete the natural END(k) answer grid of the
v2 paper figure (fig_grid) at N = 7 and 8, whose k = 1, 2 (and k = 8 at N = 8) cells S824 did not run.
Scenes are those of S824: the scene code below is S824's (exp/s824_count.py) line for line - seed 82400, 40 scenes, 8 pictures drawn per scene,
the 7-picture condition shows the first seven, menu = the eight labels + one extra word - so every scene is checked against S824's saved labels
and menu.  Conditions (shown = stated count): N = 7, k in {1, 2, 3}; N = 8, k in {1, 2, 3, 8}; END and START.  k = 3 repeats S824 cells and
checks that this run reproduces S824's answers.  Same prompt, readout (two-form menu readout, S794), t1 and logits as S824; only the batching
differs (several scenes per forward, at most 330 pictures per batch).
Usage: python s834_grid.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s834_{}_grid.json"
CELLS = {7: [1, 2, 3], 8: [1, 2, 3, 8]}


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max", "S834 reads the generated form in either tokenization (S794)"
    dev = os.environ.get("S834_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def sc(lg, s):
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    T1 = []

    def prompt(Na, Ns, q, frame, menu):
        return v.chat([("user", f"{v.IMG*Na}\nThere are {Ns} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    # scenes: S824's code, unchanged
    rng = random.Random(a.seed); out = []
    for si in range(a.n):
        labs = rng.sample(labels, 8); extra = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        out.append(dict(si=si, labs=labs, extra=extra, menu=menu, land={}, mlog={}, _pics=pics))
    nrun = 0
    for N, ks in CELLS.items():
        jobs = [(rec, k, fr) for rec in out for k in ks for fr in ("last", "first")]
        per = max(1, a.max_pics // N)
        for c0 in range(0, len(jobs), per):
            part = jobs[c0:c0 + per]
            encs = [v.encode(prompt(N, N, k, fr, rec["menu"]), rec["_pics"][:N]) for rec, k, fr in part]
            assert len({len(e["input_ids"][0]) for e in encs}) == 1, "prompts of one N must align"
            with torch.no_grad():
                lg = v.model(**batch_inputs(encs), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            for b, (rec, k, fr) in enumerate(part):
                nm = f"a{N}_s{N}_k{k}_{'E' if fr == 'last' else 'S'}"
                menu, labs = rec["menu"], rec["labs"]
                T1.append(int(int(lg[b].argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())]
                rec["land"][nm] = labs.index(w) + 1 if w in labs[:N] else 0
                rec["mlog"][nm] = [float(sc(lg[b], x)) for x in labs + rec["extra"]]
            nrun += len(part)
            del lg, encs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{a.tag}] N{N} {min(c0 + per, len(jobs))}/{len(jobs)} · {nrun} runs", flush=True)
    for rec in out:
        rec.pop("_pics")
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), cells={str(N): ks for N, ks in CELLS.items()}, seed=a.seed,
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=82400); ap.add_argument("--max_pics", type=int, default=320)
    main(ap.parse_args())
