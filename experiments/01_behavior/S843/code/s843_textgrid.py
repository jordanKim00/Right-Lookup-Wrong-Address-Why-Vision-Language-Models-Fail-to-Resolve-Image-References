"""S843 (pre-registered in knowledge/DECISIONS.md before any S843 data): text modality, user order (10-17: after the gaps) - do word lists show
the same end-relative pattern as pictures?  The image main series asks for the k-th of N pictures; here the items are the label words
themselves, in one line ('Words: w1, w2, ...'), with the same sentence otherwise: 'There are N words in a row. What is the k-th word from the
last / first? Answer with only one word from: <menu>.'  Label pool, menu recipe (the N words + the first unused words to nine, shuffled), THINK
prefix (Qwen3.5), two-form menu readout and t1 as the image series.  N in {6, 7, 8, 9}, k = 1..N, END and START, 40 lists per N, seed 84300.
Usage: python s843_textgrid.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s843_{}_textgrid.json"


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S843_DEVICE", "cuda:0")
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
    assert len(labels) >= 9
    T1 = []
    prompt = lambda N, q, frame, menu, labs: v.chat([("user", f"Words: {', '.join(labs)}\nThere are {N} words in a row. What is the {ORDW[q-1]} word "
                                                               f"from the {frame}? Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)
    rng = random.Random(a.seed); out = []; nrun = 0
    for si in range(a.n):
        rec = dict(si=si, land={}, labs={}, menu={})
        for N in (6, 7, 8, 9):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            menu = labs + extra; rng.shuffle(menu)
            rec["labs"][str(N)] = labs; rec["menu"][str(N)] = menu
            conds = [(k, fr) for k in range(1, N + 1) for fr in ("last", "first")]
            encs = [v.encode(prompt(N, k, fr, menu, labs), []) for k, fr in conds]
            assert len({len(e["input_ids"][0]) for e in encs}) == 1
            with torch.no_grad():
                for c0 in range(0, len(encs), a.batch):
                    part = encs[c0:c0 + a.batch]
                    lg = v.model(**batch_inputs(part), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                    for b, (k, fr) in enumerate(conds[c0:c0 + a.batch]):
                        T1.append(int(int(lg[b].argmax()) in t1ids(menu)))
                        w = menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())]
                        rec["land"][f"N{N}_k{k}_{'E' if fr == 'last' else 'S'}"] = labs.index(w) + 1 if w in labs else 0
            nrun += len(encs)
        out.append(rec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n} · {nrun} runs", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=out),
              open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=84300); ap.add_argument("--batch", type=int, default=60)
    main(ap.parse_args())
