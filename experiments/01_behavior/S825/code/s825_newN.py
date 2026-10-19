"""S825 (pre-registered in knowledge/DECISIONS.md before any S825 data, with numerical predictions from exp/s825_fit.py): does the END error law
fitted at N = 7 / 8 predict new N?  Natural END(k) and START(k) runs at N = 6 and N = 9 (stated = shown), k = 1..N, 40 scenes (procedural pictures,
160 px; N = 6: 6 pictures + 3 extra words, N = 9: all 9 labels, menus of 9 words), seed 82500 (new).
Readout: two-form menu readout (S794), t1, the chosen word and the two-form logits of the menu words.
Usage: python s825_newN.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s825_{}_newN.json"


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S825_DEVICE", "cuda:0")
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
    prompt = lambda N, q, frame, menu: v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                                         f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)
    rng = random.Random(a.seed); out = []; nrun = 0
    for si in range(a.n):
        rec = dict(si=si, land={}, labs={}, menu={})
        for N in (6, 9):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            rec["labs"][str(N)] = labs; rec["menu"][str(N)] = menu
            conds = [(k, fr) for k in range(1, N + 1) for fr in ("last", "first")]
            encs = [v.encode(prompt(N, k, fr, menu), pics) for k, fr in conds]
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
    ap.add_argument("--seed", type=int, default=82500); ap.add_argument("--batch", type=int, default=18)
    main(ap.parse_args())
