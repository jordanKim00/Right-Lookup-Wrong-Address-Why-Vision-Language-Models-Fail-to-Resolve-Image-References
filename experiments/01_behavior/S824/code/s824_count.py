"""S824 (pre-registered in knowledge/DECISIONS.md before any S824 data): which errors are target-relative, literal-relative or absolute, and which
count does the end-relative target use - the stated count or the pictures shown?  FINDINGS §911 addendum (saved clean runs, S801 only, 40 cells
per k): at N = 8 END(5) lands on 4 .625, 3 .175, 5 .10, 2 .000 (the N = 7 '2' errors move with the target, not to picture 2); at N = 8 END(7)
lands on 6 .475 (one step from the literal 7).  The user's advisor (10-16 14:0x): confirm on new scenes and manipulate the stated count apart from
the pictures.
Per scene (procedural pictures, 160 px; 8 pictures drawn, the 7-picture condition shows the first seven; menu = the eight labels + one extra word,
the same in every condition): shown N_a in {7, 8} x stated N_s in {7, 8} ('There are N_s images in a row.') x k in {3, 4, 5, 6, 7} x frame in
{END, START} - natural runs only, batched per N_a.
Readout: two-form menu readout (S794), t1, the chosen word and the two-form logits of the nine menu words.
Scenes: seed 82400 (new), 40 scenes.
Usage: python s824_count.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s824_{}_count.json"


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max", "S824 reads the generated form in either tokenization (S794)"
    dev = os.environ.get("S824_DEVICE", "cuda:0")
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

    rng = random.Random(a.seed); out = []; nrun = 0
    for si in range(a.n):
        labs = rng.sample(labels, 8); extra = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={})
        for Na in (7, 8):
            conds = [(Ns, k, fr) for Ns in (7, 8) for k in a.ks for fr in ("last", "first")]
            encs = [v.encode(prompt(Na, Ns, k, fr, menu), pics[:Na]) for Ns, k, fr in conds]
            assert len({len(e["input_ids"][0]) for e in encs}) == 1, "prompts of one N_a must align"
            with torch.no_grad():
                for c0 in range(0, len(encs), a.batch):
                    part = encs[c0:c0 + a.batch]
                    lg = v.model(**batch_inputs(part), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                    for b, (Ns, k, fr) in enumerate(conds[c0:c0 + a.batch]):
                        nm = f"a{Na}_s{Ns}_k{k}_{'E' if fr == 'last' else 'S'}"
                        T1.append(int(int(lg[b].argmax()) in t1ids(menu)))
                        w = menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())]
                        rec["land"][nm] = labs.index(w) + 1 if w in labs[:Na] else 0
                        rec["mlog"][nm] = [float(sc(lg[b], x)) for x in labs + extra]
            nrun += len(encs)
        out.append(rec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n} · {nrun} runs", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), ks=a.ks, t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=82400); ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--batch", type=int, default=20)
    main(ap.parse_args())
