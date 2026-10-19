"""S801 (pre-registered in knowledge/DECISIONS.md before any S801 data): are correct end-relative answers reads of the target picture?
Natural runs only (no edit).  Per scene, N in {6, 7, 8}, k in {1, 2, 3, 4, 5, N - 1, N} (both ends and the middle; amended before data),
j = N + 1 - k: END(k) and START(j), each unedited, with the picture at
the target j replaced by a picture of a menu word absent from the scene (target swap), and, as a control, with the picture at another block c
(c not in {j, k}, drawn per cell) replaced the same way (control swap).  Readout: two-form menu readout (S794), t1.
Scenes: procedural pictures (160 px), 40 per N, seed 80100.
Usage: python s801_grounding.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s801_{}_grounding.json"


def main(a):
    assert os.environ.get("R3_READ") in ("bare", "max"), "S801 reads the generated form (S794)"
    dev = os.environ.get("S801_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(s):
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    T1 = []

    def prompt(N, q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def ask(N, q, frame, pics, labs, menu):
        inp = v.encode(prompt(N, q, frame, menu), pics)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return dict(land=(labs.index(lab) + 1 if lab in labs else 0), ans=lab)

    rng = random.Random(a.seed); out = []
    for N in a.Ns:
        for si in range(a.n):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            nw = extra[0]
            r2 = random.Random(a.seed * 1000 + 100 * N + si)
            nwpic = proc_image(r2.choice(pool[nw]))[0].resize((a.pixels, a.pixels))
            ks = a.ks if a.ks else sorted({k_ for k_ in (1, 2, 3, 4, 5, N - 1, N) if 1 <= k_ <= N})      # both ends and the middle
            for k in ks:
                j = N + 1 - k
                c = r2.choice([b for b in range(1, N + 1) if b not in (j, k)])
                def swapped(p):
                    ps = list(pics); ps[p - 1] = nwpic; ls = list(labs); ls[p - 1] = nw
                    return ps, ls
                pT, lT = swapped(j); pC, lC = swapped(c)
                rec = dict(N=N, si=si, k=k, j=j, c=c, nw=nw)
                for nm, (q, fr) in (("END", (k, "last")), ("START", (j, "first"))):
                    rec[nm] = ask(N, q, fr, pics, labs, menu)
                    rec[f"{nm}|T"] = ask(N, q, fr, pT, lT, menu)
                    rec[f"{nm}|C"] = ask(N, q, fr, pC, lC, menu)
                out.append(rec)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{a.tag}] N={N} done", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=80100)
    ap.add_argument("--Ns", type=int, nargs="+", default=[6, 7, 8]); ap.add_argument("--ks", type=int, nargs="+", default=None)
    main(ap.parse_args())
