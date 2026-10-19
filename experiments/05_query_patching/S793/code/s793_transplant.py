"""S793 (pre-registered in knowledge/DECISIONS.md before any S793 data): does the natural end-relative error live in the lookup queries?
Bidirectional query transplant between the natural END(k) run and the START(j) run of the same scene (same images and menu; the two prompts
differ only in the ordinal and frame words, so every token position aligns), at the full-attention reader layers of two depth bands, for the
rows from the question mark to the answer.  Qwen3.5 only (exp/c_qpath.py recomputes the gated full attention of Qwen3.5).
Scenes: procedural pictures (160 px), N = 7, seed 79300 (new), 40 scenes, k in {2, 3, 5, 6}, j = N + 1 - k; the S730 / S780 prompt with
the stated count.  Bands: pre = full-attention layers at 33-55% depth, look = at 56-87% depth (the index-matching band of S705a / S720).
Arms per (scene, k): START, END (both capture their queries at every reader layer), NOOP (START given its own queries at every reader),
ES_pre / ES_look (END given START's queries: does the start-relative query repair the natural error?), SE_pre / SE_look (START given END's
queries: does the end-relative query reproduce the natural answer, right or wrong?).  Readout: the bare menu token (R3_READ=bare asserted),
the landed image, the vocabulary top-1 check; block shares of all heads at the answer row at every look-band layer for START, END, ES_look,
SE_look.
Usage: python s793_transplant.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath import QPath

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def main(a):
    assert os.environ.get("R3_READ") in ("bare", "max"), "S793 must read the generated form (FINDINGS §846, §851); set R3_READ=max"
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    assert v.fam == "qwen35", "QPath recomputes the gated full attention of Qwen3.5"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    dep = lambda l: l / (nl - 1)
    BANDS = {"pre": [l for l in FA_L if .33 <= dep(l) < .555], "look": [l for l in FA_L if .555 <= dep(l) <= .875]}
    READ = sorted(set(BANDS["pre"]) | set(BANDS["look"]))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(s):                                          # answer readout: the bare token the model generates
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    TOOL = QPath(v, READ, [])
    T1 = []

    def prompt(k, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, labs, menu, cfg):
        TOOL.set(cfg)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); TOOL.set(None)
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), st

    print(f"[{a.tag}] bands {BANDS}", flush=True)
    rng = random.Random(a.seed); out = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        for k in (2, 3, 5, 6):
            j = N + 1 - k
            iS, iE = v.encode(prompt(j, "first", menu), pics), v.encode(prompt(k, "last", menu), pics)
            idS, idE = iS["input_ids"][0].tolist(), iE["input_ids"][0].tolist()
            assert len(idS) == len(idE), "START and END prompts must align"
            S_ = len(idE)
            ve = [i for i, t in enumerate(idE) if t == ve_id]; vs = [i for i, t in enumerate(idE) if t == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([idE[i]]))
            assert [i for i, (x, y) in enumerate(zip(idS, idE)) if x != y] and max(i for i, (x, y) in enumerate(zip(idS, idE)) if x != y) < qm
            rows = list(range(qm, S_))
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            cap = dict(rows=[len(rows) - 1], owner=owner, nb=N)
            base = dict(rows=rows, grabq=set(READ), cap=cap)
            lS, stS = run(iS, labs, menu, base)
            lE, stE = run(iE, labs, menu, base)
            qS = {li: stS[("q", li)] for li in READ}; qE = {li: stE[("q", li)] for li in READ}
            rec = dict(si=si, k=k, j=j, land=dict(START=lS, END=lE), share={})
            sh = lambda st: {li: torch.softmax(st[("cap", li)][0].float(), -1).mean(0).tolist() for li in BANDS["look"] if ("cap", li) in st}
            rec["share"]["START"], rec["share"]["END"] = sh(stS), sh(stE)
            rec["land"]["NOOP"], _ = run(iS, labs, menu, dict(rows=rows, edit={li: dict(qd=qS[li]) for li in READ}))
            for bn, Ls in BANDS.items():
                lx, sx = run(iE, labs, menu, dict(rows=rows, cap=cap, edit={li: dict(qd=qS[li]) for li in Ls}))
                rec["land"][f"ES_{bn}"] = lx
                if bn == "look":
                    rec["share"]["ES_look"] = sh(sx)
                lx, sx = run(iS, labs, menu, dict(rows=rows, cap=cap, edit={li: dict(qd=qE[li]) for li in Ls}))
                rec["land"][f"SE_{bn}"] = lx
                if bn == "look":
                    rec["share"]["SE_look"] = sh(sx)
            out.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    TOOL.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), bands=BANDS, t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), rows=out), open(f"/vlm/results/r3/s793_{a.tag}_transplant.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s793_{a.tag}_transplant.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=79300)
    main(ap.parse_args())
