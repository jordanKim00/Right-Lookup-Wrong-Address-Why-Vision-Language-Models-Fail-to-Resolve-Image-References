"""S708: what the identified mechanism predicts for a repair of 27B.  Pre-registration in knowledge/DECISIONS.md before any S708 data.
Mechanism (FINDINGS §729-§741): one index-matching route (image keys carry a forward index code); START queries hit it exactly for every
position; 27B's END query aims one step before the visible end and blurs beyond; 27B applies the END frame as an operator to whatever
ordinal address arrives (S679).  Predictions for an early rewrite at the ordinal interface (R3-style, oracle address):
  rewriting the ordinal to the forward target AND the frame to START sends every END question through the exact forward route;
  rewriting the ordinal alone lets 27B reflect it again (back to the literal position k);
  a deliberately wrong forward address is followed.
Qwen3.5-27B, N = 7.  Calibration (20 scenes, seed 70801): decoder-layer outputs at the ordinal token of START(t), t = 1..7 -> mu_ord[t];
at the frame word of START and END questions -> mu_fr[START], mu_fr[END].  Write layer L_w in {3, 8} (relative .05 / .13; the early ordinal
interface, S474b).  Test (30 scenes, seed 70800), END(k), k in {2, 3, 4, 5, 6} (target r = 8-k):
    BASE; ORD: h_ord += mu_ord[r] - mu_ord[k];  FR: h_fw += mu_fr[START] - mu_fr[END];  ORD_FR: both;
    WRONG: ORD_FR with a wrong forward address w (w not in {r, k}; w = r - 2 if r > 3 else r + 2);
    RAND_FR: FR plus a random ordinal delta of the same norm as mu_ord[r] - mu_ord[k] (seed 70802).
Readout: landing (r, k, w, other).
Registered rules (analysis exp/s708_analyze.py), 30 scenes, per write layer, means over k:
    REBIND_REPAIRS: P(r | ORD_FR) >= .80 and >= P(r | BASE) + .30
    REREFLECT: P(k | ORD) >= .40  (27B composes the written ordinal with the END frame again)
    ADDRESS_SPECIFIC: P(w | WRONG) >= .70
    Predictions: all three at L_w = 3.
Usage: python s708_rebind.py MODEL TAG [--n 40] [--ncal 20] [--device_map balanced]
"""
from __future__ import annotations

import argparse, os, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s708_{}_rebind.json"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    LW = sorted({max(1, int(round(r * (nl - 1)))) for r in (3 / 63, 8 / 63)})
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    def aid(s):                                          # answer readout (S789): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return tid(s)
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    T = v.tok.convert_tokens_to_ids
    im_start = T("<|im_start|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}

    def hook(li):
        def h(m, i, out):
            if ST.get("pos") is None:
                return out
            t = out[0] if isinstance(out, tuple) else out
            if ST.get("grab"):
                ST["cache"][li] = {k_: t[0, p].detach().float().cpu() for k_, p in ST["pos"].items()}
            e = ST.get("edit", {}).get(li)
            if e:
                t = t.clone()
                for k_, d in e.items():
                    p = ST["pos"][k_]; t[0, p] = (t[0, p].float() + d.to(t.device)).to(t.dtype)
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            return out
        return h
    hooks = [v.layers[li].register_forward_hook(hook(li)) for li in LW]

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(q, frame, labs, pic, menu, grab=False, edit=None):
        inp = v.encode(prompt(q, frame, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        turn = max(i for i, t in enumerate(ids) if t == im_start)
        o = max(i for i, t in enumerate(ids[:turn]) if t == tid(ORDW[q - 1]) and ids[i + 1] == tid("image"))
        fw = next(i for i in range(o + 1, turn) if ids[i] == tid(frame))
        ST.clear(); ST.update(pos={"ord": o, "fw": fw}, grab=grab, cache={}, edit=edit or {})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        cache = ST["cache"]; ST.clear()
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return dict(land=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs], cache=cache)

    def scene(rng):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        return labs, pic, menu

    rc = random.Random(70801)
    acc_ord = {li: {t: [] for t in range(1, N + 1)} for li in LW}; acc_fr = {li: {"first": [], "last": []} for li in LW}
    for _ in range(a.ncal):
        labs, pic, menu = scene(rc)
        for t in range(1, N + 1):
            for frame in ("first", "last"):
                c = run(t, frame, labs, pic, menu, grab=True)["cache"]
                for li in LW:
                    acc_fr[li][frame].append(c[li]["fw"])
                    if frame == "first":
                        acc_ord[li][t].append(c[li]["ord"])
    mu_ord = {li: {t: torch.stack(acc_ord[li][t]).mean(0) for t in range(1, N + 1)} for li in LW}
    d_fr = {li: torch.stack(acc_fr[li]["first"]).mean(0) - torch.stack(acc_fr[li]["last"]).mean(0) for li in LW}
    gen = torch.Generator().manual_seed(70802)
    print(f"[{a.tag}] 쓰기 층 {LW} · 보정 {a.ncal}", flush=True)
    rng = random.Random(70800)
    rows = []
    while len(rows) < a.n:
        labs, pic, menu = scene(rng)
        rec = dict(k={})
        for k in (2, 3, 4, 5, 6):
            r = N + 1 - k; w = r - 2 if r > 3 else r + 2
            cell = {"BASE": run(k, "last", labs, pic, menu)["land"]}
            for li in LW:
                d_ord = mu_ord[li][r] - mu_ord[li][k]; d_w = mu_ord[li][w] - mu_ord[li][k]
                rnd = torch.randn(d_ord.shape, generator=gen); rnd = rnd / rnd.norm() * d_ord.norm()
                arms = {"ORD": {"ord": d_ord}, "FR": {"fw": d_fr[li]}, "ORD_FR": {"ord": d_ord, "fw": d_fr[li]},
                        "WRONG": {"ord": d_w, "fw": d_fr[li]}, "RAND_FR": {"ord": rnd, "fw": d_fr[li]}}
                for name, ed in arms.items():
                    cell[f"{name}@{li}"] = run(k, "last", labs, pic, menu, edit={li: ed})["land"]
            rec["k"][str(k)] = dict(r=r, w=w, land=cell)
        rows.append(rec)
        torch.cuda.empty_cache()
        if len(rows) % 10 == 0:
            print(f"[{a.tag}] {len(rows)}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, LW=LW, rows=rows), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--ncal", type=int, default=20)
    ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
