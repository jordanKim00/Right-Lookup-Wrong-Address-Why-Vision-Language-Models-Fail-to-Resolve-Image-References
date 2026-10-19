"""S720: small-model contrast of S705b (same script body, same scenes and seed 70510, N = 7) -- is the END failure of Qwen3.5-4B / 9B,
whose index-matching heads aim END at the literal k-th image (§740), also a failure of the matching query?  Pre-registration in
knowledge/DECISIONS.md before any S720 data.  Heads: the model's index-matching heads (exp/s720_heads.py: START diagonal share >= .5 on
the S705a discovery scenes; 4B 31 heads at L19/23/27, 9B 16 heads at L23/27); the attention-pattern arms act on the layers of those
heads (27B used L35-47).  Arms and readout as S705b (NOOP, PTR, PTR_C, PTR_W, PTR_M, Q_ONLY, G_ONLY, QG; recipient END(8-t), donor START(t)).
Registered rules (analysis exp/s720_analyze.py = exp/s705b_analyze.py logic), 40 scenes:
    I0: P(t+1 | PTR_W) >= .40 and NOOP == NATIVE in >= .95 of runs
    On NATIVE-wrong items whose START(t) is correct: PTR_RESCUE P(t | PTR) >= .60; MATCHER_SUFFICIENT P(t | PTR_M) >= .5 P(t | PTR);
    QK_ROUTE P(t | Q_ONLY) >= .5 P(t | QG) and P(t | G_ONLY) < .5 P(t | QG)
    Predictions: I0, PTR_RESCUE, MATCHER_SUFFICIENT, QK_ROUTE -- the lookup is shared and the END failure enters through the query.
Usage: python s720_select.py MODEL TAG --heads JSON [--n 40]
"""
from __future__ import annotations

import argparse, os, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn
from c_pat import PatAttn

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s720_{}_select.json"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    R = lambda r: int(round(r * (nl - 1)))
    MH = json.load(open(a.heads))["matchers"]                                      # [[layer, head], ...]
    MHd = {}
    for l, h in MH:
        MHd.setdefault(int(l), []).append(int(h))
    MH_L = sorted(MHd)
    PTR_L = list(MH_L)                                                             # S720: the index heads' layers
    ctx = Ctx(); FA = FullAttn(v, sorted(set(FA_L)), ctx); PA = PatAttn(v, PTR_L)
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

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(q, frame, labs, pic, menu, pat=None, qg=None, grab=False, noop=False):
        inp = v.encode(prompt(q, frame, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        img = [p for b in range(N) for p in range(vs[b], ve[b] + 1)]
        trows = list(range(ve[-1] + 1, len(ids)))
        qm = next(i for i in trows if "?" in v.tok.decode([ids[i]]))
        srows = list(range(qm, len(ids)))
        ctx.reset(); PA.set(None)
        if grab:                                                                  # donor capture: image weights at PTR_L, q / gate at MH_L
            PA.set(dict(rows=srows, pos=img, grab=set(PTR_L), edit={}, noop=set()))
            ctx.rows = srows; ctx.grab = True; ctx.edit = {}
        elif pat is not None:
            st, cond, heads = pat
            PA.set(dict(rows=srows, pos=img, grab=set(), edit={li: st[li] for li in PTR_L}, noop=set(), cond=cond,
                        **({"heads": heads} if heads is not None else {})))
        elif qg is not None:
            st, parts = qg
            ctx.rows = srows
            ctx.edit = {("F", li): dict(heads=MHd[li], **{p_: st[("F", li, p_)] for p_ in parts}) for li in MH_L}
        elif noop:
            PA.set(dict(rows=srows, pos=img, grab=set(), edit={}, noop=set(PTR_L)))
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        out = dict(land=None, ll=[float(lg[tid(l)]) for l in labs])
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        out["land"] = labs.index(lab) + 1 if lab in labs else 0
        if grab:
            out["pat"] = dict(PA.store); out["qg"] = {k: v_ for k, v_ in ctx.store.items()}
        ctx.reset(); PA.set(None)
        return out

    print(f"[{a.tag}] PTR {PTR_L} · MATCHER {MHd}", flush=True)
    rng = random.Random(70510)
    rows_out = []
    while len(rows_out) < a.n:
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(t={})
        for t in (2, 3, 4):
            S = run(t, "first", labs, pic, menu, grab=True); SW = run(t + 1, "first", labs, pic, menu, grab=True)
            k = N + 1 - t
            r = dict(START=dict(land=S["land"], ll=S["ll"]))
            r["NATIVE"] = run(k, "last", labs, pic, menu)
            r["NOOP"] = run(k, "last", labs, pic, menu, noop=True)
            r["PTR"] = run(k, "last", labs, pic, menu, pat=(S["pat"], False, None))
            r["PTR_C"] = run(k, "last", labs, pic, menu, pat=(S["pat"], True, None))
            r["PTR_W"] = run(k, "last", labs, pic, menu, pat=(SW["pat"], False, None))
            r["PTR_M"] = run(k, "last", labs, pic, menu, pat=(S["pat"], False, {li: MHd.get(li, []) for li in PTR_L}))
            r["Q_ONLY"] = run(k, "last", labs, pic, menu, qg=(S["qg"], ("q",)))
            r["G_ONLY"] = run(k, "last", labs, pic, menu, qg=(S["qg"], ("gate",)))
            r["QG"] = run(k, "last", labs, pic, menu, qg=(S["qg"], ("q", "gate")))
            rec["t"][str(t)] = {kk: dict(land=vv["land"], ll=vv["ll"]) for kk, vv in r.items()}
        rows_out.append(rec)
        torch.cuda.empty_cache()
        if len(rows_out) % 10 == 0:
            print(f"[{a.tag}] {len(rows_out)}/{a.n}", flush=True)
    FA.remove(); PA.remove()
    json.dump(dict(tag=a.tag, model=a.model, PTR=PTR_L, MATCHER=MHd, rows=rows_out), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
