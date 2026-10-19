"""S705b: is 27B's END failure a failure of image selection at L35-47, which part of the selection matters (which image vs how much image),
and does the matching heads' failure travel through their query or their output gate?  Pre-registration in knowledge/DECISIONS.md before data.
Motivation: S704 -- imposing the START selection at L51 and later does not steer the answer (SEL_W .19); the decisive selection is earlier
(S687a re-analysis: only correct END items transmit the target block at L35-47; S688 tables: matcher heads at L39 / L47).
Qwen3.5-27B, N = 7, new scenes (seed 70510), targets t in {2, 3, 4}; recipient = NATIVE END(8-t); donors = START(t) (same scene, same target,
same menu; image keys / values identical) and START(t+1).  Edits at the suffix rows ("?" .. answer) of the full-attention layers L35-47:
    NOOP     recompute without change
    PTR      every head's attention weights over the image tokens <- START(t)'s (text weights rescaled)
    PTR_C    the same, but keeping END's total image mass: only the distribution within images is transplanted
    PTR_W    PTR with START(t+1) as donor (instrument)
    PTR_M    PTR on the MATCHER heads only (fixed on S705a's discovery scenes, --heads)
    Q_ONLY / G_ONLY / QG   the MATCHER heads' query (pre-RoPE, after q_norm) / output gate / both <- START(t)'s (c_tools.FullAttn)
Readout: landing (t, t+1, other); m_local = l(t) - max_{j != t} l(j).
Registered rules (analysis exp/s705b_analyze.py), 40 scenes:
    I0: P(t+1 | PTR_W) >= .40 (pooled) and NOOP == NATIVE in >= .95 of runs
    On NATIVE-wrong items whose START(t) is correct:
      PTR_RESCUE: P(t | PTR) >= .60;  COND_CARRIES: P(t | PTR_C) >= .5 P(t | PTR);  MATCHER_SUFFICIENT: P(t | PTR_M) >= .5 P(t | PTR)
      QK_ROUTE: P(t | Q_ONLY) >= .5 P(t | QG) and P(t | G_ONLY) < .5 P(t | QG);  GATE_ROUTE: the same with Q and G exchanged
    Predictions: I0, PTR_RESCUE, COND_CARRIES, MATCHER_SUFFICIENT, QK_ROUTE.
Usage: python s705b_select.py MODEL TAG --heads JSON [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn
from c_pat import PatAttn

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s705b_{}_select.json"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    R = lambda r: int(round(r * (nl - 1)))
    PTR_L = [l for l in FA_L if R(35 / 63) <= l <= R(47 / 63)]                     # 27B: 35, 39, 43, 47
    MH = json.load(open(a.heads))["matchers"]                                      # [[layer, head], ...]
    MHd = {}
    for l, h in MH:
        MHd.setdefault(int(l), []).append(int(h))
    MH_L = sorted(MHd)
    ctx = Ctx(); FA = FullAttn(v, sorted(set(FA_L)), ctx); PA = PatAttn(v, PTR_L)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
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
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
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
