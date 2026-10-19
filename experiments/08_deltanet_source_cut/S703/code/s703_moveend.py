"""S703: can the end of the image sequence be moved by hiding the last images from one module only?  (the end-moving table)
Pre-registration in knowledge/DECISIONS.md before any S703 data.  Qwen3.5-27B, seven pictures, END questions.
Prior (FINDINGS §596-§599, k = 2): the question reads the end through full-attention keys / values of the last block (L47 71%,
L43 + L47 93%; the route into the recurrent state 0), and L51 then selects the block one before.  S688: 27B is accurate for k = 2 and
k = N, and blurs for k in between.
The last m in {1, 2} images are made invisible to the question's text rows (exp/c_hide.py) through
    FA_ALL  full-attention keys at every full-attention layer      DN_ALL  DeltaNet memory + short convolution at every DeltaNet layer
    FA_4347 full-attention keys at L43 and L47 only (27B; relative .683 and .746)
for k in {1, 2, 3, 7 - m} (the last case asks for the first visible-end image: "the (7-m)-th from the last").
References: the true (7 - m)-image prefix run (same pictures, same menu) and the unedited seven-image run.
Readout per (arm, m, k): landing on the moved target T_m = 8 - m - k, on the unmoved target T_0 = 8 - k (if visible), on k (literal).
Registered rules (analysis exp/s703_analyze.py), 40 items, seed 70300:
    END_MOVES(arm, k) = P(T_m | arm) >= .5 P(T_m | prefix reference) and >= P(T_m | unedited) + .20, pooled over m
    Predictions: END_MOVES(FA_ALL, 2) and END_MOVES(FA_4347, 2); not END_MOVES(DN_ALL, 2); no prediction for k = 1, 3, 7 - m.
Usage: python s703_moveend.py MODEL TAG [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn
from c_hide import HideDN

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s703_{}_moveend.json"
NR = 7


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    DN_L = [i for i, L in enumerate(v.layers) if getattr(L, "linear_attn", None) is not None]
    L43 = min([l for l in FA_L if l >= round(.667 * (nl - 1))], key=lambda l: abs(l - round(.683 * (nl - 1))))
    L47 = min([l for l in FA_L if l > L43])
    ARMS = {"FA_ALL": (set(), set(FA_L)), "DN_ALL": (set(DN_L), set()), "FA_4347": (set(), {L43, L47})}
    ctx = Ctx(); FA = FullAttn(v, FA_L, ctx); HD = HideDN(v)
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

    def prompt(n, q, menu):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. What is in the {ORDW[q-1]} image from the last? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(q, order, pic, menu, m=0, arm=None):
        inp = v.encode(prompt(len(order), q, menu), [pic[l] for l in order])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        rows = list(range(ve[-1] + 1, len(ids)))
        ctx.reset(); HD.set(None)
        if m:
            dn_on, fa_on = ARMS[arm]
            b0 = len(order) - m
            assert ids[vs[b0] - 1] == ve_id
            H = list(range(vs[b0], ve[-1] + 1)); src3 = [vs[b0] - 3, vs[b0] - 2, vs[b0] - 1]
            if dn_on:
                HD.set(dict(layers=dn_on, H=H, rows=rows, src3=src3, noop=False))
            if fa_on:
                ctx.rows = rows; ctx.edit = {("F", li): {"ko": H} for li in fa_on}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        ctx.reset(); HD.set(None)
        return menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]

    print(f"[{a.tag}] FA {FA_L} · L43/L47 = {L43}/{L47} · DN {len(DN_L)}", flush=True)
    rng = random.Random(70300)
    rows_out = []
    while len(rows_out) < a.n:
        labs = rng.sample(labels, NR); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        pos = lambda lab: labs.index(lab) + 1 if lab in labs else 0
        rec = dict(base={}, hide={}, ref={})
        for k in (1, 2, 3, 5, 6):
            rec["base"][str(k)] = pos(run(k, labs, pic, menu))
        for m in (1, 2):
            for k in sorted({1, 2, 3, NR - m}):
                rec["ref"][f"{m}|{k}"] = pos(run(k, labs[:NR - m], pic, menu))
                for arm in ARMS:
                    rec["hide"][f"{arm}|{m}|{k}"] = pos(run(k, labs, pic, menu, m, arm))
        rows_out.append(rec)
        torch.cuda.empty_cache()
        if len(rows_out) % 10 == 0:
            print(f"[{a.tag}] {len(rows_out)}/{a.n}", flush=True)
    FA.remove(); HD.remove()
    json.dump(dict(tag=a.tag, model=a.model, FA=FA_L, L43=L43, L47=L47, arms=list(ARMS), rows=rows_out),
              open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
