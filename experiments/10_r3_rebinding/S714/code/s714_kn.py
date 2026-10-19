"""S714: the k = N boundary route of Qwen3.5-27B -- how "the N-th image from the last" (= the first image) is answered although it
does not use the end anchor (S701 y = 5, S703 k = 7 - m).  Pre-registration in knowledge/DECISIONS.md before any S714 data.
Descriptive lead (S705a saved arrays, re-read 2026-10-09): for END(k = 7) the answer-row image selection of the full-attention heads sits
on the last block at L35-43 (block 7 share .30-.35, block 1 .08-.12) and moves to block 1 only at L51-55 (.26-.37; L55 matcher heads up to
.58), while END(k = 2) is decided at L35-47 (S705b) and later selection edits do not move it (S704).
Procedural scenes as S703 (training pool, 160 px), N in {5, 7}, canonical prompt "There are N images in a row. What is in the k-th image
from the last? Answer with only one word from: <menu>."  Image 1 is hidden from the full-attention keys of the question's text rows
(c_tools FullAttn "ko", as S703) at a layer window.
Arms, END(k = N) (target 1):
    BASE        unedited
    NOCOUNT     the count sentence removed
    COUNT_UP    the count sentence states N + 1 (a stated-count route N' + 1 - k would give 2)
    HIDE1_ALL   image 1 hidden at every full-attention layer
    HIDE1_EARLY image 1 hidden at the full-attention layers of L35-47 (27B; relative .54-.75)
    HIDE1_LATE  image 1 hidden at the full-attention layers from L51 on (relative >= .80)
Controls: START(1) ("What is in the first image?") under BASE, HIDE1_ALL, HIDE1_EARLY, HIDE1_LATE; END(k = N - 1) (target 2) under BASE,
NOCOUNT, COUNT_UP.
Registered rules (analysis exp/s714_analyze.py), 30 scenes per N, seed 71400:
    I0 HIDE_INSTRUMENT: P(2 | START(1), HIDE1_ALL) >= .50
    KN_COUNT:         P(1 | END(N), COUNT_UP) <= P(1 | END(N), BASE) - .30
    KN_FIRST_VISIBLE: P(2 | END(N), HIDE1_ALL) >= .50
    KN_LATE:          P(1 | END(N), HIDE1_LATE) <= P(1 | BASE) - .30 and P(1 | END(N), HIDE1_EARLY) >= P(1 | BASE) - .10
    Predictions: I0, KN_LATE (the descriptive lead), KN_COUNT; none for KN_FIRST_VISIBLE.
Usage: python s714_kn.py MODEL TAG [--n 30] [--device_map balanced] [--seed 71400]
"""
from __future__ import annotations

import argparse, json, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s714_{}_kn.json"


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    EARLY = [l for l in FA_L if .54 <= l / (nl - 1) <= .75]
    LATE = [l for l in FA_L if l / (nl - 1) >= .80]
    WIN = {"HIDE1_ALL": FA_L, "HIDE1_EARLY": EARLY, "HIDE1_LATE": LATE}
    ctx = Ctx(); FA = FullAttn(v, FA_L, ctx)
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

    def prompt(n, q, frame, menu, count):
        cs = "" if count is None else f"There are {count} images in a row. "
        qq = f"What is in the {ORDW[q-1]} image from the last?" if frame == "END" else f"What is in the {ORDW[q-1]} image?"
        return v.chat([("user", f"{v.IMG*n}\n{cs}{qq} Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(n, q, frame, pics, menu, count, hide=None):
        inp = v.encode(prompt(n, q, frame, menu, count), pics)
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        rows = list(range(ve[-1] + 1, len(ids)))
        ctx.reset()
        if hide:
            H = list(range(vs[0], ve[0] + 1))                                     # image 1 (vision_start .. vision_end)
            ctx.rows = rows; ctx.edit = {("F", li): {"ko": H} for li in WIN[hide]}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        ctx.reset()
        return menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]

    print(f"[{a.tag}] FA {FA_L} · EARLY {EARLY} · LATE {LATE}", flush=True)
    rng = random.Random(a.seed)
    rows_out = []
    for N in (7, 5):
        for si in range(a.n):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            pos = lambda lab: labs.index(lab) + 1 if lab in labs else 0
            rec = dict(N=N, si=si, EN={}, S1={}, EN1={})
            for arm, (count, hide) in {"BASE": (N, None), "NOCOUNT": (None, None), "COUNT_UP": (N + 1, None),
                                       "HIDE1_ALL": (N, "HIDE1_ALL"), "HIDE1_EARLY": (N, "HIDE1_EARLY"), "HIDE1_LATE": (N, "HIDE1_LATE")}.items():
                rec["EN"][arm] = pos(run(N, N, "END", pics, menu, count, hide))
            for arm in ("BASE", "HIDE1_ALL", "HIDE1_EARLY", "HIDE1_LATE"):
                rec["S1"][arm] = pos(run(N, 1, "START", pics, menu, N, None if arm == "BASE" else arm))
            for arm, count in (("BASE", N), ("NOCOUNT", None), ("COUNT_UP", N + 1)):
                rec["EN1"][arm] = pos(run(N, N - 1, "END", pics, menu, count))
            rows_out.append(rec)
            torch.cuda.empty_cache()
            if (si + 1) % 10 == 0:
                print(f"[{a.tag}] N{N} {si + 1}/{a.n}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, FA=FA_L, EARLY=EARLY, LATE=LATE, rows=rows_out), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=71400)                                # smoke runs use another seed
    main(ap.parse_args())
