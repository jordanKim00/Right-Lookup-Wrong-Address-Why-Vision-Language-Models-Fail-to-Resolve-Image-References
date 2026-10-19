"""S707: which heads read the end at L43 / L47 -- the matcher heads themselves, or the other heads?  Pre-registration in knowledge/DECISIONS.md
before any S707 data.  Qwen3.5-27B, N = 7, new scenes (seed 70700).  S706: hiding the last images from the 11 last-block-reading heads does
not move the end (.14), hiding them from every head at L43 + L47 does (S703 .60 / .55).  MATCHER heads (73) fixed on S705a's discovery scenes.
END_k, k in {2, 3}; the last m in {1, 2} images are made invisible to the question's text rows (keys masked) for:
    ALL_4347     every head at L43 and L47 (reference)
    MAT_4347     the MATCHER heads at L43 and L47       NONMAT_4347  every other head at L43 and L47
    MAT_ALL      the MATCHER heads at all their layers  NONMAT_ALL   every non-MATCHER head at every full-attention layer
References: the unedited run (T_0 = 8-k), the true (7-m)-image prefix (T_m = 8-m-k).
Registered rules (analysis exp/s707_analyze.py), 40 scenes, pooled over k and m:
    END_IN_MATCHERS: P(T_m | MAT_4347) >= .5 P(T_m | ALL_4347) and >= P(T_m | NONMAT_4347) + .15
    END_OUTSIDE_MATCHERS: P(T_m | NONMAT_4347) >= .5 P(T_m | ALL_4347) and >= P(T_m | MAT_4347) + .15
    (reported) MAT_ALL and NONMAT_ALL
    Prediction: END_IN_MATCHERS.
Usage: python s707_endreaders.py MODEL TAG --heads JSON [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s707_{}_endreaders.json"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    nh = v.layers[FA_L[0]].self_attn.config.num_attention_heads
    hj = json.load(open(a.heads))
    ANC = [tuple(x) for x in hj["anchors"]]; MAT = [tuple(x) for x in hj["matchers"]]
    by_layer = lambda hs: {l: sorted(h for ll, h in hs if ll == l) for l in sorted({l for l, _ in hs})}
    R = lambda r: int(round(r * (v.n_layer - 1)))
    L43 = min([l for l in FA_L if l >= R(.667)], key=lambda l: abs(l - R(.683))); L47 = min(l for l in FA_L if l > L43)
    MATS = set(MAT)
    SETS = {"ALL_4347": {l: list(range(nh)) for l in (L43, L47)},
            "MAT_4347": {l: sorted(h for ll, h in MAT if ll == l) for l in (L43, L47)},
            "NONMAT_4347": {l: [h for h in range(nh) if (l, h) not in MATS] for l in (L43, L47)},
            "MAT_ALL": by_layer(MAT),
            "NONMAT_ALL": {l: [h for h in range(nh) if (l, h) not in MATS] for l in FA_L}}
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

    def prompt(n, q, menu):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. What is in the {ORDW[q-1]} image from the last? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(k, labs, pic, menu, m=0, arm=None, donor=None, grab=False):
        inp = v.encode(prompt(len(labs), k, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        rows = list(range(ve[-1] + 1, len(ids)))
        ctx.reset(); ctx.rows = rows
        if grab:
            ctx.grab = True; ctx.edit = {}
        elif m:
            H = list(range(vs[N - m], ve[-1] + 1))
            ctx.edit = {("F", li): {"ko": H, "ko_heads": hs, "heads": []} for li, hs in SETS[arm].items() if hs}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = {k_: v_ for k_, v_ in ctx.store.items()} if grab else None
        ctx.reset()
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(land=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs], store=st)

    print(f"[{a.tag}] L43/L47 {L43}/{L47} · " + " · ".join(f"{k_} {sum(len(v_) for v_ in s_.values())}" for k_, s_ in SETS.items()), flush=True)
    rng = random.Random(70700)
    rows_out = []
    while len(rows_out) < a.n:
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(base={}, ref={}, arm={})
        for k in (2, 3):
            b = run(k, labs, pic, menu); rec["base"][str(k)] = dict(land=b["land"], ll=b["ll"])
            for m in (1, 2):
                rec["ref"][f"{m}|{k}"] = run(k, labs[:N - m], pic, menu)["land"]
                for arm in SETS:
                    x = run(k, labs, pic, menu, m=m, arm=arm)
                    rec["arm"][f"{arm}|{m}|{k}"] = dict(land=x["land"], ll=x["ll"])
        rows_out.append(rec)
        torch.cuda.empty_cache()
        if len(rows_out) % 10 == 0:
            print(f"[{a.tag}] {len(rows_out)}/{a.n}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, arms=list(SETS), sets={k: {str(l): h for l, h in v_.items()} for k, v_ in SETS.items()},
                   rows=rows_out), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
