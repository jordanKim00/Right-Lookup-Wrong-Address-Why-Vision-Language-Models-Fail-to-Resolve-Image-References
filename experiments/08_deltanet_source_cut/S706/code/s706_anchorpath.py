"""S706: do the anchor heads hand the end of the image sequence to the matcher heads' queries?  (sender -> receiver, Cheng et al. style)
Pre-registration in knowledge/DECISIONS.md before any S706 data.  Qwen3.5-27B, N = 7, new scenes (seed 70600).
Head sets fixed on S705a's discovery scenes (results/r3/s705a_qa_27b_matchers.json): ANCHOR (11; 6 of them are not matchers -- PURE) and
MATCHER (73).  END_k, k in {2, 3}; the last m in {1, 2} images are made invisible to the question's text rows (keys masked) for:
    FA_ALL     every head of every full-attention layer (reference; S703)
    ANC        the ANCHOR heads only            ANC_PURE  the PURE anchor heads only
    ANC_RAND   as many random heads (not ANCHOR, not MATCHER; same layers, seed 70601)
    ANC_Q      ANC, and the MATCHER heads' queries (pre-RoPE, after q_norm) restored from the unedited run of the same question
    ANC_G      ANC, and the MATCHER heads' output gates restored instead
References: the unedited seven-image run (T_0 = 8-k) and the true (7-m)-image prefix (T_m = 8-m-k).
Readout: landing on T_m (moved end), T_0 (original end), k (literal); margin m_local at T_m.
Registered rules (analysis exp/s706_analyze.py), 40 scenes, pooled over k and m:
    ANCHORS_SET_END: P(T_m | ANC) >= .5 P(T_m | FA_ALL) and >= P(T_m | ANC_RAND) + .20
    PURE_SUFFICE (reported): P(T_m | ANC_PURE) >= .5 P(T_m | ANC)
    END_VIA_MATCHER_Q: P(T_0 | ANC_Q) >= P(T_0 | ANC) + .20 and P(T_0 | ANC_Q) > P(T_0 | ANC_G)
    Predictions: ANCHORS_SET_END, END_VIA_MATCHER_Q.
Usage: python s706_anchorpath.py MODEL TAG --heads JSON [--n 40] [--device_map balanced]
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
OUT = "/vlm/results/r3/s706_{}_anchorpath.json"


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
    PURE = [x for x in ANC if x not in set(MAT)]
    rr = random.Random(70601); pool_h = [(l, h) for l, _ in ANC for h in range(nh) if (l, h) not in set(ANC) | set(MAT)]
    RAND = []
    for l, _ in ANC:                                                              # one random head per anchor head, same layer
        cand = [x for x in pool_h if x[0] == l and x not in RAND]
        if cand:
            RAND.append(rr.choice(cand))
    by_layer = lambda hs: {l: sorted(h for ll, h in hs if ll == l) for l in sorted({l for l, _ in hs})}
    SETS = {"ANC": by_layer(ANC), "ANC_PURE": by_layer(PURE), "ANC_RAND": by_layer(RAND)}
    MATL = by_layer(MAT)
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
            if arm == "FA_ALL":
                ctx.edit = {("F", li): {"ko": H} for li in FA_L}
            else:
                base = "ANC" if arm in ("ANC_Q", "ANC_G") else arm
                hide = SETS[base]
                ed = {("F", li): {"ko": H, "ko_heads": hs, "heads": []} for li, hs in hide.items()}
                if arm in ("ANC_Q", "ANC_G"):
                    part = "q" if arm == "ANC_Q" else "gate"
                    for li, hs in MATL.items():
                        e = ed.setdefault(("F", li), {"heads": []})
                        e["heads"] = hs; e[part] = donor[("F", li, part)]
                        if "ko" in e and "ko_heads" not in e:
                            e["ko_heads"] = []
                ctx.edit = ed
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = {k_: v_ for k_, v_ in ctx.store.items()} if grab else None
        ctx.reset()
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(land=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs], store=st)

    print(f"[{a.tag}] ANC {SETS['ANC']} · PURE {SETS['ANC_PURE']} · RAND {SETS['ANC_RAND']} · MATCHER layers {sorted(MATL)}", flush=True)
    rng = random.Random(70600)
    rows_out = []
    while len(rows_out) < a.n:
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(base={}, ref={}, arm={})
        for k in (2, 3):
            b = run(k, labs, pic, menu, grab=True); rec["base"][str(k)] = dict(land=b["land"], ll=b["ll"])
            for m in (1, 2):
                rec["ref"][f"{m}|{k}"] = run(k, labs[:N - m], pic, menu)["land"]
                for arm in ("FA_ALL", "ANC", "ANC_PURE", "ANC_RAND", "ANC_Q", "ANC_G"):
                    x = run(k, labs, pic, menu, m=m, arm=arm, donor=b["store"])
                    rec["arm"][f"{arm}|{m}|{k}"] = dict(land=x["land"], ll=x["ll"])
        rows_out.append(rec)
        torch.cuda.empty_cache()
        if len(rows_out) % 10 == 0:
            print(f"[{a.tag}] {len(rows_out)}/{a.n}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, sets={k: {str(l): h for l, h in v_.items()} for k, v_ in SETS.items()},
                   matchers={str(l): h for l, h in MATL.items()}, rows=rows_out), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
