"""S616b: choose the write layer inside the model by forward rebinding efficacy, then rebuild S609c and evaluate
(pre-registered, exp/QUEUE.md S616b). One model load.
Efficacy at layer L (forward items only, procedural pictures): question "What is in the k-th image?" -- write
Mb_L[k'] - Mb_L[k] on the ordinal, success = the answer names block k'; options "Which image shows a <label of block j>?"
whose options do not point to j -- rewrite option X's ordinal to j, success = option X is chosen.
Write layer = argmax over L of min(question efficacy, options efficacy), ties -> the earlier layer.
Usage: python s616b_pipe.py MODEL BASE [--n 60]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from r3ref import ORD

C = 10
SWEEP_ONLY = False


def main(model, base, n):
    v = Uni(model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    import os
    if not os.path.exists(f"/vlm/results/s588c_reader_s609_{base}.pt") and not SWEEP_ONLY:   # new model: S608 site + reader first
        import s608_layout, s609_reader
        out = s608_layout.main(v, f"s608_{base}", None, 300, 60, 2)
        site = max(out, key=lambda k: (min(out[k].values()), -int(k.split("|")[0][1:])))
        print(f"[s616b {base}] 끝 읽기 자리 {site} (네 배치 최소 {min(out[site].values()):.3f})", flush=True)
        s609_reader.main(v, f"s609r_{base}", base, int(site.split("|")[0][1:]), site.split("|")[1], 500, 60)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    D = len(v.layers); cands = list(range(1, round(.3 * D) + 1))
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    ord_ids = {tok1(" " + ORD[i]): i + 1 for i in range(C)}
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    rng = random.Random(61610)
    ST = dict(cap=None, write=None)

    def hook(l):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if ST["cap"] is not None and l in cands:
                ST["cap"][l] = t[0, ST["pos"]].float().cpu()
            if ST["write"] is not None and ST["write"][0] == l:
                t2 = t.clone(); _, pos, dlt = ST["write"]
                t2[0, pos] = (t[0, pos].float() + dlt.to(t.device)).to(t.dtype)
                return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
            return o
        return h
    hks = [v.layers[l].register_forward_hook(hook(l)) for l in cands]

    def scene(N):
        ps, shapes = [], set()
        for p_ in rng.sample(pids, 80):
            sh = proc_image(p_)[1]
            if sh not in shapes:
                ps.append(p_); shapes.add(sh)
            if len(ps) == N:
                break
        rs = [proc_image(p_) for p_ in ps]
        return [r_[0] for r_ in rs], [r_[1] for r_ in rs]

    def enc(body, pics):
        return v.encode(v.chat([("user", f"{v.IMG*len(pics)}\n{body}")], prefix_assistant=THINK), pics)

    @torch.no_grad()
    def logits(inp):
        return v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()

    # prototypes per candidate layer (forward questions, ordinal tokens)
    protos = {l: {k: [] for k in range(1, C + 1)} for l in cands}
    for _ in range(80):
        N = rng.randint(3, 10); pics, labs = scene(N)
        k = rng.randint(1, N)
        inp = enc(f"What is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(labs)}.", pics)
        ids = inp["input_ids"][0].tolist(); pos = max(i for i, t in enumerate(ids) if t in ord_ids)
        ST.update(cap={}, pos=pos); logits(inp); cap = ST["cap"]; ST["cap"] = None
        for l in cands:
            protos[l][ord_ids[ids[pos]]].append(cap[l])
    Mb = {l: {k: torch.stack(x).mean(0) for k, x in protos[l].items() if x} for l in cands}
    effq = {l: [] for l in cands}; effo = {l: [] for l in cands}
    for _ in range(n):
        N = rng.randint(3, 6); pics, labs = scene(N)
        k = rng.randint(1, N); k2 = rng.choice([x for x in range(1, N + 1) if x != k])
        opts = list(labs); rng.shuffle(opts); cand = [tok1(" " + o) for o in opts]
        inp = enc(f"What is in the {ORD[k-1]} image? Answer with only one word from: {', '.join(opts)}.", pics)
        ids = inp["input_ids"][0].tolist(); pos = max(i for i, t in enumerate(ids) if t in ord_ids)
        for l in cands:
            if k not in Mb[l] or k2 not in Mb[l]:
                continue
            ST["write"] = (l, pos, Mb[l][k2] - Mb[l][k]); lg = logits(inp); ST["write"] = None
            effq[l].append(opts[int(lg[cand].argmax())] == labs[k2 - 1])
        j = rng.randint(1, N); others = [x for x in range(1, N + 1) if x != j]
        ks = rng.sample(others, min(3, len(others))); X = rng.randrange(len(ks))
        body = (f"Which image shows a {labs[j-1]}?\nSelect from the following choices.\n" + "\n".join(f"{chr(65+i)}: The {ORD[kk-1]} image" for i, kk in enumerate(ks)) +
                "\nAnswer with the option's letter from the given choices directly.")
        inp = enc(body, pics); ids = inp["input_ids"][0].tolist()
        opos = [i for i, t in enumerate(ids) if t in ord_ids][-len(ks):]
        letters = [tok1(chr(65 + i)) for i in range(len(ks))]
        for l in cands:
            if ks[X] not in Mb[l] or j not in Mb[l]:
                continue
            ST["write"] = (l, opos[X], Mb[l][j] - Mb[l][ks[X]]); lg = logits(inp); ST["write"] = None
            effo[l].append(int(lg[letters].argmax()) == X)
    for h in hks:
        h.remove()
    eff = {l: (float(np.mean(effq[l])), float(np.mean(effo[l]))) for l in cands}
    print(f"[s616b {base}] 층별 전방 재결속 효력 [질문/보기]: " + " ".join(f"L{l} {a:.2f}/{b:.2f}" for l, (a, b) in eff.items()), flush=True)
    Lb = max(cands, key=lambda l: (min(eff[l]), -l))
    print(f"[s616b {base}] 쓰기 층 L{Lb} (효력 {eff[Lb][0]:.2f}/{eff[Lb][1]:.2f})", flush=True)
    json.dump(dict(base=base, eff={int(l): x for l, x in eff.items()}, Lb=int(Lb)), open(f"/vlm/results/r3/s616b_{base}_writelayer.json", "w"))
    if SWEEP_ONLY:                                                                 # validation on the studied models
        return
    Lr = round(.36 * D); Lrel = sorted({round(.36 * D), round(.44 * D), round(.56 * D)})
    import r3_build as rb
    rb.BIND, rb.BIND_MODE, rb.BIND_TH = True, "hard", "mid"
    rb.run_build(v, f"r3s616b_{base}", "/vlm/assets/r3cal_a3", Lb, Lr, Lrel, f"s609_{base}", a3=True, epochs=40)
    from r3_repair import R3Repair
    import s582_evalrepair as s5
    rep = R3Repair(v, f"/vlm/results/r3/r3s616b_{base}_r3.pt", f"s609_{base}")
    s5.run(v, rep, f"e582_s616b_{base}", {"mmiu_pos", "muir", "mmiu_other", "muir_all"}, per_task=50, per_other=25, cap=None, sealed=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("base"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--sweep_only", action="store_true")
    a = ap.parse_args()
    SWEEP_ONLY = a.sweep_only
    main(a.model, a.base, a.n)
