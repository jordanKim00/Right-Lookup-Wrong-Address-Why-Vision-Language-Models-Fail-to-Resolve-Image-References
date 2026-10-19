"""S722: does the whole R³ effect on 27B travel through the queries that read the images?  (completing S710)
Pre-registration in knowledge/DECISIONS.md before any S722 data.  R³ writes only at question tokens after the images, so image keys and
values are identical in the unedited and repaired runs (causal attention); the repaired answer can reach image content only through the
question tokens' attention queries (full attention) or their recurrent reads.  Same design as S710 (readers r327b, R³-v3 plan with the
S711 frame rule, new dev scenes seed 72200, hard cells N7 k3-6 and N5 k3-4, 25 scenes per N), positions from "?" to the answer:
    R3_QBACK      repaired run, the 73 matcher heads' queries put back (S710 replication)
    R3_QBACK_ALL  repaired run, the queries of ALL full-attention heads at every full-attention layer put back
    END_QALL      original run given ALL full-attention queries of the repaired run
    END_Q         original run given the matcher queries (S710 replication)
Readout on rescued items (R3 correct, original wrong).  Registered rules (analysis exp/s722_analyze.py):
    QUERY_COMPLETE:   P(j | R3_QBACK_ALL) <= .15  (putting back every full-attention query removes nearly the whole repair)
    QUERY_SUFFICIENT: P(j | END_QALL) >= .60       (all full-attention queries of the repaired run carry most of the repair)
    MATCHER_SHARE:    report P(j | R3_QBACK) and P(j | END_Q) next to the all-head arms
    Predictions: QUERY_COMPLETE and QUERY_SUFFICIENT.
Usage: python s722_mediation_all.py MODEL TAG --base B --heads JSON [--n 25] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3ref import ORD, load_image
from c_tools import Ctx, FullAttn
import r3_build
from r3_build_md import ModelMD

C = 10


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    st = torch.load(f"/vlm/results/r3/s709_{a.base}_cal.pt", weights_only=False)
    cal, Lb = st["cal"], st["Lb"]
    M = ModelMD(v, Lb, Lb, [Lb], st["reader"])
    fr = cal["frame"]
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    nh = v.layers[FA_L[0]].self_attn.config.num_attention_heads
    MAT = [tuple(x) for x in json.load(open(a.heads))["matchers"]]
    MATL = {}
    for l, h in MAT:
        MATL.setdefault(l, []).append(h)
    ALLL = {l: list(range(nh)) for l in FA_L}                                       # every head of every full-attention layer
    RNDL = {}
    ctx = Ctx(); FA = FullAttn(v, FA_L, ctx)
    tid = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    keys = sorted({k for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})
    from r3_proc import proc_image
    from s588c_reader import probs as rp
    by_lab = {}
    for kk in keys[:3000]:
        lb = proc_image(int(kk.split(":")[1]))[1]
        if len(v.tok(" " + lb, add_special_tokens=False)["input_ids"]) == 1:
            by_lab.setdefault(lb, []).append(kk)
    labels = sorted(l for l, x in by_lab.items() if len(x) >= 10)

    def body(N, k, menu):
        return f"There are {N} images in a row. What is in the {ORD[k-1]} image from the last? Answer with only one word from: {', '.join(menu)}."

    def land(lg, labs, menu):
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return labs.index(lab) + 1 if lab in labs else 0

    def plan_writes(c):
        """R³-v3 plan (exp/r3_min_md.py: margin .5 detection, frame radius x1.2, frame always, ordinal when t != k_hat)."""
        Hb = c["H"][Lb]
        Dm = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"])                  # margin in the ordinal chart (any det_space)
        d2, k2 = Dm.topk(2, dim=1, largest=False)
        X = torch.cat([c["bf"][l] for l in M.Rlayers], 1)
        maxcode = int((rp(M.RP, X).argmax(1) + 1).max())
        writes = []
        for jx in (d2[:, 0] / d2[:, 1].clamp_min(1e-6) <= .5).nonzero().flatten().tolist():
            kh = int(k2[jx, 0]) + 1
            L, Rr = r3_build.clause_window(v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
            hits = [(q - c["s0"], a_) for q in L + Rr for a_ in fr["mu"] if float((Hb[q - c["s0"]] - fr["mu"][a_]).norm()) <= 1.2 * fr["thr"][a_]]
            if not hits:
                continue
            t_ = maxcode + 1 - kh
            if 1 <= t_ <= C and t_ != kh:
                writes.append((c["s0"] + jx, cal["Mb"][t_ - 1] - cal["Mb"][kh - 1]))
            if 1 <= t_ <= C:
                writes.append((c["s0"] + hits[0][0], fr["mu_start"] - fr["mu"][hits[0][1]]))
        return writes

    def run(inp, writes, srows, grab=False, edit=None):
        ctx.reset(); ctx.rows = srows
        if grab:
            ctx.grab = True; ctx.edit = {}
        elif edit is not None:
            ctx.edit = edit
        lg = M.run(inp, writes)
        stv = {k_: v_ for k_, v_ in ctx.store.items()} if grab else None
        ctx.reset()
        return lg, stv

    def edit_from(store, layers, part):
        return {("F", li): {"heads": hs, part: store[("F", li, part)]} for li, hs in layers.items()}

    rng = random.Random(a.seed)                                                     # 72200 (S722, 27B); 72300 = S723 scenes (S728, 9B)
    rows = []
    for N, ks in ((7, (3, 4, 5, 6)), (5, (3, 4))):
        for si in range(a.n):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [load_image(rng.choice(by_lab[l])) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            for k in ks:
                j = N + 1 - k
                c = M.capture(body(N, k, menu), pics)
                ids = c["ids"]
                qm = next(i for i in range(c["s0"], len(ids)) if "?" in v.tok.decode([ids[i]]))
                srows = list(range(qm, len(ids)))
                writes = plan_writes(c)
                lgE, sE = run(c["inp"], [], srows, grab=True)
                lgR, sR = run(c["inp"], writes, srows, grab=True)
                rec = dict(N=N, k=k, j=j, acted=bool(writes), land=dict(NATIVE=land(lgE, labs, menu), R3=land(lgR, labs, menu)))
                if writes:
                    arms = {"R3_QBACK": (writes, edit_from(sE, MATL, "q")), "R3_QBACK_ALL": (writes, edit_from(sE, ALLL, "q")),
                            "END_Q": ([], edit_from(sR, MATL, "q")), "END_QALL": ([], edit_from(sR, ALLL, "q"))}
                    for name, (w_, ed) in arms.items():
                        rec["land"][name] = land(run(c["inp"], w_, srows, edit=ed)[0], labs, menu)
                rows.append(rec)
            torch.cuda.empty_cache()
            if (si + 1) % 5 == 0:
                print(f"[{a.tag}] N{N} {si + 1}/{a.n}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, matchers={str(l): h for l, h in MATL.items()}, all_layers=FA_L, rows=rows),
              open(f"/vlm/results/r3/s722_{a.tag}_mediation.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s710_{a.tag}_mediation.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--device_map", default=None); ap.add_argument("--seed", type=int, default=72200)
    main(ap.parse_args())
