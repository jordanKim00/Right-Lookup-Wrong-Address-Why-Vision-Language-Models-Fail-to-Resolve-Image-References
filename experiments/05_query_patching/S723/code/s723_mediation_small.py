"""S723: the S710 repair-mediation test in a smaller model (Qwen3.5-9B) -- does the repair also travel through the index-matching
heads' queries when the original failure is a literal query?  Pre-registration in knowledge/DECISIONS.md before any S723 data.
Latest R³ plan (R³-v3 rules, as exp/r3_min_md.py) on the earlier module's calibration for Qwen3.5-9B (base lm_35_9b: write layer 4, reader
s609_35_9b); index-matching heads = exp/s720_heads.py set (16 heads at L23/27).  New dev scenes (seed 72300), N7 k3-6 and N5 k3-4,
25 scenes per N, positions from "?" to the answer.  Arms as S710: R3_QBACK / R3_GBACK / R3_RBACK (repaired run, matcher queries / gates /
as many random non-matcher heads' queries put back), END_QR3 / END_GR3 / END_RR3 (original run given them).
Registered rules (analysis exp/s723_analyze.py = S710 rules on rescued items):
    MEDIATION_NECESSARY:  P(j | R3_QBACK) <= .70 and P(j | R3_RBACK) >= .90
    MEDIATION_SUFFICIENT: P(j | END_QR3) >= .20, >= P(j | END_GR3) + .10 and >= P(j | END_RR3) + .10
    Predictions: both.
Usage: python s723_mediation_small.py MODEL TAG --base lm_35_9b --heads JSON [--n 25]
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
    rr = random.Random(72301); RNDL = {}
    for l, hs in MATL.items():
        pool = [h for h in range(nh) if h not in hs]
        RNDL[l] = sorted(rr.sample(pool, min(len(hs), len(pool))))
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

    rng = random.Random(72300)
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
                    arms = {"R3_QBACK": (writes, edit_from(sE, MATL, "q")), "R3_GBACK": (writes, edit_from(sE, MATL, "gate")),
                            "R3_RBACK": (writes, edit_from(sE, RNDL, "q")), "END_QR3": ([], edit_from(sR, MATL, "q")),
                            "END_GR3": ([], edit_from(sR, MATL, "gate")), "END_RR3": ([], edit_from(sR, RNDL, "q"))}
                    for name, (w_, ed) in arms.items():
                        rec["land"][name] = land(run(c["inp"], w_, srows, edit=ed)[0], labs, menu)
                rows.append(rec)
            torch.cuda.empty_cache()
            if (si + 1) % 5 == 0:
                print(f"[{a.tag}] N{N} {si + 1}/{a.n}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, matchers={str(l): h for l, h in MATL.items()}, random={str(l): h for l, h in RNDL.items()}, rows=rows),
              open(f"/vlm/results/r3/s723_{a.tag}_mediation.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s710_{a.tag}_mediation.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
