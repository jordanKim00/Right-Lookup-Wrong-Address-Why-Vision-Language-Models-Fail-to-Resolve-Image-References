"""S710: does the R³ gain on 27B travel through the matcher heads' queries?  (repair mediation)  Pre-registration in
knowledge/DECISIONS.md before any S710 data.  Readers and R³-min plan exactly as S709 (exp/s709_eval.py, calibration exp/s709_cal.py).
Qwen3.5-27B, new procedural scenes from the dev pool (seed 71000), END questions on hard cells (N = 7, k in {3, 4, 5, 6}; N = 5, k in
{3, 4}).  Per item: NATIVE (unedited) and R3 (R³-min writes) runs grab the MATCHER heads' queries (pre-RoPE, after q_norm) and output
gates at the suffix rows ("?" .. answer) of their layers (S705a discovery set, results/r3/s705a_qa_27b_matchers.json).  Arms:
    R3_QBACK   R3 run with the matcher queries put back to NATIVE's          R3_GBACK   ... with the matcher gates put back instead
    R3_RBACK   R3 run with as many random non-matcher heads' queries put back (same layers, seed 71001)
    END_QR3    NATIVE run given the R3 run's matcher queries                  END_GR3    ... given the R3 run's matcher gates
    END_RR3    NATIVE run given the R3 run's random-head queries
Readout: landing on j; analysed on rescued items (R3 correct, NATIVE wrong) and on all items.
Registered rules (analysis exp/s710_analyze.py), 25 scenes per N:
    MEDIATION_NECESSARY: on rescued items P(j | R3_QBACK) <= 1 - .30 (the gain drops by >= 30 points) and P(j | R3_RBACK) >= .90
    MEDIATION_SUFFICIENT: on rescued items P(j | END_QR3) >= .20 and >= P(j | END_GR3) + .10 and >= P(j | END_RR3) + .10
    Predictions: both (with the S705b caveat that the matcher-query route carried only part of the oracle rescue).
Usage: python s710_mediation.py MODEL TAG --base B --heads JSON [--n 25] [--device_map balanced]
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
    rr = random.Random(71001); RNDL = {}
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
        Hb = c["H"][Lb]
        if cal["det_space"] == "full":
            dist, kh0 = torch.cdist(Hb, cal["Mb"]).min(1); th = cal["th_full"]
        else:
            dist, kh0 = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]).min(1); th = cal["th_chart"]
        X = torch.cat([c["bf"][l] for l in M.Rlayers], 1)
        maxcode = int((rp(M.RP, X).argmax(1) + 1).max())
        writes = []
        for jx in (dist <= th).nonzero().flatten().tolist():
            kh = int(kh0[jx]) + 1
            L, Rr = r3_build.clause_window(v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
            hits = [(q - c["s0"], a_) for q in L + Rr for a_ in fr["mu"] if float((Hb[q - c["s0"]] - fr["mu"][a_]).norm()) <= fr["thr"][a_]]
            if not hits:
                continue
            t_ = maxcode + 1 - kh
            if 1 <= t_ <= C and t_ != kh:
                writes.append((c["s0"] + jx, cal["Mb"][t_ - 1] - cal["Mb"][kh - 1]))
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

    rng = random.Random(71000)
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
              open(f"/vlm/results/r3/s710_{a.tag}_mediation.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s710_{a.tag}_mediation.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
