"""S709-eval: end-to-end R³ (R³-min: model-internal readers only, exp/r3_repair.py min_mode logic) on Qwen3.5-27B vs native, text and oracle
controls, with a per-item stage ledger.  Pre-registration in knowledge/DECISIONS.md before any S709 data.
Readers from exp/s709_cal.py (forward / frame-label calibration only).  Independent procedural scenes (seed 70900, R³ image pool and
size), N in {5, 7}, every k, END questions "There are N images in a row. What is in the k-th image from the last? Answer with only one
word from: <menu>."  Target j = N+1-k.
Arms:
    NATIVE      unedited
    EXPLICIT    the question rewritten as "the j-th image from the first" with the true j (text oracle)
    TEXTNORM    a text parser (regex on "the <ordinal> image from the last") + the input image count -> the same rewrite (coincides with
                EXPLICIT on this canonical form; kept as the external baseline)
    O_ORD       oracle ordinal write at the ordinal token: Mb[j] - Mb[k] at L_w
    O_ORD_FR    O_ORD + frame write at the frame word: mu_start - mu_last (the S708 upper bound with these prototypes)
    R3          R³-min plan from the readers: detector (k_hat), clause-window frame anchor (end / last), end reader (N_hat = max block
                code), t = N_hat + 1 - k_hat, the two writes; no write when nothing is detected or t = k_hat
    WRONG       R3's detected positions with a wrong forward address w (w = j - 2 if j > 3 else j + 2; only where R3 acts)
Stage ledger per item: ordinal detected, k_hat == k, frame anchor found, N_hat == N, t == j, act, landing.
Registered rules (analysis exp/s709_analyze.py), 20 scenes per N:
    E2E_REPAIR: P(j | R3) >= P(j | NATIVE) + .25 overall, and P(j | R3) >= .70 on the hard cells (N7 k in {3,4,5,6}, N5 k in {3,4})
    ORACLE_FRACTION: P(j | R3) >= .8 P(j | O_ORD_FR) overall
    ADDRESS_CONTROL: P(w | WRONG) >= .70 where R3 acts
    REREFLECT_REPLICATES: P(k | O_ORD) >= .40 over items with j != k
    Predictions: all four.
Usage: python s709_eval.py MODEL TAG --base B [--n 20] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, re, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3ref import ORD, load_image
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
    tid = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    keys = sorted({k for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})          # dev pool (disjoint from the reader draw)
    from r3_proc import proc_image
    lab_of = {}
    for kk in keys[:3000]:
        lab_of[kk] = proc_image(int(kk.split(":")[1]))[1]
    by_lab = {}
    for kk, lb in lab_of.items():
        if len(v.tok(" " + lb, add_special_tokens=False)["input_ids"]) == 1:
            by_lab.setdefault(lb, []).append(kk)
    labels = sorted(l for l, x in by_lab.items() if len(x) >= 10)

    def body(N, k, frame, menu):
        return f"There are {N} images in a row. What is in the {ORD[k-1]} image from the {frame}? Answer with only one word from: {', '.join(menu)}."

    def parse_rewrite(text, N):
        m = re.search(r"the (\w+) image from the last", text)
        if not m or m.group(1) not in ORD:
            return None
        k = ORD.index(m.group(1)) + 1
        return text.replace(m.group(0), f"the {ORD[N - k]} image from the first") if 1 <= k <= N else None

    def land(lg, labs, menu):
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return labs.index(lab) + 1 if lab in labs else 0

    def plan(c):
        """R³-min plan (r3_repair.R3Repair.plan, min_mode) on a capture."""
        Hb = c["H"][Lb]
        if cal["det_space"] == "full":
            dist, kh0 = torch.cdist(Hb, cal["Mb"]).min(1); th = cal["th_full"]
        else:
            dist, kh0 = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"]).min(1); th = cal["th_chart"]
        X = torch.cat([c["bf"][l] for l in M.Rlayers], 1)
        from s588c_reader import probs as rp
        codes = (rp(M.RP, X).argmax(1) + 1).long(); maxcode = int(codes.max())
        srcs = (dist <= th).nonzero().flatten().tolist()
        out = []
        for jx in srcs:
            kh = int(kh0[jx]) + 1
            L, Rr = r3_build.clause_window(v.tok, c["ids"], c["s0"] + jx, c["s0"], c["turn"])
            win = [q - c["s0"] for q in L + Rr]
            hits = [(qq, a_) for qq in win for a_ in fr["mu"] if float((Hb[qq] - fr["mu"][a_]).norm()) <= fr["thr"][a_]]
            best = hits[0] if hits else None
            t_ = (maxcode + 1 - kh) if best is not None else kh
            act = best is not None and 1 <= t_ <= C and t_ != kh
            out.append(dict(pos=c["s0"] + jx, kh=kh, t=t_, act=act, fpos=(c["s0"] + best[0]) if best else None,
                            fa=best[1] if best else None, N_hat=maxcode))
        return out

    rng = random.Random(70900)
    rows = []
    for N in (7, 5):
        for si in range(a.n):
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [load_image(rng.choice(by_lab[l])) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            for k in range(1, N + 1):
                j = N + 1 - k
                txt = body(N, k, "last", menu)
                c = M.capture(txt, pics)
                rec = dict(N=N, si=si, k=k, j=j, land={})
                rec["land"]["NATIVE"] = land(c["logits"], labs, menu)
                rec["land"]["EXPLICIT"] = land(M.capture(body(N, j, "first", menu), pics)["logits"], labs, menu)
                tn = parse_rewrite(txt, N)
                rec["land"]["TEXTNORM"] = land(M.capture(tn, pics)["logits"], labs, menu) if tn else rec["land"]["NATIVE"]
                ids = c["ids"]
                o = max(i for i in range(c["s0"], c["turn"]) if ids[i] == tid(ORD[k - 1]) and ids[i + 1] == tid("image"))
                fw = next(i for i in range(o + 1, c["turn"]) if ids[i] == tid("last"))
                d_ord = (cal["Mb"][j - 1] - cal["Mb"][k - 1]) if j != k else torch.zeros_like(cal["Mb"][0])
                d_fr = fr["mu_start"] - fr["mu"]["last"]
                rec["land"]["O_ORD"] = land(M.run(c["inp"], [(o, d_ord)]), labs, menu)
                rec["land"]["O_ORD_FR"] = land(M.run(c["inp"], [(o, d_ord), (fw, d_fr)]), labs, menu)
                P_ = plan(c)
                acts = [p_ for p_ in P_ if p_["act"]]
                writes = []
                for p_ in acts:
                    writes.append((p_["pos"], cal["Mb"][p_["t"] - 1] - cal["Mb"][p_["kh"] - 1]))
                    writes.append((p_["fpos"], fr["mu_start"] - fr["mu"][p_["fa"]]))
                rec["land"]["R3"] = land(M.run(c["inp"], writes), labs, menu) if writes else rec["land"]["NATIVE"]
                w = j - 2 if j > 3 else j + 2
                if acts:
                    p0 = acts[0]
                    ww = [(p0["pos"], cal["Mb"][w - 1] - cal["Mb"][p0["kh"] - 1]), (p0["fpos"], fr["mu_start"] - fr["mu"][p0["fa"]])]
                    rec["land"]["WRONG"] = land(M.run(c["inp"], ww), labs, menu)
                rec.update(w=w, ledger=dict(detected=[dict(pos=p_["pos"], kh=p_["kh"], t=p_["t"], act=p_["act"], fa=p_["fa"], N_hat=p_["N_hat"]) for p_ in P_],
                                           ord_pos=o, fw_pos=fw))
                rows.append(rec)
            torch.cuda.empty_cache()
            if (si + 1) % 5 == 0:
                print(f"[{a.tag}] N{N} {si + 1}/{a.n}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, base=a.base, Lb=Lb, reader=st["reader"], reader_layer=st["reader_layer"], rows=rows),
              open(f"/vlm/results/r3/s709_{a.tag}_eval.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s709_{a.tag}_eval.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
