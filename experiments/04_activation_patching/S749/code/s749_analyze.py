"""S749 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-12 20:1x) before any S749 data.
F = answers on the donor's target (8 - k'), R = answers on the recipient's target (8 - k), acc(k') = the donor's natural END accuracy.
I0             NOOP answers equal the natural answers on >= .95 of (scene, k), and RES@46 pooled F >= .7 * mean acc(k') over the 12 pairs
SENDERS_CARRY  pooled over the 12 pairs: F(SEND) >= .5 * mean acc(k'), F(SEND) >= R(SEND) + .10, and F(CTRL) <= .5 * F(SEND)
BLUR           pair (2 <- 5) at RES@42: F <= acc(5) + .10 -> H1_STATE; F >= acc(5) + .15 -> H2_INTERFERENCE; otherwise UNDECIDED
Descriptive: landing distributions per pair and arm, the birth curve F(RES@L) per donor k', the reverse pair (5 <- 2) on image 6, the MATCHER
(S705a) fixed-bank pointer shares at the decision layers (L35-47) and later (L51-55), scene-bootstrap 95% intervals (4,000 draws).
Usage: python s749_analyze.py TAG
"""
import json, sys
import numpy as np

N = 7


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze(tag):
    D = f"/vlm/results/r3/s749_{tag}_swapk"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz")
    M = meta["meta"]; KS = meta["KS"]; RES = meta["RES"]; CAP = meta["CAP"]; n = len(M)
    tgt = lambda k: N + 1 - k
    L = lambda key: np.array([m["land"][key] for m in M])
    acc = {k: float(np.mean(L(f"E{k}") == tgt(k))) for k in KS}
    pairs = [(k, kp) for k in KS for kp in KS if kp != k]
    arms = ["SEND", "CTRL"] + [f"RES{l}" for l in RES]
    out = dict(n=n, acc={str(k): round(v, 3) for k, v in acc.items()}, checks={}, arms={}, pairs={}, birth={}, pointer={})
    # per arm pooled F and R (per scene, averaged over pairs, for the bootstrap)
    for arm in arms:
        Fs = np.mean([[M[i]["land"][f"{arm}|{k}<{kp}"] == tgt(kp) for (k, kp) in pairs] for i in range(n)], 1)
        Rs = np.mean([[M[i]["land"][f"{arm}|{k}<{kp}"] == tgt(k) for (k, kp) in pairs] for i in range(n)], 1)
        out["arms"][arm] = dict(F=boot(Fs), R=boot(Rs))
        for (k, kp) in pairs:
            l = L(f"{arm}|{k}<{kp}")
            out["pairs"][f"{arm}|{k}<{kp}"] = dict(F=round(float(np.mean(l == tgt(kp))), 3), R=round(float(np.mean(l == tgt(k))), 3),
                                                   dist={str(b): round(float(np.mean(l == b)), 3) for b in range(0, N + 1)})
    macc = float(np.mean([acc[kp] for (_, kp) in pairs]))
    noop = float(np.mean([M[i]["land"][f"NOOP|{k}"] == M[i]["land"][f"E{k}"] for i in range(n) for k in KS]))
    res46 = f"RES{max(RES)}"
    out["checks"]["I0"] = bool(noop >= .95 - 1e-9 and out["arms"][res46]["F"][0] >= .7 * macc - 1e-9)
    out["i0"] = dict(noop=round(noop, 3), res_late_F=out["arms"][res46]["F"][0], mean_acc=round(macc, 3))
    fs, rs, fc = out["arms"]["SEND"]["F"][0], out["arms"]["SEND"]["R"][0], out["arms"]["CTRL"]["F"][0]
    out["checks"]["SENDERS_CARRY"] = bool(fs >= .5 * macc - 1e-9 and fs >= rs + .10 - 1e-9 and fc <= .5 * fs + 1e-9)
    res42 = f"RES{sorted(RES)[-2]}"
    f25 = out["pairs"][f"{res42}|2<5"]["F"]
    out["checks"]["BLUR"] = ("H1_STATE" if f25 <= acc[5] + .10 + 1e-9 else "H2_INTERFERENCE" if f25 >= acc[5] + .15 - 1e-9 else "UNDECIDED")
    out["blur_detail"] = dict(F_2from5=f25, acc5=round(acc[5], 3), reverse_5from2_on6=out["pairs"][f"{res42}|5<2"]["F"], acc2=round(acc[2], 3))
    # birth curve: F(RES@L) per donor k'
    for kp in KS:
        out["birth"][str(kp)] = {str(l): round(float(np.mean([np.mean(L(f"RES{l}|{k}<{kp}") == tgt(kp)) for k in KS if k != kp])), 3) for l in RES}
    # MATCHER pointer shares at the donor target vs recipient target
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    def ptr(key, b, layers):
        v = [int(np.argmax(z[f"{si}|{key}|{li}"][ri, h])) + 1 == b for si in range(n) for (li, h, ri) in mt if li in layers and f"{si}|{key}|{li}" in z.files]
        return round(float(np.mean(v)), 3) if v else None
    dec, late = [l for l in CAP if l <= 47], [l for l in CAP if l > 47]
    for k in KS:
        out["pointer"][f"E{k}"] = dict(dec=ptr(f"E{k}", tgt(k), dec), late=ptr(f"E{k}", tgt(k), late))
    for arm in ("SEND", res42, res46):
        for (k, kp) in pairs:
            out["pointer"][f"{arm}|{k}<{kp}"] = dict(dec_donor=ptr(f"{arm}|{k}<{kp}", tgt(kp), dec), late_donor=ptr(f"{arm}|{k}<{kp}", tgt(kp), late),
                                                     dec_recip=ptr(f"{arm}|{k}<{kp}", tgt(k), dec))
    print(f"[{tag}] acc {out['acc']} · I0 {out['i0']} -> {out['checks']['I0']}")
    for arm in arms:
        print(f"   {arm:6s} F {out['arms'][arm]['F']} · R {out['arms'][arm]['R']}")
    print(f"[{tag}] SENDERS_CARRY {out['checks']['SENDERS_CARRY']} · BLUR {out['checks']['BLUR']} {out['blur_detail']}")
    print(f"[{tag}] birth F(RES@L) by donor k': {out['birth']}")
    for key in ("SEND|2<5", "SEND|5<2", f"{res42}|2<5", f"{res42}|5<2"):
        print(f"   {key:10s} {out['pairs'][key]['dist']}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s749_analysis.json", "w"), indent=1)
