"""S772 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 01:1x) before any S772 data.
Cells c = (N, phrasing) in (7, A), (8, A), (7, B), (8, B); k = 2..6, k' = N + 1 - k.  P(arm, b) = share of scenes answering image b.
I0          per cell, the natural START(k) answers k on >= .90 of (scene, k) and F_P12 answers k on >= .85 of (scene, k) pooled over k.
FRAME_FAR   per cell, pooled over k >= 4: P(F_P1, k) >= P(E, k) + .30 (the L27-34 frame read alone switches far END(k) to the literal k).
FRAME_NEAR  per cell, k = 2: P(F_P1, 2) <= .40 and P(F_P12, 2) >= .85 (the near END needs the frame read beyond L34, S759).
REPRO       per cell, pooled over k: on the scenes where END(k) errs (answer != k'), the share where START(k') with END(k)'s L43-59 queries (Q_ES) gives
            END(k)'s answer >= .50 and >= the natural START(k') = END(k) share + .30 (END(k)'s errors are carried by its reader queries).
FIX         per cell, pooled over k >= 3: P(Q_SE, k') >= P(E, k') + .25 (START(k')'s reader queries fix END(k)).
READER_OPT  per cell, pooled over (scene, k) and the natural runs END(k), START(k), START(k'): at L51-59, the menu word with the largest head-mean
            weight in the answer row is the answered word in >= .60 of runs for START and the share for END(k) is lower than for START(k').
NEAR_ADDR42 per cell: P(D42, k') >= .30 at k = 2 (the END(2) address is in the message after L42; D42 = START(k) readers given END(k) - START(k)
            at the input of L43).
FAR_ADDR42  per cell, pooled over k >= 4 with k' != k: P(D42, k') <= P(D42 at k = 2, k') - .20 (no readable far-END address after L42).
FAITH       all of FRAME_FAR, REPRO and FIX hold in all four cells.
Rule checks (added 2026-10-14 01:3x, before any S772 data were read; registered in DECISIONS):
RULE_SPREAD per cell, natural END(k) over scenes answering an image: the mean absolute position error |answer - k'| averaged over k = 4, 5, 6 exceeds
            that of k = 2 by >= .50 positions (far steps are coarse).
RULE_CENTER per cell and every k in 4, 5, 6 with k' != k: the mean signed error (answer - k') lies in [-1, 1] and is closer to 0 than the literal's
            offset (k - k') is (far errors scatter around the target, not around the literal).
Descriptive: per cell and k the answer distributions, the menu-logit margin of the target word over the best other menu word, Jensen-Shannon
distance of the menu distributions (R_P1 to END(k) and to START(k); Q_ES to END(k) and START(k')), reader target-block shares per layer.
Usage: python s772_analyze.py TAG_N7A TAG_N8A TAG_N7B TAG_N8B
"""
import json, os, sys
import numpy as np


def p(ll):
    x = np.array(ll, float); x = np.exp(x - x.max()); return x / x.sum()


def jsd(a, b):
    m = .5 * (a + b); kl = lambda u, w: float(np.sum(u * (np.log2(u + 1e-12) - np.log2(w + 1e-12))))
    return .5 * (kl(a, m) + kl(b, m))


def sm(sc):
    x = sc - sc.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)


def cell(tag):
    D = f"/vlm/results/r3/s772_{tag}_faith"
    d = json.load(open(D + ".json")); M = d["meta"]; z = np.load(D + ".npz"); N = d["N"]; KS = d["KS"]; RL = d["RB"]["late"]
    A_ = lambda m, nm, k: m["ans"][f"{nm}|{k}"]["pos"]
    LL = lambda m, nm, k: m["ans"][f"{nm}|{k}"]["ll"]
    P = lambda nm, k, b: float(np.mean([A_(m, nm, k) == b for m in M]))
    out = dict(tag=tag, N=N, phr=d["phr"], n=len(M), checks={}, per_k={})
    for k in KS:
        kp = N + 1 - k
        r = dict(E=dict(target=round(P("E", k, kp), 3), literal=round(P("E", k, k), 3)), S=round(P("S", k, k), 3),
                 F_P1=round(P("F_P1", k, k), 3), F_P12=round(P("F_P12", k, k), 3), R_P1=dict(k=round(P("R_P1", k, k), 3)),
                 Q_ES=dict(kp=round(P("Q_ES", k, kp), 3), eq_E=round(float(np.mean([A_(m, "Q_ES", k) == A_(m, "E", k) for m in M])), 3)),
                 Q_SE=dict(kp=round(P("Q_SE", k, kp), 3)), D42=dict(kp=round(P("D42", k, kp), 3), k=round(P("D42", k, k), 3)))
        tw = lambda m, nm: (lambda x: x[kp - 1] - np.max(np.delete(x, kp - 1)))(np.array(LL(m, nm, k)))
        r["margin_target"] = {nm: round(float(np.mean([tw(m, nm) for m in M])), 3) for nm in ("E", "Q_SE", "F_P1")}
        r["jsd"] = dict(R_P1_to_E=round(float(np.mean([jsd(p(LL(m, "R_P1", k)), p(LL(m, "E", k))) for m in M])), 3),
                        R_P1_to_S=round(float(np.mean([jsd(p(LL(m, "R_P1", k)), p(LL(m, "S", k))) for m in M])), 3),
                        Q_ES_to_E=round(float(np.mean([jsd(p(LL(m, "Q_ES", k)), p(LL(m, "E", k))) for m in M])), 3),
                        Q_ES_to_P=round(float(np.mean([jsd(p(LL(m, "Q_ES", k)), p(LL(m, "P" if kp != k else "S", k))) for m in M])), 3))
        r["late_target_share"] = {nm: round(float(np.mean([np.mean([sm(z[f"{m['si']}|{k}|{nm}|{li}"].astype(np.float64))[1].mean(0)[kp - 1] for li in RL])
                                                           for m in M])), 3) for nm in ("E", "Q_SE", "F_P1", "S")}
        # descriptive (added before any S772 data were read): spread of the late readers' block distribution around the target
        pk = "P" if kp != k else "S"
        spread = lambda nm: float(np.mean([np.mean([float(np.sum(sm(z[f"{m['si']}|{k}|{nm}|{li}"].astype(np.float64))[1].mean(0) * np.abs(np.arange(1, N + 1) - kp)))
                                                    for li in RL]) for m in M]))
        r["late_spread_around_target"] = dict(END=round(spread("E"), 3), START_kp=round(spread(pk), 3))
        out["per_k"][str(k)] = r
    # checks
    nat_S = float(np.mean([A_(m, "S", k) == k for m in M for k in KS]))
    fp12 = float(np.mean([A_(m, "F_P12", k) == k for m in M for k in KS]))
    out["i0"] = dict(START=round(nat_S, 3), F_P12=round(fp12, 3))
    out["checks"]["I0"] = bool(nat_S >= .90 - 1e-9 and fp12 >= .85 - 1e-9)
    far = [k for k in KS if k >= 4]
    gF = float(np.mean([A_(m, "F_P1", k) == k for m in M for k in far])) - float(np.mean([A_(m, "E", k) == k for m in M for k in far]))
    out["frame_far_gain"] = round(gF, 3); out["checks"]["FRAME_FAR"] = bool(gF >= .30 - 1e-9)
    out["checks"]["FRAME_NEAR"] = bool(P("F_P1", 2, 2) <= .40 + 1e-9 and P("F_P12", 2, 2) >= .85 - 1e-9) if 2 in KS else None
    errs = [(m, k) for m in M for k in KS if A_(m, "E", k) != N + 1 - k]
    nat_eq = float(np.mean([A_(m, "P" if N + 1 - k != k else "S", k) == A_(m, "E", k) for m, k in errs])) if errs else float("nan")
    rep = float(np.mean([A_(m, "Q_ES", k) == A_(m, "E", k) for m, k in errs])) if errs else float("nan")
    out["repro"] = dict(n_err=len(errs), Q_ES=round(rep, 3), natural_START_kp=round(nat_eq, 3))
    out["checks"]["REPRO"] = bool(errs and rep >= .50 - 1e-9 and rep >= nat_eq + .30 - 1e-9)
    k3 = [k for k in KS if k >= 3]
    gX = float(np.mean([A_(m, "Q_SE", k) == N + 1 - k for m in M for k in k3])) - float(np.mean([A_(m, "E", k) == N + 1 - k for m in M for k in k3]))
    out["fix_gain"] = round(gX, 3); out["checks"]["FIX"] = bool(gX >= .25 - 1e-9)
    far4 = [k for k in KS if k >= 4 and N + 1 - k != k]
    d2 = P("D42", 2, N - 1) if 2 in KS else float("nan")
    dfar = float(np.mean([A_(m, "D42", k) == N + 1 - k for m in M for k in far4])) if far4 else float("nan")
    out["addr42"] = dict(near_k2=round(d2, 3), far=round(dfar, 3), far_literal=round(float(np.mean([A_(m, "D42", k) == k for m in M for k in far4])), 3))
    out["checks"]["NEAR_ADDR42"] = bool(d2 >= .30 - 1e-9)
    out["checks"]["FAR_ADDR42"] = bool(dfar <= d2 - .20 + 1e-9)
    def errs_of(k):
        x = [A_(m, "E", k) - (N + 1 - k) for m in M if A_(m, "E", k) > 0]
        return np.array(x, float)
    mae = {k: round(float(np.mean(np.abs(errs_of(k)))), 3) for k in KS}
    sgn = {k: round(float(np.mean(errs_of(k))), 3) for k in KS}
    out["rule"] = dict(mae=mae, signed=sgn)
    farm = [k for k in KS if k >= 4]
    out["checks"]["RULE_SPREAD"] = bool(np.mean([mae[k] for k in farm]) - mae[2] >= .50 - 1e-9) if 2 in KS else None
    cen = [k for k in farm if N + 1 - k != k]
    out["checks"]["RULE_CENTER"] = bool(all(-1 - 1e-9 <= sgn[k] <= 1 + 1e-9 and abs(sgn[k]) < abs(k - (N + 1 - k)) for k in cen))
    hit = {}
    for nm in ("E", "S", "P"):
        h = []
        for m in M:
            for k in KS:
                key = nm if not (nm == "P" and N + 1 - k == k) else "S"
                w = np.mean([z[f"{m['si']}|{k}|{key}|{li}|opt"].astype(np.float64)[1].mean(0) for li in RL], 0)
                h.append(int(np.argmax(w)) == int(np.argmax(LL(m, key, k))))
        hit[nm] = round(float(np.mean(h)), 3)
    out["reader_top_option_is_answer"] = hit
    out["checks"]["READER_OPT"] = bool(hit["S"] >= .60 - 1e-9 and hit["P"] >= .60 - 1e-9 and hit["E"] < hit["P"])
    print(f"[{tag}] N {N} · {d['phr']} · n {len(M)} · checks {out['checks']} · I0 {out['i0']} · frame far gain {out['frame_far_gain']} · repro {out['repro']} · fix {out['fix_gain']} · addr42 {out['addr42']} · rule {out['rule']} · top option {hit}")
    for k, r in out["per_k"].items():
        print(f"   k {k}: {r}")
    return out


if __name__ == "__main__":
    res = {t: cell(t) for t in sys.argv[1:]}
    res["FAITH"] = bool(len(sys.argv[1:]) == 4 and all(res[t]["checks"][c] for t in sys.argv[1:] for c in ("FRAME_FAR", "REPRO", "FIX")))
    print("FAITH", res["FAITH"])
    json.dump(res, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s772_analysis.json"), "w"), indent=1)
