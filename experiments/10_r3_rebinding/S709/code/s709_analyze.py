"""S709 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S709 data.
E2E_REPAIR: P(j | R3) >= P(j | NATIVE) + .25 overall and P(j | R3) >= .70 on hard cells (N7 k 3-6, N5 k 3-4)
ORACLE_FRACTION: P(j | R3) >= .8 P(j | O_ORD_FR);  ADDRESS_CONTROL: P(w | WRONG) >= .70 where R3 acts;  REREFLECT_REPLICATES: P(k | O_ORD) >= .40 (j != k)
Stage ledger: for R3 failures, the first failing stage among detected / k_hat / frame / N_hat / t / write-succeeded-but-wrong.
Usage: python s709_analyze.py TAG
"""
import collections, json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
ARMS = ("NATIVE", "EXPLICIT", "TEXTNORM", "O_ORD", "O_ORD_FR", "R3")


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s709_{tag}_eval.json")); rows = d["rows"]
    print(f"[{tag}] n {len(rows)} · reader L{d['reader_layer']} · L_b {d['Lb']}")
    P = lambda arm, rs: float(np.mean([r["land"][arm] == r["j"] for r in rs])) if rs else float("nan")
    for N in (7, 5):
        for k in range(1, N + 1):
            rs = [r for r in rows if r["N"] == N and r["k"] == k]
            print(f"[{tag}]  N{N} k{k} (j{N+1-k}): " + " | ".join(f"{a_} {P(a_, rs):.2f}" for a_ in ARMS))
    hard = [r for r in rows if (r["N"] == 7 and 3 <= r["k"] <= 6) or (r["N"] == 5 and 3 <= r["k"] <= 4)]
    acted = [r for r in rows if "WRONG" in r["land"]]
    nonmid = [r for r in rows if r["j"] != r["k"]]
    # stage ledger on R3 failures
    led = collections.Counter()
    for r in rows:
        if r["land"]["R3"] == r["j"]:
            continue
        det = r["ledger"]["detected"]; src = [x for x in det if abs(x["pos"] - r["ledger"]["ord_pos"]) == 0]
        if not src: led["ordinal not detected"] += 1; continue
        x = src[0]
        if x["kh"] != r["k"]: led["k_hat wrong"] += 1
        elif x["fa"] is None: led["frame not bound"] += 1
        elif x["N_hat"] != r["N"]: led["N_hat wrong"] += 1
        elif x["t"] != r["j"]: led["t wrong"] += 1
        elif not x["act"]: led["no write"] += 1
        else: led["address written, selection wrong"] += 1
    checks = dict(overall={a_: round(P(a_, rows), 3) for a_ in ARMS}, hard={a_: round(P(a_, hard), 3) for a_ in ARMS},
                  WRONG_follow=round(float(np.mean([r["land"]["WRONG"] == r["w"] for r in acted])), 3) if acted else None, n_acted=len(acted),
                  REREFLECT_k=round(float(np.mean([r["land"]["O_ORD"] == r["k"] for r in nonmid])), 3), R3_failure_stages=dict(led))
    checks["E2E_REPAIR"] = GE(P("R3", rows), P("NATIVE", rows) + .25) and GE(P("R3", hard), .70)
    checks["ORACLE_FRACTION"] = GE(P("R3", rows), .8 * P("O_ORD_FR", rows))
    checks["ADDRESS_CONTROL"] = bool(acted) and GE(checks["WRONG_follow"], .70)
    checks["REREFLECT_REPLICATES"] = GE(checks["REREFLECT_k"], .40)
    print(f"[{tag}] checks {checks}")
    return checks


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s709_analysis.json", "w"), indent=1, default=float)
