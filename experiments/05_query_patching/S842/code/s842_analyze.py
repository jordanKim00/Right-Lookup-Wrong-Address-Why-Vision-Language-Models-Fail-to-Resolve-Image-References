"""S842 analysis (rules fixed in knowledge/DECISIONS.md before any S842 data).  Identity cases N7 END(4), N9 END(5); P_j = share answering k;
d(arm) = P_j(arm) - P_j(END); scene bootstrap 4,000 (seed 0), cases pooled.
I0 (per case): t1 >= .95; QNOOP_E = END and QNOOP_S = START in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5;
START(k) -> k >= .90; image keys identical (0.0); positive control d(P_FULL) >= .15.  Valid cases are pooled.
  SUB   for X in W, T, R: 'X carries the reading mode' if d(X_FULL) >= .5 d(P_FULL) with interval above 0; the verdict lists those X (or 'none'),
        and names the largest.
  KOQ   'the late re-reading of the question words carries the END error' if d(KO_Q_LATE) >= .15 with interval above 0; otherwise 'not shown';
        KO_F_LATE, KO_O_LATE and KO_Q_ALL descriptive.
Usage: python exp/s842_analyze.py [TAG]  ->  results/r3/s842_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s842_27b"):
    d = json.load(open(f"{V}/results/r3/s842_{tag}_mode3.json")); R = d["rows"]; look = d["look"]; nl = d["n_layer"]
    late = [str(l) for l in look if l / (nl - 1) > .75]
    rng = np.random.default_rng(0)
    Pj = lambda rows, a: float(np.mean([r["land"][a] == r["k"] for r in rows]))
    def m(r, a):
        x = r.get("mass", {}).get(a, {})
        v = [np.mean([x[L][i] for i in (2, 5, 9)]) for L in late if L in x]
        return float(np.mean(v)) if v else np.nan
    def boot(rows, f):
        v = []
        for _ in range(B):
            ix = rng.integers(0, len(rows), len(rows)); v.append(f([rows[i] for i in ix]))
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], cases={})
    pool = []
    for N, k in [(7, 4), (9, 5)]:
        rows = [r for r in R if r["N"] == N]
        if not rows:
            continue
        ag = lambda a1, a2: float(np.mean([r["land"][a1] == r["land"][a2] for r in rows]))
        p95 = lambda a1, a2: float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"][a1], r["mlog"][a2])) for r in rows], 95))
        i0 = dict(qnoop_e=ag("QNOOP_E", "END"), qnoop_s=ag("QNOOP_S", "START"), p95=max(p95("QNOOP_E", "END"), p95("QNOOP_S", "START")),
                  start_acc=Pj(rows, "START"), pos=Pj(rows, "P_FULL") - Pj(rows, "END"))
        i0["ok"] = bool(d["t1_menu"] >= .95 and i0["qnoop_e"] >= .90 and i0["qnoop_s"] >= .90 and i0["p95"] <= .5 and i0["start_acc"] >= .90
                        and d["key_rel_diff_max"] == 0.0 and i0["pos"] >= .15)
        arms = sorted(rows[0]["land"])
        out["cases"][f"N{N}k{k}"] = dict(n=len(rows), I0=i0, P_j={a: Pj(rows, a) for a in arms},
                                         mass={a: float(np.nanmean([m(r, a) for r in rows])) for a in arms})
        if i0["ok"]:
            pool += rows
    if pool:
        dd = lambda a, q=None: Pj(q or pool, a) - Pj(q or pool, "END")
        A = ["P_FULL", "WSUB_FULL", "TSUB_FULL", "RSUB_FULL", "KO_Q_LATE", "KO_F_LATE", "KO_O_LATE", "KO_Q_ALL"]
        res = {a: dict(d=dd(a), ci=boot(pool, lambda q, a=a: dd(a, q))) for a in A}
        out["d"] = res
        carry = [x for x in ("WSUB_FULL", "TSUB_FULL", "RSUB_FULL") if res[x]["d"] >= .5 * res["P_FULL"]["d"] and res[x]["ci"][0] > 0]
        out["SUB"] = dict(carriers=carry or "none", largest=max(("WSUB_FULL", "TSUB_FULL", "RSUB_FULL"), key=lambda x: res[x]["d"]))
        out["KOQ"] = ("the late re-reading of the question words carries the END error" if res["KO_Q_LATE"]["d"] >= .15 and res["KO_Q_LATE"]["ci"][0] > 0
                      else "not shown")
    else:
        out["SUB"] = out["KOQ"] = "no valid case"
    fn = f"{V}/results/r3/s842_analysis.json" if tag == "s842_27b" else f"{V}/results/r3/s842_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for c, v in out["cases"].items():
        print(c, v["I0"], "\n   P_j", {a: round(x, 3) for a, x in v["P_j"].items()}, "\n   mass", {a: round(x, 3) for a, x in v["mass"].items()})
    print("d", json.dumps(out.get("d"))[:900]); print("SUB", out["SUB"]); print("KOQ", out["KOQ"])


if __name__ == "__main__":
    main(*sys.argv[1:])
