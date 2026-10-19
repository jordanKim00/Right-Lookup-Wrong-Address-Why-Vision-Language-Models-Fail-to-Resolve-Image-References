"""S845 analysis (rules fixed in knowledge/DECISIONS.md before any S845 data).  Writer -> reader path.  Cases as S836-S840 (identity (7, 4), (9, 5);
non-identity (7, 5), (9, 7)); P_copy(arm) = share answering image k (identity: the target; otherwise 'step removed'); scene bootstrap 4,000.
I0 (per case): t1 >= .95; QNOOP = END in >= .90 of scenes (95th percentile of the largest menu-logit change <= .5); START(k) -> k >= .90; image keys
identical (0.0); the corrected run moves the answer: P_copy(A_BAND) - P_copy(END) >= .15.  Valid cases are pooled.
Restore of an arm relative to its corrected run X (A_BAND for *_B, A_4749 for *_47): f(arm) = (P_copy(arm) - P_copy(END)) / (P_copy(X) - P_copy(END)),
pooled as sums.
  PATH   f(BOTH_B) >= .60 (lower bound > .40) -> 'the band's correction reaches the answer through the late queries'.
  SPLIT  with f_A = f(ADDR_B), f_R = f(READ_B), f_B = f(BOTH_B): 'the address part alone carries it' if f_A >= .8 f_B; 'the reading part alone
         carries it' if f_R >= .8 f_B; 'both parts are needed' if f_A < .5 f_B and f_R < .5 f_B and f_B >= .40; otherwise 'shared'.
  TSUB   f(READT_B) >= .5 f(READ_B) with lower bound of f(READT_B) > 0 -> 'the reading correction lies in the text-key subspace'; and its late picture
         mass moves toward START by >= .5 of READ_B's move (descriptive consistency of attention and answer).
  P4749  f(BOTH_47) relative to A_4749 (descriptive), and A_4749's own restore relative to A_BAND's.
Descriptive: per arm the late positional scores (layer 55 answer row: z[k] - max of the other blocks), the copy-word logit margin, picture mass.
Usage: python exp/s845_analyze.py [TAG]  ->  results/r3/s845_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s845_27b"):
    d = json.load(open(f"{V}/results/r3/s845_{tag}_wr.json")); R = d["rows"]; late = [str(x) for x in d["late"]]
    rng = np.random.default_rng(0)
    def mass(r, a):
        x = r["mass"].get(a, {})
        v = [np.mean([x[L][i] for i in (2, 5, 9)]) for L in late if L in x]
        return float(np.mean(v)) if v else np.nan
    def zmar(r, a, k):
        z = r["qk"][a][late[-1]][9]
        return z[k - 1] - max(z[i] for i in range(len(z)) if i != k - 1)
    def lmar(r, a, k):
        lg = r["mlog"][a]; return lg[k - 1] - max(lg[i] for i in range(len(lg)) if i != k - 1)
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], cases={})
    valid = {}
    ARMS = ["END", "START", "A_BAND", "A_4749", "QNOOP", "ADDR_B", "READ_B", "BOTH_B", "READT_B", "ADDR_47", "READ_47", "BOTH_47"]
    for N, k, j in d["cases"]:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        cname = f"N{N}k{k}j{j}"
        pc = lambda a, rr=rows: float(np.mean([r["land"][a] == k for r in rr]))
        ag = float(np.mean([r["land"]["QNOOP"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP"], r["mlog"]["END"])) for r in rows], 95))
        i0 = dict(qnoop=ag, p95=p95, start_acc=pc("START"), restore=pc("A_BAND") - pc("END"))
        i0["ok"] = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and i0["start_acc"] >= .90 and d["key_rel_diff_max"] == 0.0 and i0["restore"] >= .15)
        out["cases"][cname] = dict(n=len(rows), I0=i0, P_copy={a: pc(a) for a in ARMS},
                                   mass={a: float(np.nanmean([mass(r, a) for r in rows])) for a in ARMS},
                                   zmargin55={a: float(np.mean([zmar(r, a, k) for r in rows])) for a in ARMS},
                                   logit_margin={a: float(np.mean([lmar(r, a, k) for r in rows])) for a in ARMS})
        if i0["ok"]:
            valid[cname] = rows
    def kof(c):
        return int(c.split("k")[1].split("j")[0])
    def f(groups, arm, ref):
        num = den = 0.0
        for c, rows in groups.items():
            k = kof(c); g = lambda a: np.mean([r["land"][a] == k for r in rows])
            num += g(arm) - g("END"); den += g(ref) - g("END")
        return float(num / den) if abs(den) > 1e-9 else float("nan")
    def boot(fun):
        v = []
        for _ in range(B):
            v.append(fun({c: [rows[i] for i in rng.integers(0, len(rows), len(rows))] for c, rows in valid.items()}))
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out["valid_cases"] = sorted(valid)
    if valid:
        fr = {a: dict(f=f(valid, a, "A_BAND"), ci=boot(lambda g, a=a: f(g, a, "A_BAND"))) for a in ("ADDR_B", "READ_B", "BOTH_B", "READT_B", "A_4749")}
        fr.update({a: dict(f=f(valid, a, "A_4749"), ci=boot(lambda g, a=a: f(g, a, "A_4749"))) for a in ("ADDR_47", "READ_47", "BOTH_47")})
        out["f"] = fr
        fA, fR, fB = fr["ADDR_B"]["f"], fr["READ_B"]["f"], fr["BOTH_B"]["f"]
        out["PATH"] = ("the band's correction reaches the answer through the late queries" if fB >= .60 and fr["BOTH_B"]["ci"][0] > .40 else "not shown")
        out["SPLIT"] = ("the address part alone carries it" if fA >= .8 * fB else "the reading part alone carries it" if fR >= .8 * fB
                        else "both parts are needed" if fA < .5 * fB and fR < .5 * fB and fB >= .40 else "shared")
        mv = lambda a: np.mean([out["cases"][c]["mass"][a] - out["cases"][c]["mass"]["END"] for c in valid])
        out["TSUB"] = dict(verdict=("the reading correction lies in the text-key subspace" if fr["READT_B"]["f"] >= .5 * fR and fr["READT_B"]["ci"][0] > 0
                                    else "not shown"), mass_move_ratio=float(mv("READT_B") / mv("READ_B")) if abs(mv("READ_B")) > 1e-9 else None)
    fn = f"{V}/results/r3/s845_analysis.json" if tag == "s845_27b" else f"{V}/results/r3/s845_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for c, v in out["cases"].items():
        print(c, v["I0"], "\n   P_copy", {a: round(x, 3) for a, x in v["P_copy"].items()}, "\n   zmargin55", {a: round(x, 2) for a, x in v["zmargin55"].items()})
    print("f", json.dumps(out.get("f"))[:900]); print("PATH", out.get("PATH")); print("SPLIT", out.get("SPLIT")); print("TSUB", out.get("TSUB"))


if __name__ == "__main__":
    main(*sys.argv[1:])
