"""S836 analysis (rules fixed in knowledge/DECISIONS.md before any S836 data).
I0 (per case): t1 >= .95 (run); DNOOP names the same image as natural END in >= 90% of scenes and the 95th percentile of the largest absolute
menu-logit change is <= .5; DPOS51's recomputed layer-51 query equals START's (relative max difference <= .02) in >= 95% of scenes; START(k) -> k
>= .90; image keys identical between START and END (largest relative difference 0.0); the full late-query swap moves the answer:
P_copy(D_LATE) - P_copy(END) >= .15 (otherwise that case's restore fractions are undefined).  Cases failing I0 are reported, not pooled.
Measures: P_copy(arm) = share of scenes answering image k (the early-found position; = accuracy in identity cases, 'step removed' otherwise);
restore fraction rho(b) = (P_copy(D_b) - P_copy(END)) / (P_copy(D_LATE) - P_copy(END)); pooled over valid cases as sum of numerators over sum
of denominators; scene bootstrap (4,000 draws, resampled within case).  Reading mode: late picture mass m = mean over rows B1, B4, answer and
layers 51, 55 of the matcher heads' attention on image tokens; rho_m(b) the same fraction for m (descriptive unless its denominator >= .05).
Verdicts (pooled):
  WRITE: b* = the smallest b with rho(b) >= .80 and lower interval bound >= .50 -> 'the END-specific information the late queries use is in place
         by layer b*' (band = (previous bound, b*]); b* = 36 -> 'in place before the lookup band'; none -> 'written at layer 50 or later'.
  SPLIT: identity vs non-identity b* (descriptive; same / different).
  MODE:  b*_m by the same rule on rho_m -> 'the reading mode is in place by b*_m'; compare with b*.
Usage: python exp/s836_analyze.py [TAG]  ->  results/r3/s836_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
RNG = np.random.default_rng(0)


def load(tag):
    return json.load(open(f"{V}/results/r3/s836_{tag}_latewrite.json"))


def main(tag="s836_27b"):
    d = load(tag); R = d["rows"]; bounds = d["bounds"]; late = [str(x) for x in d["late"]]
    arms = ["END", "START", "DNOOP", "DPOS51", "D_LATE"] + [f"D_{b}" for b in bounds]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], dpos_q_rel_max=d["dpos_q_rel_max"], cases={})
    def mass(r, arm):
        m = r["mass"].get(arm, {})
        return float(np.mean([np.mean([m[L][i] for i in (2, 5, 9)]) for L in late if L in m])) if m else np.nan
    pooled = {}
    for N, k, j in d["cases"]:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        n = len(rows); cname = f"N{N}k{k}j{j}"
        pc = lambda a, rr=rows: float(np.mean([r["land"][a] == k for r in rr]))
        agree = float(np.mean([r["land"]["DNOOP"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["DNOOP"], r["mlog"]["END"])) for r in rows], 95))
        qok = float(np.mean([r.get("dpos_q_rel", 1.0) <= .02 for r in rows]))
        st = float(np.mean([r["land"]["START"] == k for r in rows]))
        den = pc("D_LATE") - pc("END")
        i0 = dict(dnoop_agree=agree, dnoop_p95=p95, dpos_q_ok=qok, start_acc=st, restore=den,
                  ok=bool(d["t1_menu"] >= .95 and agree >= .90 and p95 <= .5 and qok >= .95 and st >= .90 and d["key_rel_diff_max"] == 0.0 and den >= .15))
        c = dict(n=n, I0=i0, P_copy={a: pc(a) for a in arms}, P_target={a: float(np.mean([r["land"][a] == j for r in rows])) for a in arms},
                 mass={a: float(np.nanmean([mass(r, a) for r in rows])) for a in arms})
        c["rho"] = {str(b): ((pc(f"D_{b}") - pc("END")) / den if den >= .15 else None) for b in bounds}
        dm = c["mass"]["D_LATE"] - c["mass"]["END"]
        c["rho_mass"] = {str(b): ((c["mass"][f"D_{b}"] - c["mass"]["END"]) / dm if abs(dm) >= .05 else None) for b in bounds}
        out["cases"][cname] = c
        if i0["ok"]:
            pooled[cname] = rows
    def frac(groups, b, key):
        num = den = 0.0
        for cname, rows in groups.items():
            k = int(cname.split("k")[1].split("j")[0])
            if key == "copy":
                f = lambda a: np.mean([r["land"][a] == k for r in rows])
            else:
                f = lambda a: np.nanmean([mass(r, a) for r in rows])
            num += f(f"D_{b}") - f("END"); den += f("D_LATE") - f("END")
        return num / den if abs(den) > 1e-9 else np.nan
    def boot(groups, b, key):
        vals = []
        for _ in range(4000):
            g2 = {c: [rows[i] for i in RNG.integers(0, len(rows), len(rows))] for c, rows in groups.items()}
            vals.append(frac(g2, b, key))
        vals = np.array(vals)
        return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
    def verdict(groups, key):
        res = {str(b): dict(rho=float(frac(groups, b, key)), ci=boot(groups, b, key)) for b in bounds} if groups else {}
        bstar = next((b for b in bounds if res and res[str(b)]["rho"] >= .80 and res[str(b)]["ci"][0] >= .50), None)
        if not groups:
            v = "no valid case"
        elif bstar is None:
            v = "written at layer 50 or later"
        elif bstar == bounds[0]:
            v = "in place before the lookup band"
        else:
            prev = bounds[bounds.index(bstar) - 1]
            v = f"in place by layer {bstar} (written in ({prev}, {bstar}])"
        return dict(curve=res, b_star=bstar, verdict=v)
    out["valid_cases"] = sorted(pooled)
    out["WRITE"] = verdict(pooled, "copy")
    ident = {c: r for c, r in pooled.items() if c.split("j")[1] == c.split("k")[1].split("j")[0]}
    nonid = {c: r for c, r in pooled.items() if c not in ident}
    out["SPLIT"] = dict(identity=verdict(ident, "copy"), non_identity=verdict(nonid, "copy"))
    out["SPLIT"]["same"] = out["SPLIT"]["identity"]["b_star"] == out["SPLIT"]["non_identity"]["b_star"]
    mg = {c: r for c, r in pooled.items() if abs(out["cases"][c]["mass"]["D_LATE"] - out["cases"][c]["mass"]["END"]) >= .05}
    out["MODE"] = verdict(mg, "mass")
    out["MODE"]["same_as_write"] = out["MODE"]["b_star"] == out["WRITE"]["b_star"]
    fn = f"{V}/results/r3/s836_analysis.json" if tag == "s836_27b" else f"{V}/results/r3/s836_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for cname, c in out["cases"].items():
        print(cname, "I0", c["I0"], "\n   P_copy", {a: round(x, 3) for a, x in c["P_copy"].items()}, "\n   rho", c["rho"], "\n   mass", {a: round(x, 3) for a, x in c["mass"].items()})
    print("WRITE", json.dumps(out["WRITE"])[:900]); print("SPLIT", out["SPLIT"]["identity"]["verdict"], "|", out["SPLIT"]["non_identity"]["verdict"])
    print("MODE", json.dumps(out["MODE"])[:600])


if __name__ == "__main__":
    main(*sys.argv[1:])
