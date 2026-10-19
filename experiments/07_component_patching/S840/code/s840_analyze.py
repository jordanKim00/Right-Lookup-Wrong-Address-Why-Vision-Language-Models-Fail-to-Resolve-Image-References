"""S840 analysis (rules fixed in knowledge/DECISIONS.md before any S840 data).  Band = layers 46-49; cases, I0 and the restore fraction rho / rho_m
exactly as S837 (P_copy = answer at k; rho = (P_copy(arm) - P_copy(END)) / (P_copy(D_LATE) - P_copy(END)), pooled over valid cases, scene
bootstrap 4,000; rho_m the same for the late picture mass).
Verdicts (pooled):
  GROUP  rho(G_MIX) - rho(G_MLP) >= .15 with interval above 0 -> 'the mixers carry the step on direct paths'; the reverse -> 'the MLPs ...';
         otherwise 'both or neither'.
  HALF   rho(G_4849) - rho(G_4647) >= .15 (interval above 0) -> 'layers 48-49'; the reverse -> 'layers 46-47'; otherwise 'both halves'.
  DOWNSTREAM  rho(A_BAND) - rho(C_BAND) >= .20 (interval above 0) -> 'the band's outputs act through downstream processing'; rho(A_BAND) <=
         rho(C_BAND) + .05 -> 'direct paths suffice'; otherwise 'partly downstream'.
  BOTTLENECK  a single component c with rho(A_c) >= .50 (lower bound > .30) -> 'its END output is needed for the step' (list or 'none').
  MODEB  the same rule on rho_m(A_c) for the reading mode (list or 'none').
Usage: python exp/s840_analyze.py [TAG]  ->  results/r3/s840_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s840_27b"):
    d = json.load(open(f"{V}/results/r3/s840_{tag}_paths.json")); R = d["rows"]; band = d["band"]; late = [str(x) for x in d["late"]]
    comps = [f"C_{l}_{kd}" for l in band for kd in ("mix", "mlp")]
    rng = np.random.default_rng(0)
    def mass(r, a):
        x = r["mass"].get(a, {})
        v = [np.mean([x[L][i] for i in (2, 5, 9)]) for L in late if L in x]
        return float(np.mean(v)) if v else np.nan
    def step(r, a):
        land, k, j = r["land"][a], r["k"], r["j"]
        if land < 1:
            return None
        return abs(land - k) if j == k else (land - k) * np.sign(j - k)
    out = dict(tag=tag, band=band, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], cases={})
    valid = {}
    for N, k, j in d["cases"]:
        rows = [r for r in R if r["N"] == N and r["k"] == k]
        cname = f"N{N}k{k}j{j}"
        pc = lambda a, rr=rows: float(np.mean([r["land"][a] == k for r in rr]))
        agree = float(np.mean([r["land"]["DNOOP"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["DNOOP"], r["mlog"]["END"])) for r in rows], 95))
        den = pc("D_LATE") - pc("END")
        i0 = dict(dnoop_agree=agree, dnoop_p95=p95, start_acc=pc("START"), restore=den,
                  ok=bool(d["t1_menu"] >= .95 and agree >= .90 and p95 <= .5 and pc("START") >= .90 and d["key_rel_diff_max"] == 0.0 and den >= .15))
        arms = sorted(rows[0]["land"])
        st = {a: [s for s in (step(r, a) for r in rows) if s is not None] for a in ("END", "C_BAND", "A_BAND")}
        out["cases"][cname] = dict(n=len(rows), I0=i0, P_copy={a: pc(a) for a in arms},
                                   mass={a: float(np.nanmean([mass(r, a) for r in rows])) for a in arms},
                                   step_mean={a: (float(np.mean(x)) if x else None) for a, x in st.items()},
                                   step_ge2={a: (float(np.mean([s >= 2 for s in x])) if x else None) for a, x in st.items()})
        if i0["ok"]:
            valid[cname] = rows
    def kof(c):
        return int(c.split("k")[1].split("j")[0])
    def rho(groups, a, key="copy"):
        num = den = 0.0
        for c, rows in groups.items():
            k = kof(c)
            f = (lambda x: np.mean([r["land"][x] == k for r in rows])) if key == "copy" else (lambda x: np.nanmean([mass(r, x) for r in rows]))
            num += f(a) - f("END"); den += f("D_LATE") - f("END")
        return float(num / den) if abs(den) > 1e-9 else float("nan")
    def boot(groups, f):
        v = []
        for _ in range(B):
            v.append(f({c: [rows[i] for i in rng.integers(0, len(rows), len(rows))] for c, rows in groups.items()}))
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out["valid_cases"] = sorted(valid)
    ARMS = ["C_BAND", "G_MIX", "G_MLP", "G_4647", "G_4849", "A_BAND"] + [f"A_{l}_{kd}" for l in band for kd in ("mix", "mlp")]
    if valid:
        cur = {a: dict(rho=rho(valid, a), ci=boot(valid, lambda g, a=a: rho(g, a))) for a in ARMS}
        out["curve"] = cur
        def dverdict(a1, a2, thr):
            dv = cur[a1]["rho"] - cur[a2]["rho"]; ci = boot(valid, lambda g: rho(g, a1) - rho(g, a2))
            return dv, ci, (dv >= thr and ci[0] > 0), (-dv >= thr and ci[1] < 0)
        dv, ci, p1, p2 = dverdict("G_MIX", "G_MLP", .15)
        out["GROUP"] = dict(diff=dv, ci=ci, verdict=("the mixers carry the step on direct paths" if p1 else "the MLPs carry the step on direct paths" if p2
                                                        else "both or neither"))
        dv, ci, p1, p2 = dverdict("G_4849", "G_4647", .15)
        out["HALF"] = dict(diff=dv, ci=ci, verdict=("layers 48-49" if p1 else "layers 46-47" if p2 else "both halves"))
        dv, ci, p1, _ = dverdict("A_BAND", "C_BAND", .20)
        out["DOWNSTREAM"] = dict(diff=dv, ci=ci, verdict=("the band's outputs act through downstream processing" if p1 else
                                                          "direct paths suffice" if cur["A_BAND"]["rho"] <= cur["C_BAND"]["rho"] + .05 else "partly downstream"))
        singles = [f"A_{l}_{kd}" for l in band for kd in ("mix", "mlp")]
        out["BOTTLENECK"] = [a for a in singles if cur[a]["rho"] >= .50 and cur[a]["ci"][0] > .30] or "none"
        mg = {c: r for c, r in valid.items() if abs(out["cases"][c]["mass"]["D_LATE"] - out["cases"][c]["mass"]["END"]) >= .05}
        if mg:
            cm = {a: dict(rho_m=rho(mg, a, "mass"), ci=boot(mg, lambda g, a=a: rho(g, a, "mass"))) for a in ARMS}
            out["curve_mass"] = cm
            out["MODEB"] = [a for a in singles if cm[a]["rho_m"] >= .50 and cm[a]["ci"][0] > .30] or "none"
    fn = f"{V}/results/r3/s840_analysis.json" if tag == "s840_27b" else f"{V}/results/r3/s840_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for c, v in out["cases"].items():
        print(c, v["I0"], "\n   P_copy", {a: round(x, 3) for a, x in v["P_copy"].items()}, "\n   step", v["step_mean"], v["step_ge2"])
    for key in ("GROUP", "HALF", "DOWNSTREAM", "BOTTLENECK", "MODEB"):
        print(key, json.dumps(out.get(key))[:700])
    if "curve" in out:
        print("curve", {a: (round(x["rho"], 3), [round(y, 2) for y in x["ci"]]) for a, x in out["curve"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
