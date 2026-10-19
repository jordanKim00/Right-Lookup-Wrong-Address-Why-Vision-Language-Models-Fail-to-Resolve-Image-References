"""S837 analysis (rules fixed in knowledge/DECISIONS.md before any S837 data).  Band = layers 46-49 (S836: the late queries' END information is
written in (46, 50] - the reading mode in (46, 48], the copy-shift in (48, 50]).
I0 (per case): t1 >= .95; DNOOP names the same image as END in >= 90% of scenes with the 95th percentile of the largest menu-logit change <= .5;
START(k) -> k >= .90; image keys identical (0.0); full restore P_copy(D_LATE) - P_copy(END) >= .15.  Valid cases are pooled.
P_copy(arm) = share answering image k (identity: the target; otherwise the early-found copy, 'step removed'); restore fraction
rho(arm) = (P_copy(arm) - P_copy(END)) / (P_copy(D_LATE) - P_copy(END)) pooled as sum of numerators over sum of denominators; scene bootstrap 4,000.
Late picture mass m as S836 (rows B1, B4, answer; layers 51, 55; matcher heads); rho_m the same fraction for m.
Verdicts (pooled):
  COMP  a single component c (C_l_mix / C_l_mlp) is a 'writer' if rho(c) >= .30 with lower bound > .10; the list of writers (or 'none').
  BAND  rho(C_BAND) >= .60 with lower bound > .40 -> 'the band's direct paths carry the late END information'; else 'not through direct paths'.
  MODEW the component with the largest rho_m (descriptive), and 'reading-mode writer' for any c with rho_m(c) >= .30 (lower bound > .10).
  DOSE  (gap 4) answer step = signed distance of the answer from k toward the target (identity cells: absolute distance), answers on a picture
        only: 'the step size follows the band's write magnitude' if mean step(DOSE_M1) - mean step(END) >= .30 (interval above 0) and
        P(step >= 2 | DOSE_M1) - P(step >= 2 | END) >= .15 (interval above 0); MONO (descriptive): mean step END >= DOSE_H >= C_BAND.
Usage: python exp/s837_analyze.py [TAG]  ->  results/r3/s837_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000


def main(tag="s837_27b"):
    d = json.load(open(f"{V}/results/r3/s837_{tag}_writer.json")); R = d["rows"]; band = d["band"]; late = [str(x) for x in d["late"]]
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
        st = {a: [s for s in (step(r, a) for r in rows) if s is not None] for a in ("END", "DOSE_H", "DOSE_M1", "C_BAND")}
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
    if valid:
        cur = {a: dict(rho=rho(valid, a), ci=boot(valid, lambda g, a=a: rho(g, a))) for a in comps + ["C_BAND", "DOSE_H", "DOSE_M1"]}
        out["curve"] = cur
        writers = [a for a in comps if cur[a]["rho"] >= .30 and cur[a]["ci"][0] > .10]
        out["COMP"] = writers if writers else "none"
        out["BAND"] = ("the band's direct paths carry the late END information" if cur["C_BAND"]["rho"] >= .60 and cur["C_BAND"]["ci"][0] > .40
                       else "not through direct paths")
        mg = {c: r for c, r in valid.items() if abs(out["cases"][c]["mass"]["D_LATE"] - out["cases"][c]["mass"]["END"]) >= .05}
        if mg:
            cm = {a: dict(rho_m=rho(mg, a, "mass"), ci=boot(mg, lambda g, a=a: rho(g, a, "mass"))) for a in comps + ["C_BAND"]}
            out["curve_mass"] = cm
            out["MODEW"] = dict(top=max(comps, key=lambda a: cm[a]["rho_m"]),
                                writers=[a for a in comps if cm[a]["rho_m"] >= .30 and cm[a]["ci"][0] > .10] or "none")
        def stepstat(g, a, ge2=False):
            xs = [s for rows in g.values() for s in (step(r, a) for r in rows) if s is not None]
            return float(np.mean([s >= 2 for s in xs]) if ge2 else np.mean(xs)) if xs else float("nan")
        dmean = stepstat(valid, "DOSE_M1") - stepstat(valid, "END"); dge2 = stepstat(valid, "DOSE_M1", True) - stepstat(valid, "END", True)
        cm_ = boot(valid, lambda g: stepstat(g, "DOSE_M1") - stepstat(g, "END")); cg_ = boot(valid, lambda g: stepstat(g, "DOSE_M1", True) - stepstat(g, "END", True))
        out["DOSE"] = dict(mean={a: stepstat(valid, a) for a in ("END", "DOSE_H", "DOSE_M1", "C_BAND")},
                           ge2={a: stepstat(valid, a, True) for a in ("END", "DOSE_H", "DOSE_M1", "C_BAND")},
                           d_mean=dmean, ci_mean=cm_, d_ge2=dge2, ci_ge2=cg_,
                           verdict=("the step size follows the band's write magnitude" if dmean >= .30 and cm_[0] > 0 and dge2 >= .15 and cg_[0] > 0
                                    else "not shown"))
        mm = out["DOSE"]["mean"]
        out["DOSE"]["MONO"] = bool(mm["END"] >= mm["DOSE_H"] >= mm["C_BAND"])
    fn = f"{V}/results/r3/s837_analysis.json" if tag == "s837_27b" else f"{V}/results/r3/s837_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    for c, v in out["cases"].items():
        print(c, v["I0"], "\n   P_copy", {a: round(x, 3) for a, x in v["P_copy"].items()}, "\n   step", v["step_mean"], v["step_ge2"])
    for key in ("COMP", "BAND", "MODEW", "DOSE"):
        print(key, json.dumps(out.get(key))[:700])
    if "curve" in out:
        print("curve", {a: (round(x["rho"], 3), [round(y, 2) for y in x["ci"]]) for a, x in out["curve"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
