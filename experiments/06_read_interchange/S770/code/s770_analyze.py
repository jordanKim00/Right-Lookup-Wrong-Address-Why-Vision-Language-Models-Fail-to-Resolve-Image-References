"""S770 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 23:4x) before any S770 data.
END family: P5(arm) = share answering image 5; R = P5(F5P1) - P5(E5); phi(arm) = (P5(arm) - P5(E5)) / R (given fraction);
rho(X) = (P5(F5P1) - P5(REV_X@all)) / R (removed fraction).
I0          NOOP_E5 = E5, NOOP_F5P1 = F5P1 and NOOP_S3 = S3 in answers on >= .95 of scenes, and GIVE_DMTG@all = OUT@all on >= .95.
REPL        R >= .30.
ATT         phi(GIVE_DMTG@all): CARRIES if >= .80, PARTIAL if >= .30, NONE otherwise (does the L35-59 full-attention read at the question rows carry
            the frame effect, or does part of it travel through the DeltaNet / MLP sublayers of those rows).  NEC_ATT: rho(DMTG) >= .80.
Q_PATH      phi(E_Q@all) >= .50 and (P5(F5P1) - P5(REV_Q@all)) / R >= .50: the frame effect reaches the answer through the L35-59 queries of the
            question rows (sufficient and necessary);  Q_LOCUS = the band among early, mid, late with the largest phi(E_Q@band).
GAP         S768's missing part: increments dM = phi(DM@all) - phi(D@all), dT = phi(DMT@all) - phi(DM@all), dG = phi(DMTG@all) - phi(DMT@all);
            GAP = the component with the largest increment if it is >= .20, else NONE.
START family: C6(arm) = P6(arm) - P6(S3) (START(3) moved to START(6)'s image).
Q_REPL      C6(S_Q_DMT@late) >= .50 (S767 late .85).
LATE_ROUTE  c_X = C6(X) / C6(S_Q_DMT@late) for X = S_Q_M@late (mass), S_Q_T@late (text weights), S_GIVE_D@late (image distribution);
            IMAGE / MASS / TEXT if exactly that one has c >= .50; REDUNDANT if two or more; JOINT if none.  Read only if Q_REPL.
MID_ROUTE   the same at mid; read only if C6(S_Q_DMT@mid) >= .30.
Descriptive: every arm's answer distribution, phi / rho / C6 tables per band, text values (phi(T) - phi(Tw)); from the natural runs, per band layer and
row "answer": head-mean image mass, text mass on the menu words, the share of heads whose top menu word is the answered word, mass on the
ordinal and frame words.
Usage: python s770_analyze.py TAG
"""
import json, os, sys
import numpy as np


def analyze(tag):
    D = f"/vlm/results/r3/s770_{tag}_dmtg"
    d = json.load(open(D + ".json")); M = d["meta"]; n = len(M)
    P = lambda key, x: float(np.mean([m["ans"][key]["pos"] == x for m in M]))
    agree = lambda a, b: float(np.mean([m["ans"][a]["pos"] == m["ans"][b]["pos"] for m in M]))
    out = dict(n=n, checks={}, dist={k: {str(b): round(P(k, b), 3) for b in range(0, 8) if P(k, b) > 0} for k in M[0]["ans"]})
    i0 = dict(E5=agree("NOOP_E5", "E5"), F5P1=agree("NOOP_F5P1", "F5P1"), S3=agree("NOOP_S3", "S3"), OUT=agree("GIVE_DMTG@all", "OUT@all"))
    out["i0"] = {k: round(x, 3) for k, x in i0.items()}
    out["checks"]["I0"] = bool(all(x >= .95 - 1e-9 for x in i0.values()))
    e5, f5 = P("E5", 5), P("F5P1", 5); R = f5 - e5
    out["recovery"] = dict(E5=round(e5, 3), F5P1=round(f5, 3), R=round(R, 3))
    out["checks"]["REPL"] = bool(R >= .30 - 1e-9)
    phi = lambda key: (P(key, 5) - e5) / R if R > 0 else float("nan")
    out["phi"] = {b: {x: round(phi(f"GIVE_{x}@{b}"), 3) for x in ("D", "M", "T", "Tw", "G", "DM", "DG", "DMT", "DMTG")} for b in ("mid", "late", "all")}
    out["phi"]["OUT@all"] = round(phi("OUT@all"), 3)
    out["rho"] = {x: round((f5 - P(f"REV_{x}@all", 5)) / R, 3) if R > 0 else None for x in ("D", "M", "T", "G", "DMTG")}
    pa = out["phi"]["all"]
    out["checks"]["ATT"] = "CARRIES" if pa["DMTG"] >= .80 - 1e-9 else "PARTIAL" if pa["DMTG"] >= .30 - 1e-9 else "NONE"
    out["checks"]["NEC_ATT"] = bool(out["rho"]["DMTG"] is not None and out["rho"]["DMTG"] >= .80 - 1e-9)
    out["phi_Q"] = {b: round(phi(f"E_Q@{b}"), 3) for b in ("early", "mid", "late", "all")}
    out["rho_Q"] = round((f5 - P("REV_Q@all", 5)) / R, 3) if R > 0 else None
    out["checks"]["Q_PATH"] = bool(R > 0 and out["phi_Q"]["all"] >= .50 - 1e-9 and out["rho_Q"] >= .50 - 1e-9)
    out["checks"]["Q_LOCUS"] = max(("early", "mid", "late"), key=lambda b: out["phi_Q"][b]) if R > 0 else None
    inc = dict(M=pa["DM"] - pa["D"], T=pa["DMT"] - pa["DM"], G=pa["DMTG"] - pa["DMT"])
    out["increments"] = {k: round(x, 3) for k, x in inc.items()}
    best = max(inc, key=inc.get)
    out["checks"]["GAP"] = best if inc[best] >= .20 - 1e-9 else "NONE"
    out["text_values"] = {b: round(out["phi"][b]["T"] - out["phi"][b]["Tw"], 3) for b in ("mid", "late", "all")}
    s3 = P("S3", 6)
    C6 = lambda key: P(key, 6) - s3
    out["C6"] = {b: dict(**{f"GIVE_{x}": round(C6(f"S_GIVE_{x}@{b}"), 3) for x in ("D", "M", "T", "Tw", "G", "DMTG")},
                         **{f"Q_{x}": round(C6(f"S_Q_{x}@{b}"), 3) for x in ("DMT", "M", "T")}) for b in ("mid", "late")}
    out["checks"]["Q_REPL"] = bool(C6("S_Q_DMT@late") >= .50 - 1e-9)

    def route(b):
        tot = C6(f"S_Q_DMT@{b}")
        c = dict(IMAGE=C6(f"S_GIVE_D@{b}") / tot, MASS=C6(f"S_Q_M@{b}") / tot, TEXT=C6(f"S_Q_T@{b}") / tot)
        big = [k for k, x in c.items() if x >= .50 - 1e-9]
        verdict = big[0] if len(big) == 1 else "REDUNDANT" if len(big) >= 2 else "JOINT"
        return verdict, {k: round(x, 3) for k, x in c.items()}
    lr, lc = route("late") if out["checks"]["Q_REPL"] else (None, None)
    mr, mc = route("mid") if C6("S_Q_DMT@mid") >= .30 - 1e-9 else (None, None)
    out["checks"]["LATE_ROUTE"] = lr; out["checks"]["MID_ROUTE"] = mr; out["route_c"] = dict(late=lc, mid=mc)
    # descriptive reading of the natural runs (row "answer")
    z = np.load(D + ".npz")
    desc = {}
    for nm in ("E5", "F5P1", "S3", "S6"):
        desc[nm] = {}
        for li in d["PB"]["all"]:
            im, mn, hit, ordm, frm = [], [], [], [], []
            for m in M:
                A = z[f"{m['si']}|{nm}|{li}"].astype(np.float64)[2]                     # [H, S] row "answer"
                ps = m["pos"]; img = np.zeros(A.shape[1], bool)
                for b in range(7):
                    img[ps["vs"][b]:ps["ve"][b] + 1] = True
                im.append(A[:, img].sum(-1).mean())
                ow = list(ps["opt"]); op = [ps["opt"][w] for w in ow]
                mn.append(A[:, op].sum(-1).mean())
                aw = {"E5": m["ans"]["E5"], "F5P1": m["ans"]["F5P1"], "S3": m["ans"]["S3"], "S6": m["ans"]["S6"]}[nm]
                word = (m["labs"] + m["extra"])[int(np.argmax(aw["ll"]))]
                top = [ow[i] for i in A[:, op].argmax(-1)]
                hit.append(np.mean([t == word for t in top]))
                ordm.append(A[:, ps["o"]].mean()); frm.append(A[:, ps["f"]].mean())
            desc[nm][str(li)] = dict(img=round(float(np.mean(im)), 3), menu=round(float(np.mean(mn)), 3), top_menu_is_answer=round(float(np.mean(hit)), 3),
                                     ordinal=round(float(np.mean(ordm)), 4), frame=round(float(np.mean(frm)), 4))
    out["reading"] = desc
    print(f"[{tag}] n {n} · checks {out['checks']} · I0 {out['i0']} · recovery {out['recovery']}")
    print(f"   phi {out['phi']}\n   rho {out['rho']} · increments {out['increments']} · text values {out['text_values']}")
    print(f"   query path phi {out['phi_Q']} · rho {out['rho_Q']}")
    print(f"   C6 {out['C6']}\n   route {out['route_c']}")
    for nm, x in desc.items():
        print(f"   reading {nm}: " + " | ".join(f"L{li} img {v['img']} menu {v['menu']} top=ans {v['top_menu_is_answer']}" for li, v in x.items()))
    return out


if __name__ == "__main__":
    json.dump({sys.argv[1]: analyze(sys.argv[1])}, open(os.environ.get("ANALYSIS_OUT", "/vlm/results/r3/s770_analysis.json"), "w"), indent=1)
