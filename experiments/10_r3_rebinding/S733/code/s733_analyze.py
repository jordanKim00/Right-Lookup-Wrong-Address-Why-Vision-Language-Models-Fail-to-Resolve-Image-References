"""S733 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-10) before any S733 data.
Other families (InternVL3-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B) with the same calibrated pipeline as the paper models.
Held-out test: the later MMIU split (500 end-relative items), never evaluated on these families.
    XFAM_REPAIR[m]:   later MMIU end-relative accuracy, R³ - original >= .20 with the 95% paired bootstrap interval above 0, and R³ >= .8 x forward
    XFAM_LOCALITY[m]: forward answers changed by R³ <= .01 (later and development runs)
    XFAM_MECH[m]:     S730 on procedural scenes: >= 5 discovered index-matching heads, END literal share > target share at the answer row
                      (k != 4), and QUERY_RESTORED
Reported: READ choice and PlanExact (S724), reader admission (S729), development MMIU / MUIR next to the earlier module's records.
Usage: python s733_analyze.py FAM   (ivl3_8b | q25_7b | llov_7b)
"""
import json, os, sys
import numpy as np

R = "/vlm/results"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def boot(x, B=4000, seed=73300):
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main(f):
    out, chk = {}, {}
    P = (load(f"{R}/r3/s724_plan.json") or {}).get(f"rc_{f}")
    if P:
        out["read"] = dict(final=P["final_name"], val={k: v["all"]["exact"] for k, v in P["val"].items()})
        print(f"[{f}] READ: {P['final_name']} · val PlanExact " + " · ".join(f"{k} {v:.3f}" for k, v in out["read"]["val"].items()))
    rd = load(f"{R}/r3/s729_rd_{f}_reader.json")
    if rd:
        n, o = rd["res"]["new"], rd["res"]["old"]
        adm = n["all"]["count"] >= o["all"]["count"] - 1e-12 and all(n[k]["count"] >= o[k]["count"] - .01 - 1e-12 for k in ("proc", "nat", "low"))
        out["reader"] = dict(admit=bool(adm), new={k: n[k]["count"] for k in ("proc", "nat", "low", "all")}, old={k: o[k]["count"] for k in ("proc", "nat", "low", "all")})
        print(f"[{f}] 판독기 개수 정확 새 {out['reader']['new']} · 옛 {out['reader']['old']} → 채택 {adm}")
    fwd_changed = []
    for kind, tag in (("later", f"x5fr_{f}"), ("dev", f"x5dv_{f}")):
        d = load(f"{R}/s582_evalrepair_{tag}.json")
        if not d:
            continue
        for suite in sorted({r["suite"] for r in d["rows"]}):
            rows = [r for r in d["rows"] if r["suite"] == suite]
            g = lambda k: np.array([r[k] == r["gold"] for r in rows], float)
            orig, rep, fwd = g("R|base"), g("R|rep"), g("F|base")
            ch = float(np.mean([r["F|rep"] != r["F|base"] for r in rows])); fwd_changed.append(ch)
            d_ = boot(rep - orig)
            out[f"{kind}|{suite}"] = dict(n=len(rows), original=float(orig.mean()), r3=float(rep.mean()), forward=float(fwd.mean()), d=d_, fwd_changed=ch)
            print(f"[{f}] {kind} {suite}: 원래 {orig.mean():.3f} → R³ {rep.mean():.3f} (앞 방향 {fwd.mean():.3f}) · 차 {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] · 앞 방향 바뀜 {ch:.3f}")
    L = out.get("later|mmiu_pos")
    if L:
        chk["XFAM_REPAIR"] = L["d"][0] >= .20 - 1e-12 and L["d"][1] > 0 and L["r3"] >= .8 * L["forward"] - 1e-12
    if fwd_changed:
        chk["XFAM_LOCALITY"] = max(fwd_changed) <= .01 + 1e-12
    v1dev = load(f"{R}/s582_evalrepair_e582_s616b_{f}.json")
    if v1dev and "dev|mmiu_pos" in out:
        for suite in ("mmiu_pos", "muir"):
            rows = [r for r in v1dev["rows"] if r["suite"] == suite]
            if rows:
                out[f"v1dev|{suite}"] = float(np.mean([r["R|rep"] == r["gold"] for r in rows]))
        print(f"[{f}] v1(S616b) 개발: " + " · ".join(f"{s} {out[f'v1dev|{s}']:.3f}" for s in ("mmiu_pos", "muir") if f"v1dev|{s}" in out))
    S7 = (load(f"{R}/r3/s730_analysis.json") or {}).get(f"stw_{f}")
    if S7:
        s = S7["summary"]["k!=4"]
        chk["XFAM_MECH"] = S7["heads"] >= 5 and s["END"]["lit"] > s["END"]["tgt"] and S7["checks"]["QUERY_RESTORED"]
        out["mech"] = dict(heads=S7["heads"], end_lit=s["END"]["lit"], end_tgt=s["END"]["tgt"], checks=S7["checks"])
        print(f"[{f}] 기전: 정합 머리 {S7['heads']} · END 문자 몫 {s['END']['lit']:.3f} vs 목표 {s['END']['tgt']:.3f} · S730 {S7['checks']}")
    print(f"[{f}] 판정 {chk}")
    PP = f"{R}/r3/s733_analysis.json"
    allr = load(PP) or {}
    allr[f] = dict(out=out, checks=chk)
    json.dump(allr, open(PP, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
