"""S724 plan-level evaluation (amendment before any S724 data; user direction 2026-10-10: "개별 부품 선택 규칙은 유지하되, 완성된 쓰기 계획의
정확도를 별도의 보정용 검증 자료에서 확인").  No answer is read: every gold plan follows from the item's construction.
Gold plan of an input: for every END-bound reference r (ordinal token p_r, ordinal k_r, frame word token f_r, N images), the frame write at
f_r and, when j_r = N + 1 - k_r differs from k_r, the ordinal write (p_r -> j_r); nothing for START / unframed references or other tokens.
A plan is exact when its ordinal writes (position -> target) and its frame-write positions equal the gold plan's.
Arms (all recomputed offline from exp/s724_readcal.py's per-token measurements):
    min(det, bind, count)   exp/r3_min_md.R3MinMD's plan for detector det, prototype radius bind, N_hat mode count
    v1c(count)              exp/r3_v1c.R3V1C: the earlier module's detections, gate (g > .5) and frame positions + t = N_hat + 1 - k_hat
    v1                      the earlier module itself (R³-v1: its learned pointer target and its writes at the checkpoint's tau)
Reference-level errors, counted per input (an input can carry several):
    MISS     an END reference without a detection or without a frame binding
    CROSS    a reference bound to the frame word of another reference
    MISBIND  an END reference bound to a token that is no frame word
    SPURIOUS a write for a START / unframed reference or for a non-reference token
    ADDR_N   a bound END reference whose ordinal write (or its absence) is wrong while N_hat != N
    ADDR_K   the same with N_hat = N (k_hat wrong; for R³-v1: its pointer)
Selection (sel split): admissible = detectors with false detections <= .005 and binders with false binding <= .02 (component metrics,
exp/s724_select.py); the final Read = the admissible combination with the highest PlanExact on the sel split (ties: the component-rule
choice, then abs < margin .4 < .5 < .6, prototype x1.0 < x1.2 < x1.5 < learned gate, last < max).
Confirmation (val split): PlanExact and the errors for R³-v1, the uniform Read of S718 (margin .5, x1.2, max) and the final Read, with
paired bootstrap intervals over inputs; by image source, mixed vs single-frame inputs and format.
Usage: python s724_plan.py SELTAG VALTAG   (e.g. rc_4b rv_4b); merges into /vlm/results/r3/s724_plan.json
"""
import json, os, sys
import numpy as np

from s724_select import detect, window, analyze

C = 10
DETS = ["abs", "margin:.4", "margin:.5", "margin:.6"]
BINDS = ["proto:1.0", "proto:1.2", "proto:1.5"]
CNTS = ["last", "max"]
UNIFORM = ("min", "margin:.5", "proto:1.2", "max")


def nhat(rec, cnt):
    return rec["codes"][-1] if cnt == "last" else max(rec["codes"])


def plan_min(rec, det, bind, cnt):
    """sources: {pos: (k_hat, frame_pos or None, ordinal target or None)} as R3MinMD.plan decides (writes only for resolvable sources)."""
    m, kh = detect(rec, det); s = float(bind.split(":")[1]); nh = nhat(rec, cnt); toks = rec["toks"]
    src = {}
    for j in range(len(toks)):
        if not m[j]:
            continue
        k = kh[j]
        hits = [q for q in window(toks, j) if rec["d_end"][q] <= s * rec["thr"]["end"] or rec["d_last"][q] <= s * rec["thr"]["last"]]
        t = nh + 1 - k
        if hits and 1 <= t <= C:
            src[j] = (k, hits[0], t if t != k else None)
        else:
            src[j] = (k, None, None)
    return src, nh


def plan_v1c(rec, cnt):
    nh = nhat(rec, cnt); src = {}
    for x in rec["v1c"]:
        k = x["kh"]; t = nh + 1 - k; bound = x["g"] is not None and x["g"] > .5
        if bound and 1 <= t <= C and x["fpos"] is not None:
            src[x["pos"]] = (k, x["fpos"], t if t != k else None)
        elif bound and 1 <= t <= C:
            src[x["pos"]] = (k, None, t if t != k else None)
        else:
            src[x["pos"]] = (k, None, None)
    return src, nh


def plan_v1(rec):
    src = {}
    for x in rec["v1"]:
        src[x["pos"]] = (x["kh"], x["fpos"], x["t"] if x["act"] else None)
    return src, None


def score(rec, src, nh):
    gold = rec["gold"]; N = rec["N"]
    gpos = {g["pos"]: g for g in gold}
    fowner = {g["ftok"]: g["pos"] for g in gold if g["ftok"] is not None}
    end_f = {g["ftok"] for g in gold if g["frame"] == "end"}
    gw = {g["pos"]: N + 1 - g["k"] for g in gold if g["frame"] == "end" and N + 1 - g["k"] != g["k"]}
    pw = {p: x[2] for p, x in src.items() if x[2] is not None}
    pf = {x[1] for x in src.values() if x[1] is not None}
    err = dict(MISS=0, CROSS=0, MISBIND=0, SPURIOUS=0, ADDR_N=0, ADDR_K=0)
    for g in gold:
        x = src.get(g["pos"])
        if g["frame"] == "end":
            if x is None or x[1] is None:
                err["MISS"] += 1; continue
            if x[1] != g["ftok"]:
                err["CROSS" if x[1] in fowner else "MISBIND"] += 1; continue
            if pw.get(g["pos"]) != gw.get(g["pos"]):
                err["ADDR_N" if (nh is not None and nh != N) else "ADDR_K"] += 1
        else:
            if x is not None and (x[1] is not None or x[2] is not None):
                err["SPURIOUS"] += 1
                if x[1] is not None and x[1] in end_f:
                    err["CROSS"] += 1
    for p, x in src.items():
        if p not in gpos and (x[1] is not None or x[2] is not None):
            err["SPURIOUS"] += 1
    exact = pw == gw and pf == end_f
    return exact, err


def arms(rec):
    A = {}
    for d in DETS:
        for b in BINDS:
            for c in CNTS:
                A[("min", d, b, c)] = plan_min(rec, d, b, c)
    if "v1c" in rec:
        for c in CNTS:
            A[("v1c", "v1", "v1_gate", c)] = plan_v1c(rec, c)
        A[("v1", "v1", "v1", "v1")] = plan_v1(rec)
    return A


def evaluate(R):
    rows = []
    for r in R:
        A = arms(r)
        rows.append({k: score(r, s, nh) for k, (s, nh) in A.items()})
    return rows


def summary(R, rows, key, sub=None):
    idx = [i for i, r in enumerate(R) if sub is None or sub(r)]
    ex = np.array([rows[i][key][0] for i in idx], float)
    E = {e: float(np.mean([rows[i][key][1][e] > 0 for i in idx])) for e in rows[0][key][1]}
    return dict(n=len(idx), exact=float(ex.mean()) if len(idx) else float("nan"), **{f"has_{e}": v for e, v in E.items()})


def boot(R, rows, k1, k0, B=4000, seed=72400):
    x = np.array([rows[i][k1][0] for i in range(len(R))], float) - np.array([rows[i][k0][0] for i in range(len(R))], float)
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def name(k):
    return "R3-v1" if k[0] == "v1" else ("learned gate + count " + k[3] if k[0] == "v1c" else f"{k[1]} + {k[2]} + {k[3]}")


def main(sel, val):
    comp = analyze(sel)                                                            # component metrics and the component-rule choice (sel split)
    Rs = json.load(open(f"/vlm/results/r3/s724_{sel}_readcal.json"))["rec"]
    Rv = json.load(open(f"/vlm/results/r3/s724_{val}_readcal.json"))["rec"]
    rows_s, rows_v = evaluate(Rs), evaluate(Rv)
    M = comp["metrics"]["all"]
    ok_d = [d for d in DETS if M["detector"][d]["fp"] <= .005 + 1e-12] or [min(DETS, key=lambda d: M["detector"][d]["fp"])]
    ok_b = [b for b in BINDS if M["binder"][b]["false_bind"] <= .02 + 1e-12] or [min(BINDS, key=lambda b: M["binder"][b]["false_bind"])]
    cands = [("min", d, b, c) for d in ok_d for b in ok_b for c in CNTS]
    if "v1_gate" in M["binder"] and M["binder"]["v1_gate"]["false_bind"] <= .02 + 1e-12 and M["detector"]["abs"]["fp"] <= .005 + 1e-12:
        cands += [("v1c", "v1", "v1_gate", c) for c in CNTS]
    ch = comp["choice"]
    comp_key = (("v1c", "v1", "v1_gate", ch["count"]) if ch["binder"] == "v1_gate" else ("min", ch["detector"], ch["binder"], ch["count"]))
    pe = {k: summary(Rs, rows_s, k)["exact"] for k in rows_s[0]}
    order = lambda k: (-round(pe[k], 9), k != comp_key, (DETS + ["v1"]).index(k[1]), (BINDS + ["v1_gate"]).index(k[2]), CNTS.index(k[3]))
    final = sorted(cands, key=order)[0]
    print(f"[{sel}] 부품 규칙 선택: {name(comp_key)} (PlanExact sel {pe[comp_key]:.3f}) -> 최종(계획 수준): {name(final)} (PlanExact sel {pe[final]:.3f})")
    top = sorted(pe, key=lambda k: -pe[k])[:6]
    print(f"[{sel}]   sel PlanExact 상위: " + " · ".join(f"{name(k)} {pe[k]:.3f}" for k in top))
    rep = {}
    show = [k for k in ([("v1", "v1", "v1", "v1")] if ("v1", "v1", "v1", "v1") in rows_v[0] else []) + [UNIFORM, final]]
    subs = dict(all=None, procedural=lambda r: not r["natural"], natural=lambda r: r["natural"], mixed=lambda r: r["kind"] == "mixed",
                single=lambda r: r["kind"] != "mixed", options=lambda r: r["fmt"].startswith("options"), question=lambda r: r["fmt"] == "question",
                statement=lambda r: r["fmt"] == "statement")
    for k in show:
        rep[name(k)] = {s: summary(Rv, rows_v, k, f) for s, f in subs.items()}
        a = rep[name(k)]["all"]
        print(f"[{val}] {name(k):<34s} PlanExact {a['exact']:.3f} · MISS {a['has_MISS']:.3f} · CROSS {a['has_CROSS']:.3f} · MISBIND {a['has_MISBIND']:.3f} · "
              f"SPURIOUS {a['has_SPURIOUS']:.3f} · ADDR_N {a['has_ADDR_N']:.3f} · ADDR_K {a['has_ADDR_K']:.3f}  "
              f"(절차 {rep[name(k)]['procedural']['exact']:.3f} · 자연 {rep[name(k)]['natural']['exact']:.3f} · 섞인 틀 {rep[name(k)]['mixed']['exact']:.3f})")
    diffs = {"final-uniform": boot(Rv, rows_v, final, UNIFORM)}
    if ("v1", "v1", "v1", "v1") in rows_v[0]:
        diffs["final-v1"] = boot(Rv, rows_v, final, ("v1", "v1", "v1", "v1"))
    print(f"[{val}] 차이(짝 부트스트랩): " + " · ".join(f"{k} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in diffs.items()))
    return dict(component_choice=list(comp_key), final=list(final), final_name=name(final), sel_planexact={name(k): v for k, v in pe.items()},
                admissible=[name(k) for k in cands], val=rep, diffs=diffs)


if __name__ == "__main__":
    P = "/vlm/results/r3/s724_plan.json"
    out = json.load(open(P)) if os.path.exists(P) else {}
    out[sys.argv[1]] = main(sys.argv[1], sys.argv[2])
    json.dump(out, open(P, "w"), indent=1)
