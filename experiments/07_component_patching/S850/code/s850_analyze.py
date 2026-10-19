"""S850 analysis (rules fixed in knowledge/DECISIONS.md before any S850 data; user 10-17 evening).  Boundary-level stop-signal causal test.
8 pictures, k = 4, stated count N_s in {6, 7, 8} (stated target t = N_s - 3), early lookup pinned to p in {3, 4, 5}; 3 x 3 grid, match cells
p = t.  stay = P(answer = p).  Scene bootstrap 4,000 (seed 0).  Steering: boundary rows B1 + B4 at the input of layer 46 (the writer's input)
along the frozen S848 directions (sha256 705fe80fe1e057fc) by +8 / +16 / -8 within-cell SDs, or +16 SD along a random direction in the same
50-PC subspace; FULLB replaces all five boundary rows at that input by S_EARLY_p's.
I0 (per N_s; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(4) -> 4
  >= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0); NOOP_B (zero steering through the hook) answers as
  E_EARLY in >= .99 of runs.
REP-B  held-out validity of the steered signal: on these new scenes, the E_EARLY residual of B1 at the input of layer 46, centred within each scene
  over its 9 cells and projected on the frozen direction; per scene the double-centred interaction of the 3 diagonal cells; Z > 0 with the 95%
  interval above 0 -> 'the boundary goal-attainment signal holds in held-out scenes' (B4 and other layers descriptive).
POS    positive control: share of runs whose FULLB answer differs from E_EARLY >= .30 with the interval's lower end > .15 -> 'the boundary locus
  controls the update'.
WRITER per run and row (B1, B4, answer), the band's total output W (sum of the eight components at the row); rho(arm) = ((W_arm - W_E) . D) / |D|^2
  with D = W_S_EARLY - W_E (the start-relative band output at the same pinned position); rho(PLUS16) with interval above 0 AND dW = rho(PLUS16) -
  rho(RAND16) with interval above 0 -> 'strengthening the goal-attainment signal moves the writer's output toward the start-relative one'
  (amended before data: the absolute part added); else 'no writer response toward stopping', with the bound.
LATE   late hold margin h = z[p] - max(z[p - 1], z[p + 1]) (layers 51, 55; B4 and answer rows; matcher heads); h(PLUS16) - h(E) with interval above 0
  AND dH = [h(PLUS16) - h(E)] - [h(RAND16) - h(E)] with interval above 0 over all cells -> 'the late lookup holds the early position more'
  (amended before data: the absolute part added).
UPDATE (amended 10-17 before data: the smoke showed that a random direction of the same size can lower staying by itself, so a difference to
  RAND alone could read as control; a size-matched RAND8 was added for PLUS8) for each size a in {8, 16}: S_P = stay(PLUSa) - stay(E), S_R =
  stay(RANDa) - stay(E), D = S_P - S_R; each over the 9 cells ('all') and over the 3 match cells ('match').  'target-match information can control
  the update' if, for some size and in 'all' or in 'match', S_P >= .10 with interval above 0 AND D >= .10 with interval above 0 (strengthening raises
  staying above the unsteered run and above random); 'no stop-specific effect' if, for both sizes, S_P <= .05 with its upper end <= .10 in 'all'
  and S_P <= .10 with its upper end <= .15 in 'match' (strengthening does not raise staying; S_R reported for generic disruption); otherwise
  'partial'.
PRED   prior prediction of the late positional change from the writer-to-reader path (weights only; no fitted coefficient): for every PLUS16 run, the
  predicted change of h at layer 51 (B4 and answer rows) from PRED_VC16 (steering vector + the measured band-output change) against the actual
  change: sign agreement >= .70 and Pearson r >= .50 -> 'the writer-to-reader path predicts the late positional change under steering'.
CONCLUSION (registered mapping, user): REP-B fails -> 'the represented-but-unused hypothesis is weakened (the steered signal does not hold in new
  scenes)'; else POS fails -> 'insufficient causal discrimination: no "unused" conclusion'; else UPDATE control -> 'target-match information
  can control the update but is too weak in natural runs'; else UPDATE no stop-specific effect -> 'no evidence of causal use of target-match information in
  the tested writer path (B1 / B4 at the writer input)'; else 'partial'.  No conclusion about other loci or about the whole model.
Descriptive: PLUS8, MINUS8 (match cells), dose; menu-logit margin of p; PRED with the vector alone (PRED_V16); REP at layers 43-49 for B1 / B4;
the chain per arm (writer rho, late dh, stay).
Usage: python exp/s850_analyze.py [TAG]  ->  results/r3/s850_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
PS = (3, 4, 5); NSS = (6, 7, 8)


def main(tag="s850_27b"):
    d = json.load(open(f"{V}/results/r3/s850_{tag}_bstop.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s850_{tag}_bstop.npz")
    dz = np.load(f"{V}/research/ces/outputs/s850_boundary_dirs.npz")
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    RES = d["res_layers"]; KROWS = d["krows"]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in NSS:
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == 4 for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        nb = float(np.mean([r["land"][f"NOOP_B_p{p}"] == r["land"][f"E_EARLY_p{p}"] for r in rows for p in PS]))
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0 and nb >= .99)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, noop_b=nb, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    cells = [(ns, p) for ns in NSS for p in PS]; match = [(p + 3, p) for p in PS]
    def boot(f):
        v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    stay = lambda arm, cs, ss: float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] == p for ns, p in cs for s in ss]))
    # REP-B
    def repb(L, row, tagd):
        li = RES.index(L); ri = d["srows"].index(row)
        m = dz[f"m_{tagd}"].astype(np.float64)
        proj = {s: np.array([[float(Z[f"Ns{ns}_s{s}_E_EARLY_p{p}_res"][li, ri].astype(np.float64) @ m) for ns in NSS] for p in PS]) for s in sis}
        Is = {}
        for s in sis:
            a = proj[s] - proj[s].mean()
            Is[s] = float(np.mean([a[c, c] - a[c, :].mean() - a[:, c].mean() + a.mean() for c in range(3)]))
        sdw = float(np.mean(np.std(np.stack([proj[s] - proj[s].mean() for s in sis]), 0)))
        f = lambda ss: float(np.mean([Is[s] for s in ss]))
        z, ci = f(sis), boot(f)
        return dict(Z=z, ci=ci, dprime=z / sdw if sdw > 0 else float("nan"), dprime_ci=[c / sdw for c in ci])
    rb = dict(B1_L46=repb(46, 2, "L46_r2"), B4_L46=repb(46, 5, "L46_r5"), B1_L44=repb(44, 2, "L44_r2"), B4_L44=repb(44, 5, "L44_r5"))
    out["REP_B"] = dict(sites=rb, verdict=("the boundary goal-attainment signal holds in held-out scenes" if rb["B1_L46"]["Z"] > 0 and rb["B1_L46"]["ci"][0] > 0
                                           else "not shown"))
    # POS
    chg = lambda arm: (lambda ss: float(np.mean([by[ns][s]["land"][f"{arm}_p{p}"] != by[ns][s]["land"][f"E_EARLY_p{p}"] for ns, p in cells for s in ss])))
    pv, pci = chg("FULLB")(sis), boot(chg("FULLB"))
    out["POS"] = dict(changed=pv, ci=pci, stay_FULLB=stay("FULLB", cells, sis), stay_E=stay("E_EARLY", cells, sis),
                      verdict=("the boundary locus controls the update" if pv >= .30 and pci[0] > .15 else "not shown"))
    # WRITER
    def rho(arm, s, ns, p):
        W = lambda a_: Z[f"Ns{ns}_s{s}_{a_}_p{p}_comp"].astype(np.float64).sum(0)        # [rows, E]
        WE, WS, WA = W("E_EARLY"), W("S_EARLY"), W(arm)
        D = WS - WE
        return float(np.mean([((WA[i] - WE[i]) @ D[i]) / (D[i] @ D[i] + 1e-12) for i in range(len(KROWS))]))
    rhos = {s: {(ns, p): (rho("PLUS16", s, ns, p), rho("RAND16", s, ns, p)) for ns, p in cells} for s in sis}
    fw = lambda ss: float(np.mean([rhos[s][c][0] - rhos[s][c][1] for s in ss for c in cells]))
    fp = lambda ss: float(np.mean([rhos[s][c][0] for s in ss for c in cells]))
    wv, wci, pv0, pci0 = fw(sis), boot(fw), fp(sis), boot(fp)
    out["WRITER"] = dict(dW=wv, ci=wci, rho_PLUS16=pv0, rho_PLUS16_ci=pci0, rho_RAND16=float(np.mean([rhos[s][c][1] for s in sis for c in cells])),
                         verdict=("strengthening the goal-attainment signal moves the writer's output toward the start-relative one" if wci[0] > 0 and pci0[0] > 0
                                  else "no writer response toward stopping"))
    # LATE hold margins
    def hold(arm, s, ns, p, layers=("51", "55"), rws=(5, 9)):
        v = []
        for L in layers:
            for row in rws:
                z = by[ns][s]["qk"][f"{arm}_p{p}"][L][row]
                v.append(z[p - 1] - max(z[p - 2], z[p] if p < len(z) else -1e9))
        return float(np.mean(v))
    fh = lambda ss: float(np.mean([(hold("PLUS16", s, ns, p) - hold("E_EARLY", s, ns, p)) - (hold("RAND16", s, ns, p) - hold("E_EARLY", s, ns, p))
                                   for s in ss for ns, p in cells]))
    hv, hci = fh(sis), boot(fh)
    fh0 = lambda ss: float(np.mean([hold("PLUS16", s, ns, p) - hold("E_EARLY", s, ns, p) for s in ss for ns, p in cells]))
    hv0, hci0 = fh0(sis), boot(fh0)
    out["LATE"] = dict(dH=hv, ci=hci, dH_vs_base=hv0, ci_vs_base=hci0,
                       verdict=("the late lookup holds the early position more" if hci[0] > 0 and hci0[0] > 0 else "not shown"))
    # UPDATE
    res_u = {}
    for a_ in (8, 16):
        for nm_c, cs in (("all", cells), ("match", match)):
            SP = lambda ss, cs=cs, a_=a_: stay(f"PLUS{a_}", cs, ss) - stay("E_EARLY", cs, ss)
            SR = lambda ss, cs=cs, a_=a_: stay(f"RAND{a_}", cs, ss) - stay("E_EARLY", cs, ss)
            DD = lambda ss, cs=cs, a_=a_: SP(ss, cs, a_) - SR(ss, cs, a_)
            res_u[f"{nm_c}_{a_}"] = dict(S_P=[SP(sis), boot(SP)], S_R=[SR(sis), boot(SR)], D=[DD(sis), boot(DD)])
    ctrl = any(res_u[k]["S_P"][0] >= .10 and res_u[k]["S_P"][1][0] > 0 and res_u[k]["D"][0] >= .10 and res_u[k]["D"][1][0] > 0 for k in res_u)
    nochg = all(res_u[f"all_{a_}"]["S_P"][0] <= .05 and res_u[f"all_{a_}"]["S_P"][1][1] <= .10 and res_u[f"match_{a_}"]["S_P"][0] <= .10
                and res_u[f"match_{a_}"]["S_P"][1][1] <= .15 for a_ in (8, 16))
    uv = "target-match information can control the update" if ctrl else "no stop-specific effect" if nochg else "partial"
    out["UPDATE"] = dict(**res_u, verdict=uv)
    # PRED (layer 51, B4 and answer rows)
    pa, pr, pv_ = [], [], []
    for s in sis:
        for ns, p in cells:
            act = hold("PLUS16", s, ns, p, ("51",)) - hold("E_EARLY", s, ns, p, ("51",))
            pred = hold("PRED_VC16", s, ns, p, ("51",)) - hold("E_EARLY", s, ns, p, ("51",))
            predv = hold("PRED_V16", s, ns, p, ("51",)) - hold("E_EARLY", s, ns, p, ("51",))
            pa.append(act); pr.append(pred); pv_.append(predv)
    pa, pr, pv_ = map(np.array, (pa, pr, pv_))
    sg = float(np.mean(np.sign(pa) == np.sign(pr))); rr = float(np.corrcoef(pa, pr)[0, 1])
    out["PRED"] = dict(sign_agree=sg, r=rr, sign_agree_vector_only=float(np.mean(np.sign(pa) == np.sign(pv_))), r_vector_only=float(np.corrcoef(pa, pv_)[0, 1]),
                       mean_actual=float(pa.mean()), mean_pred=float(pr.mean()), mean_pred_vector_only=float(pv_.mean()),
                       verdict=("the writer-to-reader path predicts the late positional change under steering" if sg >= .70 and rr >= .50 else "not shown"))
    # conclusion
    if not ok_all:
        concl = "no valid run"
    elif out["REP_B"]["verdict"] == "not shown":
        concl = "the represented-but-unused hypothesis is weakened (the steered signal does not hold in new scenes)"
    elif out["POS"]["verdict"] == "not shown":
        concl = "insufficient causal discrimination: no 'unused' conclusion"
    elif uv == "target-match information can control the update":
        concl = "target-match information can control the update but is too weak in natural runs"
    elif uv == "no stop-specific effect":
        concl = "no evidence of causal use of target-match information in the tested writer path (B1 / B4 at the writer input)"
    else:
        concl = "partial"
    out["CONCLUSION"] = concl
    # descriptive
    desc = {}
    for arm in ("NOOP_B", "PLUS8", "PLUS16", "MINUS8", "RAND8", "RAND16", "FULLB", "S_EARLY"):
        desc[arm] = dict(stay_all=stay(arm, cells, sis), stay_match=stay(arm, match, sis), changed=chg(arm)(sis),
                         hold=float(np.mean([hold(arm, s, ns, p) for s in sis for ns, p in cells])))
    desc["E_EARLY"] = dict(stay_all=stay("E_EARLY", cells, sis), stay_match=stay("E_EARLY", match, sis),
                           hold=float(np.mean([hold("E_EARLY", s, ns, p) for s in sis for ns, p in cells])))
    out["desc"] = desc
    fn = f"{V}/results/r3/s850_analysis.json" if tag == "s850_27b" else f"{V}/results/r3/s850_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:700])
    for k in ("REP_B", "POS", "WRITER", "LATE", "UPDATE", "PRED"):
        print(k, json.dumps(out[k])[:600])
    print("CONCLUSION", concl); print("desc", json.dumps(desc))


if __name__ == "__main__":
    main(*sys.argv[1:])
