"""S849-P analysis (rules fixed in knowledge/DECISIONS.md before any S849 data; thresholds and the readout fixed on the S848 development data).
Test B: from measured changes of the band 46-49 outputs, a fixed computation predicts the late queries, the image-wise positional scores and the
answer distribution in new scenes and new (p, N_s) combinations (7 pictures, k = 4, N_s 6-8, p in {2, 3, 5, 6}).  Interventions: BANDF (END
pinned, START's band outputs) predicted by PRED_F from E_EARLY; BANDE (START(k) pinned, END's band outputs) predicted by PRED_E from S_EARLY.
PRED arms recompute each late full-attention reader's query (layers 51, 55, 59, 63) from the unchanged residual plus the measured band-output
change, with the model's own weights only (exp/c_qpath_g.py; no fitted parameters).
I0 (per N_s; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th percentile of the largest menu-logit change <= .5; START(4) -> 4
>= .90; donors START(p) -> p >= .90 pooled (>= .80 each); image keys identical (0.0).
Level 1 (late query change): per run and late layer, cosine between the predicted change (PRED query - base query) and the actual change
  (intervention query - base query), boundary B4 and answer rows, all heads; per layer 'predicted' if the median over runs >= .50 with the
  scene-bootstrap 95% interval of the median above .40 (S848 development: .74 / .53 / .48 / .34 at 51 / 55 / 59 / 63).
Level 2 (image-wise positional-score change): per layer, Pearson r between predicted and actual block-score changes (every image, boundary B4
  and answer rows; matcher-head means at 51 / 55, all-head means at 59 / 63); 'predicted' if r >= .60 with the bootstrap interval above .50
  (development: .80 / .81 / .71 / .68).
Level 3 (answer distribution): the frozen readout (research/ces/outputs/s849_readout.json, sha256 32fd3e5e6aff4144; conditional logit over the
  pictures on the late scores) applied to the PRED arms' predicted scores gives each cell's predicted answer distribution under BANDF / BANDE;
  share of (cell x picture) within .15 of the observed share >= .70 -> 'the answer distribution under the band swap is predicted from measured
  component changes' (development: .89).
Each level is reported separately ('levels 1-2 only' = the address update is predicted, the answer readout only in part).
Descriptive: R^2 of the query prediction per layer; the oracle readout (actual intervention scores); the predicted vs observed change of the answer
distribution relative to the base run; per-arm results.
Usage: python exp/s849p_analyze.py [TAG]  ->  results/r3/s849p_analysis.json"""
from __future__ import annotations

import json, sys
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 2000
PS = (2, 3, 5, 6); NSS = (6, 7, 8); N = 7
PAIRS = (("F", "BANDF", "E_EARLY", "PRED_F"), ("E", "BANDE", "S_EARLY", "PRED_E"))


def main(tag="s849p_27b"):
    d = json.load(open(f"{V}/results/r3/s849_{tag}_pred.json")); R = d["rows"]
    Z = np.load(f"{V}/results/r3/s849_{tag}_pred.npz")
    ro = json.load(open(f"{V}/research/ces/outputs/s849_readout.json")); w = np.array(ro["w"]); FEATS = [(f.split("_")[0][1:], int(f.split("row")[1])) for f in ro["feats"]]
    rng = np.random.default_rng(0)
    by = {ns: {r["si"]: r for r in R if r["Ns"] == ns} for ns in NSS}
    sis = sorted(set.intersection(*[set(x) for x in by.values()]))
    late = [str(x) for x in d["late"]]
    out = dict(tag=tag, t1=d["t1_menu"], key_rel_diff_max=d["key_rel_diff_max"], n=len(sis), I0={})
    ok_all = True
    for ns in NSS:
        rows = [by[ns][s] for s in sis]
        ag = float(np.mean([r["land"]["QNOOP_E"] == r["land"]["END"] for r in rows]))
        p95 = float(np.percentile([max(abs(x - y) for x, y in zip(r["mlog"]["QNOOP_E"], r["mlog"]["END"])) for r in rows], 95))
        st = float(np.mean([r["land"]["START"] == 4 for r in rows]))
        don = {p: float(np.mean([r["land"][f"START_p{p}"] == p for r in rows])) for p in PS}
        ok = bool(d["t1_menu"] >= .95 and ag >= .90 and p95 <= .5 and st >= .90 and np.mean(list(don.values())) >= .90 and min(don.values()) >= .80
                  and d["key_rel_diff_max"] == 0.0)
        ok_all &= ok
        out["I0"][str(ns)] = dict(qnoop=ag, p95=p95, start_acc=st, donors=don, ok=ok)
    out["I0"]["all"] = bool(ok_all)
    # level 1 and 2: per (scene) lists so that the bootstrap resamples scenes
    cos = {L: {s: [] for s in sis} for L in late}; r2 = {L: [] for L in late}; dz = {L: {s: ([], []) for s in sis} for L in late}
    for kind, act, base, pred in PAIRS:
        for ns in NSS:
            for s in sis:
                for p in PS:
                    qa = Z[f"Ns{ns}_s{s}_{act}_p{p}_q"].astype(np.float64); qb = Z[f"Ns{ns}_s{s}_{base}_p{p}_q"].astype(np.float64)
                    qp = Z[f"Ns{ns}_s{s}_{pred}_p{p}_qd"].astype(np.float64)
                    for li, L in enumerate(late):
                        da = (qa[li] - qb[li]).ravel(); dp = (qp[li] - qb[li]).ravel()
                        cos[L][s].append(float(da @ dp / (np.linalg.norm(da) * np.linalg.norm(dp) + 1e-12)))
                        r2[L].append(float(1 - ((da - dp) ** 2).sum() / ((da ** 2).sum() + 1e-12)))
                        za = np.array(by[ns][s]["qk"][f"{act}_p{p}"][L]); zb = np.array(by[ns][s]["qk"][f"{base}_p{p}"][L])
                        zp = np.array(by[ns][s]["qk"][f"{pred}_p{p}"][L])
                        for row in (5, 9):
                            dz[L][s][0].extend((za[row] - zb[row]).tolist()); dz[L][s][1].extend((zp[row] - zb[row]).tolist())
    nv = "no valid run"
    L1, L2 = {}, {}
    for L in late:
        med = lambda ss, L=L: float(np.median([c for s in ss for c in cos[L][s]]))
        cr = lambda ss, L=L: float(np.corrcoef([x for s in ss for x in dz[L][s][0]], [x for s in ss for x in dz[L][s][1]])[0, 1])
        bm = [med([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        bc = [cr([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(B)]
        v1, c1 = med(sis), [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        v2, c2 = cr(sis), [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]
        L1[L] = dict(median_cos=v1, ci=c1, r2_median=float(np.median(r2[L])), predicted=bool(v1 >= .50 and c1[0] > .40))
        L2[L] = dict(r=v2, ci=c2, predicted=bool(v2 >= .60 and c2[0] > .50))
    def lv(dd, name):
        ok = [L for L in late if dd[L]["predicted"]]
        if not ok_all:
            return nv
        if len(ok) == len(late):
            return f"the {name} is predicted at all four late layers"
        return f"the {name} is predicted at layers {', '.join(ok)}" if ok else f"the {name} is not predicted"
    out["level1"] = dict(per_layer=L1, verdict=lv(L1, "late query change"))
    out["level2"] = dict(per_layer=L2, verdict=lv(L2, "image-wise positional-score change"))
    # level 3
    def pdist(qk):
        Zm = np.array([[qk[L][row][b] for b in range(N)] for L, row in FEATS]); U = w @ (Zm - Zm.max(1, keepdims=True)); U = U - U.max()
        e = np.exp(U); return e / e.sum()
    cal, cal_or, chg = [], [], []
    for kind, act, base, pred in PAIRS:
        for ns in NSS:
            rows = [by[ns][s] for s in sis]
            for p in PS:
                obs = np.bincount([r["land"][f"{act}_p{p}"] for r in rows], minlength=N + 1)[1:] / len(rows)
                obs_b = np.bincount([r["land"][f"{base}_p{p}"] for r in rows], minlength=N + 1)[1:] / len(rows)
                pp = np.mean([pdist(r["qk"][f"{pred}_p{p}"]) for r in rows], 0)
                po = np.mean([pdist(r["qk"][f"{act}_p{p}"]) for r in rows], 0)
                pb = np.mean([pdist(r["qk"][f"{base}_p{p}"]) for r in rows], 0)
                cal += list(np.abs(pp - obs) <= .15); cal_or += list(np.abs(po - obs) <= .15)
                chg.append((float(np.abs((pp - pb) - (obs - obs_b)).max()), float(np.sign((pp - pb)[p - 1]) == np.sign((obs - obs_b)[p - 1]))))
    c3 = float(np.mean(cal))
    out["level3"] = dict(calibration=c3, oracle_calibration=float(np.mean(cal_or)), change_max_abs_err_median=float(np.median([c[0] for c in chg])),
                         stay_change_sign_agree=float(np.mean([c[1] for c in chg])),
                         verdict=(nv if not ok_all else "the answer distribution under the band swap is predicted from measured component changes" if c3 >= .70
                                  else "not shown"))
    out["answers"] = {f"Ns{ns}_p{p}": {arm: np.bincount([by[ns][s]["land"][f"{arm}_p{p}"] for s in sis], minlength=N + 1)[1:].tolist()
                                      for arm in ("E_EARLY", "S_EARLY", "BANDF", "BANDE")} for ns in NSS for p in PS}
    fn = f"{V}/results/r3/s849p_analysis.json" if tag == "s849p_27b" else f"{V}/results/r3/s849p_analysis_{tag}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out["I0"])[:600])
    for k in ("level1", "level2"):
        print(k, out[k]["verdict"], json.dumps({L: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for L, v in out[k]["per_layer"].items()}))
    print("level3", json.dumps(out["level3"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
