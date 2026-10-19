"""agent_math_1015 / d_sharper_carrier (descriptive CPU test of a rule pre-registered in knowledge/agent_exchange_2026-10-15/
MATH_MECHANISM_REVIEW.md section Q-D at 2026-10-15 16:14:42 KST, before this computation).  Second task D.
Rule: donor-following rises with dm = m(donor natural run) - m(recipient natural run), m = top - second of the answer-row L51 matcher-head mean
block shares (normalized over the 7 picture blocks).  Follow = the exchanged run lands where the donor's natural run lands; only cells where the
donor's and the recipient's natural landings differ.  Scope fixed in advance (target-carrying exchanges): S809 B2-type arms; S811 query-part arms.
Model: logistic Follow ~ stratum (arm x k) + beta * z(dm); strata with no variation in Follow are dropped (they carry no information on beta).
Development set S811 (partly seen), test set S809 (held out).  Verdict: S809 beta > 0 with the 95% scene-bootstrap lower bound > 0
(4,000 draws, seed 1015).  Reported with it: S811 beta; the held-out log-likelihood gain on S809 of beta fixed from S811 (intercepts refit)
over intercepts only; donor-only and recipient-only slopes; per-arm slopes.
Output: research/ces/outputs/agent_math_1015/d_sharper_carrier.json"""
import json
import numpy as np

R3 = "/home/storage2/ai26269345/vlm/results/r3"; O = "/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015"


def margin(v):
    v = np.asarray(v, float); v = v / v.sum(); s = np.sort(v)[::-1]
    return float(s[0] - s[1])


def cells_s809():
    R = json.load(open(f"{R3}/s809_s809_27b_circuit.json"))["rows"]
    arms = {"E<-S:B2": ("START", "END"), "E<-R:B2": ("R3", "END"), "S<-E:B2": ("END", "START"), "R<-E:B2": ("END", "R3"),
            "S<-E:B1B2": ("END", "START"), "E<-CNT8:B2": ("CNT8", "END"), "E<-CODE6:B2": ("CODE6", "END"),
            "E<-S:B2_DN": ("START", "END"), "E<-S:B2_FA": ("START", "END"), "E<-S:B2_MLP": ("START", "END"),
            "S<-E:B2_DN": ("END", "START"), "S<-E:B2_FA": ("END", "START"), "S<-E:B2_MLP": ("END", "START")}
    out = []
    for r in R:
        m = {run: margin(r["share"][run]["51"]["mh"]) for run in ("START", "END", "R3", "CNT8", "CODE6")}
        for a, (don, rec) in arms.items():
            if r["land"][don] == r["land"][rec]:
                continue
            out.append(dict(set="S809", si=f"S809:{r['si']}", arm=a, k=r["k"], md=m[don], mr=m[rec], follow=int(r["land"][a] == r["land"][don])))
    return out


def cells_s811():
    R = json.load(open(f"{R3}/s811_s811_27b_dissoc.json"))["rows"]
    arms = {"EP_Sj": ("START", "END"), "EP_Sk": ("START_k", "END"), "EP_E": ("END_kp", "END"), "EU_E": ("END_kp", "END"),
            "EU_Sk": ("START_k", "END"), "SU": ("START_k", "START"), "SP": ("START_k", "START")}
    out = []
    for r in R:
        m = {run: margin(r["read"][run]["51"]["blk_mh"]) for run in ("START", "END", "START_k", "END_kp")}
        for a, (don, rec) in arms.items():
            if r["land"][don] == r["land"][rec]:
                continue
            out.append(dict(set="S811", si=f"S811:{r['si']}", arm=a, k=r["k"], md=m[don], mr=m[rec], follow=int(r["land"][a] == r["land"][don])))
    return out


def design(cells, mu=None, sd=None, which="dm"):
    x = np.array([c["md"] - c["mr"] if which == "dm" else (c["md"] if which == "donor" else -c["mr"]) for c in cells])
    mu = x.mean() if mu is None else mu; sd = x.std() if sd is None else sd
    strata = sorted({(c["arm"], c["k"]) for c in cells})
    keep = [s for s in strata if 0 < np.mean([c["follow"] for c in cells if (c["arm"], c["k"]) == s]) < 1]
    idx = {s: i for i, s in enumerate(keep)}
    rows = [i for i, c in enumerate(cells) if (c["arm"], c["k"]) in idx]
    X = np.zeros((len(rows), len(keep) + 1)); y = np.zeros(len(rows))
    for n, i in enumerate(rows):
        c = cells[i]; X[n, idx[(c["arm"], c["k"])]] = 1; X[n, -1] = (x[i] - mu) / sd; y[n] = c["follow"]
    return X, y, rows, keep, mu, sd


def fit(X, y, offset=None, fix_beta=None, iters=60):
    off = np.zeros(len(y)) if offset is None else offset
    if fix_beta is not None:
        off = off + fix_beta * X[:, -1]; X = X[:, :-1]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ w + off; p = 1 / (1 + np.exp(-eta)); W = p * (1 - p) + 1e-9
        g = X.T @ (y - p); Hm = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g); w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ w + off; p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return w, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def beta_boot(cells, which="dm", B=4000, seed=1015):
    X, y, rows, keep, mu, sd = design(cells, which=which); w, ll = fit(X, y); b0 = w[-1]
    rng = np.random.default_rng(seed); sis = sorted({c["si"] for c in cells}); bs = []
    by = {s: [i for i, c in enumerate(cells) if c["si"] == s] for s in sis}
    for _ in range(B):
        pick = rng.choice(len(sis), len(sis)); cc = [cells[i] for s in pick for i in by[sis[s]]]
        Xb, yb, *_ = design(cc, mu, sd, which=which)
        if Xb.shape[0] < 10:
            continue
        bs.append(fit(Xb, yb)[0][-1])
    return float(b0), [float(x) for x in np.percentile(bs, [2.5, 97.5])], len(y), len(keep)


if __name__ == "__main__":
    dev, test = cells_s811(), cells_s809()
    out = {}
    for name, cells in (("S811_dev", dev), ("S809_test", test)):
        b, ci, n, ns = beta_boot(cells)
        bd, cid, *_ = beta_boot(cells, which="donor", B=1000); br, cir, *_ = beta_boot(cells, which="recipient", B=1000)
        out[name] = dict(n_cells=n, n_strata=ns, beta=b, ci=ci, beta_donor_only=bd, ci_donor=cid, beta_recipient_only=br, ci_recipient=cir,
                         follow_rate=float(np.mean([c["follow"] for c in cells])))
        print(f"{name}: cells {n} (strata {ns}) beta {b:+.3f} {np.round(ci, 3)} | donor-only {bd:+.3f} {np.round(cid, 3)} | recipient-only (sign: -m_rec) {br:+.3f} {np.round(cir, 3)}")
    # held-out gain: beta from dev (in dev's z units -> convert to raw dm units), applied to test in test's z units
    Xd, yd, _, _, mud, sdd = design(dev); wd, _ = fit(Xd, yd); beta_raw = wd[-1] / sdd
    Xt, yt, _, _, mut, sdt = design(test)
    _, ll_fixed = fit(Xt, yt, fix_beta=beta_raw * sdt)
    _, ll_null = fit(Xt, yt, fix_beta=0.0)
    out["heldout_gain_S809"] = dict(beta_raw_from_dev=float(beta_raw), ll_fixed=ll_fixed, ll_intercepts_only=ll_null, gain=ll_fixed - ll_null)
    print(f"held-out: beta from S811 (raw units) {beta_raw:+.3f}; S809 LL fixed-beta {ll_fixed:.2f} vs intercepts-only {ll_null:.2f}: gain {ll_fixed - ll_null:+.2f} nats")
    # per-arm descriptive slopes (raw dm), test and dev
    per = {}
    for name, cells in (("S811", dev), ("S809", test)):
        for a in sorted({c["arm"] for c in cells}):
            cc = [c for c in cells if c["arm"] == a]
            try:
                X, y, rows, keep, mu, sd = design(cc)
                if len(keep) == 0 or len(y) < 8:
                    continue
                w, _ = fit(X, y); per[f"{name}|{a}"] = dict(n=len(y), beta=float(w[-1]), follow=float(np.mean([c["follow"] for c in cc])))
            except Exception:
                pass
    out["per_arm"] = per
    for k_, v in per.items():
        print(f"   {k_:18s} n {v['n']:3d} follow {v['follow']:.2f} beta {v['beta']:+.2f}")
    json.dump(out, open(f"{O}/d_sharper_carrier.json", "w"), indent=1)
