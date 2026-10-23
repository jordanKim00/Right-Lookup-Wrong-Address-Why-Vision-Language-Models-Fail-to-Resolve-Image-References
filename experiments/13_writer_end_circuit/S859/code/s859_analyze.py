"""S859 analysis (rules fixed before the data; knowledge/DECISIONS.md S859 entry and its 10-20 amendment, both before any data).
Input: results/r3/s859_<tag>_anat.{json,npz}.
Notation: a pair = (scene, layout N_a, k, p).  E(k, p) = END(k) with the early lookup pinned at p; the primary reference is R(p) = START(p) pinned at
its own address (a pure pointer at p, same pass); the secondary reference is S(k, p) = START(k) pinned at p (frame-only contrast; START's own late
re-lookup returns to k from the pin, so S's k-bump enters E - S - a co-analysis round 2).  For one late layer L and anatomy row r (B1, B4, answer),
profiles over the N_a picture blocks are centred over the blocks.
  z_true  = head-mean (late_mean[L]) block scores of E minus those of the reference, as measured (pass 1)
  d_u     = u_z(E)[u] - <ref>_u_z(E)[u]: unit u's contribution difference through the direct path, both linearised at E's late input
  D       = sum over the band's units of d_u (the band's direct-path change);  Dup = the same over the upstream FA readers (35 / 39 / 43)
Registered verdicts use the primary reference R (band units only); the S-reference versions are reported as descriptive ('secondary_S').
I0 (instrument; all must hold): I0a two-pass logits bit-identical; I0b decomposition median chk_fa <= .01 and 95th pct <= .03, same for chk_dn;
  I0c Jacobian 95th pct chk_jac <= 1e-3; I0d START(q) natural answer q >= .95 in both layouts; I0e t1 >= .95; I0f picture keys identical across a
  layout's prompts; I0g pin reach (head-mean block-score argmax at L43 / L47 = p, rows B1, B4, answer) >= .95 over the pinned E, S, R runs.
M1 fidelity (first late layer; rows pooled): median over pairs of cos(D, z_true); 'direct path carries the END late change' if the median >= .60
  and the scene-bootstrap 95% lower bound >= .50; 'partly' if the median >= .40; else 'not by the direct path'.  Descriptive: cos(D + Dup, z_true).
M2 attribution (first late layer): unit share s_u = <d_u, D> / <D, D> over the band's units (per pair and row, then mean).  K50 = fewest units whose
  largest mean shares sum to >= .50: 'concentrated' if K50 <= 10, 'distributed' if >= 30, else 'intermediate' (K50 over the heads alone reported).
  'DN heads carry more than FA47 heads' if sum(DN) - sum(FA47) has a scene-bootstrap interval above 0; 'FA47 more' if below; else 'similar'.
M3 sources (first two late layers via the stored group split; rows pooled): group share g_x = <sum_u d_{u,x}, D'> / <D', D'>, D' = the group-split
  FA47 and DN differences; pooled IMG (pictures), FRAME, QT (count .. ' image from the', ordinal, '?', instruction), MENU, BND (B0-B4), THINK,
  PRE.  Registered: the largest pooled source; IMG >= .30; FRAME >= .20; text after the frame (QM + instruction + MENU + BND + THINK) >= .50;
  BND - IMG: 'BND > IMG' if the interval is above 0, 'IMG > BND' if below, else 'similar' (the frame word and the ordinal differ between E and R
  by construction, so the BND / IMG split is the informative one).
M4 shape (D at the first late layer, per pair and row; least squares on centred bases):
  R = ramp + anchor 1[b = p] + neighbours 1[|b - p| = 1];  P = target 1[b = j*] + anchor + neighbours;  S = anchor + predecessor 1[b = p - 1] +
  successor 1[b = p + 1];  SE = S + end block 1[b = N_a];  RE = R + end block.
  M4a 'ramp-anchor form' if mean R2(R) - R2(P) > 0 with the interval above 0, 'pointer form' if below, else 'undecided'; M4b 'long-range ramp' if
  R2(R) - R2(S) > 0 above 0, 'local re-addressing' if below, else 'undecided'; M4c 'one-back + end form' if R2(SE) - R2(RE) > 0 above 0, 'ramp +
  end form' if below, else 'undecided' (M4c without the p = N_a - 1 pairs, where the successor is the end block).
M5 ramp coefficient c_ramp (model R, rows pooled) regressed on (1, k, p, N_a, 1[p = N_a - 1]) over pairs of both layouts: 'k-only' if b_k's
  interval excludes 0 while |b_p| < |b_k| / 2 and |b_N| < |b_k| / 2; 'calibrated' if b_p and b_N have the needed signs (b_p with b_k, b_N
  against) with intervals excluding 0 and |b_N| >= |b_k| / 2; else 'mixed'.  M5b: the same on z_true (descriptive).
M5c / M5d gain of the one-back: the predecessor coefficient mu of SE (rows pooled) for D (M5c) and for z_true (M5d), regressed on (1, k, p, N_a,
  1[p = N_a - 1]); rho = -b_N / b_k reported with its interval; 'k-gated' if b_k's interval excludes 0 and |b_N| < |b_k| / 4; 'end-relative gain'
  if b_N's interval excludes 0, sign(b_N) = -sign(b_k) and |b_N| >= |b_k| / 4; else 'mixed'.
M6 shown count (pairs of (scene, k, p) in both layouts: k 2-4, p 3-4; first late layer; blocks 1-6): D_N = the band's sum of u_z(N8 E) - u_z(N6 E);
  z_N = measured N8 - N6 late scores of E; cos median 'band direct path carries the shown-count change' if >= .50, 'partly' if >= .30, else 'not'.
M7 identity: anchor coefficient of model R at identity pairs (p = j*) minus non-identity pairs: 'anchor left equally at identity' if the interval
  includes 0 or is negative; 'gated at identity' if positive and excluding 0.
M8 gate vs read (pairs; first late layer; rows pooled): for the FA47 heads and the DN heads, share of their summed change carried by the gate
  part, <sum gate_uz, T> / <T, T>, T = sum (gate_uz + read_uz): 'gate-dominated' if >= .60, 'read-dominated' if <= .40, else 'both'.
M9 who writes the predecessor bump (D, SE's predecessor coefficient, linear in the profile): shares of mu_D by MLPs, DN heads, FA47 heads and FA47's
  anchor-picture group img_p, and the gate part's share within FA47 img_p: 'anchor payload read by FA47' if FA47's img_p share >= .30, else 'not
  mainly FA47's anchor read'; 'END-gated' if the gate share >= .60, 'read-driven' if <= .40, else 'both'.  (Under the pin E and R read block p
  with the same address part, so the img_p difference is gate plus non-address attention by construction - M9b is the payload test.)
M9b payload (descriptive): for every FA head of the anatomy, the anchor, predecessor and successor coefficients (SE) of its own img_p read in the R
  runs (the static payload of the anchor picture at that head, through the direct path) and in the E runs; listed as 'predecessor heads' only
  if pred > 0 and pred - succ >= |anchor| / 3, and as 'anti-self heads' if the anchor coefficient is negative and more than twice either
  neighbour (amended 10-20 16:1x before the analysis was run - an anti-self read raises both neighbours and is not a pointer).
End-anchor pins (descriptive; p = N_a - 1): per k the share of E answers at p - 1, p and the target, and D's argmax.
Usage: python3 s859_analyze.py TAG"""
from __future__ import annotations

import json, sys
import numpy as np

TAG = sys.argv[1]
J = json.load(open(f"/vlm/results/r3/s859_{TAG}_anat.json"))
Z = np.load(f"/vlm/results/r3/s859_{TAG}_anat.npz", mmap_mode="r")
OUT = f"/vlm/results/r3/s859_analysis_{TAG}.json"
rng = np.random.default_rng(859)
NB = 4000
LATE = J["late"]; L0 = LATE[0]; iL0 = 0
UNITS = [tuple(u) for u in J["units"]]
GR = J["groups"]; gi_ = {g: i for i, g in enumerate(GR)}
rows = {r["si"]: r for r in J["rows"]}
scenes = sorted(rows)
BAND_L = set(J["band"])
FA_AN = J.get("fa_anat", J["fa_w"])
iFA47 = [FA_AN.index(l) for l in J["fa_w"]]
band_u = [i for i, u in enumerate(UNITS) if u[1] in BAND_L]
up_u = [i for i, u in enumerate(UNITS) if u[1] not in BAND_L]
pos_b = {u: j for j, u in enumerate(band_u)}
nm_fa = [i for i, u in enumerate(UNITS) if u[0] == "fa" and u[1] in BAND_L]
nm_dn = {l: [i for i, u in enumerate(UNITS) if u[0] == "dn" and u[1] == l] for l in J["dn_w"]}
dn_all = sum(nm_dn.values(), [])
nm_mlp = [i for i, u in enumerate(UNITS) if u[0] == "mlp"]
HU = [i for i, u in enumerate(UNITS) if u[0] in ("fa", "dn")]                               # order of the gate / read arrays
res = dict(tag=TAG, n_scenes=len(scenes), units=J["units"])


def get(si, nm, key):
    return np.asarray(Z[f"s{si}_{nm}_{key}"], dtype=np.float64)


def cen(x):
    return x - x.mean(-1, keepdims=True)


def boot(vals_by_scene, f=np.mean):
    keys = [k for k in vals_by_scene if len(vals_by_scene[k])]
    if not keys:
        return dict(est=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    allv = np.concatenate([np.asarray(vals_by_scene[k], dtype=float) for k in keys])
    est = float(f(allv)); bs = []
    for _ in range(NB):
        pick = rng.choice(len(keys), len(keys))
        bs.append(f(np.concatenate([np.asarray(vals_by_scene[keys[i]], dtype=float) for i in pick])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(est=est, lo=float(lo), hi=float(hi), n=int(len(allv)))


def zmean(si, nm, L):
    zl = np.asarray(rows[si]["zl"][nm][str(L)], dtype=np.float64)                         # [R, H, nb]
    return zl[:, J["late_mean"][str(L)]].mean(1)


def part_of(n, k):
    return next(i for i, ks in enumerate(J["layouts"][str(n)]["split"]) if k in ks)


def basis(nb, p, js, kind):
    b = np.arange(1, nb + 1, dtype=float)
    ramp = b - b.mean(); anc = (b == p).astype(float); nei = (np.abs(b - p) == 1).astype(float); tgt = (b == js).astype(float)
    pred = (b == p - 1).astype(float); succ = (b == p + 1).astype(float); endb = (b == nb).astype(float)
    cols = dict(R=[ramp, anc, nei], P=[tgt, anc, nei], S=[anc, pred, succ], SE=[anc, pred, succ, endb], RE=[ramp, anc, nei, endb],
                RS=[ramp, anc, pred, succ], ramp=[ramp], an=[anc, nei])[kind]
    return np.stack([c - c.mean() for c in cols], 1)


def fit(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - X @ coef) ** 2).sum() / tot if tot > 0 else np.nan
    return coef, r2


PAIRS = [(si, int(n_), k, p) for n_ in ("8", "6") for si in scenes for k in J["layouts"][n_]["ks"] for p in J["layouts"][n_]["ps"]]
ename = lambda n, k, p: f"N{n}_E_k{k}p{p}"
refname = {"ref": lambda n, k, p: f"N{n}_R{part_of(n, k)}_p{p}", "pair": lambda n, k, p: f"N{n}_S_k{k}p{p}"}
gate_key = {"ref": "ref_", "pair": ""}                                                      # prefixes of the gate / read arrays

# ---------------------------------------------------------------- I0
pinned = [(si, ename(n, k, p)) for si, n, k, p in PAIRS] + [(si, f"N{n}_S_k{k}p{p}") for si, n, k, p in PAIRS] + \
         sorted({(si, refname["ref"](n, k, p)) for si, n, k, p in PAIRS})
chk = {key: np.concatenate([get(si, nm, key).ravel() for si, nm in pinned]) for key in ("chk_fa", "chk_dn", "chk_jac", "chk_lin")}
st_ok = []
for n_ in ("8", "6"):
    for q in J["layouts"][n_]["starts"]:
        st_ok += [rows[si]["land"][f"N{n_}_START_q{q}"] == q for si in scenes]
reach = []
for si, nm in pinned:
    p = int(nm.split("p")[-1])
    for L in (J["early"][-2], J["early"][-1]):
        qk = np.asarray(rows[si]["qk"][nm][str(L)])
        reach += [int(np.argmax(qk[i])) + 1 == p for i in J["krows"]]
I0 = dict(I0a=J["two_pass_logit_diff_max"] == 0.0,
          I0b=bool(np.median(chk["chk_fa"]) <= .01 and np.percentile(chk["chk_fa"], 95) <= .03 and np.median(chk["chk_dn"]) <= .01
                   and np.percentile(chk["chk_dn"], 95) <= .03),
          I0c=bool(np.percentile(chk["chk_jac"], 95) <= 1e-3), I0d=bool(np.mean(st_ok) >= .95), I0e=bool((J["t1_menu"] or 0) >= .95),
          I0f=J["key_rel_diff_max"] == 0.0, I0g=bool(np.mean(reach) >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), two_pass=J["two_pass_logit_diff_max"], chk_fa_med=float(np.median(chk["chk_fa"])),
                 chk_fa_p95=float(np.percentile(chk["chk_fa"], 95)), chk_dn_med=float(np.median(chk["chk_dn"])),
                 chk_dn_p95=float(np.percentile(chk["chk_dn"], 95)), chk_jac_p95=float(np.percentile(chk["chk_jac"], 95)),
                 chk_lin_med=float(np.median(chk["chk_lin"])), start_acc=float(np.mean(st_ok)), t1=J["t1_menu"], key_rel_diff=J["key_rel_diff_max"],
                 pin_reach=float(np.mean(reach)))


# ---------------------------------------------------------------- per-pair quantities for one reference
def build(ref):
    per = []
    for si, n, k, p in PAIRS:
        e = ename(n, k, p); rn = refname[ref](n, k, p)
        uE = get(si, e, "u_z"); uP = get(si, e, ref + "_u_z")
        d = cen(uE - uP)
        D = d[band_u].sum(0); Dup = d[up_u].sum(0) if up_u else np.zeros_like(D)
        zt = cen(np.stack([zmean(si, e, L) - zmean(si, rn, L) for L in LATE]))
        per.append(dict(si=si, n=n, k=k, p=p, d=d, D=D, Dup=Dup, zt=zt, js=n + 1 - k, e=e, rn=rn))
    return per


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def m1(per):
    c0 = {si: [] for si in scenes}; cu = {si: [] for si in scenes}; byL = {L: {si: [] for si in scenes} for L in LATE}
    for q in per:
        for il, L in enumerate(LATE):
            for r in range(q["D"].shape[1]):
                c = cosv(q["D"][il, r], q["zt"][il, r]); byL[L][q["si"]].append(c)
                if il == 0:
                    c0[q["si"]].append(c); cu[q["si"]].append(cosv(q["D"][il, r] + q["Dup"][il, r], q["zt"][il, r]))
    m = boot(c0, np.median)
    return dict(cos_median=m, verdict=("direct path carries the END late change" if m["est"] >= .60 and m["lo"] >= .50 else
                                       "partly" if m["est"] >= .40 else "not by the direct path"),
                with_upstream=boot(cu, np.median), by_layer={str(L): boot(byL[L], np.median) for L in LATE})


def m2(per):
    S_all = []
    for q in per:
        D = q["D"][iL0]; den = (D * D).sum() + 1e-12
        S_all.append(np.einsum("urb,rb->u", q["d"][band_u, iL0], D) / den)
    S_all = np.stack(S_all); s_mean = S_all.mean(0)
    ob = np.argsort(-s_mean); K50 = int(np.searchsorted(np.cumsum(s_mean[ob]), .5) + 1)
    heads_b = [j for j, u in enumerate(band_u) if UNITS[u][0] != "mlp"]
    oh = sorted(heads_b, key=lambda j: -s_mean[j]); K50h = int(np.searchsorted(np.cumsum(s_mean[oh]), .5) + 1)
    def comp(idx):
        by = {si: [] for si in scenes}; jj = [pos_b[i] for i in idx]
        for q, sv in zip(per, S_all):
            by[q["si"]].append(float(sv[jj].sum()))
        return boot(by)
    diff = {si: [] for si in scenes}
    for q, sv in zip(per, S_all):
        diff[q["si"]].append(float(sv[[pos_b[i] for i in dn_all]].sum() - sv[[pos_b[i] for i in nm_fa]].sum()))
    dd = boot(diff)
    order = [band_u[j] for j in ob]
    return dict(K50=K50, K50_heads=K50h, verdict_conc=("concentrated" if K50 <= 10 else "distributed" if K50 >= 30 else "intermediate"),
                components=dict(FA47=comp(nm_fa), MLP=comp(nm_mlp), **{f"DN{l}": comp(ix) for l, ix in nm_dn.items()}), dn_minus_fa=dd,
                verdict_dnfa=("DN heads carry more than FA47 heads" if dd["lo"] > 0 else "FA47 more" if dd["hi"] < 0 else "similar"),
                top=[dict(unit=list(UNITS[i]), share=float(s_mean[pos_b[i]]),
                          by_k={f"N{n}_k{k}": float(np.mean([sv[pos_b[i]] for q, sv in zip(per, S_all) if q["k"] == k and q["n"] == n]))
                                for n in (8, 6) for k in J["layouts"][str(n)]["ks"]}) for i in order[:25]]), order


POOL = dict(IMG=[f"img{b + 1}" for b in range(8)], FRAME=["frame"], QT=["count", "qhead", "ord", "qmid", "qm", "instr"], MENU=["menu"],
            BND=["bnd"], THINK=["think"], PRE=["pre"])


def m3(per, ref, order):
    gsh = {x: {si: [] for si in scenes} for x in list(GR) + list(POOL) + ["AFTER_FRAME", "BND_minus_IMG"]}
    for q in per:
        fa = cen(get(q["si"], q["e"], "fa_gz")[iFA47] - get(q["si"], q["e"], ref + "_fa_gz")[iFA47])
        dn = cen(get(q["si"], q["e"], "dn_gz") - get(q["si"], q["e"], ref + "_dn_gz"))
        G_ = fa.sum((0, 1)) + dn.sum((0, 1)); Dp = G_.sum(0); den = (Dp * Dp).sum() + 1e-12
        sh = np.einsum("glrb,lrb->g", G_, Dp) / den
        for g in GR:
            gsh[g][q["si"]].append(float(sh[gi_[g]]))
        for x, gs in POOL.items():
            gsh[x][q["si"]].append(float(sum(sh[gi_[g]] for g in gs)))
        gsh["AFTER_FRAME"][q["si"]].append(float(sum(sh[gi_[g]] for g in ["qm", "instr", "menu", "bnd", "think"])))
        gsh["BND_minus_IMG"][q["si"]].append(float(sh[gi_["bnd"]] - sum(sh[gi_[g]] for g in POOL["IMG"])))
    m = {x: boot(v) for x, v in gsh.items()}
    big = max(POOL, key=lambda x: m[x]["est"]); bi = m["BND_minus_IMG"]
    top = [i for i in order[:12] if UNITS[i][0] != "mlp"]
    accT = {i: np.zeros(len(GR)) for i in top}
    for q in per:                                                                             # each run's arrays read once per pair (speed only)
        FAd = get(q["si"], q["e"], "fa_gz") - get(q["si"], q["e"], ref + "_fa_gz")
        DNd = get(q["si"], q["e"], "dn_gz") - get(q["si"], q["e"], ref + "_dn_gz")
        for i in top:
            kind, l, h = UNITS[i]
            x = cen(FAd[FA_AN.index(l), h]) if kind == "fa" else cen(DNd[J["dn_w"].index(l), h])
            tot = x.sum(0); accT[i] += np.einsum("glrb,lrb->g", x, tot) / ((tot * tot).sum() + 1e-12)
    top_src = []
    for i in top:
        acc = accT[i] / len(per); kind, l, h = UNITS[i]
        top_src.append(dict(unit=[kind, l, h], groups={GR[j]: float(acc[j]) for j in np.argsort(-np.abs(acc))[:6]}))
    return dict(shares=m, largest_pooled=big, IMG_ge_30=m["IMG"]["est"] >= .30, FRAME_ge_20=m["FRAME"]["est"] >= .20,
                AFTER_FRAME_ge_50=m["AFTER_FRAME"]["est"] >= .50,
                verdict_bnd_img=("BND > IMG" if bi["lo"] > 0 else "IMG > BND" if bi["hi"] < 0 else "similar"), top_unit_sources=top_src)


MODELS = ("R", "P", "S", "SE", "RE", "RS", "ramp", "an")


def shape_fits(per, key="D"):
    r2 = {m: {si: [] for si in scenes} for m in MODELS}
    dR = {si: [] for si in scenes}; dS = {si: [] for si in scenes}; dSE = {si: [] for si in scenes}
    cR, cSE, amax = [], [], []
    for q in per:
        nb = q[key].shape[-1]; ccR, ccS = [], []
        for r in range(q[key].shape[1]):
            y = q[key][iL0, r]; vals = {}
            for m in MODELS:
                coef, rr = fit(y, basis(nb, q["p"], q["js"], m)); vals[m] = rr; r2[m][q["si"]].append(rr)
                if m == "R":
                    ccR.append(coef)
                if m == "SE":
                    ccS.append(coef)
            dR[q["si"]].append(vals["R"] - vals["P"]); dS[q["si"]].append(vals["R"] - vals["S"])
            if q["p"] != q["n"] - 1:
                dSE[q["si"]].append(vals["SE"] - vals["RE"])
            amax.append((q["k"], q["p"], q["n"], int(np.argmax(y)) + 1))
        cR.append(np.mean(ccR, 0)); cSE.append(np.mean(ccS, 0))
    return r2, dR, dS, dSE, np.array(cR), np.array(cSE), amax


def design(per):
    return np.array([[1, q["k"], q["p"], q["n"], float(q["p"] == q["n"] - 1)] for q in per], dtype=float), np.array([q["si"] for q in per])


def breg(A, y, sc):
    b0, *_ = np.linalg.lstsq(A, y, rcond=None); bs = []
    for _ in range(NB):
        pick = rng.choice(scenes, len(scenes)); idx = np.concatenate([np.where(sc == s)[0] for s in pick])
        bs.append(np.linalg.lstsq(A[idx], y[idx], rcond=None)[0])
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return b0, lo, hi, bs


def ramp_verdict(b0, lo, hi):
    bk, bp, bN = b0[1], b0[2], b0[3]; ex = lambda j: lo[j] > 0 or hi[j] < 0
    if ex(1) and abs(bp) < abs(bk) / 2 and abs(bN) < abs(bk) / 2:
        return "k-only"
    if ex(2) and ex(3) and np.sign(bp) == np.sign(bk) and np.sign(bN) == -np.sign(bk) and abs(bN) >= abs(bk) / 2:
        return "calibrated"
    return "mixed"


def gain_verdict(b0, lo, hi, bs):
    bk, bN = b0[1], b0[3]; ex = lambda j: lo[j] > 0 or hi[j] < 0
    rho = -bs[:, 3] / np.where(np.abs(bs[:, 1]) > 1e-12, bs[:, 1], np.nan)
    rr = [float(-bN / bk) if abs(bk) > 1e-12 else float("nan"), *[float(x) for x in np.nanpercentile(rho, [2.5, 97.5])]]
    if ex(1) and abs(bN) < abs(bk) / 4:
        v = "k-gated"
    elif ex(3) and np.sign(bN) == -np.sign(bk) and abs(bN) >= abs(bk) / 4:
        v = "end-relative gain"
    else:
        v = "mixed"
    return v, rr


def coefd(b0, lo, hi):
    return {nm: [float(b0[j]), float(lo[j]), float(hi[j])] for j, nm in enumerate(("const", "k", "p", "N", "endpin"))}


def cellmean(vals, per, j):
    out = {}
    for c, q in zip(vals, per):
        out.setdefault(f"N{q['n']}_k{q['k']}_p{q['p']}", []).append(float(c[j]))
    return {k_: float(np.mean(v)) for k_, v in out.items()}


def m4_m5_m7(per):
    r2, dR, dS, dSE, cR, cSE, amax = shape_fits(per, "D")
    m4a, m4b, m4c = boot(dR), boot(dS), boot(dSE)
    rel = {}
    for k_, p_, n_, am in amax:
        rel.setdefault(f"N{n_}_k{k_}_p{p_}", []).append(am - p_)
    out = dict(M4=dict(R2={m: boot(v) for m, v in r2.items()}, R_minus_P=m4a, R_minus_S=m4b, SE_minus_RE=m4c,
                       verdict=("ramp-anchor form" if m4a["lo"] > 0 else "pointer form" if m4a["hi"] < 0 else "undecided"),
                       verdict_b=("long-range ramp" if m4b["lo"] > 0 else "local re-addressing" if m4b["hi"] < 0 else "undecided"),
                       verdict_c=("one-back + end form" if m4c["lo"] > 0 else "ramp + end form" if m4c["hi"] < 0 else "undecided"),
                       pred_coef_by_cell=cellmean(cSE, per, 1), anchor_coef_by_cell=cellmean(cSE, per, 0),
                       argmax_minus_p={key: {str(d): float(np.mean(np.array(v) == d)) for d in sorted(set(v))} for key, v in rel.items()}))
    A, sc = design(per)
    b0, lo, hi, _ = breg(A, cR[:, 0], sc)
    out["M5"] = dict(coef=coefd(b0, lo, hi), verdict=ramp_verdict(b0, lo, hi))
    r2t, _, _, _, cRt, cSEt, _ = shape_fits(per, "zt")
    b0t, lot, hit, _ = breg(A, cRt[:, 0], sc)
    out["M5b"] = dict(coef=coefd(b0t, lot, hit), R2_true={m: boot(v) for m, v in r2t.items()})
    b0c, loc, hic, bsc = breg(A, cSE[:, 1], sc); vc, rc = gain_verdict(b0c, loc, hic, bsc)
    out["M5c"] = dict(coef=coefd(b0c, loc, hic), rho=rc, verdict=vc)
    b0d, lod, hid, bsd = breg(A, cSEt[:, 1], sc); vd, rd = gain_verdict(b0d, lod, hid, bsd)
    out["M5d"] = dict(coef=coefd(b0d, lod, hid), rho=rd, verdict=vd, mu_true_by_cell=cellmean(cSEt, per, 1))
    a_id = {si: [] for si in scenes}; a_non = {si: [] for si in scenes}
    for q, c in zip(per, cR):
        (a_id if q["p"] == q["js"] else a_non)[q["si"]].append(float(c[1]))
    dif = {si: ([np.mean(a_id[si]) - np.mean(a_non[si])] if a_id[si] and a_non[si] else []) for si in scenes}
    m7 = boot(dif)
    out["M7"] = dict(anchor_identity=boot(a_id), anchor_nonidentity=boot(a_non), diff=m7,
                     verdict=("gated at identity" if m7["lo"] > 0 else "anchor left equally at identity"))
    return out, cSE


def m8(per, ref):
    fa_h = [j for j, i in enumerate(HU) if UNITS[i][0] == "fa" and UNITS[i][1] in BAND_L]
    dn_h = [j for j, i in enumerate(HU) if UNITS[i][0] == "dn"]
    gsh = {"FA47": {si: [] for si in scenes}, "DN": {si: [] for si in scenes}}
    for q in per:
        gt = cen(get(q["si"], q["e"], gate_key[ref] + "gate_uz")[:, iL0]); rd_ = cen(get(q["si"], q["e"], gate_key[ref] + "read_uz")[:, iL0])
        for nm_, ix in (("FA47", fa_h), ("DN", dn_h)):
            G1 = gt[ix].sum(0); T_ = G1 + rd_[ix].sum(0)
            gsh[nm_][q["si"]].append(float((G1 * T_).sum() / ((T_ * T_).sum() + 1e-12)))
    out = {}
    for nm_, v in gsh.items():
        b = boot(v)
        out[nm_] = dict(gate_share=b, verdict=("gate-dominated" if b["est"] >= .60 else "read-dominated" if b["est"] <= .40 else "both"))
    return out


def m9(per, ref):
    def pc(y, q):
        return fit(y, basis(y.shape[-1], q["p"], q["js"], "SE"))[0][1]
    m = {x: {si: [] for si in scenes} for x in ("MLP", "DN", "FA47", "FA47_img_p", "FA47_img_p_gate")}
    for q in per:
        R = q["D"].shape[1]
        tot = np.mean([pc(q["D"][iL0, r], q) for r in range(R)])
        if abs(tot) < 1e-9:
            continue
        part = lambda ix: np.mean([pc(q["d"][ix, iL0, r].sum(0), q) for r in range(R)])
        m["MLP"][q["si"]].append(part(nm_mlp) / tot); m["DN"][q["si"]].append(part(dn_all) / tot); m["FA47"][q["si"]].append(part(nm_fa) / tot)
        fa = cen(get(q["si"], q["e"], "fa_gz")[iFA47] - get(q["si"], q["e"], ref + "_fa_gz")[iFA47])[:, :, gi_[f"img{q['p']}"], 0]
        fimg = np.mean([pc(fa[..., r, :].sum((0, 1)), q) for r in range(R)])
        m["FA47_img_p"][q["si"]].append(fimg / tot)
        gfa = cen(get(q["si"], q["e"], gate_key[ref] + "gate_fa_gz")[iFA47])[:, :, gi_[f"img{q['p']}"], 0]
        gimg = np.mean([pc(gfa[..., r, :].sum((0, 1)), q) for r in range(R)])
        if abs(fimg) > 1e-9:
            m["FA47_img_p_gate"][q["si"]].append(gimg / fimg)
    b = {x: boot(v, np.median) for x, v in m.items()}
    return dict(shares_median=b, verdict_payload=("anchor payload read by FA47" if b["FA47_img_p"]["est"] >= .30 else "not mainly FA47's anchor read"),
                verdict_gate=("END-gated" if b["FA47_img_p_gate"]["est"] >= .60 else "read-driven" if b["FA47_img_p_gate"]["est"] <= .40 else "both"))


# ---------------------------------------------------------------- primary (R) and secondary (S)
perR = build("ref")
res["M1"] = m1(perR)
res["M2"], orderR = m2(perR)
res["M3"] = m3(perR, "ref", orderR)
o45, cSE_R = m4_m5_m7(perR); res.update(o45)
res["M8"] = m8(perR, "ref")
res["M9"] = m9(perR, "ref")
perS = build("pair")
m2S, orderS = m2(perS)
o45S, _ = m4_m5_m7(perS)
res["secondary_S"] = dict(M1=m1(perS), M2=dict(K50=m2S["K50"], verdict_conc=m2S["verdict_conc"], verdict_dnfa=m2S["verdict_dnfa"], top=m2S["top"][:12]),
                          M4={k_: o45S["M4"][k_] for k_ in ("verdict", "verdict_b", "verdict_c", "R_minus_P", "R_minus_S", "SE_minus_RE")},
                          M5=o45S["M5"], M5c=o45S["M5c"], M5d=o45S["M5d"], M7=o45S["M7"], M8=m8(perS, "pair"), M9=m9(perS, "pair"))

# ---------------------------------------------------------------- M9b: the static payload of the anchor picture, per FA head (descriptive)
pay = {}
accs = {(li_, h): {x: [] for x in ("R_anchor", "R_pred", "R_succ", "E_anchor", "E_pred", "E_succ")} for li_ in range(len(FA_AN)) for h in range(24)}
for q in perR:                                                                               # each run's fa_gz read once per pair (speed only)
    for nm, tg in ((q["rn"], "R"), (q["e"], "E")):
        X = cen(get(q["si"], nm, "fa_gz")[:, :, gi_[f"img{q['p']}"], 0])                        # [nFA, H, R, nb] own reads of the anchor picture
        for li_ in range(len(FA_AN)):
            for h in range(24):
                x = X[li_, h]
                cs = [fit(x[r], basis(x.shape[-1], q["p"], q["js"], "SE"))[0] for r in range(x.shape[0])]
                acc = accs[(li_, h)]
                if q["p"] < x.shape[-1] - 1:                                                     # successor defined apart from the end block
                    acc[tg + "_succ"].append(np.mean([c[2] for c in cs]))
                acc[tg + "_anchor"].append(np.mean([c[0] for c in cs])); acc[tg + "_pred"].append(np.mean([c[1] for c in cs]))
for (li_, h), acc in accs.items():
    pay[f"{FA_AN[li_]}_{h}"] = {k_: float(np.mean(v)) for k_, v in acc.items()}
# 10-20 16:1x amendment (before the analysis is run; a co-analysis round 3): an anti-self read (s(p) strongly negative, both neighbours up
# together) is not a predecessor pointer - a head counts as 'predecessor' only if its predecessor coefficient exceeds its successor coefficient
# by at least a third of |anchor| and is positive; 'anti-self' if the anchor coefficient is negative and larger in size than twice either neighbour.
pred_heads = [k_ for k_, v in pay.items() if v["R_pred"] > 0 and v["R_pred"] - v["R_succ"] >= abs(v["R_anchor"]) / 3]
anti_heads = [k_ for k_, v in pay.items() if v["R_anchor"] < 0 and abs(v["R_anchor"]) > 2 * max(v["R_pred"], v["R_succ"], 0)]
res["M9b"] = dict(per_head=pay, predecessor_heads=sorted(pred_heads, key=lambda k_: -(pay[k_]["R_pred"] - pay[k_]["R_succ"]))[:20],
                  anti_self_heads=sorted(anti_heads, key=lambda k_: pay[k_]["R_anchor"])[:20])

# ---------------------------------------------------------------- M6 (reference-free: E across layouts)
cN = {si: [] for si in scenes}; cU = {si: [] for si in scenes}; faN = {si: [] for si in scenes}; upN = {si: [] for si in scenes}
for si in scenes:
    for k in [2, 3, 4]:
        for p in [3, 4]:
            e8, e6 = ename(8, k, p), ename(6, k, p)
            u8 = get(si, e8, "u_z")[:, iL0][..., :6]; u6 = get(si, e6, "u_z")[:, iL0][..., :6]
            DN_ = cen(u8 - u6)[band_u].sum(0); DU_ = cen(u8 - u6)[up_u].sum(0) if up_u else np.zeros_like(DN_)
            zN = cen(zmean(si, e8, L0)[:, :6] - zmean(si, e6, L0))
            for r in range(DN_.shape[0]):
                cN[si].append(cosv(DN_[r], zN[r])); cU[si].append(cosv(DU_[r], zN[r]))
            den = (zN * zN).sum() + 1e-12
            g8 = get(si, e8, "fa_gz")[iFA47][:, :, [gi_["img7"], gi_["img8"]], 0][..., :6]
            faN[si].append(float((cen(g8).sum((0, 1, 2)) * zN).sum() / den))
            if J.get("fa_up"):
                gu = get(si, e8, "fa_gz")[[FA_AN.index(l) for l in J["fa_up"]]][:, :, [gi_["img7"], gi_["img8"]], 0][..., :6]
                upN[si].append(float((cen(gu).sum((0, 1, 2)) * zN).sum() / den))
m6 = boot(cN, np.median)
res["M6"] = dict(cos_median=m6, verdict=("band direct path carries the shown-count change" if m6["est"] >= .50 else "partly" if m6["est"] >= .30 else "not"),
                 fa47_img78_projection=boot(faN), upstream_cos_median=boot(cU, np.median), upstream_img78_projection=boot(upN))
# ---------------------------------------------------------------- descriptive
upd = {}
for l in J.get("fa_up", []):
    ix = [i for i, u in enumerate(UNITS) if u[0] == "fa" and u[1] == l]
    cs = {si: [] for si in scenes}
    for q in perR:
        x = q["d"][ix, iL0].sum(0)
        for r in range(x.shape[0]):
            cs[q["si"]].append(cosv(x[r], q["zt"][iL0, r]))
    upd[str(l)] = dict(cos_with_measured=boot(cs, np.median),
                       norm_ratio_to_band=float(np.median([np.linalg.norm(q["d"][ix, iL0].sum(0)) / (np.linalg.norm(q["D"][iL0]) + 1e-12) for q in perR])))
res["upstream_fa"] = upd
endpin = {}
for n in (8, 6):
    pe = n - 1
    if pe not in J["layouts"][str(n)]["ps"]:
        continue
    for k in J["layouts"][str(n)]["ks"]:
        la = np.array([rows[si]["land"][ename(n, k, pe)] for si in scenes]); js = n + 1 - k
        endpin[f"N{n}_k{k}"] = dict(p_minus_1=float(np.mean(la == pe - 1)), p=float(np.mean(la == pe)), target=float(np.mean(la == js)), target_is=js,
                                     D_argmax=[int(np.argmax(q["D"][iL0].mean(0))) + 1 for q in perR if q["n"] == n and q["k"] == k and q["p"] == pe])
res["end_anchor_pins"] = endpin
prof = {}
for n in (8, 6):
    for k in J["layouts"][str(n)]["ks"]:
        for p in J["layouts"][str(n)]["ps"]:
            qq = [q for q in perR if q["n"] == n and q["k"] == k and q["p"] == p]
            prof[f"N{n}_k{k}p{p}"] = dict(D=np.mean([q["D"][iL0] for q in qq], 0).mean(0).round(4).tolist(),
                                          z_true=np.mean([q["zt"][iL0] for q in qq], 0).mean(0).round(4).tolist())
res["profiles_L0"] = prof
nat = {}
for n in (8, 6):
    lay = J["layouts"][str(n)]
    for k in lay["ends"]:
        if k not in lay["starts"]:
            continue
        ds = [cen(get(si, f"N{n}_END_k{k}", "u_z")[band_u, iL0] - get(si, f"N{n}_START_q{k}", "u_z")[band_u, iL0]) for si in scenes]
        dm = np.mean(ds, 0); Dm = dm.sum(0); sh = np.einsum("urb,rb->u", dm, Dm) / ((Dm * Dm).sum() + 1e-12)
        nat[f"N{n}_k{k}"] = dict(D=Dm.mean(0).round(4).tolist(), top=[[list(UNITS[band_u[i]]), float(sh[i])] for i in np.argsort(-sh)[:8]],
                                 land=[rows[si]["land"][f"N{n}_END_k{k}"] for si in scenes])
res["natural"] = nat
links = []
for i in orderR[:10]:
    acc = np.zeros(24)
    for q in perR:
        x = get(q["si"], q["e"], "u_hz")[i, 0] - get(q["si"], q["e"], "ref_u_hz")[i, 0]
        acc += np.linalg.norm(cen(x), axis=-1).sum(-1)
    acc /= len(perR)
    ms = J["matchers"].get(str(J["perhead"][0]), list(range(24)))
    links.append(dict(unit=list(UNITS[i]), late_heads=[[int(h), float(acc[h])] for h in np.argsort(-acc)[:6]], matcher_share=float(acc[ms].sum() / acc.sum())))
res["links"] = links
ans = {}
for n in (8, 6):
    for k in J["layouts"][str(n)]["ks"]:
        for p in J["layouts"][str(n)]["ps"]:
            ans[f"N{n}_k{k}p{p}"] = dict(E=np.bincount([rows[si]["land"][ename(n, k, p)] for si in scenes], minlength=n + 1).tolist(),
                                         S=np.bincount([rows[si]["land"][f"N{n}_S_k{k}p{p}"] for si in scenes], minlength=n + 1).tolist(),
                                         R=np.bincount([rows[si]["land"][refname["ref"](n, k, p)] for si in scenes], minlength=n + 1).tolist())
res["answers"] = ans
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], M1=res["M1"]["verdict"], M2=[res["M2"]["K50"], res["M2"]["verdict_conc"], res["M2"]["verdict_dnfa"]],
                      M3=[res["M3"]["largest_pooled"], res["M3"]["verdict_bnd_img"]], M4=[res["M4"]["verdict"], res["M4"]["verdict_b"], res["M4"]["verdict_c"]],
                      M5=res["M5"]["verdict"], M5c=res["M5c"]["verdict"], M5d=res["M5d"]["verdict"], M6=res["M6"]["verdict"], M7=res["M7"]["verdict"],
                      M8={k_: v_["verdict"] for k_, v_ in res["M8"].items()}, M9=[res["M9"]["verdict_payload"], res["M9"]["verdict_gate"]]), indent=1))
print("->", OUT)
