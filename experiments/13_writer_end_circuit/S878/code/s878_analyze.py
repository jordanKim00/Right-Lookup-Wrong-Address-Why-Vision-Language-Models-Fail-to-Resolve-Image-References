"""S878 analysis (rules fixed before the data; knowledge/DECISIONS.md S878 entry).  Inputs results/r3/s878_<tag>_qk.json and the per-scene npz
(post-RoPE queries of every head at rows B1 / B4 / answer for E, R, E_BANDR; post-RoPE keys at every image position, once per scene), and
S873's transplant JSON for the reproducibility check.
Address score of a query set q (rows x heads) against a scene's image keys K at late layer L, for pin p:
  T1(q, K) = mean over the three rows and the late_mean heads of [LSE_{t in block p-1}(q_h . k_{g(h), t} / 16) - LSE_{t in block p}(...)],
  g(h) = the key-value head of query head h (GQA groups of 6) - exactly the readout's block score difference (token level).
Writer address effect for query source A and key source B: W(A, B) = T1(q_A^E, K_B) - T1(q_A^BANDR, K_B); natural W(A, A).
Q1 (primary, L51): over ordered pairs A != B of the same pin, r_q = Pearson(W(A, B), W(A, A)) (follows the query source) and r_k =
  Pearson(W(A, B), W(B, B)) (follows the key source), within-pin centred; scene bootstrap 4,000 (seed 878; pairs among the resampled scenes).
  r_k - r_q >= .20 with the interval above 0 -> 'H1: the image-key geometry sets the per-scene address effect of the writer';
  <= -.20 with the interval below 0 -> 'H2: the writer's query update itself is scene-specific'; else 'both'.
Descriptive: the same at L55 / L59 / L63; per pin the two-way decomposition of the 48 x 48 matrix W(A, B) (query-source main effect,
key-source main effect, interaction: variance shares); the first-order block-mean form W_lin(A, B) = mean_rows,heads (q_A^E - q_A^BANDR) .
(kbar_B(p-1) - kbar_B(p)) / 16 against the exact W (Pearson, R^2); the mean natural effect.
I0: the offline T1(q_A^E, K_A) reproduces the run's stored readout (median |diff| <= .01 at L51); the image keys are the same in every run
of a scene (largest recorded difference <= .05, float16 storage); the natural W(A, A) at L51 reproduces S873's natural address effect of the
same scenes and pins (Pearson >= .90); t1 >= .95; START(p) answers p >= .95.  Nothing is read if I0 fails.
Usage: python3 s878_analyze.py TAG [--s873 s873_27b]"""
from __future__ import annotations

import argparse, json
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--s873", default="s873_27b"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s878_{A.tag}_qk.json")); NPD = J["npz_dir"]
OUT = f"{R3}/s878_analysis_{A.tag}.json"
rng = np.random.default_rng(878); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; LAYS = [str(l) for l in J["lateq"]]
GRP = 6


def lse(x, axis=-1):
    m = x.max(axis, keepdims=True); return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def T1(q, K, own, p, heads):
    qs = q[:, heads].astype(np.float64)                                            # [3, H, 256]
    Kh = K[:, [h // GRP for h in heads]].astype(np.float64)                          # [T, H, 256]
    lg = np.einsum("rhd,thd->rht", qs, Kh) / 16.0
    return float((lse(lg[:, :, own == p - 2]) - lse(lg[:, :, own == p - 1])).mean())


Q, K, OWN = {}, {}, {}
for si in range(S):
    Z = np.load(f"{NPD}/scene{si:02d}.npz"); OWN[si] = Z["img_owner"].astype(int)
    for L in LAYS:
        K[(si, L)] = Z[f"kimg_{L}"].astype(np.float32)
        for p in PINS:
            for nm in ("E", "B"):
                Q[(si, p, nm, L)] = Z[f"q_{nm}{p}_{L}"].astype(np.float32)
    del Z
H = {L: lm[L] for L in LAYS}
W = {}
for L in LAYS:
    for p in PINS:
        M = np.zeros((S, S))
        for a_ in range(S):
            for b_ in range(S):
                M[a_, b_] = T1(Q[(a_, p, "E", L)], K[(b_, L)], OWN[b_], p, H[L]) - T1(Q[(a_, p, "B", L)], K[(b_, L)], OWN[b_], p, H[L])
        W[(p, L)] = M
res = dict(tag=A.tag, n_scenes=S)
# ---------------------------------------------------------------- I0
def capT1(r, key, p):
    z = np.asarray(r["zl"][key]["51"], dtype=float)[:, lm["51"]].mean(1).mean(0); return z[p - 2] - z[p - 1]
rep = [abs(T1(Q[(s, p, "E", "51")], K[(s, "51")], OWN[s], p, H["51"]) - capT1(rows[s], f"Ep{p}", p)) for s in range(S) for p in PINS]
kd = max(float(r.get("kdiff") or 0.0) for r in rows)
try:
    J73 = json.load(open(f"{R3}/s873_{A.s873}_transplant.json")); r73 = J73["rows"]; lm73 = J73["late_mean"]
    def c73(r, key, p):
        z = np.asarray(r["zl"][key]["51"], dtype=float)[:, lm73["51"]].mean(1).mean(0); return z[p - 2] - z[p - 1]
    w73 = [c73(r73[s], f"Ep{p}", p) - c73(r73[s], f"Ep{p}_E_BANDR", p) for s in range(S) for p in PINS]
    w78 = [W[(p, "51")][s, s] for s in range(S) for p in PINS]
    repro = float(np.corrcoef(w73, w78)[0, 1])
except FileNotFoundError:
    repro = float("nan")
st = float(np.mean([rows[s]["land"][f"START{p}"] == p for s in range(S) for p in PINS]))
I0 = dict(readout=bool(np.median(rep) <= .01), keys_same=bool(kd <= .05), reproducible=bool(repro >= .90), t1=bool((J["t1_menu"] or 0) >= .95),
          START=bool(st >= .95))
res["I0"] = dict(**I0, all=bool(all(I0.values())), readout_median=float(np.median(rep)), readout_max=float(np.max(rep)), kdiff=kd, repro_r=repro,
                 t1_value=J["t1_menu"], START_acc=st)
ok = res["I0"]["all"]
# ---------------------------------------------------------------- Q1
def q1(ix, L):
    xs, nq, nk, P = [], [], [], []
    ix = list(ix)
    for p in PINS:
        M = W[(p, L)]
        for i, a_ in enumerate(ix):
            for j, b_ in enumerate(ix):
                if a_ == b_:
                    continue
                xs.append(M[a_, b_]); nq.append(M[a_, a_]); nk.append(M[b_, b_]); P.append(p)
    xs, nq, nk, P = map(np.array, (xs, nq, nk, P))
    for p in PINS:
        m = P == p
        for v_ in (xs, nq, nk):
            v_[m] -= v_[m].mean()
    rq, rk = np.corrcoef(xs, nq)[0, 1], np.corrcoef(xs, nk)[0, 1]
    return float(rk - rq), float(rq), float(rk)
def bt(fn):
    est = fn(range(S)); bs = [fn(rng.integers(0, S, S)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))
main = bt(lambda ix: q1(ix, "51")[0]); e = q1(range(S), "51")
res["Q1"] = dict(diff=main, r_query=e[1], r_key=e[2],
                 verdict=(("H1: the image-key geometry sets the per-scene address effect of the writer" if main["est"] >= .20 and main["lo"] > 0 else
                           "H2: the writer's query update itself is scene-specific" if main["est"] <= -.20 and main["hi"] < 0 else "both") if ok else "not read"))
# ---------------------------------------------------------------- descriptive
desc = {}
for L in LAYS:
    d_ = q1(range(S), L); desc[f"L{L}"] = dict(diff=d_[0], r_query=d_[1], r_key=d_[2])
    for p in PINS:
        M = W[(p, L)]; mu = M.mean(); al = M.mean(1) - mu; be = M.mean(0) - mu; ga = M - mu - al[:, None] - be[None, :]
        tot = np.var(M); desc[f"L{L}_p{p}_shares"] = dict(query=float(np.var(al) / tot), key=float(np.var(be) / tot), interaction=float(np.var(ga) / tot),
                                                         mean_natural=float(np.mean(np.diag(M))))
lin = {}
for L in LAYS:
    xs, ys = [], []
    for p in PINS:
        for a_ in range(S):
            dq = (Q[(a_, p, "E", L)] - Q[(a_, p, "B", L)])[:, H[L]].astype(np.float64)
            for b_ in range(S):
                Kb = K[(b_, L)]; own = OWN[b_]
                kd_ = (Kb[own == p - 2].mean(0) - Kb[own == p - 1].mean(0))[[h // GRP for h in H[L]]].astype(np.float64)       # [H, 256]
                xs.append(float(np.einsum("rhd,hd->rh", dq, kd_).mean() / 16.0)); ys.append(W[(p, L)][a_, b_])
    xs, ys = np.array(xs), np.array(ys); b = (xs @ ys) / (xs @ xs)
    lin[f"L{L}"] = dict(pearson=float(np.corrcoef(xs, ys)[0, 1]), R2_through_origin=float(1 - np.sum((ys - b * xs) ** 2) / np.sum((ys - ys.mean()) ** 2)), slope=float(b))
res["descriptive"] = dict(by_layer=desc, first_order_blockmean=lin)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], Q1=[res["Q1"]["verdict"], main["est"], e[1], e[2]],
                      shares_L51={f"p{p}": desc[f"L51_p{p}_shares"] for p in PINS}, linear=lin["L51"]), indent=1))
print("->", OUT)
