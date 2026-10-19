"""S853 stored data: where does the non-additivity of ordinal k and early position p appear? (post hoc, NOT registered; user 10-18 after the
S853 report - S853's R2_add .32 shows a low fit of the tested additive model, not by itself a causal interaction term.)  No new run.
Natural 2 x 2 per scene: E1 (k3, p3), E2 (k3, p4), E3 (k5, p3), E4 (k5, p4), N_s 8.  Orthogonal contrasts of a vector-valued readout X:
  K = (-E1 - E2 + E3 + E4) / 2   (ordinal),   P = (-E1 + E2 - E3 + E4) / 2   (early position),
  I = ( E1 - E2 - E3 + E4) / 2   (= half of the user's I_W = [W(5,4) - W(5,3)] - [W(3,4) - W(3,3)])
  interaction share IS = sum |I|^2 / sum (|K|^2 + |P|^2 + |I|^2) (sums over scenes; scene bootstrap 4,000, seed 3)
  consistency C(X) = |mean over scenes|^2 / mean |.|^2 of each contrast (a direction shared by the scenes; ~1/30 = .033 for scene-specific noise)
Levels (in the order of the computation): writer outputs W (token mixer of layer 47 = FA47, of layer 49 = DN49, MLPs of 46-49; rows B1, B4,
answer), late queries Q (51-63, rows B1, B4, answer), late positional scores Z (centred 8-vectors, rows B1, B4, answer, layers 51-63), answer
logits L (eight picture words, centred).  The same for the transplants of S853 (recipient corner R; single swaps X_K, X_P; both X_KP):
R2_add = 1 - sum |D_KP - D_K - D_P|^2 / sum |D_KP|^2 at each level.
Usage: python3 exp/p4_s853_int.py  ->  research/ces/outputs/s853_int.json"""
from __future__ import annotations

import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
B = 4000
LATE = ("51", "55", "59", "63")


def main():
    d = json.load(open(f"{V}/results/r3/s853_s853_27b_writer.json")); Z = np.load(f"{V}/results/r3/s853_s853_27b_writer.npz")
    by = {r["si"]: r for r in d["rows"]}; sis = sorted(by)
    comps = [f"{l}{kd}" for l, kd in [tuple(c) for c in d["comps"]]]
    rng = np.random.default_rng(3); idx = rng.integers(0, len(sis), (B, len(sis)))
    cen = lambda x: np.asarray(x, dtype=float) - np.mean(x)
    def W(s, nm, cn, rows=(0, 1)):
        c = Z[f"s{s}_{nm}_comp"].astype(np.float64); return np.concatenate([c[comps.index(cn), r] for r in rows])
    def Q(s, nm, rows=(0, 1, 2)):
        q = Z[f"s{s}_{nm}_q"].astype(np.float64); return np.concatenate([q[:, r].ravel() for r in rows])
    def Zs(s, nm, rows=(2, 5, 9)):
        return np.concatenate([cen(by[s]["qk"][nm][L][r]) for L in LATE for r in rows])
    def Ls(s, nm):
        return cen(by[s]["mlog"][nm][:8])
    levels = {"W_FA47_B1B4": lambda s, nm: W(s, nm, "47mix"), "W_DN49_B1B4": lambda s, nm: W(s, nm, "49mix"),
              "W_FA47_answer": lambda s, nm: W(s, nm, "47mix", (2,)), "W_DN49_answer": lambda s, nm: W(s, nm, "49mix", (2,)),
              "W_all8_B1B4": lambda s, nm: np.concatenate([W(s, nm, cn) for cn in comps]),
              "Q_late_B1B4ans": Q, "Z_late_B1B4ans": Zs, "L_answer": Ls}
    out = dict(note="post hoc, not registered; natural 2 x 2 contrasts and S853 transplant additivity per level", natural={}, transplant={})
    for lv, f in levels.items():
        K, P, I = [], [], []
        for s in sis:
            e1, e2, e3, e4 = (f(s, nm) for nm in ("E1", "E2", "E3", "E4"))
            K.append((-e1 - e2 + e3 + e4) / 2); P.append((-e1 + e2 - e3 + e4) / 2); I.append((e1 - e2 - e3 + e4) / 2)
        K, P, I = map(np.asarray, (K, P, I))
        sk, sp, si_ = (np.sum(x ** 2, 1) for x in (K, P, I))
        tot = sk + sp + si_
        bs = si_[idx].sum(1) / tot[idx].sum(1)
        cons = lambda X: float(np.sum(X.mean(0) ** 2) / np.mean(np.sum(X ** 2, 1)))
        out["natural"][lv] = dict(IS=[float(si_.sum() / tot.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]],
                                  share_K=float(sk.sum() / tot.sum()), share_P=float(sp.sum() / tot.sum()),
                                  consistency=dict(K=cons(K), P=cons(P), I=cons(I)))
    corners = {"E1": ("XK_1_3", "XP_1_2", "XKP_1_4"), "E2": ("XK_2_4", "XP_2_1", "XKP_2_3"), "E3": ("XK_3_1", "XP_3_4", "XKP_3_2"),
               "E4": ("XK_4_2", "XP_4_3", "XKP_4_1")}
    for lv, f in levels.items():
        num, den = [], []
        for s in sis:
            a_ = b_ = 0.0
            for c, (k_, p_, kp_) in corners.items():
                z0 = f(s, c); dk, dp, dkp = f(s, k_) - z0, f(s, p_) - z0, f(s, kp_) - z0
                a_ += float(np.sum((dkp - dk - dp) ** 2)); b_ += float(np.sum(dkp ** 2))
            num.append(a_); den.append(b_)
        num, den = np.asarray(num), np.asarray(den)
        bs = 1 - num[idx].sum(1) / den[idx].sum(1)
        out["transplant"][lv] = dict(R2_add=[float(1 - num.sum() / den.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]])
    json.dump(out, open(f"{V}/research/ces/outputs/s853_int.json", "w"), indent=1)
    print(f"{'level':16s} | natural: interaction share [95%]   share K  share P | consistency K / P / I | transplants: R2_add [95%]")
    for lv in levels:
        n = out["natural"][lv]; t = out["transplant"][lv]
        print(f"{lv:16s} | {n['IS'][0]:.3f} [{n['IS'][1][0]:.3f}, {n['IS'][1][1]:.3f}]  {n['share_K']:.3f}  {n['share_P']:.3f} | "
              f"{n['consistency']['K']:.2f} / {n['consistency']['P']:.2f} / {n['consistency']['I']:.2f} | "
              f"{t['R2_add'][0]:+.3f} [{t['R2_add'][1][0]:+.3f}, {t['R2_add'][1][1]:+.3f}]")


if __name__ == "__main__":
    main()
