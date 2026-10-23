"""Descriptive (S927; no verdict): the model's own first-order address-to-answer transfer, from the gradients.
For every natural END(k) run, T[q][b] = predicted change of menu logit b when the early read-row address is pinned to q (exact local change x
gradient), Dp[q][b] = the part through paths around the band (46-49) read-row outputs, BAND[q][b] = band <- START(q).  Summaries per k:
  R_rel[d][j] = mean over runs and q of T[q][q + j] - mean_b T[q][b] with d = q - a (where the pin moves the address relative to the natural
  anchor a) and j = b - q (which answer, relative to the pinned position, gains);
  band share of the pin response = 1 - <Dp, T> / <T, T> (projection of the bypass part on the total, per run, then the median);
  target column: mean over q != t of (T[q][t] - mean_b T[q][b]) - does pinning the address anywhere move the target's logit?
Usage: python3 p_s927_transfer.py LRTAG [LRTAG ...]"""
import collections, json, sys
import numpy as np

out = {}
for TAG in sys.argv[1:]:
    L = json.load(open(f"/vlm/results/r3/s927_{TAG}_lr.json")); N = L["N"]
    byk = collections.defaultdict(lambda: dict(rel=collections.defaultdict(list), share=[], tcol=[], diag=[], step=[], bandrel=collections.defaultdict(list)))
    for rec in L["rows"]:
        for k_, o in rec["k"].items():
            k = int(k_); a = o["anchor"]; t = o["t"]
            T = np.array(o["T"])[:, :N]; Dp = np.array(o["Dp"])[:, :N]; BD = np.array(o["BAND"])[:, :N]
            Tc = T - T.mean(1, keepdims=True); Bc = BD - BD.mean(1, keepdims=True)
            B = byk[k]
            for q in range(1, N + 1):
                for b in range(1, N + 1):
                    B["rel"][(q - a, b - q)].append(Tc[q - 1, b - 1]); B["bandrel"][(q - a, b - q)].append(Bc[q - 1, b - 1])
                if q != t:
                    B["tcol"].append(Tc[q - 1, t - 1])
                B["diag"].append(Tc[q - 1, q - 1])
                if q >= 2:
                    B["step"].append(Tc[q - 1, q - 2])
            tt = (T * T).sum()
            if tt > 0:
                B["share"].append(1 - (Dp * T).sum() / tt)
    res = {}
    for k in sorted(byk):
        B = byk[k]
        rel = {f"{d},{j}": float(np.mean(v)) for (d, j), v in B["rel"].items() if len(v) >= 10}
        res[str(k)] = dict(diag=float(np.mean(B["diag"])), step=float(np.mean(B["step"])), target_col=float(np.mean(B["tcol"])),
                           band_share_median=float(np.median(B["share"])), rel=rel,
                           band_rel={f"{d},{j}": float(np.mean(v)) for (d, j), v in B["bandrel"].items() if len(v) >= 10})
        print(TAG, "k", k, "pin q -> logit change at q (diag) %.2f, at q-1 (step) %.2f, target column %.2f, band share %.2f"
              % (res[str(k)]["diag"], res[str(k)]["step"], res[str(k)]["target_col"], res[str(k)]["band_share_median"]))
    out[TAG] = res
json.dump(out, open("/vlm/research/ces/outputs/writer_circuit_2026-10-20/main/p_s927_transfer.json", "w"), indent=1)
print("-> p_s927_transfer.json")
