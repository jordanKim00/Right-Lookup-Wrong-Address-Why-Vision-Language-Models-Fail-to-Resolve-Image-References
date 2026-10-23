"""S927 descriptive figure: the model's own first-order address-to-answer transfer (no fitted parameter).  For every natural END(k) run,
T[q][b] = <dz_b/dh, Delta(pin to q)> at the early readers (exact local change x gradient); shown: the mean over scenes of T[q][b] centred
over the answer options b, per k (rows: pinned position q, columns: answer b).  Data: results/r3/s927_<tag>_lr.json.
Usage: python fig_s927_transfer.py [TAG] -> results/figures/s927/transfer_<TAG>.{pdf,png}"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/vlm/exp/figs")
import style

TAG = sys.argv[1] if len(sys.argv) > 1 else "s927a_27b"
L = json.load(open(TAG if TAG.endswith(".json") else f"/vlm/results/r3/s927_{TAG}_lr.json")); N = L["N"]; KS = L["ks"]
TAG = os.path.basename(TAG).replace(".json", "") if TAG.endswith(".json") else TAG
M = {k: np.zeros((N, N)) for k in KS}; n = {k: 0 for k in KS}
for rec in L["rows"]:
    for k_, o in rec["k"].items():
        k = int(k_); T = np.array(o["T"])[:, :N]
        M[k] += T - T.mean(1, keepdims=True); n[k] += 1
style.apply()
fig, axs = plt.subplots(1, len(KS), figsize=(style.TWO, 1.25), sharey=True)
for ax, k in zip(axs, KS):
    A = M[k] / n[k]; lim = np.abs(A).max()                                         # each panel scaled to its own maximum (printed in the title)
    im = ax.imshow(A / lim, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax.set_title(f"k = {k}  ({lim:.1f})", pad=6)
    ax.plot([N - k], [-.95], marker="v", ms=2.5, color=style.TARGET, clip_on=False)   # target column t = N + 1 - k
    ax.set_xticks([0, N - 1]); ax.set_xticklabels([1, N]); ax.set_yticks([0, N - 1]); ax.set_yticklabels([1, N])
    for q in range(N):                                                             # outline the step-back cell (b = q - 1) and the copy cell (b = q)
        if q >= 1:
            ax.add_patch(plt.Rectangle((q - 1 - .5, q - .5), 1, 1, fill=False, lw=.4, ec=style.INK, ls=":"))
    ax.set_xlabel("answer $b$", labelpad=1)
axs[0].set_ylabel("pinned address $q$", labelpad=1)
cb = fig.colorbar(im, ax=axs, fraction=.012, pad=.01); cb.set_label("$\\Delta z_b$ / panel max", labelpad=2); cb.ax.tick_params(labelsize=5.5)
os.makedirs("/vlm/results/figures/s927", exist_ok=True)
style.save(fig, f"/vlm/results/figures/s927/transfer_{TAG}")
print("->", f"/vlm/results/figures/s927/transfer_{TAG}.png", {k: round(float(np.trace(M[k] / n[k], -1) / (N - 1)), 2) for k in KS})
