"""S719 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S719 data.
Fixed matcher set: results/r3/s705a_qa_27b_matchers.json (S705a discovery scenes, N = 7).  Block score L_h(j) = logsumexp over image j
of the QK logits; share = softmax over blocks.  Primary row = answer; tail1 reported.
    START_DIAG[N]   mean START diagonal share of the fixed set >= .5
    ANCHOR_SHIFT[N] END k = 2, last m in {1, 2} hidden: argmax over visible blocks lands on N-m-1 in >= .5 of (scene, head, m) and on N-1 in <= .25
    STEP_LAW[N]     mean END target share k = 2 > k = 3 and > k = 4
Behaviour (reported): END accuracy by k, START accuracy.
Usage: python s719_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def analyze(tag):
    D = f"/vlm/results/r3/s719_{tag}_qkanatomy"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; N = meta["N"]
    MAT = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_qa_27b_matchers.json"))["matchers"]]

    def sc_of(si, key, L, ri):
        own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{L}|limg"][ri].astype(np.float64)
        return np.stack([lse(limg[:, own == b]) for b in range(N)], -1)          # [H, N]

    out = {}
    for rn, ri in (("ans", 2), ("tail1", 1)):
        diag, step = [], {k: [] for k in (2, 3, 4)}; moved = stay = tot = 0
        for si in range(len(M)):
            for (L, h) in MAT:
                if f"{si}|S1m0|{L}|limg" not in z.files:
                    continue
                for t in range(1, N + 1):
                    sc = sc_of(si, f"S{t}m0", L, ri)[h]; sh = np.exp(sc - lse(sc)); diag.append(sh[t - 1])
                for k in (2, 3, 4):
                    sc = sc_of(si, f"E{k}m0", L, ri)[h]; sh = np.exp(sc - lse(sc)); step[k].append(sh[N - k])
                for m in (1, 2):
                    sc = sc_of(si, f"E2m{m}", L, ri)[h][:N - m]
                    am = int(np.argmax(sc)) + 1; tot += 1; moved += am == N - m - 1; stay += am == N - 1
        r = dict(start_diag=float(np.mean(diag)), step={k: float(np.mean(v)) for k, v in step.items()},
                 shift=dict(n=tot, moved=moved / max(tot, 1), stay=stay / max(tot, 1)))
        out[rn] = r
        print(f"[{tag} N{N} {rn}] START 대각 {r['start_diag']:.3f} · END 목표 몫 k2 {r['step'][2]:.3f} k3 {r['step'][3]:.3f} k4 {r['step'][4]:.3f} · "
              f"기준점 이동 {r['shift']['moved']:.3f} (제자리 {r['shift']['stay']:.3f}, n {tot})")
    acc = {}
    for k in range(1, N + 1):
        acc[f"E{k}"] = float(np.mean([M[si]["land"][f"E{k}m0"] == N + 1 - k for si in range(len(M))]))
        acc[f"S{k}"] = float(np.mean([M[si]["land"][f"S{k}m0"] == k for si in range(len(M))]))
    print(f"[{tag} N{N}] 행동 END: " + " · ".join(f"k{k} {acc[f'E{k}']:.2f}" for k in range(1, N + 1)) + " | START: " + " · ".join(f"{acc[f'S{k}']:.2f}" for k in range(1, N + 1)))
    a = out["ans"]
    checks = dict(START_DIAG=GE(a["start_diag"], .5), ANCHOR_SHIFT=GE(a["shift"]["moved"], .5) and a["shift"]["stay"] <= .25 + 1e-9,
                  STEP_LAW=a["step"][2] > a["step"][3] and a["step"][2] > a["step"][4])
    print(f"[{tag} N{N}] checks {checks}")
    return dict(N=N, rows=out, behaviour=acc, checks=checks)


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s719_analysis.json", "w"), indent=1, default=float)
