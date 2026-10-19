"""S705a analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S705a data.
Block score of head h for image j at a row: L_h(j) = logsumexp_{p in image j} logit_p; block share = softmax mass of image j / total image
mass (image-conditional selection).  Rows: qm, tail1, ans.  Discovery = scenes 0..14, held-out = 15..29.
MATCHER (discovery): START diagonal share >= .5 (mean over t) and END k = 2 target-share >= .4 at the ans or tail1 row
ANCHOR  (discovery): END last-visible share >= .35 (mean over k, unmoved anchor) and START last-block share <= .2
QK_ANCHOR_SHIFT (held-out, MATCHER heads, END k = 2): argmax_j L_h(j) with anchor moved by m lands on 7-m-1 in >= .5 of (scene, head, m)
    cases and on 6 in <= .25
QK_STEP_LAW (held-out, MATCHER heads): mean END target share at k = 2 > at k = 4 and > at k = 5, with START diagonal share at the same
    targets (t = 6, 4, 3) >= .5
Contrastive cross-moment bases (Lee et al.) are an exploratory follow-up on the same saved arrays (not part of these rules).
Usage: python s705a_analyze.py TAG
"""
import json, sys
import numpy as np

GE = lambda a, b: a >= b - 1e-9
N = 7


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def analyze(tag):
    D = f"/vlm/results/r3/s705a_{tag}_qkanatomy"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz")
    CAP = meta["CAP"]; M = meta["meta"]; ns = len(M)
    disc, hold = list(range(ns // 2)), list(range(ns // 2, ns))
    ROWS = {"qm": 0, "tail1": 1, "ans": 2}

    def shares(si, key, li, ri):
        """image-conditional block shares [H, N] and block scores [H, N] for a row."""
        own = np.array(M[si]["owner"]); limg = z[f"{si}|{key}|{li}|limg"][ri].astype(np.float64)   # [H, P]
        sc = np.stack([lse(limg[:, own == b]) for b in range(N)], -1)                                # [H, N]
        sh = np.exp(sc - lse(sc)[:, None])
        return sh, sc

    def mean_share(sis, key_fn, target_fn, li, ri, ks):
        vals = []
        for si in sis:
            for k in ks:
                key = key_fn(k)
                if f"{si}|{key}|{li}|limg" not in z.files:
                    continue
                sh, _ = shares(si, key, li, ri)
                vals.append(sh[:, target_fn(k) - 1])
        return np.mean(vals, 0) if vals else None                                                    # [H]

    out = {"heads": {}, "checks": {}}
    matchers, anchors = [], []
    for li in CAP:
        for rn, ri in ROWS.items():
            sdiag = mean_share(disc, lambda t: f"S{t}m0", lambda t: t, li, ri, range(1, N + 1))
            e2 = mean_share(disc, lambda k: f"E{k}m0", lambda k: N + 1 - k, li, ri, [2])
            elast = mean_share(disc, lambda k: f"E{k}m0", lambda k: N, li, ri, range(1, N + 1))
            slast = mean_share(disc, lambda t: f"S{t}m0", lambda t: N, li, ri, [t for t in range(1, N)])
            for h in range(len(sdiag)):
                if rn in ("ans", "tail1") and sdiag[h] >= .5 and e2[h] >= .4:
                    matchers.append((li, h, ri))
                if elast[h] >= .35 and slast[h] <= .2:
                    anchors.append((li, h, ri))
    out["heads"] = dict(matchers=matchers, anchors=anchors)
    print(f"[{tag}] MATCHER {len(matchers)}: {matchers[:20]}")
    print(f"[{tag}] ANCHOR  {len(anchors)}: {anchors[:20]}")
    # QK_ANCHOR_SHIFT on held-out
    moved, stay, tot = 0, 0, 0
    for (li, h, ri) in matchers:
        for si in hold:
            for m in (1, 2):
                key = f"E2m{m}"
                if f"{si}|{key}|{li}|limg" not in z.files:
                    continue
                _, sc = shares(si, key, li, ri)
                vis = sc[h, :N - m]; am = int(np.argmax(vis)) + 1
                tot += 1; moved += am == N - m - 1; stay += am == N - 1
    shift = dict(n=tot, moved=moved / max(tot, 1), stay=stay / max(tot, 1))
    out["checks"]["QK_ANCHOR_SHIFT"] = bool(tot) and GE(shift["moved"], .5) and shift["stay"] <= .25 + 1e-9
    out["checks"]["anchor_shift_detail"] = shift
    # QK_STEP_LAW on held-out
    law = {}
    for k, t in ((2, 6), (4, 4), (5, 3)):
        e = [mean_share(hold, lambda kk: f"E{kk}m0", lambda kk: N + 1 - kk, li, ri, [k])[h] for (li, h, ri) in matchers]
        s = [mean_share(hold, lambda tt: f"S{tt}m0", lambda tt: tt, li, ri, [t])[h] for (li, h, ri) in matchers]
        law[k] = (float(np.mean(e)) if e else float("nan"), float(np.mean(s)) if s else float("nan"))
    out["checks"]["step_detail"] = law
    out["checks"]["QK_STEP_LAW"] = bool(matchers) and law[2][0] > law[4][0] and law[2][0] > law[5][0] and all(GE(law[k][1], .5) for k in law)
    out["checks"]["classes_nonempty"] = bool(matchers) and bool(anchors)
    # behaviour on held-out for reference
    beh = {k: float(np.mean([M[si]["land"][f"E{k}m0"] == N + 1 - k for si in hold])) for k in range(1, N + 1)}
    out["checks"]["behaviour_END"] = beh
    print(f"[{tag}] anchor shift {shift} | step law {law} | END behaviour {beh}")
    print(f"[{tag}] checks {{QK_ANCHOR_SHIFT: {out['checks']['QK_ANCHOR_SHIFT']}, QK_STEP_LAW: {out['checks']['QK_STEP_LAW']}, nonempty: {out['checks']['classes_nonempty']}}}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s705a_analysis.json", "w"), indent=1, default=float)
