"""S748 analysis, frozen with the amended pre-registration (knowledge/DECISIONS.md, 2026-10-12 19:4x) before any S748 data.
The first registration (19:05) labelled "count" as "the k = 2 answer moves to image 5 when image 3 is hidden".  That conflated the visible
rank with the physical index (a count of visible blocks predicts physical 5 only if it is used as a code in the images' own index space),
so the outcome labels were replaced before any data by four mechanisms with predicted blocks per condition:
    ANCHOR    code of the last visible block minus (k - 1)
    RANK      the k-th visible block from the end
    COUNT     code n_vis + 1 - k (the number of visible blocks used as a code)
    CENTROID  code 2 * mean(visible codes) - 1 + 1 - k, rounded and clamped to 1..7 (a set statistic other than the count)
I0         NOOP answers equal NONE answers on >= .95 of (scene, question); LAST1 sends END k = 2 to image 5 in >= .50 (answers) and
           the MATCHER fixed-bank argmax to block 5 in >= .50
Primary    over (scene, MATCHER head-row of S705a), the share of the fixed-bank END k = 2 argmax on each mechanism's predicted block,
           averaged over MID1, FIRST1, FIRST2, HIDE6; the winner needs a mean share >= .40 and >= .10 above the runner-up, else MIXED
Secondary  the same with answers over MID1, FIRST1, FIRST2 (ANCHOR and RANK coincide there and are scored together; HIDE6 is left out
           because ANCHOR predicts a hidden block)
Descriptive: every landing distribution, k = 1 and 3, START controls, the START-only head set (E0) as a robustness read, QK by layer;
scene-bootstrap 95% intervals (4,000 draws) for the main shares.
Usage: python s748_analyze.py TAG
"""
import json, sys
import numpy as np

N = 7
VIS = {"NONE": [1, 2, 3, 4, 5, 6, 7], "NOOP": [1, 2, 3, 4, 5, 6, 7], "LAST1": [1, 2, 3, 4, 5, 6], "MID1": [1, 2, 4, 5, 6, 7],
       "FIRST1": [2, 3, 4, 5, 6, 7], "FIRST2": [3, 4, 5, 6, 7], "HIDE6": [1, 2, 3, 4, 5, 7]}


def pred(mech, cond, k):
    v = VIS[cond]
    if mech == "ANCHOR":
        return max(v) - (k - 1)
    if mech == "RANK":
        return v[-k]
    if mech == "COUNT":
        return len(v) + 1 - k
    if mech == "CENTROID":
        return int(min(N, max(1, round(2 * np.mean(v) - 1 + 1 - k))))
    raise ValueError(mech)


MECHS = ["ANCHOR", "RANK", "COUNT", "CENTROID"]
TEST = ["MID1", "FIRST1", "FIRST2", "HIDE6"]


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze(tag):
    D = f"/vlm/results/r3/s748_{tag}_whatend"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz")
    M = meta["meta"]; conds = list(meta["conds"]); qs = meta["questions"]; CAP = meta["CAP"]; n = len(M)
    land = lambda c, q: np.array([m["land"][f"{c}|{q}"] for m in M])
    A5 = json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]
    sets = {"MATCHER": [tuple(x) for x in A5]}

    def qk_arg(c, q, hs):
        """per scene: list of argmax blocks over head-rows (fixed bank, all seven blocks)."""
        out = []
        for si in range(n):
            row = [int(np.argmax(z[f"{si}|{c}|{q}|{li}"][ri, h])) + 1 for (li, h, ri) in hs if f"{si}|{c}|{q}|{li}" in z.files]
            out.append(row)
        return out

    res = dict(n=n, checks={}, qk={}, land={}, mech={})
    # I0
    agree = float(np.mean([M[i]["land"][f"NOOP|{q}"] == M[i]["land"][f"NONE|{q}"] for i in range(n) for q in qs]))
    qa = qk_arg("LAST1", "E2", sets["MATCHER"])
    last1_qk5 = float(np.mean([b == 5 for r in qa for b in r]))
    last1_ans5 = float(np.mean(land("LAST1", "E2") == 5))
    res["checks"]["I0"] = bool(agree >= .95 - 1e-9 and last1_ans5 >= .50 - 1e-9 and last1_qk5 >= .50 - 1e-9)
    res["i0"] = dict(noop_agree=round(agree, 3), last1_answer_5=round(last1_ans5, 3), last1_qk_5=round(last1_qk5, 3))
    # primary: QK shares on each mechanism's prediction
    shares = {}
    for c in conds:
        a = qk_arg(c, "E2", sets["MATCHER"])
        flat = np.array([b for r in a for b in r])
        shares[c] = {b: float(np.mean(flat == b)) for b in range(1, N + 1)}
        res["qk"][c] = {str(b): round(v, 3) for b, v in shares[c].items()}
        res["qk"][c + "_scene_share5_6"] = dict(five=boot([np.mean(np.array(r) == 5) for r in a]), six=boot([np.mean(np.array(r) == 6) for r in a]))
    score = {m: float(np.mean([shares[c][pred(m, c, 2)] for c in TEST])) for m in MECHS}
    order = sorted(MECHS, key=lambda m: -score[m])
    win = order[0] if score[order[0]] >= .40 - 1e-9 and score[order[0]] - score[order[1]] >= .10 - 1e-9 else "MIXED"
    res["checks"]["PRIMARY"] = win
    res["mech"]["qk_score"] = {m: round(v, 3) for m, v in score.items()}
    res["mech"]["pred_k2"] = {m: {c: pred(m, c, 2) for c in TEST} for m in MECHS}
    # secondary: answers (ANCHOR = RANK on MID1, FIRST1, FIRST2)
    sec = {}
    for m in ("ANCHOR", "COUNT", "CENTROID"):
        sec["ANCHOR/RANK" if m == "ANCHOR" else m] = float(np.mean([np.mean(land(c, "E2") == pred(m, c, 2)) for c in ("MID1", "FIRST1", "FIRST2")]))
    o2 = sorted(sec, key=lambda m: -sec[m])
    res["checks"]["SECONDARY"] = o2[0] if sec[o2[0]] >= .40 - 1e-9 and sec[o2[0]] - sec[o2[1]] >= .10 - 1e-9 else "MIXED"
    res["mech"]["answer_score"] = {m: round(v, 3) for m, v in sec.items()}
    # descriptive
    for c in conds:
        for q in qs:
            l = land(c, q)
            res["land"][f"{c}|{q}"] = {str(b): round(float(np.mean(l == b)), 3) for b in range(0, N + 1)}
    res["mech"]["k_fit"] = {f"{c}|E{k}": {m: round(float(np.mean(land(c, f"E{k}") == pred(m, c, k))), 3) for m in MECHS}
                            for c in conds for k in (1, 2, 3)}
    res["start"] = {f"{c}|S{t}": round(float(np.mean(land(c, f"S{t}") == t)), 3) for c in conds for t in (5, 6)}
    by = {}
    for c in TEST + ["NONE", "LAST1"]:
        by[c] = {}
        for li in CAP:
            hs = [x for x in sets["MATCHER"] if x[0] == li]
            flat = np.array([b for r in qk_arg(c, "E2", hs) for b in r])
            if len(flat):
                by[c][str(li)] = {str(b): round(float(np.mean(flat == b)), 3) for b in (4, 5, 6, 7)}
    res["qk_bylayer"] = by
    print(f"[{tag}] I0 {res['i0']} -> {res['checks']['I0']}")
    print(f"[{tag}] PRIMARY {win} · QK scores {res['mech']['qk_score']}")
    print(f"[{tag}] SECONDARY {res['checks']['SECONDARY']} · answer scores {res['mech']['answer_score']}")
    for c in conds:
        print(f"   {c:7s} QK E2 argmax " + " ".join(f"{b}:{shares[c][b]:.2f}" for b in range(1, N + 1)) +
              " | answers E2 " + " ".join(f"{b}:{res['land'][c + '|E2'][str(b)]:.2f}" for b in range(1, N + 1)))
    print(f"[{tag}] START controls {res['start']}")
    return res


if __name__ == "__main__":
    out = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(out, open("/vlm/results/r3/s748_analysis.json", "w"), indent=1)
