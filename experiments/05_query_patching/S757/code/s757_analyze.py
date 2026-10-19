"""S757 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-13 09:3x) before any S757 data.
I0              the "?" row (before the menu) has the same block scores under every menu order (max |difference| <= .01)
ANSWER_CHANGES  END(5)'s answered position differs from M0 in >= .30 of (scene, order in M1-M3)
Query pointer   the image with the largest mean block share over the S705a MATCHER head-rows at L35-47, rows tail1 and answer
Among (scene, order) whose END(5) answer changed:
    QK_STABLE   the pointer equals M0's pointer in >= .70 and moves to the new answer's image in <= .20      (outcome A)
    QK_TRACKS   the pointer moves to the new answer's image in >= .50                                       (outcome B)
    otherwise MIXED
Descriptive: target margin (score of image 3 - best other) per order and its change, the same for controls, unchanged-answer pairs.
Usage: python s757_analyze.py TAG
"""
import json, sys
import numpy as np

N = 7


def lse(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def analyze(tag):
    D = f"/vlm/results/r3/s757_{tag}_menuqk"
    meta = json.load(open(D + ".json")); z = np.load(D + ".npz"); M = meta["meta"]; CAP = meta["CAP"]; n = len(M)
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    dec = [x for x in mt if x[0] <= 47 and x[2] in (1, 2)]

    def pointer(si, key):
        sh = []
        for (li, h, ri) in dec:
            sc = z[f"{si}|{key}|{li}"][ri, h].astype(np.float64)
            sh.append(np.exp(sc - lse(sc)))
        return np.mean(sh, 0)                                                    # [N] mean block share

    def margin(si, key, t):
        v = []
        for (li, h, ri) in dec:
            sc = z[f"{si}|{key}|{li}"][ri, h].astype(np.float64)
            v.append(sc[t - 1] - np.max(np.delete(sc, t - 1)))
        return float(np.mean(v))

    out = dict(n=n, checks={})
    qd = max(float(np.max(np.abs(z[f"{si}|{m}|E5|{li}"][0] - z[f"{si}|M0|E5|{li}"][0]))) for si in range(n) for m in ("M1", "M2", "M3") for li in CAP)
    out["checks"]["I0"] = bool(qd <= .01 + 1e-9); out["qm_max_diff"] = qd
    ch, stay, move, pairs = [], 0, 0, 0
    rows = []
    for si in range(n):
        a0 = M[si]["land"]["M0|E5"]; p0 = int(np.argmax(pointer(si, "M0|E5"))) + 1
        for m in ("M1", "M2", "M3"):
            a = M[si]["land"][f"{m}|E5"]; p = int(np.argmax(pointer(si, f"{m}|E5"))) + 1
            changed = a != a0; ch.append(changed)
            rows.append(dict(si=si, m=m, a0=a0, a=a, p0=p0, p=p, m0=margin(si, "M0|E5", 3), mm=margin(si, f"{m}|E5", 3)))
            if changed:
                pairs += 1; stay += p == p0; move += p == a
    out["checks"]["ANSWER_CHANGES"] = bool(np.mean(ch) >= .30 - 1e-9)
    out["answer_change_rate"] = round(float(np.mean(ch)), 3)
    ps, pm = (stay / pairs, move / pairs) if pairs else (float("nan"), float("nan"))
    out["checks"]["QK"] = ("QK_STABLE" if pairs and ps >= .70 - 1e-9 and pm <= .20 + 1e-9 else "QK_TRACKS" if pairs and pm >= .50 - 1e-9 else "MIXED")
    out["qk_detail"] = dict(changed_pairs=pairs, pointer_stays=round(ps, 3), pointer_moves_to_new_answer=round(pm, 3))
    unch = [r for r in rows if r["a"] == r["a0"]]
    out["unchanged_pointer_stays"] = round(float(np.mean([r["p"] == r["p0"] for r in unch])), 3) if unch else None
    out["margin"] = dict(mean_M0=round(float(np.mean([r["m0"] for r in rows])), 3),
                         mean_abs_change=round(float(np.mean([abs(r["mm"] - r["m0"]) for r in rows])), 3),
                         mean_abs_change_changed=round(float(np.mean([abs(r["mm"] - r["m0"]) for r in rows if r["a"] != r["a0"]])), 3) if pairs else None)
    ctrl = {}
    for key, t in (("E2", 6), ("S3", 3)):
        ctrl[key] = dict(acc_M0=round(float(np.mean([M[si]["land"][f"M0|{key}"] == t for si in range(n)])), 3),
                         acc_M1=round(float(np.mean([M[si]["land"][f"M1|{key}"] == t for si in range(n)])), 3),
                         answer_change=round(float(np.mean([M[si]["land"][f"M1|{key}"] != M[si]["land"][f"M0|{key}"] for si in range(n)])), 3),
                         pointer_change=round(float(np.mean([np.argmax(pointer(si, f"M1|{key}")) != np.argmax(pointer(si, f"M0|{key}")) for si in range(n)])), 3))
    out["controls"] = ctrl
    out["E5_dist_M0"] = {str(b): round(float(np.mean([M[si]["land"]["M0|E5"] == b for si in range(n)])), 3) for b in range(0, N + 1)}
    print(f"[{tag}] checks {out['checks']} · ? row max diff {qd:.4f} · answer change {out['answer_change_rate']}")
    print(f"[{tag}] qk {out['qk_detail']} · unchanged-answer pointer stays {out['unchanged_pointer_stays']} · margin {out['margin']}")
    print(f"[{tag}] controls {ctrl} · END(5) M0 {out['E5_dist_M0']}")
    return out


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s757_analysis.json", "w"), indent=1)
