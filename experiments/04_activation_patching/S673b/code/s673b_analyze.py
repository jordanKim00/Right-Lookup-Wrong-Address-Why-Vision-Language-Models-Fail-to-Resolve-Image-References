"""S673b analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-08) before any data.
Items: donor-correct (FWD_j on D answers D[j]).  Landing in B: j (untransformed address), k (own surface / 27B-transformed address),
y_j (donor content), other.
D0 P(j | SAME) >= .50;  D1 at L_port P(j) - P(y_j) >= .30;  D2 some layer with P(y_j) - P(j) >= .15;
D3 the shallowest layer with P(y_j) > P(j) lies in [13, 22] (Qwen3-VL-4B) / [14, 22] (Qwen3-VL-8B).
DEREF_AT_ORDINAL = D1 and D2 and D3 (dense models).  Item-bootstrap 95% intervals (10,000, seed 67350) for j - y_j per layer.
Usage: python s673b_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np

WIN = {"Qwen3-VL-4B-Instruct": (13, 22), "Qwen3-VL-8B-Instruct": (14, 22)}


def analyze(tag):
    d = json.load(open(f"/vlm/results/r3/s673b_{tag}_deref.json"))
    rows = [r for r in d["rows"] if r["donor_ok"]]
    cat = lambda r, a: "j" if a == r["j"] else ("y" if a == r["yj"] else ("k" if a == r["k"] else "o"))
    keys = ["NONE", "SAME"] + [f"ORD@{L}" for L in d["LS"]] + [x for x in ("ORD_Q", "ORD_O") if x in rows[0]["land"]]
    sh = {key: {c: float(np.mean([cat(r, r["land"][key]) == c for r in rows])) for c in "jyko"} for key in keys}
    rng = np.random.default_rng(67350); idx = rng.integers(0, len(rows), (10000, len(rows)))
    ci = {}
    for L in d["LS"]:
        x = np.array([(cat(r, r["land"][f"ORD@{L}"]) == "j") - (cat(r, r["land"][f"ORD@{L}"]) == "y") for r in rows], float)
        ci[L] = tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]).round(3).tolist())
    d0 = sh["SAME"]["j"] >= .50
    Lp = d["L_port"]
    d1 = sh[f"ORD@{Lp}"]["j"] - sh[f"ORD@{Lp}"]["y"] >= .30
    d2 = any(sh[f"ORD@{L}"]["y"] - sh[f"ORD@{L}"]["j"] >= .15 for L in d["LS"])
    flip = [L for L in d["LS"] if sh[f"ORD@{L}"]["y"] > sh[f"ORD@{L}"]["j"]]
    Lc = min(flip) if flip else None
    base = d["model"].rstrip("/").split("/")[-1]
    win = WIN.get(base)
    d3 = Lc is not None and win is not None and win[0] <= Lc <= win[1]
    checks = dict(D0=d0, D1=d1, D2=d2, D3=d3, L_content=Lc, DEREF_AT_ORDINAL=(d1 and d2 and d3) if win else None)
    print(f"[{tag}] donor-correct {len(rows)}/{len(d['rows'])} | L_port {Lp} | ORD heads {d['OH']}")
    for key in keys:
        s = sh[key]
        extra = f"  j−y_j {ci[int(key.split('@')[1])]}" if key.startswith("ORD@") else ""
        print(f"[{tag}]  {key:9s} j {s['j']:.2f}  y_j {s['y']:.2f}  k {s['k']:.2f}  other {s['o']:.2f}{extra}")
    print(f"[{tag}] checks {checks}")
    return dict(share=sh, checks=checks, n=len(rows))


if __name__ == "__main__":
    res = {t: analyze(t) for t in sys.argv[1:]}
    json.dump(res, open("/vlm/results/r3/s673b_analysis.json", "w"), indent=1, default=float)
