"""S745b analysis (amended after the S745 gate, before any main-stage data; knowledge/DECISIONS.md 2026-10-11 20:29): a copy of
exp/s745_analyze.py with the gate at .40 and the landing thresholds relative to the model's own start-relative accuracy.
    GATE              start-relative accuracy >= .40 on the discovery scenes and >= 5 index-matching heads (else the lookup cannot be traced)
    LITERAL_QUERY     END: the heads' share on the literal block > on the target, and END accuracy <= START accuracy - .20
    ADDRESS_CONTROL   pooled over the early layers 2, 4, 6: answers on the written position x >= .5 x START accuracy and above the
                      answers on the donor's named picture and on k, for the right (x = j) and the wrong (x = w) address separately
    QUERY_MEDIATION   answers on x after the START(x) queries >= .5 x START accuracy and above the answers on k, for x = j and x = w
Reported: every layer of ADDRESS (on x, on the donor's named picture, on k), shares, cosine, scene-bootstrap intervals (2,000 draws).
Predictions: all four.  Usage: python s745_analyze.py TAG [TAG ...]
"""
import json, sys
import numpy as np


def main(tags):
    out = {}
    for tag in tags:
        d = json.load(open(f"/vlm/results/r3/s745_{tag}_core.json")); R = d["rows"]; g = d["gate"]
        chk = dict(GATE=g["start_acc"] >= .40 - 1e-12 and len(g["heads"]) >= 5)
        res = dict(gate=g)
        print(f"[{tag}] GATE start accuracy {g['start_acc']:.3f} · heads {len(g['heads'])}")
        if R:
            sc = sorted({r["si"] for r in R}); rng = np.random.default_rng(74502)
            def m(f, rs=None):
                rs = R if rs is None else rs; return float(np.mean([f(r) for r in rs]))
            def ci(f):
                bs = [m(f, [r for s in rng.choice(sc, len(sc)) for r in R if r["si"] == s]) for _ in range(2000)]
                return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            stats = {"START acc": lambda r: r["START"] == r["j"], "END acc": lambda r: r["END"] == r["j"], "END literal": lambda r: r["END"] == r["k"],
                     "END tgt share": lambda r: r["tgt_END"], "END lit share": lambda r: r["lit_END"], "START tgt share": lambda r: r["tgt_START"],
                     "cos END-START": lambda r: r["cos_END"]}
            for nm in ("j", "w"):
                x = "j" if nm == "j" else "w"
                stats[f"Q_{nm} on x"] = (lambda r, nm=nm: r[f"Q_{nm}"] == r[nm])
                stats[f"ADDR_{nm} early on x"] = (lambda r, nm=nm: np.mean([r[f"ADDR_{nm}@{L}"] == r[nm] for L in (2, 4, 6)]))
                for L in d["layers"]:
                    stats[f"ADDR_{nm}@{L} on x"] = (lambda r, nm=nm, L=L: r[f"ADDR_{nm}@{L}"] == r[nm])
                    stats[f"ADDR_{nm}@{L} on named"] = (lambda r, nm=nm, L=L: r[f"ADDR_{nm}@{L}"] == r[f"named_{nm}"])
                    stats[f"ADDR_{nm}@{L} on k"] = (lambda r, nm=nm, L=L: r[f"ADDR_{nm}@{L}"] == r["k"])
            for key, f in stats.items():
                res[key] = dict(v=m(f), ci=ci(f))
            v_ = lambda key: res[key]["v"]
            S0 = v_("START acc")
            early = lambda nm, what: float(np.mean([np.mean([r[f"ADDR_{nm}@{L}"] == (r[nm] if what == "x" else r[f"named_{nm}"] if what == "named" else r["k"]) for L in (2, 4, 6)]) for r in R]))
            qk = lambda nm: float(np.mean([r[f"Q_{nm}"] == r["k"] for r in R]))
            chk.update(LITERAL_QUERY=v_("END lit share") > v_("END tgt share") and v_("END acc") <= S0 - .20 + 1e-12,
                       ADDRESS_CONTROL=all(early(nm, "x") >= .5 * S0 - 1e-12 and early(nm, "x") > max(early(nm, "named"), early(nm, "k")) for nm in ("j", "w")),
                       QUERY_MEDIATION=all(v_(f"Q_{nm} on x") >= .5 * S0 - 1e-12 and v_(f"Q_{nm} on x") > qk(nm) for nm in ("j", "w")))
            for key in stats:
                print(f"   {key:22s} {res[key]['v']:.3f} [{res[key]['ci'][0]:.2f}, {res[key]['ci'][1]:.2f}]")
        print(f"   checks {chk}")
        out[tag] = dict(res=res, checks=chk, n=len(R))
    json.dump(out, open("/vlm/results/r3/s745b_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
