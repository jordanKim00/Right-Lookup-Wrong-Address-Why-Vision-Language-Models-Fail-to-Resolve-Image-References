"""S846 descriptive re-analysis of the band swap (post hoc, NOT registered).  p in {3, 5, 6} (p != k = 4), pooled over N_s; scene bootstrap (seed 0).
  agreement  paired same-answer shares: BANDF vs S_EARLY and vs E_EARLY; BANDE vs E_EARLY and vs S_EARLY; baseline E_EARLY vs S_EARLY
  outcome    shares of answer = k, = p, other (neither k nor p: END's departure to other pictures) per arm
  stay       P(E_EARLY = p) by p (pooled N_s), for the departure-depends-on-p reading
Usage: python3 exp/p4_s846_swapdesc.py  ->  research/ces/outputs/s846_swapdesc.json"""
import json
import numpy as np

V = "/home/storage2/ai26269345/vlm"
d = json.load(open(f"{V}/results/r3/s846_s846_27b_swap.json")); R = d["rows"]
NSS = (6, 7, 8, 9); PW = (3, 5, 6); k = 4
by = {(r["si"], r["Ns"]): r for r in R}; sis = sorted({r["si"] for r in R})
rng = np.random.default_rng(0)
def ans(s, ns, arm, p):
    return by[(s, ns)]["land"][f"{arm}_p{p}"]
def boot(f):
    v = [f([sis[i] for i in rng.integers(0, len(sis), len(sis))]) for _ in range(2000)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def agree(a, b):
    return lambda ss: float(np.mean([ans(s, ns, a, p) == ans(s, ns, b, p) for s in ss for ns in NSS for p in PW]))
out = dict(note="post hoc, not registered", agreement={}, outcome={}, stay={})
for a, b in (("BANDF", "S_EARLY"), ("BANDF", "E_EARLY"), ("BANDE", "E_EARLY"), ("BANDE", "S_EARLY"), ("E_EARLY", "S_EARLY")):
    f = agree(a, b); out["agreement"][f"{a}={b}"] = [f(sis), boot(f)]
for arm in ("E_EARLY", "S_EARLY", "BANDF", "BANDE"):
    kk = lambda ss, arm=arm: float(np.mean([ans(s, ns, arm, p) == k for s in ss for ns in NSS for p in PW]))
    pp = lambda ss, arm=arm: float(np.mean([ans(s, ns, arm, p) == p for s in ss for ns in NSS for p in PW]))
    oo = lambda ss, arm=arm: float(np.mean([ans(s, ns, arm, p) not in (k, p) for s in ss for ns in NSS for p in PW]))
    out["outcome"][arm] = dict(k=[kk(sis), boot(kk)], p=[pp(sis), boot(pp)], other=[oo(sis), boot(oo)])
for p in (3, 4, 5, 6):
    out["stay"][str(p)] = float(np.mean([ans(s, ns, "E_EARLY", p) == p for s in sis for ns in NSS]))
json.dump(out, open(f"{V}/research/ces/outputs/s846_swapdesc.json", "w"), indent=1)
for k_, v in out["agreement"].items():
    print("agree", k_, "%.3f [%.3f, %.3f]" % (v[0], *v[1]))
for a, v in out["outcome"].items():
    print("outcome", a, {x: "%.3f [%.3f, %.3f]" % (y[0], *y[1]) for x, y in v.items()})
print("stay by p", out["stay"])
