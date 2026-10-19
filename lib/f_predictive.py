"""agent_math_1015 / f_predictive (descriptive CPU test of predictions pre-registered in knowledge/agent_exchange_2026-10-15/
MATH_MECHANISM_REVIEW.md section Q-F at 2026-10-15 16:14:42 KST, before this computation).  Second task F.
Natural END landings (clean readouts).  Train on S801 (N = 6, 7, 8); test 1 = the other ten N = 7 experiments (S786, S791, S793, S797, S804,
S807, S808, S809, S810, S811), k in {2, 3, 4, 5, 6} where present; test 2 = leave one N out within S801 (k = 1..5).
Models (parameters per k, shared across N; positions from N and k):
  G   (S803 model D)  eps[b=0] + (1-eps)(c[b=k] + e G_N(b; N+1-k+mu, sigma) + u/N)
  MX  (mixture)        eps[b=0] + (1-eps)(w_t G_N(b; N+1-k, s_t) + w_tm1[b=N-k] + w_km1[b=k-1] + w_k[b=k] + w_1[b=1] + w_u/N); absent positions -> uniform
  LC  (logit compet.)  eps[b=0] + (1-eps) softmax_b(-a_t (b-(N+1-k))^2 + a_tm1[b=N-k] + a_km1[b=k-1] + a_k[b=k] + a_1[b=1])
Maximum likelihood with a numpy Nelder-Mead (24 random starts, seed 1015; counts grouped by (N, k)).  Score: held-out log-likelihood; differences with a cluster
(scene) bootstrap 95% interval (4,000, seed 1016).  Output: research/ces/outputs/agent_math_1015/f_predictive.json"""
import ast, json, sys
import numpy as np
sys.path.insert(0, "/home/storage2/ai26269345/vlm/exp/agent_math_1015")
from nm import minimize

R3 = "/home/storage2/ai26269345/vlm/results/r3"; O = "/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015"


def sm(z):
    z = np.asarray(z, float); w = np.exp(z - z.max()); return w / w.sum()


def gk(N, c, s):
    b = np.arange(1, N + 1); w = np.exp(-.5 * ((b - c) / s) ** 2); return w / w.sum() if w.sum() > 0 else np.full(N, 1 / N)


def probs(model, th, N, k):
    t = N + 1 - k; p = np.zeros(N + 1)
    if model == "G":
        c, e, u = sm([th[0], th[1], 0.]); eps = .5 / (1 + np.exp(-th[2])); mu = 4 * np.tanh(th[3]); s = .25 + np.log1p(np.exp(th[4]))
        p[1:] = e * gk(N, t + mu, s) + u / N
        if 1 <= k <= N:
            p[k] += c
        else:
            p[1:] += c / N
    elif model == "MX":
        w = sm(list(th[:5]) + [0.]); eps = .5 / (1 + np.exp(-th[5])); s = .25 + np.log1p(np.exp(th[6]))
        p[1:] = w[0] * gk(N, t, s) + w[5] / N
        for wi, pos in zip(w[1:5], (N - k, k - 1, k, 1)):
            if 1 <= pos <= N:
                p[pos] += wi
            else:
                p[1:] += wi / N
    else:  # LC
        at = np.log1p(np.exp(th[0])); eps = .5 / (1 + np.exp(-th[5]))
        b = np.arange(1, N + 1); z = -at * (b - t) ** 2
        for a, pos in zip(th[1:5], (N - k, k - 1, k, 1)):
            if 1 <= pos <= N:
                z[pos - 1] += a
        p[1:] = sm(z)
    p[1:] *= (1 - eps); p[0] = eps
    return np.clip(p, 1e-12, 1)


NP = dict(G=5, MX=7, LC=6)


def groups(cells):
    g = {}
    for (N, k, b) in cells:
        g.setdefault((N, k), np.zeros(N + 1))[b] += 1
    return list(g.items())


def nll(model, th, grp):
    return -sum(float(cnt @ np.log(probs(model, th, N, k))) for (N, k), cnt in grp)


def fit(model, cells, seed=1015, starts=24):
    rng = np.random.default_rng(seed); best = None; grp = groups(cells)
    for _ in range(starts):
        r = minimize(lambda th: nll(model, th, grp), rng.normal(0, 1.5, NP[model]), maxiter=3000)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def load_s801():
    out = []
    for r in json.load(open(f"{R3}/s801_s801_27b_grounding.json"))["rows"]:
        e = r["END"]; e = ast.literal_eval(e) if isinstance(e, str) else e
        out.append(dict(src="S801", si=f"S801:{r['N']}:{r['si']}", N=int(r["N"]), k=int(r["k"]), b=int(e["land"])))
    return out


def load_test1():
    SRC = [("S786", "s730_s786_27b_stagewise.json", lambda r: r["land_END"]), ("S791", "s791_s791_27b_chain.json", lambda r: r["land_END"]),
           ("S793", "s793_s793_27b_transplant.json", lambda r: r["land"]["END"]), ("S797", "s797_s797_27b_writer.json", lambda r: r["land"]["END"]),
           ("S804", "s804_s804_27b_cumulative.json", lambda r: r["land"]["END"]), ("S807", "s807_s807_27b_source.json", lambda r: r["land"]["END"]),
           ("S808", "s808_s808_27b_smear.json", lambda r: r["land"]["END"]), ("S809", "s809_s809_27b_circuit.json", lambda r: r["land"]["END"]),
           ("S810", "s810_s810_27b_control.json", lambda r: r["land"]["END"]), ("S811", "s811_s811_27b_dissoc.json", lambda r: r["land"]["END"])]
    out = []
    for lab, f, get in SRC:
        for r in json.load(open(f"{R3}/{f}"))["rows"]:
            out.append(dict(src=lab, si=f"{lab}:{r['si']}", N=7, k=int(r["k"]), b=int(get(r))))
    return out


def ll_cells(model, th, cells):
    cache = {}
    def lp(N, k):
        if (N, k) not in cache:
            cache[(N, k)] = np.log(probs(model, th, N, k))
        return cache[(N, k)]
    return np.array([float(lp(c["N"], c["k"])[c["b"]]) for c in cells])


def boot_diff(cells, la, lb, B=4000, seed=1016):
    sis = sorted({c["si"] for c in cells}); idx = {s: [i for i, c in enumerate(cells) if c["si"] == s] for s in sis}
    d = la - lb; rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        pick = rng.choice(len(sis), len(sis)); v.append(sum(d[i] for s in pick for i in idx[sis[s]]))
    return float(d.sum()), [float(x) for x in np.percentile(v, [2.5, 97.5])]


if __name__ == "__main__":
    s801 = load_s801(); t1 = load_test1(); out = dict(test1={}, test2={})
    # test 1: train per k on all S801 cells with that k, predict N = 7 cells of the other experiments
    tot = {m: np.zeros(0) for m in NP}; allcells = []
    for k in (2, 3, 4, 5, 6):
        tr = [(c["N"], c["k"], c["b"]) for c in s801 if c["k"] == k]
        te = [c for c in t1 if c["k"] == k]
        if not te:
            continue
        th = {m: fit(m, tr) for m in NP}
        ll = {m: ll_cells(m, th[m], te) for m in NP}
        res = dict(n_train=len(tr), n_test=len(te), ll={m: float(ll[m].sum()) for m in NP})
        for a, b in (("MX", "G"), ("LC", "G"), ("MX", "LC")):
            res[f"{a}-{b}"] = boot_diff(te, ll[a], ll[b])
        res["pred_N7"] = {m: probs(m, th[m], 7, k).round(3).tolist() for m in NP}
        obs = np.bincount([c["b"] for c in te], minlength=8); res["obs_N7"] = (obs / obs.sum()).round(3).tolist()
        out["test1"][k] = res
        for m in NP:
            tot[m] = np.concatenate([tot[m], ll[m]])
        allcells += te
        print(f"[test1] k{k}: train {len(tr)} test {len(te)} | LL G {res['ll']['G']:.1f} MX {res['ll']['MX']:.1f} LC {res['ll']['LC']:.1f} | "
              f"MX-G {res['MX-G'][0]:+.1f} {np.round(res['MX-G'][1], 1)} · LC-G {res['LC-G'][0]:+.1f} {np.round(res['LC-G'][1], 1)} · MX-LC {res['MX-LC'][0]:+.1f} {np.round(res['MX-LC'][1], 1)}")
        print(f"         obs {res['obs_N7']}\n         G   {res['pred_N7']['G']}\n         MX  {res['pred_N7']['MX']}\n         LC  {res['pred_N7']['LC']}")
    keep = [i for i, c in enumerate(allcells) if c["k"] in (2, 3, 5, 6)]
    sub = [allcells[i] for i in keep]
    out["test1"]["sum_k2356"] = {f"{a}-{b}": boot_diff(sub, tot[a][keep], tot[b][keep]) for a, b in (("MX", "G"), ("LC", "G"), ("MX", "LC"))}
    out["test1"]["sum_k2356"]["ll"] = {m: float(tot[m][keep].sum()) for m in NP}
    print("[test1] sum k2,3,5,6:", {k_: (round(v[0], 1), [round(x, 1) for x in v[1]]) if isinstance(v, tuple) else v for k_, v in out["test1"]["sum_k2356"].items()})
    # test 2: leave one N out within S801, k = 1..5
    for Nout in (6, 7, 8):
        r2 = {}
        for k in (1, 2, 3, 4, 5):
            tr = [(c["N"], c["k"], c["b"]) for c in s801 if c["k"] == k and c["N"] != Nout]
            te = [c for c in s801 if c["k"] == k and c["N"] == Nout]
            th = {m: fit(m, tr, starts=16) for m in NP}
            ll = {m: float(ll_cells(m, th[m], te).sum()) for m in NP}
            acc = {m: float(probs(m, th[m], Nout, k)[Nout + 1 - k]) for m in NP}
            obs_acc = float(np.mean([c["b"] == Nout + 1 - k for c in te]))
            r2[k] = dict(ll=ll, pred_acc=acc, obs_acc=obs_acc)
        out["test2"][Nout] = r2
        print(f"[test2] N{Nout} held out: " + " | ".join(f"k{k} LL G {v['ll']['G']:.1f} MX {v['ll']['MX']:.1f} LC {v['ll']['LC']:.1f} acc obs {v['obs_acc']:.2f} G {v['pred_acc']['G']:.2f} MX {v['pred_acc']['MX']:.2f} LC {v['pred_acc']['LC']:.2f}" for k, v in r2.items()))
    json.dump(out, open(f"{O}/f_predictive.json", "w"), indent=1, default=float)
