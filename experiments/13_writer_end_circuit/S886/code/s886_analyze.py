"""S886 analysis (rules fixed before the data; knowledge/DECISIONS.md S886 entry): which band component writes which content into the late
query?  Inputs results/r3/s886_<tag>_compmap.json + per-scene npz and the layer-51 weights (input_layernorm, q_proj, q_norm; safetensors).
Late query map at L51 linearised at the midpoint of the E and E_BANDR read-row residuals (as S885): dq = J dh.  Component write difference
d_c = out_c(E) - out_c(R) at rows B1 / B4 / answer for c in {DN46, MLP46, FA47, MLP47, DN48, MLP48, DN49, MLP49} (each E output already contains
the effect of upstream E outputs; the sum over c is exactly the band's read-row write difference between E and E_BANDR).
Direct path of component c: dq_c = J d_c; its block effect with the block-mean L51 keys kbar: dz_c(b) = mean over rows and the L51 matcher
heads of dq_c . kbar_b / 16; W_c = dz_c(p-1) - dz_c(p).
Crossed content analysis per component (as post hoc p21, now registered): W_c(A, B) = mean dq_c,A . (kbar_B,p-1 - kbar_B,p) / 16 for every
ordered pair of scenes of the same pin; two-way demeaning (query source, key source) within pin; OLS on the centred block-mean key cosine
similarities aa (A's anchor ~ B's anchor), pp (predecessors), x1 (A's predecessor ~ B's anchor), x2 (A's anchor ~ B's predecessor).
Predecessor-content signature P_c = (beta_pp - beta_x1) / 2; anchor-content signature A_c = (beta_aa - beta_x2) / 2 (the predecessor-minus-
anchor content vector gives P, A > 0).  Scene bootstrap 2,000 (seed 886; pairs among the resampled scenes).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the summed component direct path reproduces the measured writer
  effect W(T1) = T1(E) - T1(E_BANDR) at L51 (within-pin Pearson >= .80 over (scene, pin)); the summed signatures are positive (P_sum and A_sum
  > 0 with intervals above 0).
C1 (predecessor content): share_c = P_c / sum over components of max(P_c', 0); the largest share >= .50 with the lower bound > .30 -> 'component c
  writes the predecessor content'; else 'distributed'.
C2 (anchor content): the same with A_c -> 'component c writes the anchor content (suppression)'; else 'distributed'.
Descriptive: per-component W_c (mean, share of the summed direct path, within-pin Pearson with W), P_c, A_c, at L51.
Usage: python3 s886_analyze.py TAG [--model /models/Qwen3.5-27B]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--model", default="/models/Qwen3.5-27B"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s886_{A.tag}_compmap.json")); NPD = J["npz_dir"]
OUT = f"{R3}/s886_analysis_{A.tag}.json"
rng = np.random.default_rng(886); NB = 2000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; MH = lm["51"]
COMPS = ["mix46", "mlp46", "mix47", "mlp47", "mix48", "mlp48", "mix49", "mlp49"]
NAMES = dict(mix46="DN46", mlp46="MLP46", mix47="FA47", mlp47="MLP47", mix48="DN48", mlp48="MLP48", mix49="DN49", mlp49="MLP49")
GRP = 6; NHEAD = 24; HD = 256; RD = 64; EPS = 1e-6
idx = json.load(open(f"{A.model}/model.safetensors.index.json"))["weight_map"]
def W_(name):
    with safe_open(f"{A.model}/{idx[name]}", framework="pt") as f:
        return f.get_tensor(name).double()
pre = "model.language_model.layers."
w_in = W_(pre + "51.input_layernorm.weight"); Wq = W_(pre + "51.self_attn.q_proj.weight"); w_qn = W_(pre + "51.self_attn.q_norm.weight")
Wq_q = Wq.view(NHEAD, 2 * HD, -1)[:, :HD, :]


def rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]; return torch.cat([-x2, x1], -1)


def qmap(h, cos, sin):
    n = h / torch.sqrt((h * h).mean(-1, keepdim=True) + EPS) * (1 + w_in)
    q = torch.einsum("hdk,k->hd", Wq_q, n)
    q = q / torch.sqrt((q * q).mean(-1, keepdim=True) + EPS) * (1 + w_qn)
    return torch.cat([q[:, :RD] * cos + rot_half(q[:, :RD]) * sin, q[:, RD:]], -1)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def T1cap(r, key, p):
    z = np.asarray(r["zl"][key]["51"], dtype=float)[:, MH].mean(1).mean(0); return z[p - 2] - z[p - 1]


DQ = {}; KB = {}; REC = {}
for si in range(S):
    Z = np.load(f"{NPD}/scene{si:02d}.npz"); own = Z["img_owner"].astype(int)
    K = torch.tensor(Z["k51"].astype(np.float64))
    KB[si] = torch.stack([K[torch.tensor(own == b)].mean(0) for b in range(8)])                  # [8, 4, 256]
    cos = torch.tensor(Z["cos51"].astype(np.float64)); sin = torch.tensor(Z["sin51"].astype(np.float64))
    r = rows[si]
    for p in PINS:
        nE, nR, nB = f"Ep{p}", f"Rp{p}", f"Ep{p}_E_BANDR"
        hm = (torch.tensor(Z[f"h51_{nE}"].astype(np.float64)) + torch.tensor(Z[f"h51_{nB}"].astype(np.float64))) / 2
        D = torch.stack([torch.tensor(Z[f"{c}_{nE}"].astype(np.float64)) - torch.tensor(Z[f"{c}_{nR}"].astype(np.float64)) for c in COMPS], 1)  # [3, 8c, 5120]
        dq = []
        for ri in range(3):
            f = lambda x: qmap(x, cos[ri], sin[ri])
            dq.append(torch.func.vmap(lambda v: torch.func.jvp(f, (hm[ri],), (v,))[1])(D[ri])[:, MH])   # [8c, H', 256]
        dq = torch.stack(dq)                                                                       # [3, 8c, H', 256]
        DQ[(si, p)] = dq
        REC[(si, p)] = dict(si=si, p=p, W=T1cap(r, nE, p) - T1cap(r, nB, p), anch=(anchor(r["qk47"][nE]), anchor(r["qk47"][nR])))
    del Z
    print("scene", si, flush=True)
kvh = torch.tensor([h // GRP for h in MH])
def dz(dq, kb):                                                                                    # dq [3, C, H', 256], kb [8, 4, 256] -> [C, 8]
    return torch.einsum("rchd,bhd->cb", dq, kb[:, kvh]) / 16.0 / dq.shape[0] / dq.shape[2]
for (si, p), d in REC.items():
    z = dz(DQ[(si, p)], KB[si]).numpy()
    d["Wc"] = z[:, p - 2] - z[:, p - 1]
SC = sorted({k[0] for k in REC})
# crossed matrices per component and pin
KBc = torch.stack([KB[s] for s in range(S)]); KBc = KBc - KBc.mean(0, keepdim=True)
def cs(x, y):
    return float((x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum()))
CR = {}
for p in PINS:
    Wm = np.zeros((len(COMPS), S, S))
    for b_ in range(S):
        dk = (KB[b_][p - 2] - KB[b_][p - 1])[kvh]                                                  # [H', 256]
        for a_ in range(S):
            Wm[:, a_, b_] = (torch.einsum("rchd,hd->c", DQ[(a_, p)], dk) / 16.0 / 3 / len(MH)).numpy()
    sim = dict(aa=np.array([[cs(KBc[a, p - 1], KBc[b, p - 1]) for b in range(S)] for a in range(S)]),
               pp=np.array([[cs(KBc[a, p - 2], KBc[b, p - 2]) for b in range(S)] for a in range(S)]),
               x1=np.array([[cs(KBc[a, p - 2], KBc[b, p - 1]) for b in range(S)] for a in range(S)]),
               x2=np.array([[cs(KBc[a, p - 1], KBc[b, p - 2]) for b in range(S)] for a in range(S)]))
    CR[p] = (Wm, sim)


def fit(ix):
    ix = np.asarray(ix); X, Y = [], []
    off = ix[:, None] != ix[None, :]
    def dm(M):
        M = np.where(off, M, np.nan); M = M - np.nanmean(M, 1, keepdims=True); M = M - np.nanmean(M, 0, keepdims=True); return M[off]
    for p in PINS:
        Wm, sim = CR[p]; sub = np.ix_(ix, ix)
        Y.append(np.stack([dm(Wm[c][sub]) for c in range(len(COMPS))], 1))
        X.append(np.stack([dm(sim[k][sub]) for k in ("aa", "pp", "x1", "x2")], 1))
    X = np.concatenate(X); Y = np.concatenate(Y)
    return np.linalg.lstsq(X, Y, rcond=None)[0].T                                                 # [C, 4]


def sig(beta):
    return (beta[:, 1] - beta[:, 2]) / 2, (beta[:, 0] - beta[:, 3]) / 2                            # P_c, A_c


b0 = fit(np.arange(S)); P0, A0 = sig(b0)
BS = [fit(rng.integers(0, S, S)) for _ in range(NB)]
Pb = np.array([sig(b)[0] for b in BS]); Ab = np.array([sig(b)[1] for b in BS])
def share(Pv):
    pos = np.clip(Pv, 0, None); return Pv / pos.sum(-1, keepdims=True)
shP0, shA0 = share(P0[None])[0], share(A0[None])[0]; shPb, shAb = share(Pb), share(Ab)


def corr(ix, f1, f2):
    x, y = [], []
    for p in PINS:
        u = np.array([f1(REC[(SC[i], p)]) for i in ix]); w = np.array([f2(REC[(SC[i], p)]) for i in ix])
        x += list(u - u.mean()); y += list(w - w.mean())
    return float(np.corrcoef(x, y)[0, 1])


res = dict(tag=A.tag, n_scenes=S, components=[NAMES[c] for c in COMPS])
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
rsum = corr(range(len(SC)), lambda d: float(d["Wc"].sum()), lambda d: d["W"])
Ps, As_ = Pb.sum(1), Ab.sum(1)
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), direct_sum=bool(rsum >= .80),
          signatures=bool(np.percentile(Ps, 2.5) > 0 and np.percentile(As_, 2.5) > 0))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, r_directsum_W=rsum,
                 ratio_directsum_W=float(np.mean([d["Wc"].sum() for d in REC.values()]) / np.mean([d["W"] for d in REC.values()])),
                 P_sum=dict(est=float(P0.sum()), lo=float(np.percentile(Ps, 2.5)), hi=float(np.percentile(Ps, 97.5))),
                 A_sum=dict(est=float(A0.sum()), lo=float(np.percentile(As_, 2.5)), hi=float(np.percentile(As_, 97.5))))
ok = res["I0"]["all"]
def verdict(sh0, shb, what):
    c = int(np.argmax(sh0)); lo = float(np.percentile(shb[:, c], 2.5))
    v = (f"{NAMES[COMPS[c]]} writes the {what}" if sh0[c] >= .50 and lo > .30 else "distributed") if ok else "not read"
    return dict(top=NAMES[COMPS[c]], share=float(sh0[c]), lo=lo, hi=float(np.percentile(shb[:, c], 97.5)), verdict=v)
res["C1"] = verdict(shP0, shPb, "predecessor content")
res["C2"] = verdict(shA0, shAb, "anchor content (suppression)")
allx = range(len(SC))
res["descriptive"] = {NAMES[c]: dict(W_mean=float(np.mean([d["Wc"][i] for d in REC.values()])),
                                     W_share=float(np.mean([d["Wc"][i] for d in REC.values()]) / np.mean([d["Wc"].sum() for d in REC.values()])),
                                     r_with_W=corr(allx, lambda d, i=i: float(d["Wc"][i]), lambda d: d["W"]),
                                     P=float(P0[i]), P_lo=float(np.percentile(Pb[:, i], 2.5)), P_hi=float(np.percentile(Pb[:, i], 97.5)),
                                     A=float(A0[i]), A_lo=float(np.percentile(Ab[:, i], 2.5)), A_hi=float(np.percentile(Ab[:, i], 97.5)),
                                     beta=dict(zip(("aa", "pp", "x1", "x2"), map(float, b0[i]))))
                         for i, c in enumerate(COMPS)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], C1=[res["C1"]["verdict"], res["C1"]["top"], res["C1"]["share"]],
                      C2=[res["C2"]["verdict"], res["C2"]["top"], res["C2"]["share"]]), indent=1))
print("->", OUT)
