"""S885 analysis (rules fixed before the data; knowledge/DECISIONS.md S885 entry): the composition map of FA47 with the L51 lookup, from the
model's own weights and the captured states.  Inputs results/r3/s885_<tag>_compcap.json + per-scene npz, the weights of layers 47 (o_proj)
and 51 (input_layernorm, q_proj, q_norm) read from the safetensors, S878 (same scenes / pins: captured L51 queries and keys) and S882
(same scenes / pins: the causal KEEP_FA47 arm).
Late query map at L51 for read row r: q(h) = RoPE(q_norm(W_Q^q RMSNorm(h))) (RMSNorm with (1 + w), the query half of each head, RoPE on the
first 64 dims with the captured cos / sin), linearised at the midpoint of the E_BANDR and E read-row residuals (forward-mode, float64; amended
before data from the E_BANDR point after the smoke's instrument check): dq = J dh.
Block readout (linear): dz_b = mean over rows B1 / B4 / answer and the L51 matcher heads h' of dq_h' . kt_b,g(h') / 16, kt_b = the block's keys
weighted by the operating point's within-block softmax (q(midpoint) . k / 16).
FA47 direct path: dh_X = W_O^47 o^X (per head and part: the gated per-part outputs of run X); writer part dh = dh_E - dh_R.
Composition matrix of head h: C_h(b -> b') = dz_b' of J W_O^h (sigmoid(g_h^E) * vbar_b,g(h)) (head h reading picture b's block-mean value
under E's gate), centred over b'; copy score cs_h = mean_b C_h(b -> b); argmax accuracy acc_h = P(argmax_b' C_h(b -> b') = b).
Scene bootstrap 4,000 (seed 885) over (scene, pin, row).
I0: t1 >= .95; START(p) answers p >= .95; L47 anchor = p for E and R >= .90; the offline query map reproduces S878's captured L51 query of
  the same run (median cosine >= .999, E_BANDR rows) and the captured L51 keys match S878's (max |diff| <= .05); the linearised map on the
  whole band effect (dh = h51_E - h51_E_BANDR) reproduces the measured writer effect W(T1) (within-pin Pearson >= .85 over (scene, pin)).
K1: cs_h16 > 0 with the interval above 0 and acc_h16 >= .50 -> 'FA47 h16's output, composed with the L51 lookup, points at the picture it
  reads (content pointer)'; else 'not'.
K2: mean cs over the anti-self heads (0, 2, 5, 13, 15) < 0 with the interval below 0 -> 'the anti-self heads write a negative pointer to the
  picture they read'; else 'not'.
K3: mean cs over the self-address heads (1, 4, 12, 17, 20) > 0 with the interval above 0 -> 'the self-address heads write a positive pointer
  to the picture they read'; else 'not'.
K4: the FA47 direct-path writer effect Wd(T1) = dz_(p-1) - dz_p of J dh (dh = sum over heads and parts of dh_E - dh_R) against the causal
  KEEP_FA47 effect of S882 (same scenes and pins): within-pin Pearson >= .60 with the lower bound > .40 and mean ratio in [.5, 2] -> 'the FA47
  direct path (attention x composition with the L51 lookup) accounts for FA47's causal END effect'; else 'not'.
Descriptive: per-head cs / acc for all 24 heads; Wd by head group and by part (anchor, predecessor, other pictures, text); the split of Wd
into attention-mass change x E's composition vs gate change; the per-scene Pearson of Wd(T1) with W(T1).
Usage: python3 s885_analyze.py TAG [--s878 s878_27b] [--s882 s882_27b] [--model /models/Qwen3.5-27B]"""
from __future__ import annotations

import argparse, collections, json
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--s878", default="s878_27b"); ap.add_argument("--s882", default="s882_27b")
ap.add_argument("--model", default="/models/Qwen3.5-27B"); A = ap.parse_args()
R3 = "/vlm/results/r3"
J = json.load(open(f"{R3}/s885_{A.tag}_compcap.json")); NPD = J["npz_dir"]
OUT = f"{R3}/s885_analysis_{A.tag}.json"
rng = np.random.default_rng(885); NB = 4000
rows = J["rows"]; S = len(rows); PINS = J["pins"]; lm = J["late_mean"]; MH = lm["51"]
SA, AN, H16 = [1, 4, 12, 17, 20], [0, 2, 5, 13, 15], [16]
GRP = 6; NHEAD = 24; HD = 256; RD = 64
idx = json.load(open(f"{A.model}/model.safetensors.index.json"))["weight_map"]
def W_(name):
    with safe_open(f"{A.model}/{idx[name]}", framework="pt") as f:
        return f.get_tensor(name).double()
pre = "model.language_model.layers."
Wo = W_(pre + "47.self_attn.o_proj.weight")                                       # [5120, 24 * 256]
w_in = W_(pre + "51.input_layernorm.weight"); Wq = W_(pre + "51.self_attn.q_proj.weight"); w_qn = W_(pre + "51.self_attn.q_norm.weight")
Wq_q = Wq.view(NHEAD, 2 * HD, -1)[:, :HD, :]                                     # query half of every head [24, 256, 5120]
EPS = 1e-6


def rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]; return torch.cat([-x2, x1], -1)


def qmap(h, cos, sin):
    n = h / torch.sqrt((h * h).mean(-1, keepdim=True) + EPS) * (1 + w_in)
    q = torch.einsum("hdk,k->hd", Wq_q, n)
    q = q / torch.sqrt((q * q).mean(-1, keepdim=True) + EPS) * (1 + w_qn)
    return torch.cat([q[:, :RD] * cos + rot_half(q[:, :RD]) * sin, q[:, RD:]], -1)   # [24, 256]


def lse(x, axis=-1):
    m = x.max(axis, keepdims=True); return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def anchor(q):
    q = np.asarray(q); am = [int(np.argmax(q[i])) + 1 for i in (2, 5, 9)]
    c = collections.Counter(am).most_common(); return c[0][0] if c[0][1] >= 2 else am[2]


def T1cap(r, key, p):
    z = np.asarray(r["zl"][key]["51"], dtype=float)[:, MH].mean(1).mean(0); return z[p - 2] - z[p - 1]


J82 = json.load(open(f"{R3}/s882_{A.s882}_core.json")); r82 = J82["rows"]; lm82 = J82["late_mean"]["51"]
def T1_82(r, key, p):
    z = np.asarray(r["zl"][key]["51"], dtype=float)[:, lm82].mean(1).mean(0); return z[p - 2] - z[p - 1]
J78 = json.load(open(f"{R3}/s878_{A.s878}_qk.json")); NPD78 = J78["npz_dir"]

REC = {}; QCOS = []; KDIFF = []
for si in range(S):
    Z = np.load(f"{NPD}/scene{si:02d}.npz"); Z78 = np.load(f"{NPD78}/scene{si:02d}.npz")
    own = Z["img_owner"].astype(int); K = torch.tensor(Z["k51"].astype(np.float64))      # [T, 4, 256]
    KDIFF.append(float(np.abs(Z["k51"].astype(np.float32) - Z78["kimg_51"].astype(np.float32)).max()))
    cos = torch.tensor(Z["cos51"].astype(np.float64)); sin = torch.tensor(Z["sin51"].astype(np.float64))
    vbar = torch.tensor(Z["vbar47"].astype(np.float64))                                   # [8, 4, 256]
    r = rows[si]
    for p in PINS:
        nmE, nmR, nmB = f"Ep{p}", f"Rp{p}", f"Ep{p}_E_BANDR"
        h0 = torch.tensor(Z[f"h51_{nmB}"].astype(np.float64)); hE = torch.tensor(Z[f"h51_{nmE}"].astype(np.float64))
        oE = torch.tensor(Z[f"opart_{nmE}"].astype(np.float64)); oR = torch.tensor(Z[f"opart_{nmR}"].astype(np.float64))   # [3, 24, 9, 256]
        mE = Z[f"mass_{nmE}"].astype(np.float64); mR = Z[f"mass_{nmR}"].astype(np.float64)
        gE = torch.tensor(Z[f"gate_{nmE}"].astype(np.float64)); gR = torch.tensor(Z[f"gate_{nmR}"].astype(np.float64))     # [3, 24, 256]
        q78 = Z78[f"q_B{p}_51"].astype(np.float64)                                          # [3, 24, 256]
        d = dict(si=si, p=p, W=T1cap(r, nmE, p) - T1cap(r, nmB, p), anch=(anchor(r["qk47"][nmE]), anchor(r["qk47"][nmR])))
        d["Wkeep"] = T1_82(r82[si], f"Ep{p}_KEEP_FA47", p) - T1_82(r82[si], f"Ep{p}_E_BANDR", p)
        per_row = []
        for ri in range(3):
            QCOS.append(float(torch.nn.functional.cosine_similarity(qmap(h0[ri], cos[ri], sin[ri]).flatten(), torch.tensor(q78[ri]).flatten(), 0)))
            hm_ = (h0[ri] + hE[ri]) / 2                                                         # linearisation point: midpoint of E_BANDR and E
            q0 = qmap(hm_, cos[ri], sin[ri])
            # operating-point within-block key weights for the matcher heads
            kt = torch.zeros(len(MH), 8, HD, dtype=torch.float64)
            for j, hh in enumerate(MH):
                kk = K[:, hh // GRP]; s_ = kk @ q0[hh] / 16.0
                for b in range(8):
                    m = torch.tensor(own == b); w = torch.softmax(s_[m], 0); kt[j, b] = (w[:, None] * kk[m]).sum(0)
            def dz(dh):                                                                         # dh [..., 5120] -> [..., 8]
                sh = dh.shape; dh2 = dh.reshape(-1, sh[-1])
                f = lambda x: qmap(x, cos[ri], sin[ri])
                outs = []
                for i0 in range(0, dh2.shape[0], 64):
                    dq = torch.func.vmap(lambda v: torch.func.jvp(f, (hm_,), (v,))[1])(dh2[i0:i0 + 64])        # [n, 24, 256]
                    outs.append(torch.einsum("nhd,hbd->nb", dq[:, MH], kt) / 16.0 / len(MH))
                return torch.cat(outs).reshape(*sh[:-1], 8)
            # whole band effect (instrument)
            band = dz(hE[ri] - h0[ri])
            # FA47 direct path per head and part
            WoH = Wo.view(-1, NHEAD, HD)                                                        # [5120, 24, 256]
            dhE = torch.einsum("ohd,hkd->hko", WoH, oE[ri]); dhR = torch.einsum("ohd,hkd->hko", WoH, oR[ri])          # [24, 9, 5120]
            zE, zR = dz(dhE), dz(dhR)                                                          # [24, 9, 8]
            # composition matrices under E's gate: head h reading picture b's block-mean value
            vb = vbar[:, [h // GRP for h in range(NHEAD)]]                                       # [8, 24, 256]
            comp = dz(torch.einsum("ohd,bhd->hbo", WoH, vb * gE[ri][None]))                      # [24, 8, 8]
            per_row.append(dict(band=band.numpy(), zE=zE.numpy(), zR=zR.numpy(), comp=comp.numpy(), mE=mE[ri], mR=mR[ri]))
        d["rows"] = per_row
        REC[(si, p)] = d
    del Z, Z78
    print("scene", si, flush=True)
SC = sorted({k[0] for k in REC})


def t1v(z, p):
    return z[..., p - 2] - z[..., p - 1]


for d in REC.values():
    p = d["p"]
    d["Wband"] = float(np.mean([t1v(x["band"], p) for x in d["rows"]]))
    wd = np.mean([(x["zE"] - x["zR"]) for x in d["rows"]], 0)                             # [24, 9, 8]
    d["Wd"] = float(t1v(wd.sum((0, 1)), p))
    d["Wd_group"] = {g: float(t1v(wd[hs].sum((0, 1)), p)) for g, hs in (("SELFADDR", SA), ("ANTI", AN), ("H16", H16),
                                                                       ("REST", [h for h in range(24) if h not in SA + AN + H16]))}
    parts = {"anchor": [p - 1], "pred": [p - 2], "other_pics": [b for b in range(8) if b not in (p - 1, p - 2)], "text": [8]}
    d["Wd_part"] = {k: float(t1v(wd[:, v].sum((0, 1)), p)) for k, v in parts.items()}
    cm = np.mean([x["comp"] for x in d["rows"]], 0)                                        # [24, 8, 8]
    cm = cm - cm.mean(-1, keepdims=True)
    d["cs"] = np.array([np.mean([cm[h, b, b] for b in range(8)]) for h in range(24)])
    d["acc"] = np.array([np.mean([int(np.argmax(cm[h, b]) == b) for b in range(8)]) for h in range(24)])
    # attention-mass change x E's composition (direct path, pictures only) vs the rest (gate change, within-block reweighting, text)
    mE = np.mean([x["mE"] for x in d["rows"]], 0); mR = np.mean([x["mR"] for x in d["rows"]], 0)       # [24, 9]
    cr = np.mean([x["comp"] for x in d["rows"]], 0)                                        # [24, 8, 8] uncentred
    att = np.einsum("hb,hbc->c", mE[:, :8] - mR[:, :8], cr)
    d["Wd_attention_part"] = float(t1v(att, p))


def bt(fn):
    n = len(SC); est = fn(np.arange(n)); bs = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return dict(est=float(est), lo=float(lo), hi=float(hi))


def corr(ix, k1, k2):
    x, y = [], []
    for p in PINS:
        u = np.array([REC[(SC[i], p)][k1] for i in ix]); w = np.array([REC[(SC[i], p)][k2] for i in ix])
        x += list(u - u.mean()); y += list(w - w.mean())
    return float(np.corrcoef(x, y)[0, 1])


res = dict(tag=A.tag, n_scenes=S)
st = float(np.mean([r["land"][f"START{p}"] == p for r in rows for p in PINS]))
reach = float(np.mean([a_ == d["p"] for d in REC.values() for a_ in d["anch"]]))
rb = corr(range(len(SC)), "Wband", "W")
I0 = dict(t1=bool((J["t1_menu"] or 0) >= .95), START=bool(st >= .95), pin_reach=bool(reach >= .90), qmap=bool(np.median(QCOS) >= .999),
          keys=bool(max(KDIFF) <= .05), band_linear=bool(rb >= .85))
res["I0"] = dict(**I0, all=bool(all(I0.values())), t1_value=J["t1_menu"], START_acc=st, reach=reach, qcos_median=float(np.median(QCOS)),
                 qcos_min=float(np.min(QCOS)), key_maxdiff=float(max(KDIFF)), r_band_W=rb,
                 ratio_band_W=float(np.mean([d["Wband"] for d in REC.values()]) / np.mean([d["W"] for d in REC.values()])))
ok = res["I0"]["all"]
def mcs(ix, hs):
    return float(np.mean([REC[(SC[i], p)]["cs"][hs].mean() for i in ix for p in PINS]))
def macc(ix, hs):
    return float(np.mean([REC[(SC[i], p)]["acc"][hs].mean() for i in ix for p in PINS]))
k1, k1a = bt(lambda ix: mcs(ix, H16)), macc(range(len(SC)), H16)
res["K1"] = dict(cs=k1, acc=k1a, verdict=(("FA47 h16's output, composed with the L51 lookup, points at the picture it reads (content pointer)"
                                          if k1["lo"] > 0 and k1a >= .50 else "not") if ok else "not read"))
k2 = bt(lambda ix: mcs(ix, AN))
res["K2"] = dict(cs=k2, acc=macc(range(len(SC)), AN), verdict=(("the anti-self heads write a negative pointer to the picture they read" if k2["hi"] < 0 else "not")
                                                             if ok else "not read"))
k3 = bt(lambda ix: mcs(ix, SA))
res["K3"] = dict(cs=k3, acc=macc(range(len(SC)), SA), verdict=(("the self-address heads write a positive pointer to the picture they read" if k3["lo"] > 0 else "not")
                                                             if ok else "not read"))
k4r = bt(lambda ix: corr(ix, "Wd", "Wkeep"))
k4m = float(np.mean([d["Wd"] for d in REC.values()]) / np.mean([d["Wkeep"] for d in REC.values()]))
res["K4"] = dict(r=k4r, ratio=k4m, verdict=(("the FA47 direct path (attention x composition with the L51 lookup) accounts for FA47's causal END effect"
                                             if k4r["est"] >= .60 and k4r["lo"] > .40 and .5 <= k4m <= 2 else "not") if ok else "not read"))
allx = range(len(SC))
res["descriptive"] = dict(
    cs_by_head={h: float(np.mean([d["cs"][h] for d in REC.values()])) for h in range(24)},
    acc_by_head={h: float(np.mean([d["acc"][h] for d in REC.values()])) for h in range(24)},
    Wd_mean=float(np.mean([d["Wd"] for d in REC.values()])), W_mean=float(np.mean([d["W"] for d in REC.values()])),
    Wkeep_mean=float(np.mean([d["Wkeep"] for d in REC.values()])), r_Wd_W=corr(allx, "Wd", "W"),
    Wd_group={g: float(np.mean([d["Wd_group"][g] for d in REC.values()])) for g in ("SELFADDR", "ANTI", "H16", "REST")},
    Wd_part={k: float(np.mean([d["Wd_part"][k] for d in REC.values()])) for k in ("anchor", "pred", "other_pics", "text")},
    Wd_attention_part=float(np.mean([d["Wd_attention_part"] for d in REC.values()])),
    by_pin={f"p{p}": dict(W=float(np.mean([REC[(s, p)]["W"] for s in SC])), Wd=float(np.mean([REC[(s, p)]["Wd"] for s in SC])),
                          Wkeep=float(np.mean([REC[(s, p)]["Wkeep"] for s in SC]))) for p in PINS})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(dict(I0=res["I0"]["all"], K1=[res["K1"]["verdict"], k1["est"], k1a], K2=[res["K2"]["verdict"], k2["est"]],
                      K3=[res["K3"]["verdict"], k3["est"]], K4=[res["K4"]["verdict"], k4r["est"], k4m]), indent=1))
print("->", OUT)
