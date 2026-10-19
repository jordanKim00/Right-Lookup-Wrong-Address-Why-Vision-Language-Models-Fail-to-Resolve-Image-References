"""S805 (descriptive re-analysis of saved S705a QK captures, no model call; proposed by the user 2026-10-15, not pre-registered): is the
blurred far end-relative lookup of Qwen3.5-27B (A) diffuse within each head, (B) sharp heads that disagree, or (C) variable across scenes?
For each lookup layer L of the S786 index-matching heads (39, 43, 47, 51, 55; L35 has one head), each question (START t, END k, natural
wording m0, N = 7, 30 scenes) and the answer row, p[s, h] = the head's attention over the 7 image blocks (softmax over the 462 image-token
logits, summed per block).  Exact decomposition of the entropy of the grand mean (nats, max ln 7 = 1.95):
    H(E_{s,h} p) = H_within + D_heads + D_scenes
    H_within = E_{s,h} H(p[s,h]);  D_heads = E_s [H(E_h p[s,h]) - E_h H(p[s,h])];  D_scenes = H(E_{s,h} p) - E_s H(E_h p[s,h]).
Also per (scene, head) the peak block: on the target j, on the literal block k, on another block, and the head-to-head peak agreement within a scene.
QK only, so the S782 readout defect does not apply.  Usage: python3 exp/s805_entropy_decomp.py -> results/s805_entropy_decomp.json"""
import json, collections
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm"
J = json.load(open(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.json"))
Z = np.load(f"{ROOT}/results/r3/s705a_qa_27b_qkanatomy.npz")
MH = collections.defaultdict(list)
for l, h in json.load(open(f"{ROOT}/exp/s786_heads/27b.json"))["matchers"]:
    MH[int(l)].append(int(h))
N, ROW = 7, 2                                                    # rows: qm, tail1, ans
H = lambda p: float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(-1).mean()) if p.ndim > 1 else float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())


def block_dist(si, q, L):
    lg = Z[f"{si}|{q}|{L}|limg"][ROW].astype(np.float64)                     # [24 heads, 462]
    own = np.array(J["meta"][si]["owner"])
    w = np.exp(lg - lg.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
    return np.stack([w[:, own == b].sum(-1) for b in range(N)], -1)          # [24, 7]


out = {}
for L in (39, 43, 47, 51, 55):
    hs = MH[L]
    for fr in ("S", "E"):
        for t in range(1, N + 1):
            q = f"{fr}{t}m0"; tgt = t if fr == "S" else N + 1 - t; lit = t
            P = np.stack([block_dist(si, q, L)[hs] for si in range(len(J["meta"]))])   # [scenes, heads, 7]
            Hw = float(np.mean([[H(P[s, h]) for h in range(P.shape[1])] for s in range(P.shape[0])]))
            Hs = float(np.mean([H(P[s].mean(0)) for s in range(P.shape[0])]))
            Hg = H(P.mean((0, 1)))
            pk = P.argmax(-1)                                                   # [scenes, heads]
            agree = float(np.mean([np.mean([pk[s, a] == pk[s, b] for a in range(len(hs)) for b in range(len(hs)) if a < b]) for s in range(P.shape[0])]))
            out[f"L{L}|{q}"] = dict(L=L, frame=fr, t=t, target=tgt, n_heads=len(hs), H_total=Hg, H_within=Hw, D_heads=Hs - Hw, D_scenes=Hg - Hs,
                                    share_target=float(P[..., tgt - 1].mean()), share_literal=float(P[..., lit - 1].mean()),
                                    peak_target=float(np.mean(pk == tgt - 1)), peak_literal=float(np.mean(pk == lit - 1)),
                                    peak_other=float(np.mean((pk != tgt - 1) & (pk != lit - 1))), peak_agree=agree,
                                    peak_hist=np.bincount(pk.ravel(), minlength=N).tolist())
json.dump(out, open(f"{ROOT}/research/ces/outputs/s805_entropy_decomp.json", "w"), indent=1)
for L in (39, 43, 47, 51, 55):
    print(f"== L{L} ({len(MH[L])} heads)   H_total = H_within + D_heads + D_scenes | target share · peak on target / literal / other · head agreement")
    for fr in ("S", "E"):
        for t in range(1, N + 1):
            r = out[f"L{L}|{fr}{t}m0"]
            print(f"  {'START' if fr == 'S' else 'END'}({t}) tgt {r['target']}: {r['H_total']:.2f} = {r['H_within']:.2f} + {r['D_heads']:.2f} + {r['D_scenes']:.2f}"
                  f" | tgt {r['share_target']:.2f} · peaks {r['peak_target']:.2f}/{r['peak_literal']:.2f}/{r['peak_other']:.2f} · agree {r['peak_agree']:.2f}"
                  f" | hist {r['peak_hist']}")
