"""S626 (3): where the effect sits, layer by layer -- shares of each AtP map in token groups (target block 5, surface block 3,
other blocks, the ordinal word, the other reference words, the candidate names, the rest of the question, the chat tail, the
answer position) and the handover depth (first layer at which the answer position holds >= 50% of the layer's total effect).
At one layer the patches of all positions together are the whole input from that layer on, so the shares of a layer add up to
the (linearised) full-swap effect.
Usage: python3 exp/s626_flow.py [TAG ...]  ->  results/s626_flow.json
"""
import json, os, sys
import numpy as np

R = "/home/storage2/ai26269345/vlm/results/r3"
TAGS = sys.argv[1:] or ["atp_4b", "atp_8b", "atp2_35_4b", "atp2_35_9b", "atp_27b", "atp_4b_r3", "atpseg_35_4b_r3", "atpseg_35_9b_r3"]


def groups(d):
    toks, B = d["toks"], d["blocks"]
    T = len(toks); g = {}
    for b, (s, e) in enumerate(B):
        g.setdefault("tgt" if b == 4 else "surf" if b == 2 else "blk", []).extend(range(s - 1, e + 2))
    s0 = B[-1][1] + 2
    words = [t.strip() for t in toks]
    q_end = next(i for i in range(s0, T) if words[i] == "?")
    colon = next(i for i in range(q_end, T) if words[i] == ":")
    dot = max(i for i in range(colon, T) if words[i] == ".")
    for i in range(s0, T):
        if i == T - 1:
            k = "last"
        elif i < q_end + 1:
            k = "ord" if words[i] in ("third", "second") else "ref" if words[i] in ("image", "from", "the", "end", "?") else "q"
        elif i <= colon:
            k = "q"
        elif i < dot:
            k = "cand"
        else:
            k = "tail"
        g.setdefault(k, []).append(i)
    return g


out = {}
for tag in TAGS:
    f = f"{R}/s626_{tag}_atp.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); g = groups(d); D = d["D"]
    res = {}
    for m in ("A", "B", "Cc", "Cs"):
        E = np.array(d["maps"][m])
        tot = E.sum(1)
        sh = {k: (E[:, ix].sum(1) / np.where(np.abs(tot) > 1e-6, tot, np.nan)).tolist() for k, ix in g.items()}
        last = np.array(sh["last"])
        hand = next((l for l in range(D) if np.isfinite(last[l]) and last[l] >= .5), None)
        res[m] = dict(share=sh, layer_total=tot.tolist(), handover=hand, handover_rel=(hand / (D - 1) if hand is not None else None),
                      full=d["total"][m])
        line = " · ".join(f"{k} " + "/".join(f"{sh[k][l]:+.2f}" for l in (int(.25 * D), int(.5 * D), int(.75 * D), D - 2)) for k in ("tgt", "surf", "ord", "ref", "cand", "tail", "last") if k in sh)
        print(f"{d['tag'] if 'tag' in d else tag:>16s} {m:>2s} 전체 {d['total'][m]:+6.2f} · 넘김 L{hand} ({(hand or 0) / (D - 1):.2f}) · 몫 @25/50/75%/끝-1: {line}")
    out[tag] = dict(D=D, behaviour=d["behaviour"], maps=res)
json.dump(out, open("/home/storage2/ai26269345/vlm/results/s626_flow.json", "w"))
print("-> results/s626_flow.json")
