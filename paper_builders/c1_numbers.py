import json, sys
sys.path.insert(0, "/vlm/exp/figs")
import numpy as np
import make_c_figs as F
R = "/vlm/results/r3"
for tag, hf in (("qa_4b", f"{R}/s720_4b_heads.json"), ("qa_9b", f"{R}/s720_9b_heads.json"), ("qa_27b", f"{R}/s705a_qa_27b_matchers.json")):
    heads = [tuple(x) for x in json.load(open(hf))["matchers"]]
    for ri, rn in ((2, "ans"), (1, "tail1")):
        S, E = F.qk_tables(tag, heads, ri=ri)
        N = 7
        diag = np.mean([S[t, t] for t in range(N)])
        tgt = {k: E[k - 1, N - k] for k in range(1, N + 1)}; lit = {k: E[k - 1, k - 1] for k in range(1, N + 1)}
        print(tag, rn, "START diag %.3f" % diag, "| END target", " ".join(f"k{k}:{tgt[k]:.2f}" for k in tgt), "| literal", " ".join(f"k{k}:{lit[k]:.2f}" for k in lit))
