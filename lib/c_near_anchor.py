"""agent_math_1015 / c_near_anchor (descriptive CPU re-analysis, not registered; second task C).  Near-anchor rule (k = 2) from every
saved code-transplant record with clean readouts: S807 (CODE6, CODE8, BOTH8; count sentence N = 7), S809 (CODE6 arm), S810 (A5_CODE5),
S549 re-measured in S787 (m1, p1, id, none; no count sentence; several N), S545 re-measured in S787 (pre = all codes +2, app control; count
sentence).  Each condition is a stratum defined by its block-code vector; per stratum: max-code position(s), the (max - 1)-code candidates,
duplicates, whether a candidate precedes the max-code block, each hypothesis's predicted block, and the observed landing share there.
Hypotheses: H_max (block whose code = max - 1), H_conj (that code AND before the first max-code block), H_predmax (block just before the first
max-code block), H_last (code = last block's code - 1), H_pos (block N - 1).  Output: research/ces/outputs/agent_math_1015/c_near_anchor.json"""
import json
from collections import Counter

R3 = "/home/storage2/ai26269345/vlm/results/r3"; RES = "/home/storage2/ai26269345/vlm/results"


def codes_for(cond, N):
    base = list(range(1, N + 1))
    return {"none": base, "id": base, "END": base, "CODE6": base[:-1] + [N - 1], "m1": base[:-1] + [N - 1], "CODE8": base[:-1] + [N + 1],
            "p1": base[:-1] + [N + 1], "BOTH8": base[:-1] + [N + 1], "A5_CODE5": base[:-1] + [N - 2], "pre": [c + 2 for c in base],
            "app": base}[cond]


def predict(codes):
    N = len(codes); mx = max(codes); first_max = codes.index(mx) + 1
    cand = [b + 1 for b, c in enumerate(codes) if c == mx - 1]
    p = {}
    p["H_max"] = cand if cand else "blur"
    conj = [b for b in cand if b < first_max]
    p["H_conj"] = conj if conj else "blur"
    p["H_predmax"] = [first_max - 1] if first_max > 1 else "blur"
    last = [b + 1 for b, c in enumerate(codes) if c == codes[-1] - 1]
    p["H_last"] = last if last else "blur"
    p["H_pos"] = [N - 1]
    info = dict(codes=codes, max_code=mx, max_blocks=[b + 1 for b, c in enumerate(codes) if c == mx], cand_blocks=cand,
                n_cand=len(cand), dup_max=sum(c == mx for c in codes) > 1, cand_before_max=any(b < first_max for b in cand))
    return p, info


def collect():
    strata = {}
    def add(key, N, land):
        strata.setdefault(key, dict(N=N, lands=[]))["lands"].append(int(land))
    for r in json.load(open(f"{R3}/s807_s807_27b_source.json"))["rows"]:
        if r["k"] == 2:
            for a in ("END", "CODE6", "CODE8", "BOTH8"):
                add(f"S807|{a}", 7, r["land"][a])
    for r in json.load(open(f"{R3}/s809_s809_27b_circuit.json"))["rows"]:
        if r["k"] == 2:
            for a in ("END", "CODE6"):
                add(f"S809|{a}", 7, r["land"][a])
    for r in json.load(open(f"{R3}/s810_s810_27b_control.json"))["rows"]:
        if r["k"] == 2:
            for a in ("END", "A5_CODE5"):
                add(f"S810|{a}", 7, r["land"][a])
    for f, tag, conds in (("s549_endshift_s787_27b.json", "S549", ("none", "id", "m1", "p1")), ("s545_indexshift_s787_27b.json", "S545", ("none", "app", "pre"))):
        for r in json.load(open(f"{RES}/{f}"))["rows"]:
            if int(r["k"]) == 2:
                for c in conds:
                    add(f"{tag}|{c}|N{r['N']}", int(r["N"]), r[f"R|{c}"])
    return strata


if __name__ == "__main__":
    S = collect(); out = {}
    for key, v in sorted(S.items()):
        cond = key.split("|")[1]; N = v["N"]; codes = codes_for(cond, N)
        pred, info = predict(codes); n = len(v["lands"]); cnt = Counter(v["lands"])
        dist = [cnt.get(b, 0) for b in range(0, N + 1)]
        score = {}
        for h, pb in pred.items():
            if pb == "blur":
                score[h] = dict(pred="blur", max_share=max(dist[1:]) / n)
            else:
                score[h] = dict(pred=pb, share=sum(cnt.get(b, 0) for b in pb) / n)
        out[key] = dict(n=n, dist=dist, info=info, scores=score)
        print(f"{key:18s} n {n:3d} codes {codes} | dist(0..N) {dist}")
        print("     " + " · ".join(f"{h}: {s['pred']} -> {s.get('share', s.get('max_share')):.2f}{' (max share)' if s['pred'] == 'blur' else ''}" for h, s in score.items()))
    json.dump(out, open("/home/storage2/ai26269345/vlm/research/ces/outputs/agent_math_1015/c_near_anchor.json", "w"), indent=1)
