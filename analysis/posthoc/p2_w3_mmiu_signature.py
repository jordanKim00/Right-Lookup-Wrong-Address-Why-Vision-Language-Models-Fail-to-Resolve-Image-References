"""Phase-2 W3 (descriptive, not registered; 10-17 01:0x): which signature do Qwen3.5-27B's wrong answers on reverse-form MMIU items carry?
Saved rows: results/s489_mmiu_fix_489a27b.json ('R') and results/s509_mmiu27_509a27b.json ('R|none') - natural reverse-form answers,
restricted option-letter argmax (older readout; check the S782 registry before any main-text use).  Per item: N pictures, ks = the forward
positions of the four options' pictures, gold option.  An option naming its picture as the kr-th from the end (kr = N + 1 - pos) carries the
'literal' signature when kr equals the gold picture's position (reading the ordinal from the start lands on the gold picture), and the 'step'
signature when its copy kr moved one step toward its own target pos lands on the gold picture (identity options: +-1).  Chance = the mean
over the three wrong options.  Usage: python3 exp/p2_w3_mmiu_signature.py -> research/ces/outputs/p2_w3_mmiu_signature.json"""
import collections, json
V = "/home/storage2/ai26269345/vlm"
out = {}
for f, rkey in (("results/s489_mmiu_fix_489a27b.json", "R"), ("results/s509_mmiu27_509a27b.json", "R|none")):
    R = json.load(open(f"{V}/{f}"))["rows"]; L = "ABCD"; n = 0; nerr = 0; cnt = collections.Counter(); chance = collections.Counter()
    by_task = collections.defaultdict(lambda: [0, 0])
    for r in R:
        if rkey not in r or r.get("ks") is None or len(r["ks"]) != 4 or r["gold"] not in L or r[rkey] not in L:
            continue
        N, ks, gold, ans = r["N"], r["ks"], r["gold"], r[rkey]; n += 1; gpos = ks[L.index(gold)]
        if ans == gold:
            continue
        nerr += 1
        def sig(x):
            pos = ks[L.index(x)]; kr = N + 1 - pos
            step = gpos in (pos - 1, pos + 1) if kr == pos else (kr + (1 if pos > kr else -1)) == gpos
            return kr == gpos, step
        lit, step = sig(ans); cnt["literal"] += lit; cnt["step"] += step
        by_task[r["task"]][0] += 1; by_task[r["task"]][1] += lit
        for x in [y for y in L if y != gold]:
            l_, s_ = sig(x); chance["literal"] += l_ / 3; chance["step"] += s_ / 3
    out[f] = dict(items=n, errors=nerr, accuracy=1 - nerr / n, literal=cnt["literal"] / nerr, literal_chance=chance["literal"] / nerr,
                  step=cnt["step"] / nerr, step_chance=chance["step"] / nerr, literal_by_task={t: v[1] / v[0] for t, v in by_task.items() if v[0]})
    print(f, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[f].items() if k != "literal_by_task"}))
    print("   literal share by task:", {t: round(v, 2) for t, v in out[f]["literal_by_task"].items()})
json.dump(out, open(f"{V}/research/ces/outputs/p2_w3_mmiu_signature.json", "w"), indent=1)
