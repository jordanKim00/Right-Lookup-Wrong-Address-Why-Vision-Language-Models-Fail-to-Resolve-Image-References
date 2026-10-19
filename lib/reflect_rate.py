"""Re-reflection rate vs native composition rate (saved records only, no model call; post-hoc descriptive analysis, 2026-10-10).
For a written address t left under the end frame, the account predicts that the model re-reads it as "the t-th image from the end" and composes
it as often as it composes that reference by itself.  Per model and written address t:
    compose(t) = unedited end-relative accuracy for references whose ordinal is t (same scenes, same question form; the middle reference,
                 correct without composing, left out)
    reflect(t) = share of address-only answers on the literal image (N + 1 - t), over references with written address t
Sources: S730 stagewise (N = 7; 27B corrected split stw2), S736 frame map (N = 5, 7, 9 per N), the address-specificity test (Qwen3-VL-4B,
its own question form, ordinal-only arm O against the unedited BASE).
Usage: python3 exp/reflect_rate.py   (writes research/ces/outputs/r3_lure_latest/reflect_rate.json)
"""
import json, os, sys, collections, importlib.util
import numpy as np

ROOT = "/home/storage2/ai26269345/vlm" if os.path.exists("/home/storage2/ai26269345/vlm") else "/vlm"
R3 = f"{ROOT}/results/r3"
cells = []


def from_rows(R, src, model, by_n=False):
    for N in sorted({r.get("N", 7) for r in R}) if by_n else [None]:
        rows = [r for r in R if (N is None or r.get("N", 7) == N)]
        for t in sorted({r["j"] for r in rows}):
            comp = [r["land_END"] == r["j"] for r in rows if r["k"] == t and r["j"] != r["k"]]          # the middle reference (j = k) is correct without composing
            refl = [r["land_ADDR"] == r["k"] for r in rows if r["j"] == t and r["j"] != r["k"]]
            acc = [r["land_ADDR"] == r["j"] for r in rows if r["j"] == t and r["j"] != r["k"]]
            if comp and refl:
                cells.append(dict(source=src, model=model, N=N or 7, t=t, compose=float(np.mean(comp)), reflect=float(np.mean(refl)),
                                  address_acc=float(np.mean(acc)), n=len(refl)))


for tag, model in (("stw_4b", "Qwen3-VL-4B"), ("stw_8b", "Qwen3-VL-8B"), ("stw_35_4b", "Qwen3.5-4B"), ("stw_35_9b", "Qwen3.5-9B"), ("stw2_27b", "Qwen3.5-27B")):
    p = f"{R3}/s730_{tag}_stagewise.json"
    if os.path.exists(p):
        from_rows(json.load(open(p))["rows"], "S730", model)
for tag, model in (("fr736_4b", "Qwen3-VL-4B"), ("fr736_8b", "Qwen3-VL-8B"), ("fr736_35_4b", "Qwen3.5-4B"), ("fr736_35_9b", "Qwen3.5-9B"), ("fr736_27b", "Qwen3.5-27B")):
    p = f"{R3}/s736_{tag}_frame.json"
    if os.path.exists(p):
        from_rows(json.load(open(p))["rows"], "S736", model, by_n=True)
# address-specificity test, Qwen3-VL-4B, its own form
OUTS = f"{ROOT}/research/ces/outputs"
spec = importlib.util.spec_from_file_location("A", f"{OUTS}/r3_semantic_analysis_v1/analysis.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
man = json.load(open(f"{OUTS}/r3_semantic_cohort_v1/cohort_manifest_draft.json"))
J = json.load(open(f"{OUTS}/r3_semantic_executor_v1b/worker_runs/r3-science-root-20261007b/outputs/science-002/study/logical_journal.json"))["rows"]
rec = {r["trial_id"]: r for r in J}
comp, refl = collections.defaultdict(list), collections.defaultdict(list)
for c in man["clusters"]:
    for var in ("V0", "VP", "VL"):
        v = c.get("variants", {}).get(var)
        if not v:
            continue
        k, j = v["reference_k"], v["j_physical"]
        b, o = A.response(rec.get(f"{c['cluster_id']}/{var}/BASE")), A.response(rec.get(f"{c['cluster_id']}/{var}/O"))
        if b is not None and j != k:
            comp[k].append(A.value_at(b, "j", c, var))                               # unedited accuracy for references with ordinal k (middle left out)
        if o is not None and j != k:
            refl[j].append(A.value_at(o, "k", c, var))                               # ordinal-only write of address j: answer on the literal image k
for t in sorted(set(comp) & set(refl)):
    cells.append(dict(source="address test", model="Qwen3-VL-4B", N=None, t=t, compose=float(np.mean(comp[t])), reflect=float(np.mean(refl[t])),
                      address_acc=None, n=len(refl[t])))
x = np.array([c["compose"] for c in cells]); y = np.array([c["reflect"] for c in cells])
res = dict(cells=cells, n_cells=len(cells), pearson=float(np.corrcoef(x, y)[0, 1]), mean_abs_diff=float(np.mean(np.abs(x - y))),
           max_abs_diff=float(np.max(np.abs(x - y))))
json.dump(res, open(f"{OUTS}/r3_lure_latest/reflect_rate.json", "w"), indent=1)
print(f"cells {len(cells)} · r = {res['pearson']:.3f} · mean |compose - reflect| = {res['mean_abs_diff']:.3f} · max {res['max_abs_diff']:.3f}")
for s in ("S730", "S736", "address test"):
    cs = [c for c in cells if c["source"] == s]
    if cs:
        xs = np.array([c["compose"] for c in cs]); ys = np.array([c["reflect"] for c in cs])
        print(f"  {s}: {len(cs)} cells, r = {np.corrcoef(xs, ys)[0, 1]:.3f}, mean |d| = {np.mean(np.abs(xs - ys)):.3f}")
