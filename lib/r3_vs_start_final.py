"""R³ − Start for the runs the paper's Table 3 uses (exp/figs/make_r3_tables.bench_tags: S732 f7 for the small models, S738 c8 for 27B, once
complete; otherwise the earlier tags).  Calls exp/r3_vs_start.py on held-out MMIU and MuirBench and on the later MMIU split.
Usage (container): python /vlm/exp/r3_vs_start_final.py   -> /vlm/research/ces/outputs/r3_lure_latest/r3_vs_start_final.json"""
import subprocess, sys
sys.path.insert(0, "/vlm/exp/figs")
import make_r3_tables as T

tags = T.bench_tags()
args = []
for b, _ in T.MODELS:
    args += [f"{tags[b]['sealed']}:sealed:mmiu_pos", f"{tags[b]['sealed']}:sealed:muir", f"{tags[b]['later']}:fresh:mmiu_pos"]
out = "/vlm/research/ces/outputs/r3_lure_latest/r3_vs_start_final.json"
subprocess.run(["python3", "/vlm/exp/r3_vs_start.py", *args, "--out", out], check=True)
print("tags", {b: tags[b] for b, _ in T.MODELS})
