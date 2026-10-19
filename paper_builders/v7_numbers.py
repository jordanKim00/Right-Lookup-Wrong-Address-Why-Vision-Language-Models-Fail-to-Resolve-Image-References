"""Numbers for paper v7: the v6 numbers (numbers_v6.json, written from the raw files by exp/figs/v6_numbers.py) plus the values added for the
mock-review revision, each from a saved file:
  stop_visual      research/ces/outputs/p6_stop_visual_target.json (exp/p6_stop_visual_target.py; post hoc, S854-1 and S846 runs)
  bench_paired     research/ces/paper_v7_20261019/tables/bench_paired_values.json (exp/figs/make_v7_tables.py; item bootstrap)
  cloze_picks      results/s582_evalrepair_c8cz_27b.json (gold letters and the letters chosen per wording)
  latency_27b      results/r3/s721_analysis.json
  config_sha       sha256 of each checkpoint's config.json
  read_grid        exp/s738_plan.json (the Read grid) and results/r3/s724_{rc,rvl8}_27b_readcal.json (split sizes)
Usage: python3 exp/figs/v7_numbers.py"""
import collections, hashlib, json

V = "/home/storage2/ai26269345/vlm"
P7 = f"{V}/research/ces/paper_v7_20261019"
N = json.load(open(f"{V}/research/ces/paper_v6_20261019/numbers_v6.json"))
put = lambda k, v, src: N.__setitem__(k, dict(value=v, source=src))
put("stop_visual", json.load(open(f"{V}/research/ces/outputs/p6_stop_visual_target.json")), "research/ces/outputs/p6_stop_visual_target.json")
put("bench_paired", json.load(open(f"{P7}/tables/bench_paired_values.json")), "research/ces/paper_v7_20261019/tables/bench_paired_values.json")
rows = [r for r in json.load(open(f"{V}/results/s582_evalrepair_c8cz_27b.json"))["rows"] if r["suite"] == "cloze"]
put("cloze_picks", {k: dict(collections.Counter(r[k] for r in rows)) for k in ("gold", "F|base", "R|base", "R|rep")} | {"n": len(rows)},
    "results/s582_evalrepair_c8cz_27b.json")
put("latency_27b", json.load(open(f"{V}/results/r3/s721_analysis.json"))["latency"]["lat_27b"], "results/r3/s721_analysis.json")
CK = {"Qwen3-VL-4B": "/dev/shm/vlm_models/Qwen3-VL-4B-Instruct", "Qwen3-VL-8B": "/dev/shm/vlm_models/Qwen3-VL-8B-Instruct",
      "Qwen3.5-4B": "/dev/shm/vlm_models/Qwen3.5-4B", "Qwen3.5-9B": "/home/storage2/ai26269345/lambo/models/Qwen3.5-9B",
      "Qwen3.5-27B": "/home/storage2/ai26269345/lambo/models/Qwen3.5-27B"}
put("config_sha", {m: hashlib.sha256(open(f"{p}/config.json", "rb").read()).hexdigest()[:12] for m, p in CK.items()}, "checkpoint config.json files")
plan = json.load(open(f"{V}/exp/s738_plan.json"))["rc_27b"]["grid"]
grid = json.loads(plan.replace("'", '"')) if isinstance(plan, str) else plan
put("read_grid", dict(grid=grid, sel_n=len(json.load(open(f"{V}/results/r3/s724_rc_27b_readcal.json"))["rec"]),
                      val_n=len(json.load(open(f"{V}/results/r3/s724_rvl8_27b_readcal.json"))["rec"])), "exp/s738_plan.json, results/r3/s724_*_readcal.json")
json.dump(N, open(f"{P7}/numbers_v7.json", "w"), indent=1, default=float)
print(len(N), "keys ->", f"{P7}/numbers_v7.json")
