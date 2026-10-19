"""S652: the algorithm-fit battery on one model, same seeds/items as the 27B runs of S647 (knowledge/CAUSAL_MODEL_PREDICTIONS.md).
  I1 all block codes +2      exp/s545_indexshift.run   seed 54501, N 5-7, k 2-4, ntest 8
  I2/I3 last block code -1/+1 exp/s549_endshift.run    seed 54901, N 6-9, k 2-4, ntest 8   (= S647 27B endshift2_27b)
  I4 forward-ordinal transplant exp/s474b_fwddonor.run seed 47451, N 7-9, k 2-4, 96 pairs, early relative depths (= S647 fwddonor2_27b)
Each run is isolated; a failure is logged and is not a result.  Usage: python run_battery.py MODEL TAG"""
import argparse, sys, time, traceback
sys.path.insert(0, "/vlm/exp")
from uni import Uni

ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--only", default=None)
a = ap.parse_args()
v = Uni(a.model, device="cuda:0", attn="sdpa")
for p in v.model.parameters():
    p.requires_grad_(False)
suf = a.tag.replace("bat_", "")


def r1():
    import s545_indexshift
    s545_indexshift.run(v, tag=f"bat_{suf}", ntest=8, Ns=(5, 6, 7), seed=54501)


def r2():
    import s549_endshift
    s549_endshift.run(v, tag=f"bat_{suf}", ntest=8, Ns=(6, 7, 8, 9), ks=(2, 3, 4), seed=54901)


def r4():
    import s474b_fwddonor
    s474b_fwddonor.run(v, tag=f"bat_{suf}", npair=96, Ns=(7, 8, 9), seed=47451, rels=[0.05, 0.13, 0.2, 0.3, 0.46])


only = set(a.only.split(",")) if a.only else None
for name, fn in (("I1", r1), ("I2", r2), ("I4", r4)):
    if only and name not in only:
        continue
    t0 = time.time(); print(f"[{a.tag}] {name} 시작", flush=True)
    try:
        fn(); print(f"[{a.tag}] {name} 끝 ({time.time()-t0:.0f}s)", flush=True)
    except Exception:
        print(f"[{a.tag}] {name} 실패 — 실행 실패이며 결과가 아님\n{traceback.format_exc()}", flush=True)
print(f"[{a.tag}] 전부 끝", flush=True)
