"""S609 results table: base vs ours (S609c, model-internal signals only) vs A3.1 (clause window), forward ceiling and
recovery ratio = (ours_rev - base_rev) / (base_fwd - base_rev) (reported only when the gap >= .10).
Usage: python3 exp/s609_table.py > results/r3/S609_TABLE.md
"""
import json, os

R = "/home/storage2/ai26269345/vlm/results"
MODELS = [("4b", "Qwen3-VL-4B"), ("8b", "Qwen3-VL-8B"), ("35_4b", "Qwen3.5-4B"), ("35_9b", "Qwen3.5-9B")]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rec(fb, rb, rr):
    g = fb - rb
    return f"{(rr - rb) / g:.0%}" if g >= .10 else "—"


print("# S609 결과표 — 원 모델 / 우리(S609c, 모델 내부 신호만) / A3.1(절 창), 상한 = 같은 문항 전방\n")
print("## 1. 위치 벤치(S582): 전방 F 와 역순 R\n")
print("| 모델 | 과제 | 전방 원→우리 | 역순 원 | 역순 우리(S609c) | 역순 A3.1 | 상한 | 회복률(S609c) |")
print("|---|---|---|---|---|---|---|---|")
for b, nm in MODELS:
    c = load(f"{R}/s582_evalrepair_e582_s609c_{b}.json"); a = load(f"{R}/s582_evalrepair_e582_a3_{b}.json")
    if not c:
        continue
    for t in ("mmiu_pos", "muir", "cloze"):
        F, Rv = c["res"].get(f"{t}|F"), c["res"].get(f"{t}|R"); Ra = a["res"].get(f"{t}|R") if a else None
        if not F or not Rv:
            continue
        print(f"| {nm} | {t} | {F['base']:.3f}→{F['rep']:.3f} | {Rv['base']:.3f} | {Rv['rep']:.3f} | {Ra['rep']:.3f} | {F['base']:.3f} | {rec(F['base'], Rv['base'], Rv['rep'])} |"
              if Ra else f"| {nm} | {t} | {F['base']:.3f}→{F['rep']:.3f} | {Rv['base']:.3f} | {Rv['rep']:.3f} | — | {F['base']:.3f} | {rec(F['base'], Rv['base'], Rv['rep'])} |")
print("\n## 2. 보존(원래형, 위치 무관): 원 → 우리 (바뀐 답 비율)\n")
print("| 모델 | MMIU 12 | MUIRBench 12 | imagenette | BLINK | Q-Bench2 | Mantis | MMMU | MIBench | MIRB | MIBench 추론 | MIRB 추론 | MathVista |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for b, nm in MODELS:
    c = load(f"{R}/s582_evalrepair_e582_s609c_{b}.json"); g = load(f"{R}/r3/egen_s609c_{b}_general.json"); r = load(f"{R}/r3/ereason_s609c_{b}_general.json")
    cell = []
    for src, key in ((c, "mmiu_other|O"), (c, "muir_all|O"), (c, "imagenette|O"), (g, "blink|O"), (g, "qbench2|O"), (g, "mantis|O"), (g, "mmmu|O"),
                     (g, "mibench|O"), (g, "mirb|O"), (r, "mibench_reason|O"), (r, "mirb_reason|O"), (r, "mathvista|O")):
        x = src["res"].get(key) if src else None
        cell.append(f"{x['base']:.3f}→{x['rep']:.3f}" + (f" ({x['changed']:.3f})" if "changed" in x else "") if x else "…")
    print(f"| {nm} | " + " | ".join(cell) + " |")
print("\n## 3. 일반성 역순(서수 바꿔 쓰기 R): 원 → 우리, 상한 = 같은 문항 원래형, 회복률\n")
print("| 모델 | 묶음 | n | 상한 | 역 원 | 역 우리 | 회복률 |")
print("|---|---|---|---|---|---|---|")
for b, nm in MODELS:
    g = load(f"{R}/r3/egen_s609c_{b}_general.json")
    if not g:
        continue
    rows = g["rows"]
    for s in ("blink", "qbench2", "mantis", "mibench", "mirb"):
        S = [x for x in rows if x["suite"] == s and "R|base" in x]
        if not S:
            continue
        fb = sum(x["O|base"] == x["gold"] for x in S) / len(S); rb = sum(x["R|base"] == x["gold"] for x in S) / len(S); rr = sum(x["R|rep"] == x["gold"] for x in S) / len(S)
        print(f"| {nm} | {s} | {len(S)} | {fb:.3f} | {rb:.3f} | {rr:.3f} | {rec(fb, rb, rr)} |")
print("\n## 4. R³-Ref 영점 분할: END(역순) 원 → 우리 / A3.1, START(전방) 바뀐 답\n")
print("| 모델 | 분할 | END 원 | END 우리 | END A3.1 | START 바뀜(우리) |")
print("|---|---|---|---|---|---|")
for b, nm in MODELS:
    e = load(f"{R}/r3/eref_s609c_{b}_eval.json"); a = load(f"{R}/r3/eref_a3_{b}_eval.json")
    if not e:
        continue
    for sp, x in e["res"].items():
        y = a["res"].get(sp) if a else None
        print(f"| {nm} | {sp} | {x['end_base']:.3f} | {x['end_rep']:.3f} | {y['end_rep']:.3f} | {x['start_changed']:.3f} |" if y else
              f"| {nm} | {sp} | {x['end_base']:.3f} | {x['end_rep']:.3f} | — | {x['start_changed']:.3f} |")
