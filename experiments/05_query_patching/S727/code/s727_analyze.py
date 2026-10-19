"""S727 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-10) before any S727 data.
Share = softmax over the N image blocks of a matcher head's block log-sum-exp score at a row; means over scenes x options x heads.
    I0_OPT:        OPT_START option rows: mean share at block k >= 2/N (twice chance: the option lines do run the index lookup)
    OPT_LITERAL:   OPT_END option rows, k != 4: mean share(literal k) - mean share(target N+1-k) >= .05
    Q_TARGET:      Q_END2 answer row: mean share(block N-1) > mean share(block 2)            (27B only; §740 replication)
    R3_OPT_TARGET: OPT_END_R3 option rows, k != 4: mean share(target) - mean share(literal) >= .05
Reported: argmax-block rates (literal / target / other) per condition, per-k shares, answer-row shares, behaviour (option accuracy,
lure rate among errors, R³ accuracy, question-form landings).
Amendment (2026-10-10, before the 27B data and before any ordinal-row data except a one-scene software smoke on Qwen3.5-4B), on each
option's ordinal-token row: I0_ORD = OPT_START diagonal share >= 2/N; OPT_LITERAL_ORD = OPT_END literal - target >= .05 (k != 4);
R3_ORD_TARGET = OPT_END_R3 target - literal >= .05.  Line-end rows are reported.
Usage: python s727_analyze.py TAG [--q]   (--q: also judge Q_TARGET)
"""
import json, sys
import numpy as np


def shares(x):
    S = np.array(x["scores"], float)
    E = np.exp(S - S.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True), x["rowlab"]


def main(tag, judge_q):
    d = json.load(open(f"/vlm/results/r3/s727_{tag}_optqk.json")); N = d["N"]; R = d["rec"]
    res = {}
    for cond in ("OPT_END", "OPT_START", "OPT_END_R3"):
        lit, tgt, arg = [], [], {"literal": 0, "target": 0, "other": 0, "n": 0}
        perk = {}
        for r in R:
            sh, lab = shares(r[cond]); oi = [i for i, l in enumerate(lab) if l == "opt"]
            assert len(oi) == len(r["ks"]), (len(oi), r["ks"])
            for i, k in zip(oi, r["ks"]):
                L = sh[i, :, k - 1]; T_ = sh[i, :, N - k]
                if cond == "OPT_START":
                    lit.append(L.mean()); perk.setdefault(k, []).append(L.mean())
                    am = sh[i].argmax(-1)
                    arg["literal"] += int((am == k - 1).sum()); arg["other"] += int((am != k - 1).sum()); arg["n"] += len(am)
                    continue
                if k == (N + 1) // 2 and N % 2 == 1:
                    continue
                lit.append(L.mean()); tgt.append(T_.mean()); perk.setdefault(k, []).append((L.mean(), T_.mean()))
                am = sh[i].argmax(-1)
                arg["literal"] += int((am == k - 1).sum()); arg["target"] += int((am == N - k).sum())
                arg["other"] += int(((am != k - 1) & (am != N - k)).sum()); arg["n"] += len(am)
        r_ = {k_: v / max(arg["n"], 1) for k_, v in arg.items() if k_ != "n"}
        res[cond] = dict(literal=float(np.mean(lit)), target=float(np.mean(tgt)) if tgt else None, argmax=r_, n_rows=len(lit),
                         perk={int(k): (float(np.mean([x for x in v])) if cond == "OPT_START" else [float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))])
                               for k, v in sorted(perk.items())})
    extra = {}                                                                      # amendment: reported rows (ordinal token, line end)
    for cond in ("OPT_END", "OPT_START", "OPT_END_R3"):
        for rl in ("ord", "nl"):
            L_, T_ = [], []
            for r in R:
                sh, lab = shares(r[cond]); oi = [i for i, l in enumerate(lab) if l == rl]
                if len(oi) != len(r["ks"]):
                    continue
                for i, k in zip(oi, r["ks"]):
                    if cond != "OPT_START" and k == (N + 1) // 2 and N % 2 == 1:
                        continue
                    L_.append(sh[i, :, k - 1].mean()); T_.append(sh[i, :, N - k].mean())
            if L_:
                extra[f"{cond}|{rl}"] = dict(literal=float(np.mean(L_)), target=float(np.mean(T_)), n=len(L_))
    if extra:
        print(f"[{tag}] (보고) 다른 행: " + " · ".join(f"{k} 문자 {v['literal']:.3f} / 목표 {v['target']:.3f}" for k, v in extra.items()))
    ordchk = {}
    if all(f"{c}|ord" in extra for c in ("OPT_END", "OPT_START", "OPT_END_R3")):         # S727 amendment rules on the ordinal-token rows
        ordchk = dict(I0_ORD=extra["OPT_START|ord"]["literal"] >= 2 / N,
                      OPT_LITERAL_ORD=extra["OPT_END|ord"]["literal"] - extra["OPT_END|ord"]["target"] >= .05,
                      R3_ORD_TARGET=extra["OPT_END_R3|ord"]["target"] - extra["OPT_END_R3|ord"]["literal"] >= .05)
    beh = {}
    for cond in ("OPT_END", "OPT_START", "OPT_END_R3"):
        L = ["A", "B", "C", "D"]
        ok = [L[r[cond]["pick"]] == r["gold_letter"] for r in R]
        lure = [L[r[cond]["pick"]] == r["lure_letter"] for r in R]
        err = [not o for o in ok]
        beh[cond] = dict(acc=float(np.mean(ok)), lure_rate=float(np.mean(lure)), lure_of_errors=float(sum(l for l, e in zip(lure, err) if e) / max(sum(err), 1)))
    for k in (2, 3):
        land = [r[f"Q_END{k}"]["land"] for r in R]
        beh[f"Q_END{k}"] = dict(target=float(np.mean([x == N + 1 - k for x in land])), literal=float(np.mean([x == k for x in land])))
        sh = np.stack([shares(r[f"Q_END{k}"])[0][-1] for r in R])                      # answer row: [scenes, heads, N]
        res[f"Q_END{k}"] = dict(target=float(sh[:, :, N - k].mean()), literal=float(sh[:, :, k - 1].mean()))
    chk = dict(I0_OPT=res["OPT_START"]["literal"] >= 2 / N,
               OPT_LITERAL=res["OPT_END"]["literal"] - res["OPT_END"]["target"] >= .05,
               R3_OPT_TARGET=res["OPT_END_R3"]["target"] - res["OPT_END_R3"]["literal"] >= .05)
    if judge_q:
        chk["Q_TARGET"] = res["Q_END2"]["target"] > res["Q_END2"]["literal"]
    print(f"[{tag}] OPT_START 대각 몫 {res['OPT_START']['literal']:.3f} (최댓값 = k {res['OPT_START']['argmax']['literal']:.2f})")
    for c in ("OPT_END", "OPT_END_R3"):
        x = res[c]
        print(f"[{tag}] {c}: 문자 그대로 몫 {x['literal']:.3f} · 목표 몫 {x['target']:.3f} · 최댓값 문자 {x['argmax']['literal']:.2f} · 목표 {x['argmax']['target']:.2f} · 기타 {x['argmax']['other']:.2f}"
              f"  k 별(문자, 목표) " + " ".join(f"k{k}:{a:.2f}/{b:.2f}" for k, (a, b) in x["perk"].items()))
    print(f"[{tag}] 질문 꼴 답 행: k2 목표 {res['Q_END2']['target']:.3f} vs 문자 {res['Q_END2']['literal']:.3f} · k3 목표 {res['Q_END3']['target']:.3f} vs 문자 {res['Q_END3']['literal']:.3f}")
    print(f"[{tag}] 행동: " + " · ".join(f"{c} 정확 {b['acc']:.2f}" + (f" (미끼 {b['lure_rate']:.2f}, 오류 중 미끼 {b['lure_of_errors']:.2f})" if c.startswith("OPT") else "")
                                       if "acc" in b else f"{c} 목표 {b['target']:.2f} · 문자 {b['literal']:.2f}" for c, b in beh.items()))
    chk.update(ordchk)
    print(f"[{tag}] 판정 {chk}")
    json.dump(dict(res=res, beh=beh, checks=chk, extra=extra), open(f"/vlm/results/r3/s727_{tag}_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], "--q" in sys.argv)
