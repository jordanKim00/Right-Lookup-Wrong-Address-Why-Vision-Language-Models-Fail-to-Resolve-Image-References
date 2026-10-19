"""S721 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S721 data.
Qwen3.5-27B, later MMIU split (500), end-relative form; full = R³-v3 (m3fr3_27b), every arm on the same items.
    ABL_ORD_NEEDED:       P(no_ord)  <= P(full) - .15          (frame write alone: START(k) = literal k)
    ABL_FRAME_NEEDED:     P(no_frame) <= P(full) - .15         (ordinal write alone is re-reflected by the END frame, S708)
    ABL_ADDRESS_SPECIFIC: P(rand_target) <= P(base) + .05      (a wrong valid address does not help)
    ABL_READ_OK:          |P(oracle_n) - P(full)| <= .03       (the internal end reader is not the bottleneck)
Reported: forward answers changed per arm; prompt baselines (one-pass explanation, two-pass text address) = base answers under the
modified prompt, against the forward form (text normalization) and R³; latency / memory from s721_*_latency.json.
Usage: python s721_analyze.py [FULLTAG]   (S725 amendment, before data: the full arm is the assembled run a5fr_27b when S724 changes the 27B Read)
"""
import json, os
import numpy as np

R = "/vlm/results"
ARMS = {"full": "m3fr3_27b", "no_ord": "m3abno_27b", "no_frame": "m3abnf_27b", "rand_target": "m3abrt_27b", "oracle_n": "m3abon_27b",
        "explain": "m3pmex_27b", "twopass": "m3pmtp_27b"}


def rows(tag):
    p = f"{R}/s582_evalrepair_{tag}.json"
    return [r for r in json.load(open(p))["rows"] if r["suite"] == "mmiu_pos"] if os.path.exists(p) else None


def main():
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("--prefix=")]
    pre = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--prefix=")), None)
    if pre:                                                                         # S738: the same arms re-run under the clean-room Read (tags PREFIX*)
        ARMS.update(full=f"{pre}fr_27b", no_ord=f"{pre}abno_27b", no_frame=f"{pre}abnf_27b", rand_target=f"{pre}abrt_27b", oracle_n=f"{pre}abon_27b",
                    explain=f"{pre}pmex_27b", twopass=f"{pre}pmtp_27b")
    if args:                                                                       # S725 amendment: full = the assembled R³ when S724 changes the 27B Read
        ARMS["full"] = args[0]
    full = rows(ARMS["full"]); g = lambda rs, k: np.array([r[k] == r["gold"] for r in rs], float)
    base, F = g(full, "R|base"), g(full, "F|base")
    P = {"base": float(base.mean()), "forward(normalization)": float(F.mean())}
    out = {}
    for arm, tag in ARMS.items():
        rs = rows(tag)
        if rs is None:
            print(f"  {arm}: 기록 없음"); continue
        assert len(rs) == len(full) and all(x["gold"] == y["gold"] for x, y in zip(rs, full))
        rep = g(rs, "R|rep"); b = g(rs, "R|base"); fch = float(np.mean([r["F|rep"] != r["F|base"] for r in rs]))
        out[arm] = dict(rep=float(rep.mean()), base=float(b.mean()), forward_changed=fch)
        print(f"  {arm:12s} R 원래(이 프롬프트) {b.mean():.3f} · R³ 뒤 {rep.mean():.3f} · 앞 방향 바뀜 {fch:.3f}")
    print(f"  기준: 원래 끝 기준 {P['base']:.3f} · 앞 방향(글 정규화) {P['forward(normalization)']:.3f}")
    full_p = out["full"]["rep"]
    checks = {}
    if "no_ord" in out: checks["ABL_ORD_NEEDED"] = out["no_ord"]["rep"] <= full_p - .15 + 1e-9
    if "no_frame" in out: checks["ABL_FRAME_NEEDED"] = out["no_frame"]["rep"] <= full_p - .15 + 1e-9
    if "rand_target" in out: checks["ABL_ADDRESS_SPECIFIC"] = out["rand_target"]["rep"] <= P["base"] + .05 + 1e-9
    if "oracle_n" in out: checks["ABL_READ_OK"] = abs(out["oracle_n"]["rep"] - full_p) <= .03 + 1e-9
    lat = {}
    for t in ("lat_27b", "lat_4b"):
        p = f"{R}/r3/s721_{t}_latency.json"
        if os.path.exists(p):
            lat[t] = json.load(open(p))["res"]; print(f"  지연 {t}: {lat[t]}")
    print(f"checks {checks}")
    json.dump(dict(ref=P, arms=out, latency=lat, checks=checks), open(f"{R}/r3/s721_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
