"""S783 descriptive comparison (not a registered verdict): the registered analyses of S780 / S769 / S772 / S778 / S770 / S758 / S781 under the
space-token readout (original runs) and the bare-token readout (S783 re-runs, same scenes), side by side, plus the share of answer readouts whose
vocabulary-wide top-1 token is a bare menu token (when it is, the bare readout equals greedy generation).
Usage: python s783_compare.py"""
import glob, json, os
import numpy as np

R = "/vlm/results/r3"
load = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None


def t1_rate(pattern):
    """Share of readouts with t1m = 1 over every answer record in the raw files matching pattern."""
    n = k = 0
    for f in glob.glob(f"{R}/{pattern}"):
        d = json.load(open(f))
        if "t1_menu" in d and d["t1_menu"] is not None:
            n += d["t1_n"]; k += d["t1_menu"] * d["t1_n"]; continue
        for m in d.get("meta", []):
            for a in (m.get("ans") or {}).values():
                if isinstance(a, dict) and "t1m" in a:
                    n += 1; k += a["t1m"]
    return (k / n if n else None), n


def s780():
    A, B = load("s780_analysis.json"), load("s780_analysis_bare.json")
    if not B:
        return
    print("== S780 (old -> bare)")
    for N, o, b in (("7", "s780_27b7", "s783_27b7"), ("8", "s780_27b8", "s783_27b8")):
        x, y = A[o], B[b]
        print(f" N={N} checks old {x['checks']}\n       bare {y['checks']}")
        print(f"   i0 old {x['i0']} | bare {y['i0']}")
        for key in ("visual_r3", "visual_e2", "visual_end", "menu_reversal_change", "r3_part"):
            print(f"   {key}: old {x.get(key)} | bare {y.get(key)}")
        print(f"   r3 removed / suf: old {x.get('r3_removed_ALL59')} / {x.get('r3_suf')} | bare {y.get('r3_removed_ALL59')} / {y.get('r3_suf')}")
    print(" t1 menu (S783 S780 runs):", t1_rate("s780_s783_27b*_handoff.json"))


def s769():
    A, B = load("s769_analysis.json"), load("s769_analysis_bare.json")
    if not B:
        return
    print("== S769 (old -> bare)")
    print(" checks old", A["checks"], "\n        bare", B["checks"])
    for N in ("N7", "N8"):
        for arm in ("E4", "S4", "F4:P1", "F4:P1conv", "F4:P1fa", "F4:P23", "R4:P1"):
            print(f"  {N} {arm}: acc4 old {A[N]['acc'][arm]['4'][0]:.3f} bare {B[N]['acc'][arm]['4'][0]:.3f} | share4 L39 old {A[N]['share4_L39'][arm]:.3f} bare {B[N]['share4_L39'][arm]:.3f}")
    print(" t1 menu (S783 S769 runs):", t1_rate("s769_s783e_27b*_end4.json"))


if __name__ == "__main__":
    s780(); s769()
