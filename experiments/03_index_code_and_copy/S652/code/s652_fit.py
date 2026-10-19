"""S652 analysis: intervention x model matrix of predicted-landing rates for A_copy vs A_comp (and A_count where it differs),
on rows where the algorithms predict different blocks (knowledge/CAUSAL_MODEL_PREDICTIONS.md; registration exp/QUEUE.md S652).
  I1 all codes +2 (s545 'R|pre'):     A_copy k-2 (k >= 3) · A_comp j · A_count j-2
  I2 last code -1 (s549 'R|m1'):      A_copy k · A_comp j-1 · A_count j
  I3 last code +1 (s549 'R|p1'):      A_copy k · A_comp j+1 · A_count j
  I4 forward-ordinal transplant (s474b grid 'ord|L' at the layer nearest relative depth .13, donor-correct rows): A_copy x · A_comp N+1-x
  baseline (no intervention, same rows): A_copy k · A_comp j
Item bootstrap (2,000, seed 65200) 95% intervals; rows are items, not independent scenes for the battery's paired arms.
Usage: python3 s652_fit.py   (reads results/, writes research/ces/outputs/s652_fit/s652_fit.json)"""
import json, os, random

R = "/home/storage2/ai26269345/vlm/results" if os.path.isdir("/home/storage2/ai26269345/vlm/results") else "/vlm/results"
MODELS = {"4b": ("bat_4b", "bat_4b", "bat_4b"), "8b": ("bat_8b", "bat_8b", "bat_8b"),
          "35_4b": ("bat_35_4b", "bat_35_4b", "bat_35_4b"), "35_9b": ("bat_35_9b", "bat_35_9b", "bat_35_9b"),
          "27b": (None, "endshift2_27b", "fwddonor2_27b")}


def boot(xs, rng, B=2000):
    if not xs:
        return None
    n = len(xs); m = [sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)]
    m.sort()
    return [round(sum(xs) / n, 3), round(m[int(.025 * B)], 3), round(m[int(.975 * B)], 3), n]


def load(path):
    p = f"{R}/{path}"
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def main():
    rng = random.Random(65200); out = {}
    for m, (t1, t2, t4) in MODELS.items():
        res = {}
        rows = load(f"s545_indexshift_{t1}.json") if t1 else None
        if rows:
            sel = [r for r in rows if r["k"] >= 3 and r["k"] - 2 != r["j"]]
            res["I1"] = dict(copy=boot([r["R|pre"] == r["k"] - 2 for r in sel], rng), comp=boot([r["R|pre"] == r["j"] for r in sel], rng),
                             count=boot([r["R|pre"] == r["j"] - 2 for r in sel], rng),
                             base_copy=boot([r["R|none"] == r["k"] for r in sel], rng), base_comp=boot([r["R|none"] == r["j"] for r in sel], rng))
        rows = load(f"s549_endshift_{t2}.json")
        if rows:
            sel = [r for r in rows if r["j"] != r["k"] and r["j"] - 1 != r["k"]]
            res["I2"] = dict(copy=boot([r["R|m1"] == r["k"] for r in sel], rng), comp=boot([r["R|m1"] == r["j"] - 1 for r in sel], rng),
                             count=boot([r["R|m1"] == r["j"] for r in sel], rng), base_comp_target=boot([r["R|none"] == r["j"] - 1 for r in sel], rng))
            sel = [r for r in rows if r["j"] != r["k"] and r["j"] + 1 != r["k"] and r["j"] + 1 <= r["N"]]
            res["I3"] = dict(copy=boot([r["R|p1"] == r["k"] for r in sel], rng), comp=boot([r["R|p1"] == r["j"] + 1 for r in sel], rng),
                             jm1=boot([r["R|p1"] == r["j"] - 1 for r in sel], rng), base_comp_target=boot([r["R|none"] == r["j"] + 1 for r in sel], rng))
            sel = [r for r in rows if r["j"] != r["k"]]
            res["I0"] = dict(copy=boot([r["R|none"] == r["k"] for r in sel], rng), comp=boot([r["R|none"] == r["j"] for r in sel], rng))
        d = json.load(open(f"{R}/s474b_fwddonor_{t4}.json")) if os.path.exists(f"{R}/s474b_fwddonor_{t4}.json") else None
        if d:
            L = d["res"]["layers"]; D = max(L) + 1
            nl = 64 if m == "27b" else (32 if m.startswith("35") else 36)
            lay = min(L, key=lambda l: abs(l / (nl - 1) - .13))
            sel = [r for r in d["rows"] if r["donor_ok"] and r["x"] != r["N"] + 1 - r["x"] and f"ord|{lay}" in r["grid"]]
            res["I4"] = dict(layer=lay, copy=boot([r["grid"][f"ord|{lay}"] == r["x"] for r in sel], rng),
                             comp=boot([r["grid"][f"ord|{lay}"] == r["N"] + 1 - r["x"] for r in sel], rng))
        out[m] = res
    od = os.path.join(os.path.dirname(R), "research/ces/outputs/s652_fit"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/s652_fit.json", "w"), indent=1)
    for m, res in out.items():
        print(m, " | ".join(f"{i}: copy {v['copy'][0] if v.get('copy') else '-'} comp {v['comp'][0] if v.get('comp') else '-'} (n {v['comp'][3] if v.get('comp') else 0})"
                            for i, v in sorted(res.items())))


if __name__ == "__main__":
    main()
