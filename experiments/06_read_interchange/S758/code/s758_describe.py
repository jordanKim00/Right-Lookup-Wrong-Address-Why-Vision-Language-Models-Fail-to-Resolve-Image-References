"""S758 descriptive tables (exploratory, after the frozen analysis; not a registered verdict).
Per arm: answer shares on 6 / 3 / 5 (literal of "fifth") / 2 (literal of "second"), the natural END(5) answer a5 (scenes where a5 is not 3 or 6),
the matcher pointer distribution (S705a head-rows, L35-47, tail1 / answer) and the mean block share on each image.
Usage: python s758_describe.py TAG
"""
import json, sys
import numpy as np

N = 7


def lse(x):
    m = x.max(); return m + np.log(np.exp(x - m).sum())


def main(tag):
    D = f"/vlm/results/r3/s758_{tag}_paths"
    M = json.load(open(D + ".json"))["meta"]; z = np.load(D + ".npz"); n = len(M)
    mt = [tuple(x) for x in json.load(open("/vlm/results/r3/s705a_analysis.json"))["qa_27b"]["heads"]["matchers"]]
    dec = [x for x in mt if x[0] <= 47 and x[2] in (1, 2)]

    def share(si, a):
        sh = []
        for (li, h, ri) in dec:
            sc = z[f"{si}|{a}|{li}"][ri, h].astype(np.float64); sh.append(np.exp(sc - lse(sc)))
        return np.mean(sh, 0)
    a5 = np.array([m["land"]["E5"] for m in M]); far = [i for i in range(n) if a5[i] not in (3, 6)]
    print(f"[{tag}] n {n} · far-error scenes {len(far)}")
    print(f"{'arm':14s} {'P6':>5s} {'P3':>5s} {'P5':>5s} {'P2':>5s} {'a5':>5s} | pointer 1..7 | mean share 1..7")
    out = {}
    for a in M[0]["land"]:
        L = np.array([m["land"][a] for m in M])
        S = np.array([share(m["si"], a) for m in M]); P = S.argmax(1) + 1
        rep = float(np.mean([L[i] == a5[i] for i in far])) if far else float("nan")
        out[a] = dict(P6=float(np.mean(L == 6)), P3=float(np.mean(L == 3)), P5=float(np.mean(L == 5)), P2=float(np.mean(L == 2)), a5=rep,
                      ptr=[float(np.mean(P == b)) for b in range(1, N + 1)], share=[float(x) for x in S.mean(0)])
        o = out[a]
        print(f"{a:14s} {o['P6']:5.2f} {o['P3']:5.2f} {o['P5']:5.2f} {o['P2']:5.2f} {o['a5']:5.2f} | "
              + " ".join(f"{x:.2f}" for x in o["ptr"]) + " | " + " ".join(f"{x:.2f}" for x in o["share"]))
    json.dump(out, open(f"/vlm/results/r3/s758_{tag}_describe.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
