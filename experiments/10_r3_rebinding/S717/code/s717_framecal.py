"""S717-cal: R³ frame anchors for other end-relative wordings (frame identification only -- no end-relative answers anywhere).
R³-v3 binds an ordinal to an END frame only through the "end" / "last" anchors of exp/r3_build.calibrate ("... from the end / last");
on the S628 wordings "the k-th-to-last image", "the k-th image in reverse order" and "the k-th image counting backwards" it finds no anchor
and writes nothing (S716 dev runs).  Same procedure as calibrate's anchors: procedural scenes (N 3-10, the calibrate image source), the
three calibrate formats (question / look-at / option list), the L_b state of the frame word ("-last" after "-to", " reverse",
" backwards") averaged into an anchor mu[family]; threshold = midpoint between the family's farthest own state and the nearest other
text token (non-ordinal, non-frame), as calibrate does for "end" / "last".  The S709 calibration (base r327b) is copied with the three
anchors added to cal["frame"] and saved as base r327b4 (readers, ordinal prototypes, START anchor and the two old anchors unchanged).
Usage: python s717_framecal.py MODEL TAG [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3ref import ORD, load_image
from r3_build_md import ModelMD

FAM = {"tolast": lambda o, cap: f"{'T' if cap else 't'}he {o}-to-last image",
       "reverse": lambda o, cap: f"{'T' if cap else 't'}he {o} image in reverse order",
       "backwards": lambda o, cap: f"{'T' if cap else 't'}he {o} image counting backwards"}
ANCHOR = {"tolast": "-last", "reverse": " reverse", "backwards": " backwards"}
SKIP = {"end", "last", "start", "beginning", "-last", "reverse", "backwards", "-to"}


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    st = torch.load(f"/vlm/results/r3/s709_{a.base}_cal.pt", weights_only=False)                      # S734: any base (default r327b)
    cal, Lb = st["cal"], st["Lb"]
    M = ModelMD(v, Lb, Lb, [Lb], st["reader"])
    ords = {v.tok.decode([v.tok(" " + o, add_special_tokens=False)["input_ids"][0]]).strip() for o in ORD}
    rng = random.Random(71700)
    own = {f: [] for f in FAM}; other = []
    for fam, ref in FAM.items():
        for i in range(a.n):
            k = rng.randint(2 if fam == "tolast" else 1, 9)
            N = rng.randint(max(3, k), 10)
            pics = [load_image(f"proc:{rng.randint(0, 59_999)}") for _ in range(N)]
            fmt = rng.choice(["q", "s", "o"])
            if fmt == "o":
                ks = rng.sample(range(2 if fam == "tolast" else 1, N + 1), min(4, N - (1 if fam == "tolast" else 0)))
                body = "Which image shows a circle?\nSelect from the following choices.\n" + "\n".join(
                    f"{chr(65 + j)}: {ref(ORD[kk - 1], True)}" for j, kk in enumerate(ks)) + "\nAnswer with the option's letter from the given choices directly."
            elif fmt == "q":
                body = f"What is in {ref(ORD[k - 1], False)}? Answer with only one word from: circle, star, ring."
            else:
                body = f"Look at {ref(ORD[k - 1], False)}. What is in it? Answer with only one word from: circle, star, ring."
            c = M.capture(body, pics)
            span = c["ids"][c["s0"]:c["turn"]]; H = c["H"][Lb]
            for jx, t in enumerate(span):
                w = v.tok.decode([t])
                if w == ANCHOR[fam] and (fam != "tolast" or v.tok.decode([span[jx - 1]]) == "-to"):
                    own[fam].append(H[jx])
                elif w.strip() not in ords and w.strip() not in SKIP and w not in SKIP:
                    other.append(H[jx])
        print(f"[s717cal] {fam}: 기준점 {len(own[fam])}", flush=True)
    OB = torch.stack(other)
    fr = cal["frame"]
    for fam in FAM:
        X = torch.stack(own[fam]); mu = X.mean(0)
        o = (X - mu).norm(dim=1); d = (OB - mu).norm(dim=1)
        omax, dmin = float(o.max()), float(d.min())
        thr = (omax + dmin) / 2 if dmin > omax else float(torch.quantile(o, .99))
        fr["mu"][fam] = mu; fr["thr"][fam] = thr
        dist_old = {a_: round(float((mu - fr["mu"][a_]).norm()), 2) for a_ in ("end", "last")}
        print(f"[s717cal] {fam}: 자기 최대 {omax:.2f} · 다른 토큰 최소 {dmin:.2f} → 문턱 {thr:.2f} (×1.2 {1.2 * thr:.2f}) · end/last 기준점까지 {dist_old}", flush=True)
    st["cal"] = cal
    st["s717"] = dict(n=a.n, families=list(FAM), n_other=len(other))
    torch.save(st, f"/vlm/results/r3/s709_{a.base}4_cal.pt")
    print(f"[s717cal] 끝 -> /vlm/results/r3/s709_{a.base}4_cal.pt", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--device_map", default=None)
    ap.add_argument("--base", default="r327b")
    main(ap.parse_args())
