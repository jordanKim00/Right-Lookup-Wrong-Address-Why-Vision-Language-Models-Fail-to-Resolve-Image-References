"""S626 (4): pack the token x layer maps for the explanatory page. One entry per model run: layers, token labels (text tokens
after the images; image blocks as ranges), behaviour, full-swap totals, the exact-patching check, and the four maps A/B/Cc/Cs as
fractions of the model's full-swap scale (A/B: max(|tot A|, |tot B|); C: max(|tot Cc|, |tot Cs|)), sqrt-companded to int8 and
base64-encoded (decode: x = sign(q) (q/127)^2). The prefix tokens before the first image are dropped (identical in base and
source, effect 0).
Usage: python3 exp/s626_pack.py OUT.js
"""
import base64, json, os, sys
import numpy as np

R = "/home/storage2/ai26269345/vlm/results/r3"
RUNS = [("atp_4b", "Qwen3-VL-4B", "fail"), ("atp_8b", "Qwen3-VL-8B", "fail"), ("atp2_35_4b", "Qwen3.5-4B", "fail"),
        ("atp2_35_9b", "Qwen3.5-9B", "fail"), ("atp_27b", "Qwen3.5-27B", "compose"),
        ("atp_4b_r3", "Qwen3-VL-4B + R³", "repair"), ("atpseg_35_4b_r3", "Qwen3.5-4B + R³", "repair"),
        ("atpseg_35_9b_r3", "Qwen3.5-9B + R³", "repair")]


def q8(x):
    y = np.sign(x) * np.sqrt(np.clip(np.abs(x), 0, 1))
    return base64.b64encode(np.round(y * 127).astype(np.int8).tobytes()).decode()


out = []
for tag, name, kind in RUNS:
    f = f"{R}/s626_{tag}_atp.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    B = d["blocks"]; t0 = B[0][0] - 1                                              # first vision_start
    toks = d["toks"][t0:]
    T = len(toks)
    blocks = [[b[0] - t0, b[1] - t0] for b in B]
    tot = d["total"]
    sAB = max(abs(tot["A"]), abs(tot["B"])); sC = max(abs(tot["Cc"]), abs(tot["Cs"]))
    maps = {}
    for m in ("A", "B", "Cc", "Cs"):
        E = np.array(d["maps"][m])[:, t0:] / (sAB if m in ("A", "B") else sC)
        maps[m] = q8(E)
    lab = []
    for i, t in enumerate(toks):
        if any(b[0] <= i <= b[1] for b in blocks):
            lab.append("")
        else:
            lab.append(t.replace("\n", "\\n"))
    chk = {m: [dict(layer=r["layer"], tok=r["tok"].replace("\n", "\\n"), atp=round(r["atp"], 2), exact=round(r["exact"], 2))
               for r in rows] for m, rows in d["check"].items()}
    out.append(dict(tag=tag, name=name, kind=kind, D=d["D"], T=T, toks=lab, blocks=blocks,
                    beh={k: round(x, 3) for k, x in d["behaviour"].items()}, tot={k: round(x, 2) for k, x in tot.items()},
                    sAB=round(sAB, 2), sC=round(sC, 2), check=chk, maps=maps))
    print(f"{name}: D {d['D']} T {T} 행동 {d['behaviour']} 전체 {tot}", file=sys.stderr)
# exact interchange maps (s626_patch.py): units x grid layers, fractions of the same run's full swap, plus the AtP map summed
# over the same unit positions (for the side-by-side comparison)
PATCH = [("patch_4b", "Qwen3-VL-4B", "fail", "atp_4b"), ("patch_8b", "Qwen3-VL-8B", "fail", "atp_8b"),
         ("patch_35_4b", "Qwen3.5-4B", "fail", "atp2_35_4b"), ("patch2_35_9b", "Qwen3.5-9B", "fail", "atp2_35_9b"),
         ("patch_27b", "Qwen3.5-27B", "compose", "atp_27b"), ("patch_4b_r3", "Qwen3-VL-4B + R³", "repair", "atp_4b_r3")]
ex = []
for tag, name, kind, atag in PATCH:
    f = f"{R}/s626_{tag}_patch.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); A = json.load(open(f"{R}/s626_{atag}_atp.json"))
    toksA, BA = A["toks"], A["blocks"]
    # unit positions in the AtP run (same tokenisation): blocks with markers, then text tokens by label order after the images
    s0 = BA[-1][1] + 2
    txt = [(toksA[p_], p_) for p_ in range(s0, len(toksA))]
    pos = []; ti = 0
    for u in d["units"]:
        if u.startswith("img") and u[3:].isdigit():
            b = int(u[3:]) - 1; pos.append(list(range(BA[b][0] - 1, BA[b][1] + 2)))
        else:
            while ti < len(txt) and txt[ti][0] != u:
                ti += 1
            pos.append([txt[ti][1]] if ti < len(txt) else []); ti += 1
    full = d["full"]
    sAB = max(abs(full["A"]), abs(full["B"])); sC = max(abs(full["Cc"]), abs(full["Cs"]))
    mp, ap = {}, {}
    for m in ("A", "B", "Cc", "Cs"):
        sc = sAB if m in ("A", "B") else sC
        X = np.array(d["maps"][m]) / sc
        At = np.array(A["maps"][m]); scA = max(abs(A["total"]["A"]), abs(A["total"]["B"])) if m in ("A", "B") else max(abs(A["total"]["Cc"]), abs(A["total"]["Cs"]))
        P = np.array([[At[l, ix].sum() if ix else 0.0 for ix in pos] for l in d["layers"]]) / scA
        mp[m] = q8(X); ap[m] = q8(P)
    ex.append(dict(tag=tag, name=name, kind=kind, layers=d["layers"], D=d["D"], units=[u.replace("\n", "\\n") for u in d["units"]],
                   beh={k: round(x, 3) for k, x in d["behaviour"].items()}, tot={k: round(x, 2) for k, x in full.items()},
                   compare=d.get("compare", {}), maps=mp, atp=ap))
    print(f"실제 패칭 {name}: 단위 {len(d['units'])} 층 {len(d['layers'])}", file=sys.stderr)
TL = "/home/storage2/ai26269345/vlm/results/s626_timeline.json"                      # exp/s626_timeline.py output, if built
tl = open(TL).read() if os.path.exists(TL) else "[]"
open(sys.argv[1], "w").write("window.S626 = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\nwindow.S626X = " +
                             json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + ";\nwindow.S626T = " + tl + ";\n")
print(f"-> {sys.argv[1]} ({os.path.getsize(sys.argv[1]) / 1e6:.2f} MB, 모델 {len(out)})", file=sys.stderr)
