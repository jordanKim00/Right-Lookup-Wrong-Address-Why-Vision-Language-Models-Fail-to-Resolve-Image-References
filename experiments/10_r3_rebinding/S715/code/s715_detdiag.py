"""S715-diag: why R³-v2's ordinal detector misses option-line ordinals on MMIU / MuirBench (27B) -- a READ-stage measurement on the
first-use (sealed) items, which are development data for the READ fix from now on (knowledge/DECISIONS.md 2026-10-09 16:00).
For every input the S582 harness builds (same items, prompts and order; forward and end-relative forms), the capture forward of
R³-v2 (exp/r3_min_md.R3MinMD) is run once and, for every token of the text span after the last image, the L_b state's distances are
saved: nearest ordinal prototype and its distance in the detector chart and in the full state, the second-nearest distance, and the
distances to the end / last frame anchors; plus the end-reader codes.  No answer is read and no write is made (the harness's base /
repaired answers in the dummy record are placeholders).
Usage: python s715_detdiag.py MODEL TAG --base B --suites mmiu_pos,muir --sealed [--device_map balanced]
       python s715_detdiag.py MODEL TAG --base B --procedural 4      (S712's procedural inputs, first 4 scenes per N)
"""
from __future__ import annotations

import argparse, json, sys
import torch

sys.path.insert(0, "/vlm/exp")
import s582_evalrepair as E
from uni import Uni
from r3_min_md import R3MinMD
from s588c_reader import probs as reader_probs


class DiagRep(R3MinMD):
    def __init__(self, v, calfile):
        super().__init__(v, calfile, v2=True)
        self.rec = []

    @torch.no_grad()
    def both(self, text, pics, cand):
        v, cal = self.v, self.cal
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = [torch.tensor(b) for b in v.image_blocks(inp["input_ids"], len(pics))]
        turn = max(i for i, x in enumerate(ids) if x == self.im_start)
        s0 = int(B[-1][-1]) + 2
        self.ST = dict(cap=True, span=torch.arange(s0, turn), B=B)
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        H, bf = self.ST["H"], self.ST.get("bf", {})
        self.ST = {}
        Hb = H[self.Lb]
        Dc = torch.cdist(Hb @ cal["Pbb"], cal["Mb"] @ cal["Pbb"])                  # [T, C] chart distances
        Df = torch.cdist(Hb, cal["Mb"])                                             # [T, C] full-state distances
        dc2 = Dc.topk(2, largest=False)
        fr = cal["frame"]
        X = torch.cat([bf[l] for l in self.Rlayers], 1)
        codes = (reader_probs(self.RP, X).argmax(1) + 1).long().tolist()
        r = lambda x: [round(float(y), 3) for y in x]
        self.rec.append(dict(n_img=len(B), codes=codes, toks=[v.tok.decode([t]) for t in ids[s0:turn]],
                             kh=(dc2.indices[:, 0] + 1).tolist(), dc=r(dc2.values[:, 0]), dc2=r(dc2.values[:, 1]),
                             df=r(Df.min(1).values), d_end=r((Hb - fr["mu"]["end"]).norm(dim=1)), d_last=r((Hb - fr["mu"]["last"]).norm(dim=1))))
        return 0, 0


def procedural(v, rep, n):
    """S712's procedural inputs (same generator, seed 71200, first n scenes per N): END and START texts, distances only."""
    import random
    from r3ref import ORD, load_image
    from r3_proc import proc_image
    from s608_layout import body as layout_body
    LAYS, ENDW, STARTP = ("first", "newline", "picture"), ("last", "end"), (" from the start", " from the beginning", "")
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    tok1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    keys = sorted({k for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})
    by_lab = {}
    for kk in keys[:3000]:
        lb = proc_image(int(kk.split(":")[1]))[1]
        if len(tok1(lb)) == 1:
            by_lab.setdefault(lb, []).append(kk)
    labels = sorted(l for l, x in by_lab.items() if len(x) >= 10)
    rng = random.Random(71200)
    for N in (3, 4, 6, 8, 9):
        for si in range(12):
            lay, ew = LAYS[si % 6 // 2], ENDW[si % 2]
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [load_image(rng.choice(by_lab[l])) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            if si >= n:
                continue
            ans = f" Answer with only one word from: {', '.join(menu)}."
            for k in range(1, N + 1):
                for q in (f"What is in the {ORD[k-1]} image from the {ew}?" + ans, f"What is in the {ORD[k-1]} image{STARTP[(k - 1) % 3]}?" + ans):
                    rep.both(v.chat([("user", layout_body(v, lay, N, q))], prefix_assistant=THINK), pics, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True)
    ap.add_argument("--suites", default="mmiu_pos,muir"); ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--per_task", type=int, default=50); ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--procedural", type=int, default=0)                             # n scenes per N of S712's inputs instead of the harness
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    rep = DiagRep(v, f"/vlm/results/r3/s709_{a.base}_cal.pt")
    if a.procedural:
        procedural(v, rep, a.procedural)
    else:
        E.run(v, rep, a.tag, set(a.suites.split(",")), per_task=a.per_task, cap=a.cap, sealed=a.sealed)
    cal = rep.cal
    json.dump(dict(tag=a.tag, base=a.base, th_chart=cal["th_chart"], th_full=cal["th_full"], thr=cal["frame"]["thr"], rec=rep.rec),
              open(f"/vlm/results/r3/s715_{a.tag}_detdiag.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s715_{a.tag}_detdiag.json ({len(rep.rec)} inputs)", flush=True)
