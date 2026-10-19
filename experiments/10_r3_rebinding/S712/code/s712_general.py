"""S712: independent procedural check of R³-v2 on Qwen3.5-27B before natural benchmarks -- other scene sizes, prompt layouts and end
words than S709 / S711, and forward (START) questions as the no-harm control.  Pre-registration in knowledge/DECISIONS.md before any
S712 data.  R³-v2 frozen as registered in S711 (exp/r3_min_md.R3MinMD, v2 = True; readers and calibration exp/s709_cal.py, base r327b).
Fresh procedural scenes from the dev pool (seed 71200), N in {3, 4, 6, 8, 9} (S709 / S711 used 5 and 7), 12 scenes per N; each scene
gets one prompt layout (exp/s608_layout: images first / newline-separated / "Picture k: <image>") and one end word ("last" / "end"),
cycled over the six combinations.  Every k is asked END ("What is in the k-th image from the <end word>?") and START (cycled over
"from the start" / "from the beginning" / no frame phrase) with the S709 answer menu (scene labels + up to 2 unused labels; the dev
pool has 9 single-token labels, so N8 gets 1 extra and N9 none -- amendment before data).
Arms: END -- NATIVE, R3V2, O_ORD_FR (oracle ordinal + frame writes with the same prototypes); START -- NATIVE, R3V2.
Registered rules (analysis exp/s712_analyze.py):
    GEN_REPAIR:      END P(j | R3V2) >= .90 overall and >= P(j | NATIVE) + .20
    GEN_FACTORS:     END P(j | R3V2) >= .85 within every N, every layout and every end word (marginals)
    ORACLE_FRACTION: END P(j | R3V2) >= .90 P(j | O_ORD_FR) overall
    START_NO_HARM:   START P(k | R3V2) >= P(k | NATIVE) - .02, and R3V2 writes something on <= .05 of START items
    Predictions: all four.
Usage: python s712_general.py MODEL TAG --base B [--n 12] [--device_map balanced] [--seed 71200]
"""
from __future__ import annotations

import argparse, json, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3ref import ORD, load_image
from s608_layout import body as layout_body
from r3_min_md import R3MinMD

LAYS = ("first", "newline", "picture")
ENDW = ("last", "end")
STARTP = (" from the start", " from the beginning", "")


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rep = R3MinMD(v, f"/vlm/results/r3/s709_{a.base}_cal.pt", v2=True)
    cal, fr = rep.cal, rep.cal["frame"]
    tok1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: tok1(s)[0]
    keys = sorted({k for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})          # dev pool, as S709
    from r3_proc import proc_image
    by_lab = {}
    for kk in keys[:3000]:
        lb = proc_image(int(kk.split(":")[1]))[1]
        if len(tok1(lb)) == 1:
            by_lab.setdefault(lb, []).append(kk)
    labels = sorted(l for l, x in by_lab.items() if len(x) >= 10)
    assert len(labels) >= 9, len(labels)                                         # 9 labels: menu extras 2 (N <= 7) · 1 (N8) · 0 (N9)

    def land(lg, labs, menu):
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return labs.index(lab) + 1 if lab in labs else 0

    @torch.no_grad()
    def run(inp, writes):
        rep.ST = dict(writes=writes)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        rep.ST = {}
        return out.logits[0, -1].float()

    @torch.no_grad()
    def native_and_plan(text, pics):
        inp, writes = rep.plan(text, pics)                                       # the capture forward is unedited: its logits = NATIVE
        return inp, writes, rep.last_logits

    def find_ord(ids, k, lo, hi):
        tw, im = tok1(ORD[k - 1]), tid("image")
        hits = [i + len(tw) - 1 for i in range(lo, hi - len(tw)) if ids[i:i + len(tw)] == tw and ids[i + len(tw)] == im]
        return hits[-1]

    def ledger():
        return [dict(kh=d["kh"], t=d["t"], act=d["act"], fonly=d["fonly"], anchor=d["anchor"], N_hat=d["N_hat"], pos=d["pos"]) for d in rep.last_detail]

    rng = random.Random(a.seed)
    rows = []
    for N in (3, 4, 6, 8, 9):
        for si in range(a.n):
            lay, ew = LAYS[si % 6 // 2], ENDW[si % 2]
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
            pics = [load_image(rng.choice(by_lab[l])) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            ans = f" Answer with only one word from: {', '.join(menu)}."
            for k in range(1, N + 1):
                j = N + 1 - k
                q = f"What is in the {ORD[k-1]} image from the {ew}?" + ans
                text = v.chat([("user", layout_body(v, lay, N, q))], prefix_assistant=THINK)
                inp, writes, lg = native_and_plan(text, pics)
                rec = dict(kind="END", N=N, si=si, k=k, j=j, layout=lay, word=ew, land=dict(NATIVE=land(lg, labs, menu)),
                           wrote=bool(writes), ledger=ledger())
                rec["land"]["R3V2"] = land(run(inp, writes), labs, menu) if writes else rec["land"]["NATIVE"]
                ids = inp["input_ids"][0].tolist()
                s0 = int(v.image_blocks(inp["input_ids"], N)[-1][-1]) + 2
                turn = max(i for i, x in enumerate(ids) if x == rep.im_start)
                o = find_ord(ids, k, s0, turn)
                fw = next(i for i in range(o + 1, turn) if ids[i] == tid(ew))
                d_ord = (cal["Mb"][j - 1] - cal["Mb"][k - 1]) if j != k else torch.zeros_like(cal["Mb"][0])
                rec["land"]["O_ORD_FR"] = land(run(inp, [(o, d_ord), (fw, fr["mu_start"] - fr["mu"][ew])]), labs, menu)
                rec.update(ord_pos=o, fw_pos=fw)
                rows.append(rec)
                sp = STARTP[(k - 1) % 3]
                qs = f"What is in the {ORD[k-1]} image{sp}?" + ans
                text = v.chat([("user", layout_body(v, lay, N, qs))], prefix_assistant=THINK)
                inp, writes, lg = native_and_plan(text, pics)
                rec = dict(kind="START", N=N, si=si, k=k, j=k, layout=lay, word=sp.strip() or "none", land=dict(NATIVE=land(lg, labs, menu)),
                           wrote=bool(writes), ledger=ledger())
                rec["land"]["R3V2"] = land(run(inp, writes), labs, menu) if writes else rec["land"]["NATIVE"]
                rows.append(rec)
            torch.cuda.empty_cache()
            if (si + 1) % 4 == 0:
                print(f"[{a.tag}] N{N} {si + 1}/{a.n}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, base=a.base, Lb=rep.Lb, stats=rep.stats, rows=rows),
              open(f"/vlm/results/r3/s712_{a.tag}_general.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s712_{a.tag}_general.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=71200)                                # smoke runs use another seed
    main(ap.parse_args())
