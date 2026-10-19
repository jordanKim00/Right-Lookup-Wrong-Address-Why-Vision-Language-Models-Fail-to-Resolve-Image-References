"""S626 (2b): exact interchange at the units the page shows -- each image block (all its tokens with its two markers) and each
text token after the images -- on a layer grid, for the same scenes as s626_atp.py (same seed and sampling order), to check
the attribution-patching maps beyond their top cells (mid-layer image-block and ordinal-word cells).
Unit u at layer l: the base's layer-l output at u's positions <- the source's. Effect = metric(patched) - metric(unpatched),
both rows of the same batched forward (row 0 is the unpatched base), averaged over scenes. Sources and metrics as s626_atp.py:
A (block 5 swapped), B (block 3 swapped), C ("third" -> "second": Cc composed, Cs surface). C skips image units (identical).
The AtP map of the same run tag (--atp TAG) is summed over the same positions and compared cell by cell.
--units key: blocks 2,3,5,6,7 + the question's reference words + the tokens after the question (for the 64-layer model).
Usage: python s626_patch.py MODEL TAG --atp ATPTAG [--n 8] [--every 2] [--units all|key] [--bs 16]
                            [--repair CKPT --reader R] [--device_map ... --max_memory ...]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N, K = 7, 3


def main(a, v=None):
    mm = ({(int(k) if k.isdigit() else k): x for k, x in (kv.split("=") for kv in a.max_memory.split(","))} if a.max_memory else None)
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map, max_memory=mm) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    rep = None
    if a.repair:
        from r3_repair import R3Repair
        rep = R3Repair(v, a.repair, a.reader)
    if a.assembled:                                                                # S725: the per-model R³ (exp/r3_select.py)
        import r3_select
        rep = r3_select.build(v, a.base, a.assembled)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    D = len(v.layers)
    layers = [l for l in range(D) if (l + 1) % a.every == 0]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda l: ids1(l)[0]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(62610)                                                     # = s626_atp.py: the same scenes in the same order
    ST = {}

    def mk(li):
        def h(m, i, o):
            mode = ST.get("mode")
            if mode is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if mode == "cap" and li in layers:
                ST["H"][li] = t[0].detach().clone()
            elif mode == "patch" and li == ST["layer"]:
                t = t.clone()
                for b, pos in enumerate(ST["poss"]):
                    t[b + 1, pos] = ST["src"][li][pos].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return h
    for li, L in enumerate(v.layers):
        L.register_forward_hook(mk(li))

    def enc(ordw, cands, pics):
        q = f"What is in the {ordw} image from the end? Answer with only one word from: {', '.join(cands)}."
        t = v.chat([("user", f"{v.IMG*N}\n{q}")], prefix_assistant=THINK)
        return rep.plan(t, pics) if rep is not None else (v.encode(t, pics), None)

    @torch.no_grad()
    def run(x, mode=None, nrow=1, **kw):
        inp, writes = x
        if nrow > 1:
            inp = {k: (t.repeat(nrow, *[1] * (t.dim() - 1)) if torch.is_tensor(t) else t) for k, t in inp.items()}
        ST.clear(); ST.update(mode=mode, H={}, **kw)
        if rep is not None:
            rep.ST = dict(writes=writes)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        H = ST["H"]; ST.clear()
        if rep is not None:
            rep.ST = {}
        return out.logits[:, -1].float(), H

    E = None; full = {m: [] for m in ("A", "B", "Cc", "Cs")}; beh = {"rev": 0, "surf": 0}; drift = []
    t0 = time.time()
    split = a.cond == "split"                                                      # S646: success-/failure-conditioned maps
    got = {"rev": 0, "fail": 0}; per = []; scanned = 0; Esp = None; fsp = {"rev": {m: [] for m in full}, "fail": {m: [] for m in full}}
    n_ = -1
    while (sum(got.values()) < 2 * a.want and scanned < a.max_scan) if split else (n_ + 1 < a.n):
        n_ += 1; scanned += 1
        labs = rng.sample(labels, N + 2); scene_l, newA, newB = labs[:N], labs[N], labs[N + 1]
        imgs = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene_l]
        cands = list(labs); rng.shuffle(cands)
        tgt, surf = N + 1 - K, K
        picsA = list(imgs); picsA[tgt - 1] = proc_image(rng.choice(pool[newA]))[0].resize((a.pixels, a.pixels))
        picsB = list(imgs); picsB[surf - 1] = proc_image(rng.choice(pool[newB]))[0].resize((a.pixels, a.pixels))
        xb = enc("third", cands, imgs)
        lg0 = run(xb)[0][0]
        ans = cands[int(torch.stack([lg0[tid(c)] for c in cands]).argmax())]
        cls = "rev" if ans == scene_l[tgt - 1] else "fail"
        if split and got[cls] >= a.want:
            continue
        got[cls] += 1
        beh["rev"] += ans == scene_l[tgt - 1]; beh["surf"] += ans == scene_l[surf - 1]
        fm = {"A": lambda g: g[tid(newA)] - g[tid(scene_l[tgt - 1])], "B": lambda g: g[tid(newB)] - g[tid(scene_l[surf - 1])],
              "Cc": lambda g: g[tid(scene_l[N + 1 - 2 - 1])] - g[tid(ans)], "Cs": lambda g: g[tid(scene_l[2 - 1])] - g[tid(ans)]}
        ids = xb[0]["input_ids"][0].tolist()
        if E is None:
            B = v.image_blocks(xb[0]["input_ids"], N)
            units = [("img%d" % (b + 1), list(range(int(B[b][0]) - 1, int(B[b][-1]) + 2))) for b in range(N)]
            s0 = int(B[-1][-1]) + 2
            txt = [(v.tok.decode([ids[p]]), [p]) for p in range(s0, len(ids))]
            if a.units == "key":
                keep_img = {"img2", "img3", "img5", "img6", "img7"}
                dot = max(i for i, (w, _) in enumerate(txt) if w.strip() == ".")
                refw = {"third", "image", "from", "the", "end", "?"}
                q_end = next(i for i, (w, _) in enumerate(txt) if w.strip() == "?")
                units = [u for u in units if u[0] in keep_img] + [u for i, u in enumerate(txt) if (i <= q_end and u[0].strip() in refw) or i > dot]
            else:
                units = units + txt
            E = {m: np.zeros((len(layers), len(units))) for m in fm}
            Esp = {c: {m: np.zeros((len(layers), len(units))) for m in fm} for c in ("rev", "fail")}
            toks = [u for u, _ in units]
            print(f"[{a.tag}] 단위 {len(units)} · 층 {layers}", flush=True)
        Es = {m: np.zeros((len(layers), len(units))) for m in fm}
        for s_, ms in (("A", ["A"]), ("B", ["B"]), ("C", ["Cc", "Cs"])):
            xs = {"A": enc("third", cands, picsA), "B": enc("third", cands, picsB), "C": enc("second", cands, imgs)}[s_]
            assert xs[0]["input_ids"].shape == xb[0]["input_ids"].shape
            lgs, Hs = run(xs, "cap")
            for m in ms:
                full[m].append(float(fm[m](lgs[0]) - fm[m](lg0))); fsp[cls][m].append(full[m][-1])
            todo = [(ui, pos) for ui, (nm, pos) in enumerate(units) if not (s_ == "C" and nm.startswith("img"))]
            for lj, li in enumerate(layers):
                for c0 in range(0, len(todo), a.bs):
                    chunk = todo[c0:c0 + a.bs]
                    lg, _ = run(xb, "patch", nrow=len(chunk) + 1, layer=li, poss=[torch.tensor(p) for _, p in chunk], src=Hs)
                    for m in ms:
                        base = float(fm[m](lg[0]))
                        drift.append(abs(base - float(fm[m](lg0))))
                        for r_, (ui, _) in enumerate(chunk):
                            Es[m][lj, ui] += float(fm[m](lg[r_ + 1])) - base
            del Hs
        for m in Es:
            E[m] += Es[m]; Esp[cls][m] += Es[m]
        per.append(dict(cls=cls, ans=ans, full={m: fsp[cls][m][-1] for m in fm}, maps={m: Es[m].tolist() for m in fm}))
        torch.cuda.empty_cache()
        print(f"[{a.tag}] 장면 {len(per)} (훑음 {scanned}, 정답 {got['rev']} · 실패 {got['fail']})  ({time.time()-t0:.0f}s)", flush=True)
    nused = len(per)
    E = {m: x / nused for m, x in E.items()}
    beh = {k_: x * a.n / nused for k_, x in beh.items()}                          # keeps the json's behaviour = rate below
    tot = {m: float(np.mean(x)) for m, x in full.items()}
    print(f"[{a.tag}] 원 행동: 역순 정답 {beh['rev']/a.n:.2f} · 표면 {beh['surf']/a.n:.2f} · 전체 교체 " +
          " · ".join(f"{m} {x:+.2f}" for m, x in tot.items()) + f" · 배치 기준 어긋남 최대 {max(drift):.4f}", flush=True)
    cmp_ = {}
    if a.atp:
        d = json.load(open(f"/vlm/results/r3/s626_{a.atp}_atp.json"))
        for m in E:
            At = np.array(d["maps"][m])
            P = np.array([[At[li, pos].sum() for _, pos in units] for li in layers])
            X = E[m]; ok = np.abs(X) > 0
            r = float(np.corrcoef(P[ok].ravel(), X[ok].ravel())[0, 1]) if ok.sum() > 2 else float("nan")
            big = np.abs(X) >= .1 * max(1e-6, np.abs(X).max())
            ratio = float(np.median(P[big] / X[big])) if big.any() else float("nan")
            sign = float(np.mean(np.sign(P[big]) == np.sign(X[big]))) if big.any() else float("nan")
            cmp_[m] = dict(r=r, ratio_med=ratio, sign=sign, n_big=int(big.sum()))
            print(f"[{a.tag}] {m}: AtP 합 대 실제 상관 {r:.3f} · 큰 칸 {int(big.sum())} 비 중앙 {ratio:.2f} · 부호 일치 {sign:.2f}", flush=True)
    json.dump(dict(tag=a.tag, D=D, layers=layers, units=toks, behaviour={k_: x / a.n for k_, x in beh.items()}, full=tot,
                   maps={m: x.tolist() for m, x in E.items()}, atp=a.atp, compare=cmp_, drift=max(drift),
                   split=({c: dict(n=got[c], full={m: float(np.mean(fsp[c][m])) if fsp[c][m] else None for m in fsp[c]},
                                   maps={m: (Esp[c][m] / got[c]).tolist() if got[c] else None for m in Esp[c]}) for c in got} if split else None),
                   scanned=scanned, per=per),
              open(f"/vlm/results/r3/s626_{a.tag}_patch.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s626_{a.tag}_patch.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--atp", default=None)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--units", default="all", choices=["all", "key"]); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--repair", default=None); ap.add_argument("--reader", default=None)
    ap.add_argument("--assembled", default=None); ap.add_argument("--base", default=None)
    ap.add_argument("--device_map", default=None); ap.add_argument("--max_memory", default=None)
    ap.add_argument("--cond", default="all", choices=["all", "split"])                # S646
    ap.add_argument("--want", type=int, default=8); ap.add_argument("--max_scan", type=int, default=80)
    main(ap.parse_args())
