"""S626 (1)-(2): full token x layer maps of the reverse-ordinal computation (exp/QUEUE.md S626, amended design).
Fixed example: 7 PROC images (160 px, distinct single-token shapes), "What is in the third image from the end? Answer with only one
word from: <7 scene shapes + 2 new shapes>" (target block 5, surface block 3), 24 scenes. Counterfactual sources, token-aligned:
  A  block 5 (target) replaced by a new shape        metric logit(new) - logit(old block-5 shape)
  B  block 3 (surface) replaced by a new shape       metric logit(new) - logit(old block-3 shape)
  C  "third" -> "second" (target 6, surface 2)       metrics  Cc: logit(block 6) - logit(base answer)   Cs: logit(block 2) - logit(base answer)
Attribution patching at every layer output and every token (image patches included):
    effect(l, t) = sum_d (h_src[l, t] - h_base[l, t]) * dM/dh_base[l, t]
averaged over scenes. Check: the top cells of each map are re-measured by exact patching (h_base[l, t] <- h_src[l, t]).
--repair CKPT: the R3 module runs in base and sources (the transplanted computation).
--seg S (models whose full backward does not fit, e.g. Qwen3.5-27B on three cards): the same gradients by segment-wise
backprop, top segment first. For the segment [a, b] the forward runs with the input of layer a as a fresh leaf and stops
after layer b; backward(h_b, g_b) with g_b = dM/dh_b from the segment above gives dM/dh_l for l in [a, b] and, at the leaf,
g_{a-1}. Only one segment's graph is alive at a time; the maps are the same gradients (checked against the full backward).
Usage: python s626_atp.py MODEL TAG [--n 24] [--repair CKPT --reader R] [--device_map balanced] [--seg S]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N, K = 7, 3


def main(a):
    mm = ({(int(k) if k.isdigit() else k): x for k, x in (kv.split("=") for kv in a.max_memory.split(","))} if a.max_memory else None)
    v = (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map, max_memory=mm) if a.device_map
         else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    rep = None
    if a.repair:
        from r3_repair import R3Repair
        rep = R3Repair(v, a.repair, a.reader)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    D = len(v.layers)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    print(f"[{a.tag}] 한 토큰 도형 {labels}", flush=True)
    rng = random.Random(62610)

    ST = {}

    class _Stop(Exception):
        pass

    def mkpre(li):                                                                 # a leaf at layer li's input: grads for every layer above
        def pre(m, args, kwargs):
            if ST.get("grad") and ST.get("leaf_at", 0) == li:
                h = args[0] if args else kwargs["hidden_states"]
                h = h.detach().requires_grad_(True); ST["leaf"] = h
                if args:
                    return (h,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs); kwargs["hidden_states"] = h
                return args, kwargs
            return None
        return pre

    def mk(li):
        def h(m, i, o):
            mode = ST.get("mode")
            if mode is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if mode == "cap":
                ST["H"][li] = t[0].detach()
            elif mode == "grad":
                if li >= ST.get("leaf_at", 0):
                    t.retain_grad(); ST["H"][li] = t
                if li == ST.get("stop_at", -1):
                    raise _Stop
            elif mode == "patch" and li == ST["layer"]:
                t = t.clone(); t[0, ST["pos"]] = ST["src"].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return h
    for li, L in enumerate(v.layers):
        L.register_forward_pre_hook(mkpre(li), with_kwargs=True)
    for li, L in enumerate(v.layers):
        L.register_forward_hook(mk(li))

    def text(ordw, cands):
        q = f"What is in the {ordw} image from the end? Answer with only one word from: {', '.join(cands)}."
        return v.chat([("user", f"{v.IMG*N}\n{q}")], prefix_assistant=THINK)

    def enc(ordw, cands, pics):
        """(input, R3 writes) -- the writes depend on the input only and are re-applied in every forward."""
        t = text(ordw, cands)
        if rep is not None:
            return rep.plan(t, pics)
        return v.encode(t, pics), None

    def forward(x, mode, **kw):
        inp, writes = x
        ST.clear(); ST.update(mode=mode, H={}, grad=(mode == "grad"), **kw)
        if rep is not None:
            rep.ST = dict(writes=writes)
        ctx = torch.enable_grad() if mode == "grad" else torch.no_grad()
        with ctx:
            out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        H = ST["H"]; ST.clear()
        if rep is not None:
            rep.ST = {}
        return out.logits[0, -1].float(), H

    maps = {m: None for m in ("A", "B", "Cc", "Cs")}
    tot = {m: [] for m in maps}; beh = {"rev": 0, "surf": 0}
    keep = []
    t0 = time.time()
    for n_ in range(a.n):
        labs = rng.sample(labels, N + 2); scene_l, newA, newB = labs[:N], labs[N], labs[N + 1]
        imgs = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene_l]
        cands = list(labs); rng.shuffle(cands)
        tgt, surf = N + 1 - K, K                                                   # blocks 5 and 3
        picsA = list(imgs); picsA[tgt - 1] = proc_image(rng.choice(pool[newA]))[0].resize((a.pixels, a.pixels))
        picsB = list(imgs); picsB[surf - 1] = proc_image(rng.choice(pool[newB]))[0].resize((a.pixels, a.pixels))
        tid = lambda l: ids1(l)[0]
        inpb = enc("third", cands, imgs)
        if a.seg:                                                                  # segment-wise backprop (see the docstring)
            srcs = {"A": enc("third", cands, picsA), "B": enc("third", cands, picsB), "C": enc("second", cands, imgs)}
            lg, _ = forward(inpb, "none")
            ans = cands[int(torch.stack([lg[tid(c)] for c in cands]).argmax())]
            beh["rev"] += ans == scene_l[tgt - 1]; beh["surf"] += ans == scene_l[surf - 1]
            fm = {"A": lambda g: g[tid(newA)] - g[tid(scene_l[tgt - 1])], "B": lambda g: g[tid(newB)] - g[tid(scene_l[surf - 1])],
                  "Cc": lambda g: g[tid(scene_l[N + 1 - 2 - 1])] - g[tid(ans)], "Cs": lambda g: g[tid(scene_l[2 - 1])] - g[tid(ans)]}
            Hs = {}
            for s_, inp in srcs.items():
                lgs, H = forward(inp, "cap")
                Hs[s_] = H
                for m in fm:
                    if (m.startswith("C") and s_ == "C") or m == s_:
                        tot[m].append(float(fm[m](lgs) - fm[m](lg)))
            Es = {m: torch.zeros(D, inpb[0]["input_ids"].shape[1]) for m in fm}
            g = None
            for b_ in range(D - 1, -1, -a.seg):
                a_ = max(0, b_ - a.seg + 1)
                inp, writes = inpb
                ST.clear(); ST.update(mode="grad", H={}, grad=True, leaf_at=a_, stop_at=(b_ if b_ < D - 1 else -1))
                if rep is not None:
                    rep.ST = dict(writes=writes)
                lgt = None
                with torch.enable_grad():
                    try:
                        lgt = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
                    except _Stop:
                        pass
                H, leaf = ST["H"], ST.get("leaf"); ST.clear()
                if rep is not None:
                    rep.ST = {}
                gn = {}
                for m in fm:
                    for li in H:
                        H[li].grad = None
                    leaf.grad = None
                    if lgt is not None:
                        fm[m](lgt).backward(retain_graph=True)
                    else:
                        torch.autograd.backward(H[b_], grad_tensors=g[m].to(H[b_].device), retain_graph=True)
                    s_ = "C" if m.startswith("C") else m
                    for li in range(a_, b_ + 1):
                        gl = g[m].to(H[li].device) if (lgt is None and li == b_) else H[li].grad
                        Es[m][li] = ((Hs[s_][li].float() - H[li][0].detach().float()) * gl[0].float()).sum(-1).cpu()
                    gn[m] = leaf.grad.detach().clone() if a_ > 0 else None
                g = gn
                del H, leaf, lgt
            for m in maps:
                maps[m] = Es[m] if maps[m] is None else maps[m] + Es[m]
            if len(keep) < 8:
                keep.append(dict(inpb=inpb, srcs=srcs, metric_fn=(newA, newB, scene_l, ans, tgt, surf, cands), Hs=None))
            del Hs
            torch.cuda.empty_cache()
            if (n_ + 1) % 6 == 0:
                print(f"[{a.tag}] 장면 {n_+1}/{a.n}  ({time.time()-t0:.0f}s)", flush=True)
            continue
        lg, Hb = forward(inpb, "grad")
        ans = cands[int(torch.stack([lg[tid(c)] for c in cands]).argmax())]
        beh["rev"] += ans == scene_l[tgt - 1]; beh["surf"] += ans == scene_l[surf - 1]
        metric = {"A": lg[tid(newA)] - lg[tid(scene_l[tgt - 1])], "B": lg[tid(newB)] - lg[tid(scene_l[surf - 1])],
                  "Cc": lg[tid(scene_l[N + 1 - 2 - 1])] - lg[tid(ans)], "Cs": lg[tid(scene_l[2 - 1])] - lg[tid(ans)]}
        grads = {}
        for m, M in metric.items():
            for li in Hb:
                if Hb[li].grad is not None:
                    Hb[li].grad = None
            M.backward(retain_graph=True)
            grads[m] = {li: Hb[li].grad[0].detach().float() for li in Hb}
        hb = {li: Hb[li][0].detach().float() for li in Hb}
        srcs = {"A": enc("third", cands, picsA), "B": enc("third", cands, picsB), "C": enc("second", cands, imgs)}
        Hs, Ms = {}, {}
        for s_, inp in srcs.items():
            lgs, H = forward(inp, "cap")
            Hs[s_] = {li: H[li].float() for li in H}
            Ms[s_] = lgs
        for m in maps:
            s_ = "C" if m.startswith("C") else m
            E = torch.stack([((Hs[s_][li] - hb[li]) * grads[m][li]).sum(-1).cpu() for li in range(D)])   # [D, T]
            maps[m] = E if maps[m] is None else maps[m] + E
            lgs = Ms[s_]
            mval = {"A": lgs[tid(newA)] - lgs[tid(scene_l[tgt - 1])], "B": lgs[tid(newB)] - lgs[tid(scene_l[surf - 1])],
                    "Cc": lgs[tid(scene_l[N + 1 - 2 - 1])] - lgs[tid(ans)], "Cs": lgs[tid(scene_l[2 - 1])] - lgs[tid(ans)]}[m]
            tot[m].append(float(mval - metric[m].detach()))
        if len(keep) < 8:
            keep.append(dict(inpb=inpb, srcs=srcs, metric_fn=(newA, newB, scene_l, ans, tgt, surf, cands), Hs=None))
        del Hb, grads, hb, Hs
        torch.cuda.empty_cache()
        if (n_ + 1) % 6 == 0:
            print(f"[{a.tag}] 장면 {n_+1}/{a.n}  ({time.time()-t0:.0f}s)", flush=True)
    maps = {m: (x / a.n).numpy() for m, x in maps.items()}
    ids = inpb[0]["input_ids"][0].tolist()
    B = v.image_blocks(inpb[0]["input_ids"], N)
    toks = [v.tok.decode([t]) for t in ids]
    print(f"[{a.tag}] 원 행동: 역순 정답 {beh['rev']/a.n:.2f} · 표면 {beh['surf']/a.n:.2f} · 전체 교체 효과(로짓 차) " +
          " · ".join(f"{m} {np.mean(x):+.2f}" for m, x in tot.items()), flush=True)
    # exact patching check at the top cells of each map
    check = {}
    for m, E in maps.items():
        flat = np.argsort(-np.abs(E).ravel())[:5]
        rows = []
        for f in flat:
            li, tpos = int(f // E.shape[1]), int(f % E.shape[1])
            vals = []
            for kp in keep:
                newA, newB, scene_l, ans, tgt, surf, cands = kp["metric_fn"]
                tid = lambda l: ids1(l)[0]
                s_ = "C" if m.startswith("C") else m
                _, Hsrc = forward(kp["srcs"][s_], "cap")
                inpb = kp["inpb"]
                lg0, _ = forward(inpb, "cap")
                lg1, _ = forward(inpb, "patch", layer=li, pos=tpos, src=Hsrc[li][tpos])
                f_ = {"A": lambda g: g[tid(newA)] - g[tid(scene_l[tgt - 1])], "B": lambda g: g[tid(newB)] - g[tid(scene_l[surf - 1])],
                      "Cc": lambda g: g[tid(scene_l[N + 1 - 2 - 1])] - g[tid(ans)], "Cs": lambda g: g[tid(scene_l[2 - 1])] - g[tid(ans)]}[m]
                vals.append(float(f_(lg1) - f_(lg0)))
            rows.append(dict(layer=li, pos=tpos, tok=toks[tpos], atp=float(E[li, tpos]), exact=float(np.mean(vals))))
        check[m] = rows
        print(f"[{a.tag}] {m} 상위 칸 확인: " + " · ".join(f"L{r['layer']}@{r['pos']}({r['tok'].strip()[:8]!r}) AtP {r['atp']:+.2f}/실제 {r['exact']:+.2f}" for r in rows), flush=True)
    json.dump(dict(tag=a.tag, D=D, N=N, K=K, toks=toks, blocks=[[int(b[0]), int(b[-1])] for b in B], behaviour={k_: x / a.n for k_, x in beh.items()},
                   total={m: float(np.mean(x)) for m, x in tot.items()}, maps={m: x.tolist() for m, x in maps.items()}, check=check),
              open(f"/vlm/results/r3/s626_{a.tag}_atp.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s626_{a.tag}_atp.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--repair", default=None); ap.add_argument("--reader", default=None)
    ap.add_argument("--device_map", default=None); ap.add_argument("--seg", type=int, default=0)
    ap.add_argument("--max_memory", default=None)                                   # e.g. 0=19GiB,1=19GiB,cpu=60GiB
    main(ap.parse_args())
