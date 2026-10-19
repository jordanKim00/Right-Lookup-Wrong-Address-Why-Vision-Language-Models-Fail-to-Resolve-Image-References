"""S626: DAS heatmaps -- token x layer map of where the composed target address lives (exp/QUEUE.md S626).
Task (content answer): "What is in the k-th image from the end? Answer with only one word from: ..." over N PROC images.
Pairs: base (N_b, k_b) and source (N_s, k_s) on different scenes, N and k both differ, the four predictions differ and lie
in 1..N_b:
    j    = N_s+1-k_s   the source's computed target address
    k    = N_b+1-k_s   the source's pointer, reversed in the base context (composition after the site)
    surf = k_s         the source's pointer used as a forward address (no composition)
    base = no effect (reported, never trained)
At each cell (token, layer) and hypothesis a rank-r subspace Q (orthonormal, d x r) is trained so that
    h_b <- h_b + Q Q^T (h_s - h_b)
makes the answer follow the hypothesis (Adam, batch 1, as S463b); IIA on held-out pairs (disjoint scenes). Source states for
every cell are captured once per pair. An interchange moves a component; it cannot compute (rule 288).
--repair CKPT: the R3 module runs in base and source alike (the transplanted computation).
Usage: python s626_dasmap.py MODEL TAG [--nl 10] [--steps 120] [--ntrain 120] [--neval 40] [--rank 4]
                                        [--hyps j,k,surf] [--repair CKPT --reader R] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from r3ref import ORD

SITES = ["ord", "img", "from", "end", "qm", "last"]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    rep = None
    if a.repair:
        from r3_repair import R3Repair
        rep = R3Repair(v, a.repair, a.reader)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    D = len(v.layers)
    layers = sorted({int(round(x)) for x in np.linspace(1, D - 2, a.nl)}) if not a.layers else [int(x) for x in a.layers.split(",")]
    cfg = v.model.config
    d_model = cfg.get_text_config().hidden_size if hasattr(cfg, "get_text_config") else cfg.text_config.hidden_size
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
    qm_ids = set(v.tok("?", add_special_tokens=False)["input_ids"]) | set(v.tok(" ?", add_special_tokens=False)["input_ids"])
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})

    ST = {}

    def mk(li):
        def h(m, i, o):
            if ST.get("mode") is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST["mode"] == "cap":
                for nm, p in ST["pos"].items():
                    ST["got"][(nm, li)] = t[0, p].detach().float().cpu() if li in layers else None
                return o
            if ST["layer"] != li:
                return o
            p = ST["pos"][ST["site"]]; Q = ST["Q"].to(t.device)
            hb = t[0, p].float(); hs = ST["src"].to(t.device)
            new = hb + ((hs - hb) @ Q) @ Q.T
            t = t.clone(); t[0, p] = new.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return h
    hooks = [L.register_forward_hook(mk(li)) for li, L in enumerate(v.layers)]

    def scene(rng, N):
        out, firsts = [], set()
        for _ in range(500):
            r_ = proc_image(rng.choice(pids)); im, lab = r_[0], r_[1]
            f = tid1(lab)
            if f in firsts:
                continue
            out.append((im.resize((a.pixels, a.pixels)), lab)); firsts.add(f)
            if len(out) == N:
                return [x[0] for x in out], [x[1] for x in out]
        raise RuntimeError("scene")

    def sampler(rng):
        while True:
            Nb, Ns = rng.sample([5, 6, 7, 8], 2)
            kb, ks = rng.sample([2, 3, 4, 5], 2)
            p = {"j": Ns + 1 - ks, "k": Nb + 1 - ks, "surf": ks}
            vals = list(p.values()) + [Nb + 1 - kb, kb]
            if len(set(vals)) == 5 and all(1 <= x <= Nb for x in vals) and kb <= Nb and ks <= Ns:
                pb, lb = scene(rng, Nb); ps, ls = scene(rng, Ns)
                cb = list(lb); rng.shuffle(cb); cs = list(ls); rng.shuffle(cs)
                return dict(Nb=Nb, kb=kb, Ns=Ns, ks=ks, pred=p, pb=pb, lb=lb, cb=cb, ps=ps, ls=ls, cs=cs)

    def text(k, cl, N):
        q = f"What is in the {ORD[k-1]} image from the end? Answer with only one word from: {', '.join(cl)}."
        return v.chat([("user", f"{v.IMG*N}\n{q}")], prefix_assistant=THINK)

    def positions(ids, k):
        turn = max(i for i, x in enumerate(ids) if x == im_start)
        o = max(i for i in range(turn) if ids[i] == tid1(ORD[k - 1]))
        e = next(i for i in range(o, turn) if ids[i] == tid1("end"))
        fr = next(i for i in range(o, e) if ids[i] == tid1("from"))
        qm = next((i for i in range(e + 1, min(e + 3, turn)) if ids[i] in qm_ids), e + 1)
        return dict(ord=o, img=o + 1, **{"from": fr}, end=e, qm=qm, last=len(ids) - 1)

    def prep(s, role):
        """encoded input (and the R3 writes, which depend on the input only) cached per pair and role."""
        c = s.setdefault("cache", {})
        if role not in c:
            k, cl, N, pics = (s["kb"], s["cb"], s["Nb"], s["pb"]) if role == "b" else (s["ks"], s["cs"], s["Ns"], s["ps"])
            t = text(k, cl, N)
            if rep is not None:
                inp, writes = rep.plan(t, pics)
            else:
                inp, writes = v.encode(t, pics), None
            c[role] = (inp, writes, positions(inp["input_ids"][0].tolist(), k))
        inp, writes, pos = c[role]
        if rep is not None:
            rep.ST = dict(writes=writes)
        return inp, pos

    def run(inp, grad=False):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        return out.logits[0, -1].float()

    def capture(s):
        inp, pos = prep(s, "s")
        ST.update(mode="cap", pos=pos, got={})
        run(inp)
        got = ST["got"]; ST.clear()
        if rep is not None:
            rep.ST = {}
        return {key: x.half() for key, x in got.items() if x is not None}

    def answer(s, site, layer, Q, src, grad=False):
        inp, pos = prep(s, "b")
        ST.update(mode="int", pos=pos, site=site, layer=layer, Q=Q, src=src)
        lg = run(inp, grad=grad)
        ST.clear()
        if rep is not None:
            rep.ST = {}
        return torch.stack([lg[tid1(c)] for c in s["cb"]])

    rng_tr, rng_ev = random.Random(62600), random.Random(62699)
    t0 = time.time()
    TR = [sampler(rng_tr) for _ in range(a.ntrain)]; EV = [sampler(rng_ev) for _ in range(a.neval)]
    for s in TR + EV:
        s["src"] = capture(s)
        s.pop("ps", None); s.pop("cache", {}).pop("s", None) if False else None
    print(f"[{a.tag}] 층 {layers} · 짝 학습 {len(TR)}/보류 {len(EV)} · 기증 캐시 끝 ({time.time()-t0:.0f}s)", flush=True)
    # base behaviour (no intervention) on the held-out pairs
    beh = {"rev": 0, "surf": 0}
    for s in EV:
        with torch.no_grad():
            inp, _ = prep(s, "b"); lg = run(inp)
            if rep is not None:
                rep.ST = {}
        c = s["cb"][int(torch.stack([lg[tid1(x)] for x in s["cb"]]).argmax())]
        beh["rev"] += c == s["lb"][s["Nb"] - s["kb"]]; beh["surf"] += c == s["lb"][s["kb"] - 1]
    print(f"[{a.tag}] 원 행동(보류): 역순 정답 {beh['rev']/len(EV):.2f} · 표면 {beh['surf']/len(EV):.2f}", flush=True)
    res = {"behaviour": {k_: x / len(EV) for k_, x in beh.items()}}
    hyps = a.hyps.split(",")
    for site in SITES:
        for layer in layers:
            for hyp in hyps:
                W = torch.nn.Parameter(torch.randn(d_model, a.rank, device=v.device) * 0.02)
                opt = torch.optim.Adam([W], lr=a.lr)
                rng = random.Random(62600 + 97 * layer + 13 * SITES.index(site) + hyps.index(hyp))
                for step in range(a.steps):
                    s = TR[rng.randrange(len(TR))]
                    src = s["src"][(site, layer)].float()
                    lg = answer(s, site, layer, torch.linalg.qr(W)[0], src, grad=True)
                    tgt = s["cb"].index(s["lb"][s["pred"][hyp] - 1])
                    loss = torch.nn.functional.cross_entropy(lg[None], torch.tensor([tgt], device=lg.device))
                    opt.zero_grad(); loss.backward(); opt.step()
                Q = torch.linalg.qr(W)[0].detach()
                hit = {h: 0 for h in ("j", "k", "surf", "base_rev", "base_surf")}
                for s in EV:
                    with torch.no_grad():
                        lg = answer(s, site, layer, Q, s["src"][(site, layer)].float())
                    c = s["cb"][int(lg.argmax())]
                    for h in ("j", "k", "surf"):
                        hit[h] += c == s["lb"][s["pred"][h] - 1]
                    hit["base_rev"] += c == s["lb"][s["Nb"] - s["kb"]]; hit["base_surf"] += c == s["lb"][s["kb"] - 1]
                res[f"{site}/L{layer}/{hyp}"] = {h: x / len(EV) for h, x in hit.items()}
                print(f"[{a.tag}] {site:>4s} L{layer:<3d} {hyp:>4s} IIA {hit[hyp]/len(EV):.2f} (j {hit['j']/len(EV):.2f} k {hit['k']/len(EV):.2f} "
                      f"surf {hit['surf']/len(EV):.2f} 원 {hit['base_rev']/len(EV):.2f}/{hit['base_surf']/len(EV):.2f})  ({time.time()-t0:.0f}s)", flush=True)
                torch.cuda.empty_cache()
            json.dump(dict(tag=a.tag, layers=layers, sites=SITES, hyps=hyps, D=D, res=res), open(f"/vlm/results/r3/s626_{a.tag}_dasmap.json", "w"))
    for h in hooks:
        h.remove()
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s626_{a.tag}_dasmap.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--nl", type=int, default=10); ap.add_argument("--layers", default=None)
    ap.add_argument("--steps", type=int, default=120); ap.add_argument("--ntrain", type=int, default=120)
    ap.add_argument("--neval", type=int, default=40); ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--hyps", default="j,k,surf")
    ap.add_argument("--repair", default=None); ap.add_argument("--reader", default=None)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
