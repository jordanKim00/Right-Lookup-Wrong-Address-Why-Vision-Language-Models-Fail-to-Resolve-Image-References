"""S595: reflect the frame as well as the index (pre-registered in exp/QUEUE.md after S594).

A reverse reference is (index k, frame end); its reflection is (N+1-k, frame start). S594: rebinding the index alone leaves
"the j-th image from the end", which both dense models re-read in the end frame for small j (errors on the last image).
Items = S594's generator and seed. Prototypes at the write layer L_b (three formats, ordinals 1-10, 16 each): the ordinal
token (forward texts) and the anchor token of " from the start" / " from the beginning" (forward texts) and of
" from the end" / " from the last" (reverse texts).
Arms: base | index only (ord@L_b) | index + frame (anchor h += mu_start - mu_end|last) | index + frame (beginning) |
frame only (check) | diagnostics: the text "j-th image from the start" / "from the beginning".
"""
from __future__ import annotations
import argparse, collections, json, random, sys, time
import numpy as np
import torch
sys.path.insert(0, "/vlm/exp")
from uni import Uni
from s588p_syneval import load_cifar, ORD
from s588b_nrr import REV_TRAIN

C = 10
FWD_FRAMES = {"start": " from the start", "beginning": " from the beginning"}
REV_FRAMES = {" from the last": "last", " from the end": "end"}


def main(v, tag, Lb, nper=40, nproto=16, seed=58830):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar()
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
    for ph in list(FWD_FRAMES.values()) + list(REV_FRAMES):
        assert len(v.tok(ph, add_special_tokens=False)["input_ids"]) == 3, ph
    ST = {}

    def hook(m, i, o):
        if not ST:
            return o
        t = o[0] if isinstance(o, tuple) else o
        if "cap" in ST:
            ST["got"] = t[0, ST["cap"]].detach().float().cpu()
            return o
        t2 = t.clone()
        for p, dlt in ST["adds"]:
            t2[0, p] = (t[0, p].float() + dlt.to(t.device)).to(t.dtype)
        return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
    hk = v.layers[Lb].register_forward_hook(hook)

    def text_of(fmt, k, N, phr, opts, ks=None, cl0=None):
        if fmt == "options":
            lines = [f"{chr(65 + i)}: The {ORD[x-1]} image{phr}" for i, x in enumerate(ks)]
            return (f"{v.IMG*N}\nWhich image shows a {cl0}?\nSelect from the following choices.\n" + "\n".join(lines) +
                    "\nAnswer with the option's letter from the given choices directly.")
        if fmt == "question":
            return f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{phr}? Answer with only one word from: {', '.join(opts)}."
        return f"{v.IMG*N}\nLook at the {ORD[k-1]} image{phr}. What is in it? Answer with only one word from: {', '.join(opts)}."

    def src_pos(ids, k):
        turn = max(i for i, x in enumerate(ids) if x == im_start)
        return max(i for i, x in enumerate(ids[:turn]) if x == tok1(" " + ORD[k - 1]))

    def ids_of(text, pics):
        return v.encode(v.chat([("user", text)], prefix_assistant=THINK), pics)["input_ids"][0].tolist()

    @torch.no_grad()
    def go(text, pics, cand=None, adds=None, cap=None):
        inp = v.encode(v.chat([("user", text)], prefix_assistant=THINK), pics)
        ST.clear()
        if cap is not None:
            ST.update(cap=cap)
        elif adds:
            ST.update(adds=adds)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        got = ST.get("got"); ST.clear()
        return (int(out.logits[0, -1, cand].float().argmax()) if cand is not None else None), got

    t0 = time.time()
    rng = random.Random(seed + 1)
    acc = {i: [] for i in range(1, C + 1)}
    anc = collections.defaultdict(list)
    frames = [("", None)] + [(ph, nm) for nm, ph in FWD_FRAMES.items()] + [(ph, nm) for ph, nm in REV_FRAMES.items()]
    for i in range(1, C + 1):
        for phr, nm in frames:
            for _ in range(nproto if nm is None else max(4, nproto // 2)):
                fmt = rng.choice(["question", "statement", "options"])
                N = rng.randint(max(3, i), 10); ci = rng.sample(range(len(names)), N)
                cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
                opts = cl[:]; rng.shuffle(opts)
                if fmt == "options":
                    m = min(4, N); ks = rng.sample(range(1, N + 1), m)
                    if i not in ks:
                        ks[0] = i
                    text = text_of(fmt, i, N, phr, opts, ks=ks, cl0=cl[0])
                else:
                    text = text_of(fmt, i, N, phr, opts)
                ids = ids_of(text, pics); p = src_pos(ids, i)
                if nm is None:
                    acc[i].append(go(text, pics, cap=p)[1])
                else:
                    assert v.tok.decode([ids[p + 4]]).strip() == nm, (v.tok.decode([ids[p + 4]]), nm)
                    anc[nm].append(go(text, pics, cap=p + 4)[1])
    mu = torch.stack([torch.stack(acc[i]).mean(0) for i in range(1, C + 1)])
    mua = {nm: torch.stack(x).mean(0) for nm, x in anc.items()}
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"[{tag}] L{Lb} 프로토타입 · 앵커 수 {dict((k_, len(x)) for k_, x in anc.items())} · cos(end, last) {cos(mua['end'], mua['last']):.3f}"
          f" · cos(start, beginning) {cos(mua['start'], mua['beginning']):.3f} · |μ_start − μ_end| / |μ_end| "
          f"{float((mua['start'] - mua['end']).norm() / mua['end'].norm()):.3f}  ({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(seed)
    rows = []
    for fmt in ("question", "statement"):
        for phr in REV_TRAIN:
            an = REV_FRAMES[phr]
            for _ in range(nper):
                N = rng.randint(3, 8); k = rng.randint(1, N); j = N + 1 - k
                if j == k:
                    k = 1 if k != 1 else 2; j = N + 1 - k
                ci = rng.sample(range(len(names)), N)
                cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
                opts = cl[:]; rng.shuffle(opts)
                cand = [tok1(" " + c) for c in cl]
                text = text_of(fmt, k, N, phr, opts)
                ids = ids_of(text, pics); p = src_pos(ids, k)
                assert v.tok.decode([ids[p + 4]]).strip() == an
                land = lambda a: ("ok" if a == j - 1 else "k" if a == k - 1 else "last" if a == N - 1 else
                                  "j-1" if a == j - 2 else "j+1" if a == j else "other")
                dk = mu[j - 1] - mu[k - 1]
                r = dict(fmt=fmt, phr=phr, N=N, k=k, j=j)
                r["base"] = land(go(text, pics, cand)[0])
                r["index"] = land(go(text, pics, cand, adds=[(p, dk)])[0])
                r["index+frame_start"] = land(go(text, pics, cand, adds=[(p, dk), (p + 4, mua["start"] - mua[an])])[0])
                r["index+frame_begin"] = land(go(text, pics, cand, adds=[(p, dk), (p + 4, mua["beginning"] - mua[an])])[0])
                r["frame_only"] = land(go(text, pics, cand, adds=[(p + 4, mua["start"] - mua[an])])[0])
                r["text_start"] = land(go(text_of(fmt, j, N, FWD_FRAMES["start"], opts), pics, cand)[0])
                r["text_begin"] = land(go(text_of(fmt, j, N, FWD_FRAMES["beginning"], opts), pics, cand)[0])
                rows.append(r)
            sub = [r for r in rows if r["fmt"] == fmt and r["phr"] == phr]
            keys = ["base", "index", "index+frame_start", "index+frame_begin", "frame_only", "text_start", "text_begin"]
            print(f"[{tag}] {fmt:<9s}{phr:<15s} " + " · ".join(f"{k_} {np.mean([r[k_] == 'ok' for r in sub]):.3f}" for k_ in keys)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            for k_ in ("index", "index+frame_start", "text_start"):
                byj = collections.defaultdict(list)
                for r in sub:
                    byj[min(r["j"], 3)].append(r[k_] == "ok")
                print(f"      {k_:<18s} j=1 {np.mean(byj[1]) if byj[1] else float('nan'):.2f} · j=2 {np.mean(byj[2]) if byj[2] else float('nan'):.2f}"
                      f" · j>=3 {np.mean(byj[3]):.2f} · 오답 {dict(collections.Counter(r[k_] for r in sub if r[k_] != 'ok'))}", flush=True)
    hk.remove()
    keys = ["base", "index", "index+frame_start", "index+frame_begin", "frame_only", "text_start", "text_begin"]
    cell = lambda k_, f, ph: float(np.mean([r[k_] == "ok" for r in rows if r["fmt"] == f and r["phr"] == ph]))
    end = lambda k_: float(np.mean([cell(k_, f, " from the end") for f in ("question", "statement")]))
    print(f"\n[{tag}] from the end 질문·서술 평균: " + " · ".join(f"{k_} {end(k_):.3f}" for k_ in keys), flush=True)
    for arm in ("index+frame_start", "index+frame_begin"):
        drop = max(cell("index", f, ph) - cell(arm, f, ph) for f in ("question", "statement") for ph in REV_TRAIN)
        print(f"  (f1) {arm}: from the end {end(arm) - end('index'):+.3f}, 최대 하락 {drop:.3f} → "
              f"{'통과' if end(arm) - end('index') >= .10 and drop <= .03 else '기각'}")
    allm = lambda k_: float(np.mean([r[k_] == "ok" for r in rows]))
    print(f"  (f2) 기준만 {allm('frame_only'):.3f} vs 원 모델 {allm('base'):.3f} → {'점검 통과' if allm('frame_only') <= allm('base') + .10 else '점검 실패(기준 쓰기 혼자 역순을 품)'}")
    json.dump(dict(rows=rows, Lb=Lb), open(f"/vlm/results/s595_frame_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--Lb", type=int, required=True)
    ap.add_argument("--nper", type=int, default=40)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    main(v, a.tag, a.Lb, nper=a.nper)
