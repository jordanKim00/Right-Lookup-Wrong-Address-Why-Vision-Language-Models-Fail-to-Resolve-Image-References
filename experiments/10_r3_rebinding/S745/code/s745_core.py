"""S745: the core causal tests in a famous VLM with a non-Qwen language model, without R³ (reply to the mock reviews; the user:
"LLaVA 말고 선행 연구에서 자주 쓰는 유명한 모델로").  Pre-registration in knowledge/DECISIONS.md before any S745 data; analysis
exp/s745_analyze.py, frozen with it.  Procedural scenes of seven pictures (the S730 question form and menu, 160 px).
    0 GATE      discovery scenes (seed 74500, 15 scenes, START t = 1..7): start-relative accuracy; index-matching heads = mean START
                diagonal share >= .5 at the answer row (the S730 rule for dense models)
    1 MEASURE   test scenes (seed 74501, 30), k in {2, 3, 5, 6} (target j = 8 - k, literal image k): START(j) and END(k) answers,
                the heads' image-attention share on the target and the literal block, query cosine END vs START at the answer row
    2 ADDRESS   the residual state of the ordinal token of a donor "What is in the x-th image from the first?" about the same pictures in
                another order, written into END(k) at one layer L in {2, 4, 6, 8, 10}; x = j (the right address) and x = w (a wrong
                address, w not in {j, k}); readout: answer on position x, on the picture the donor named, or on k
    3 QUERY     END(k) given the query projections (q_proj output) of a donor START(x) on the same pictures, at every row from the
                ordinal to the answer, in every layer (keys and values stay the recipient's); x = j and x = w
Usage: python s745_core.py MODEL TAG [--n 30] [--pixels 160]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
LAYERS = (2, 4, 6, 8, 10)


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    FA = [l for l in range(v.n_layer) if getattr(v.layers[l], "self_attn", None) is not None]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}

    def attn_hook(li):                                                                     # block shares and queries at the answer row
        def h(mod, args, kwargs, out):
            if not ST.get("cap_attn"):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos.float(), sin.float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rows = [S - 1]
            qp = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
            gated = qp.shape[-1] == 2 * nh * hd
            q = qp.view(1, nh, 2 * hd)[..., :hd] if gated else qp.view(1, nh, hd)
            qn, kn = getattr(mod, "q_norm", None), getattr(mod, "k_norm", None)
            q = qn(q.to(qn.weight.dtype)).float() if qn is not None else q
            kp = torch.nn.functional.linear(hs, mod.k_proj.weight.float(), None if mod.k_proj.bias is None else mod.k_proj.bias.float())
            k = kp.view(S, -1, hd); k = kn(k.to(kn.weight.dtype)).float() if kn is not None else k
            cos, sin = cos[0], sin[0]; rd = cos.shape[-1]
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[:, None] + rotate_half(k[..., :rd]) * sin[:, None], k[..., rd:]], -1)
            heads = torch.arange(nh, device=hs.device); kh = heads // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling
            sc = torch.stack([torch.logsumexp(lg[:, :, b0:b1 + 1], -1) for b0, b1 in ST["blocks"]], -1)
            ST["cap"][li] = dict(sh=torch.softmax(sc[0], -1).cpu(), q=q[0].cpu())
            return out
        return h

    def res_hook(li):                                                                      # ordinal residual: capture or write
        def h(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            if ST.get("res_cap") is not None and li in ST["res_cap"]:
                ST["res"][li] = t[0, ST["pos"]].detach().clone()
            if ST.get("res_put") is not None and ST["res_put"][0] == li:
                t = t.clone(); t[0, ST["pos"]] = ST["res_put"][1].to(t.device, t.dtype)
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            return out
        return h

    def q_hook(li):                                                                        # query projections: capture or write
        def h(mod, args, out):
            if ST.get("q_cap"):
                ST["q"][li] = out[0, ST["rows"]].detach().clone()
            if ST.get("q_put") is not None:
                out = out.clone(); out[0, ST["rows"]] = ST["q_put"][li].to(out.device, out.dtype)
            return out
        return h
    hks = [v.layers[l].self_attn.register_forward_hook(attn_hook(l), with_kwargs=True) for l in FA]
    hks += [v.layers[l].register_forward_hook(res_hook(l)) for l in range(v.n_layer)]
    hks += [v.layers[l].self_attn.q_proj.register_forward_hook(q_hook(l)) for l in FA]

    def prompt(o, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {o} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(o, frame, pics, menu, labs, **kw):
        inp = v.encode(prompt(o, frame, menu), pics); ids = inp["input_ids"][0].tolist()
        oid = ids1(o); assert len(oid) == 1, o
        img_end = max(i for i, t in enumerate(ids) if t == v.image_token_id)
        pos = next(i for i in range(img_end + 1, len(ids)) if ids[i] == oid[0])          # the question's ordinal, not a frame word "first"
        B = v.image_blocks(inp["input_ids"], len(pics))
        ST.clear(); ST.update(pos=pos, rows=list(range(pos, len(ids))), blocks=[(int(b[0]), int(b[-1])) for b in B], cap={}, res={}, q={}, **kw)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        out = dict(cap=ST["cap"], res=ST["res"], q=ST["q"], ntok=len(ids), pos=pos); ST.clear()
        lab = menu[int(torch.stack([lg[ids1(c)[0]] for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), lab, out

    def scene(rng):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        return labs, pics, menu

    la, lb = (len(v.tok(prompt("third", fr, ["cat"]), add_special_tokens=False)["input_ids"]) for fr in ("first", "last"))
    assert la == lb, ("frame words must be single tokens so that rows align", la, lb)
    # 0 GATE
    drng = random.Random(74500); diag = {}; acc0 = []
    for _ in range(15):
        labs, pics, menu = scene(drng)
        for t in range(1, N + 1):
            land, _, o = run(ORDW[t - 1], "first", pics, menu, labs, cap_attn=True)
            acc0.append(land == t)
            for li, c in o["cap"].items():
                diag.setdefault(li, []).append(c["sh"][:, t - 1])
    HL = {}
    for li, xs in diag.items():
        m = torch.stack(xs).mean(0); hs = [int(h) for h in (m >= .5).nonzero().flatten().tolist()]
        if hs:
            HL[li] = hs
    nheads = sum(len(h) for h in HL.values())
    print(f"[{a.tag}] GATE start accuracy {np.mean(acc0):.3f} · index-matching heads {nheads} · layers {sorted(HL)}", flush=True)
    rows = []
    if nheads >= 5 and np.mean(acc0) >= a.gate:
        rng = random.Random(74501)
        for si in range(a.n):
            labs, pics, menu = scene(rng)
            for k in (2, 3, 5, 6):
                j = N + 1 - k
                w = rng.choice([x for x in range(1, N + 1) if x not in (j, k)])
                r = dict(si=si, k=k, j=j, w=w)
                r["START"], _, oS = run(ORDW[j - 1], "first", pics, menu, labs, cap_attn=True)
                r["END"], _, oE = run(ORDW[k - 1], "last", pics, menu, labs, cap_attn=True)
                for tag_, o in (("START", oS), ("END", oE)):
                    sh = torch.cat([o["cap"][l]["sh"][HL[l]] for l in sorted(HL)])
                    r[f"tgt_{tag_}"] = float(sh[:, j - 1].mean()); r[f"lit_{tag_}"] = float(sh[:, k - 1].mean())
                qS = torch.cat([oS["cap"][l]["q"][HL[l]] for l in sorted(HL)]); qE = torch.cat([oE["cap"][l]["q"][HL[l]] for l in sorted(HL)])
                r["cos_END"] = float(torch.nn.functional.cosine_similarity(qS, qE, dim=-1).mean())
                # 2 ADDRESS: donor about the same pictures in another order
                for nm, x in (("j", j), ("w", w)):
                    perm = list(range(N))
                    for _ in range(50):
                        rng.shuffle(perm)
                        named = perm[x - 1] + 1                                            # recipient position of the picture the donor names
                        if named not in (x, k):
                            break
                    dpics = [pics[p] for p in perm]; dlabs = [labs[p] for p in perm]
                    _, _, oD = run(ORDW[x - 1], "first", dpics, menu, dlabs, res_cap=set(LAYERS))
                    r[f"named_{nm}"] = named
                    for L in LAYERS:
                        r[f"ADDR_{nm}@{L}"] = run(ORDW[k - 1], "last", pics, menu, labs, res_put=(L, oD["res"][L]))[0]
                # 3 QUERY: query projections of START(x) on the same pictures, rows from the ordinal to the answer
                for nm, x in (("j", j), ("w", w)):
                    _, _, oQ = run(ORDW[x - 1], "first", pics, menu, labs, q_cap=True)
                    r[f"Q_{nm}"] = run(ORDW[k - 1], "last", pics, menu, labs, q_put=oQ["q"])[0]
                rows.append(r)
            torch.cuda.empty_cache()
            if (si + 1) % 5 == 0:
                print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, gate=dict(start_acc=float(np.mean(acc0)), heads=[[l, h] for l in sorted(HL) for h in HL[l]]),
                   layers=LAYERS, rows=rows), open(f"/vlm/results/r3/s745_{a.tag}_core.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s745_{a.tag}_core.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--gate", type=float, default=.70)                                  # S745b: .40 (DECISIONS 2026-10-11 20:29)
    main(ap.parse_args())
