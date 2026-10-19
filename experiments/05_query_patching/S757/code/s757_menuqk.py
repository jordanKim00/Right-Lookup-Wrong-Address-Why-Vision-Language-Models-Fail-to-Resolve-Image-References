"""S757: when the option order changes an imprecise END(5) answer, does the L35-47 query matching change with it?
Pre-registration in knowledge/DECISIONS.md (2026-10-13 09:3x) before any S757 data; analysis exp/s757_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt.  Per scene: END(5) with four menu orders (M0 as drawn, M1 reversed, M2, M3
random), END(2) and START(3) with M0 and M1.  Readouts: the menu-restricted answer, and at L35, 39, 43, 47, 51, 55 (27B) for the rows "?" (before
the menu, so it cannot see it), tail1 (newline after the user turn) and the answer position, the block scores of every head over the seven
image blocks with the image keys (as S748).
Usage: python s757_menuqk.py MODEL TAG [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import Ctx, FullAttn, rotate_half

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s757_{}_menuqk"
RUNS = [("E", 5, m) for m in ("M0", "M1", "M2", "M3")] + [("E", 2, m) for m in ("M0", "M1")] + [("S", 3, m) for m in ("M0", "M1")]


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    R = lambda r: int(round(r * (nl - 1)))
    CAP = sorted({min(FA_L, key=lambda l: abs(l - R(x / 63))) for x in (35, 39, 43, 47, 51, 55)})
    ctx = Ctx(); FA = FullAttn(v, FA_L, ctx)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}

    def hook(li):
        def h(mod, args, kwargs, out):
            if not ST.get("on"):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            rows = ST["rows"]; img = torch.tensor(ST["img"], device=hs.device)
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(),
                                            None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
            k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[img][:, None] + rotate_half(k[..., :rd]) * sin[img][:, None], k[..., rd:]], -1)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe, ke[:, kh]) * mod.scaling                       # [rows, heads, image positions]
            own = torch.tensor(ST["owner"], device=q.device)
            sc = torch.stack([torch.logsumexp(lg[:, :, own == b], -1) for b in range(N)], -1)  # [rows, heads, N], unmasked bank
            ST["cap"][li] = sc.cpu().numpy().astype(np.float32)
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in CAP]

    def prompt(fr, q, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(fr, q, labs, pic, menu, hide):
        inp = v.encode(prompt(fr, q, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        assert len(vs) == N and len(ve) == N
        trows = list(range(ve[-1] + 1, len(ids)))
        qm = next(i for i in trows if "?" in v.tok.decode([ids[i]]))
        uend = max(i for i, t in enumerate(ids) if t == T("<|im_end|>"))
        rows = [qm, uend + 1, len(ids) - 1]
        img = [p for b in range(N) for p in range(vs[b], ve[b] + 1)]
        owner = [b for b in range(N) for _ in range(vs[b], ve[b] + 1)]
        ctx.reset()
        if hide is not None:
            H = [p for b in hide for p in range(vs[b - 1], ve[b - 1] + 1)]
            ctx.rows = trows; ctx.edit = {("F", li): ({"ko": H} if H else {}) for li in FA_L}      # {} = NOOP recomputation
        ST.clear(); ST.update(on=True, rows=rows, img=img, owner=owner, cap={})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        cap = ST["cap"]; ST.clear(); ctx.reset()
        ll = [float(lg[tid(l)]) for l in labs]
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return labs.index(lab) + 1 if lab in labs else 0, ll, cap

    print(f"[{a.tag}] FA {FA_L} · 포착 {CAP}", flush=True)
    rng = random.Random(75700)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        orders = {"M0": list(menu), "M1": list(menu[::-1])}
        for nm in ("M2", "M3"):
            o = list(menu); rng.shuffle(o); orders[nm] = o
        rec = dict(si=si, land={}, ll={}, orders=orders)
        for fr, q, m in RUNS:
            key = f"{m}|{fr}{q}"
            land, ll, cap = run(fr, q, labs, pic, orders[m], None)
            rec["land"][key] = land; rec["ll"][key] = ll
            for li, sc in cap.items():
                arrays[f"{si}|{key}|{li}"] = sc
        meta.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 10 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    FA.remove()
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, FA=FA_L, CAP=CAP, rows=["qm", "tail1", "ans"], runs=[f"{m}|{fr}{q}" for fr, q, m in RUNS],
                   meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
