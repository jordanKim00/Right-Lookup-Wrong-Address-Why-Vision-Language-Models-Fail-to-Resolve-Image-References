"""S705a: the QK anatomy of END selection (capture run; analysis offline).  Pre-registration in knowledge/DECISIONS.md before any data.
Methods adopted after verification (knowledge/prior_work_verification_2026-10-09.md): per-head k x j compatibility tables in the real QK
(Gurnee et al. 2601.04480), contrastive query-key cross-moments (Lee et al. 2602.04752), and moving the end anchor by hiding the last images
from full attention (S703).  Qwen3.5, N = 7, one menu.
Per scene: START(t), t = 1..7; END(k), k = 1..7; END(k) with the last m in {1, 2} images hidden from every full-attention layer for the
question's text rows (anchor = image 7-m; k = 1..7-m).  At full-attention layers L35, 39, 43, 47, 51, 55 (27B; relative .556-.873) and rows
"?", tail1 (newline after the user turn), answer (last): the post-RoPE query per head, the pre-sigmoid output gate per head, and the scaled,
causally masked QK logits of the row over the image positions and the log-sum-exp of its text logits (hidden positions masked as in the run).  Once per scene: post-RoPE keys and the
values of the image positions per KV head, and the position map.  Landing of every run.
Offline (exp/s705a_analyze.py): block scores L_h(j) = logsumexp_{p in image j} logit_p; per head the START diagonal, the END anti-diagonal,
the END last-visible column; the anchor test below; contrastive bases from START on discovery scenes and their reading of END queries on
held-out scenes.
Registered (analysis exp/s705a_analyze.py), 30 scenes, seed 70500:
    Head classes are fixed on the first 15 scenes (discovery) and evaluated on the last 15 (held-out):
      MATCHER: START diagonal share (target block mass / image mass) >= .5 and END k = 2 target share >= .4 at the answer or tail1 row
      ANCHOR:  END last-visible-block share >= .35 averaged over k, with START last-block share <= .2
    QK_ANCHOR_SHIFT: in held-out scenes, for MATCHER heads, the END k = 2 block-score argmax with the anchor moved by m lands on 7-m-1
      in >= .5 of (scene, head, m) cases and on 7-1 = 6 (the unmoved target) in <= .25
    QK_STEP_LAW: for MATCHER heads (held-out), the END target share decreases from k = 2 to k = 4 and k = 5 (mean over heads), with
      the START diagonal share at the same targets >= .5
    Predictions: MATCHER and ANCHOR classes are non-empty; QK_ANCHOR_SHIFT; QK_STEP_LAW.
Usage: python s705a_qkanatomy.py MODEL TAG [--n 30] [--device_map balanced]
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
OUT = "/vlm/results/r3/s705a_{}_qkanatomy"


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
            rows = ST["rows"]
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[:, None] + rotate_half(k[..., :rd]) * sin[:, None], k[..., rd:]], -1)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe, ke[:, kh]) * mod.scaling
            mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
            if ST.get("hidden"):
                hm = torch.zeros(S, dtype=torch.bool, device=q.device); hm[torch.tensor(ST["hidden"], device=q.device)] = True
                mask = mask | hm[None, :]
            lg = lg.masked_fill(mask[:, None, :], float("-inf"))
            img = torch.tensor(ST["img"], device=q.device)
            txt = torch.ones(S, dtype=torch.bool, device=q.device); txt[img] = False
            lse_txt = torch.logsumexp(lg[:, :, txt], -1)                                  # [rows, heads]: text part of the softmax denominator
            ST["cap"][li] = dict(q=qe.half().cpu(), gate=gate.half().cpu(), limg=lg[:, :, img].float().cpu(), lsetxt=lse_txt.float().cpu())
            if ST.get("keys"):
                v_ = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
                ST["kv"][li] = dict(k=ke[img].half().cpu(), v=v_[img].half().cpu())
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in CAP]

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(q, frame, labs, pic, menu, hide=0, keys=False):
        inp = v.encode(prompt(q, frame, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        trows = list(range(ve[-1] + 1, len(ids)))
        qm = next(i for i in trows if "?" in v.tok.decode([ids[i]]))
        uend = max(i for i, t in enumerate(ids) if t == T("<|im_end|>"))
        rows = [qm, uend + 1, len(ids) - 1]
        img = [p for b in range(N) for p in range(vs[b], ve[b] + 1)]
        owner = [b for b in range(N) for _ in range(vs[b], ve[b] + 1)]
        hidden = list(range(vs[N - hide], ve[-1] + 1)) if hide else []
        ctx.reset()
        if hide:
            ctx.rows = trows; ctx.edit = {("F", li): {"ko": hidden} for li in FA_L}
        ST.clear(); ST.update(on=True, rows=rows, hidden=hidden, img=img, keys=keys, cap={}, kv={})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        cap, kv = ST["cap"], ST["kv"]
        ST.clear(); ctx.reset()
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(land=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs], cap=cap, kv=kv,
                    img=img, owner=owner, rows=rows, S=len(ids))

    print(f"[{a.tag}] FA {FA_L} · 포착 {CAP}", flush=True)
    rng = random.Random(70500)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, land={}, ll={})
        runs = [("S", t, 0) for t in range(1, N + 1)] + [("E", k, 0) for k in range(1, N + 1)] + \
               [("E", k, m) for m in (1, 2) for k in range(1, N - m + 1)]
        for j, (fr, q, m) in enumerate(runs):
            r = run(q, "first" if fr == "S" else "last", labs, pic, menu, hide=m, keys=(j == 0))
            key = f"{fr}{q}m{m}"
            rec["land"][key] = r["land"]; rec["ll"][key] = r["ll"]
            for li, c in r["cap"].items():
                for nm, t in c.items():
                    arrays[f"{si}|{key}|{li}|{nm}"] = t.numpy()
            if j == 0:
                for li, c in r["kv"].items():
                    for nm, t in c.items():
                        arrays[f"{si}|img|{li}|{nm}"] = t.numpy()
                rec.update(img=r["img"], owner=r["owner"], rows=r["rows"], S=r["S"])
        meta.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    FA.remove()
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, FA=FA_L, CAP=CAP, rows=["qm", "tail1", "ans"], meta=meta), open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
