"""S795 (pre-registered in knowledge/DECISIONS.md before any S795 data): an address lens -- how the end-relative query is formed over depth.
No answer is read (QK only), so the menu-token readout plays no part.  For each lookup layer L* in the index-matching band, the residual stream
at the answer position after every earlier layer l is passed through L*'s own input norm, query projection, q_norm and rotary embedding, and
matched against L*'s actual keys of the image tokens: the block shares of the index-matching heads at L* tell which image the address carried
by the stream up to layer l would select.  The last point (l = L* - 1) is the query L* actually uses.
Scenes: procedural pictures (160 px), N = 7, seed 79500 (new), 30 scenes; START(t) for t = 1..7 and END(k) for k = 1..7, the S730 prompt with
the stated count.  Qwen3.5 (gated full attention, as exp/c_qpath.py).
Usage: python s795_lens.py MODEL TAG --heads JSON [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def main(a):
    dev = os.environ.get("S795_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    HL = {}
    for l, h in json.load(open(a.heads))["matchers"]:
        HL.setdefault(int(l), []).append(int(h))
    LSTAR = sorted(HL)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    RES, KV = {}, {}

    def res_hook(li):
        def h(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            RES[li] = t[0, -1].detach().float().cpu()
            return None
        return h

    def attn_hook(li):
        def h(mod, args, kwargs, out):
            KV[li] = dict(hs=kwargs["hidden_states"][0].detach().float(), pe=tuple(x.detach().float() for x in kwargs["position_embeddings"]))
            return None
        return h
    hooks = [v.layers[li].register_forward_hook(res_hook(li)) for li in range(nl)]
    hooks += [v.layers[li].self_attn.register_forward_hook(attn_hook(li), with_kwargs=True) for li in LSTAR]

    def prompt(k, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def lens(text, pics):
        RES.clear(); KV.clear()
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist(); S_ = len(ids)
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        out = {}
        for Ls in LSTAR:
            mod = v.layers[Ls].self_attn; ln = v.layers[Ls].input_layernorm
            hs = KV[Ls]["hs"]; cos, sin = KV[Ls]["pe"]; cos, sin = cos[0], sin[0]
            dvc = hs.device; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            img = [p for b in range(N) for p in range(vs[b], ve[b] + 1)]; own = torch.tensor([b for b in range(N) for _ in range(vs[b], ve[b] + 1)], device=dvc)
            k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
            ke = torch.cat([k[..., :rd] * cos[img][:, None] + rotate_half(k[..., :rd]) * sin[img][:, None], k[..., rd:]], -1)
            heads = torch.tensor(HL[Ls], device=dvc); kh = heads // mod.num_key_value_groups
            c1, s1 = cos[S_ - 1][None], sin[S_ - 1][None]
            sh = []
            for li in range(Ls):                                                          # the stream after layer li, read by L*'s query
                r = RES[li].to(dvc)
                x = ln(r[None].to(ln.weight.dtype)).float()
                qg = torch.nn.functional.linear(x, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(1, nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
                qe = torch.cat([q[..., :rd] * c1[:, None] + rotate_half(q[..., :rd]) * s1[:, None], q[..., rd:]], -1)
                lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling           # [1, H, img]
                sc = torch.stack([torch.logsumexp(lg[0, :, own == b], -1) for b in range(N)], -1)    # [H, N]
                sh.append(torch.softmax(sc, -1).mean(0).cpu().tolist())
            out[str(Ls)] = sh
        return out

    rng = random.Random(a.seed); rows = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, START={}, END={})
        for q in range(1, N + 1):
            rec["START"][str(q)] = lens(prompt(q, "first", menu), pics)
            rec["END"][str(q)] = lens(prompt(q, "last", menu), pics)
        rows.append(rec)
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, lookup_layers=LSTAR, heads={str(k): v_ for k, v_ in HL.items()}, rows=rows),
              open(f"/vlm/results/r3/s795_{a.tag}_lens.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s795_{a.tag}_lens.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--heads", required=True)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=79500)
    main(ap.parse_args())
