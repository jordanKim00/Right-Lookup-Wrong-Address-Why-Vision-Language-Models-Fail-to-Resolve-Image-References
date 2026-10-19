"""S764: same-scene START / END pairs at N = 7 and N = 8 - where every full-attention head from L27 looks, how open its output gate is, and what
the model answers.  Pre-registration in knowledge/DECISIONS.md (2026-10-13 18:3x) before any S764 data; analysis exp/s764_analyze.py.
Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count.  N = 7 (seed 76400): END(k) k = 2..6 and START(8 - k).
N = 8 (seed 76401): END(k) k = 2..6 and START(9 - k).  Each pair names the same image.  At N = 8, END(4) and END(5) separate three readings
(literal k; two before the last, N - 2 = 6; the correct N + 1 - k).  For every head of every full-attention layer from L27, at the rows "?", tail1 and
the answer: the fixed-bank block scores over the N image blocks (as S748) and the mean of the output gate sigmoid(gate).  Readout: the
menu-restricted answer and the menu logits.
Usage: python s764_pairs.py MODEL TAG --N 7 [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s764_{}_pairs"


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    N = a.N
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    CAPL = [l for l in FA_L if l >= mp(27)]
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
    ST = dict(on=False)

    def cap_hook(li):
        def h(mod, args, kwargs, out):
            if not ST["on"]:
                return None
            hs = kwargs["hidden_states"][0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            rows = ST["rows"]; img = torch.tensor(ST["img"], device=hs.device)
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(),
                                            None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = torch.sigmoid(qg[..., hd:]).mean(-1)
            k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[img][:, None] + rotate_half(k[..., :rd]) * sin[img][:, None], k[..., rd:]], -1)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe, ke[:, kh]) * mod.scaling
            own = torch.tensor(ST["owner"], device=q.device)
            ST["cap"][li] = torch.stack([torch.logsumexp(lg[:, :, own == b], -1) for b in range(N)], -1).cpu().numpy().astype(np.float32)
            ST["gate"][li] = gate.cpu().numpy().astype(np.float32)
            return None
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(cap_hook(li), with_kwargs=True) for li in CAPL]

    def prompt(fr, k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(fr, k, labs, pic, menu, extra):
        inp = v.encode(prompt(fr, k, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        assert len(vs) == N and len(ve) == N
        qm = next(i for i in range(ve[-1] + 1, len(ids)) if "?" in v.tok.decode([ids[i]]))
        uend = max(i for i, t in enumerate(ids) if t == T("<|im_end|>"))
        ST.update(on=True, rows=[qm, uend + 1, len(ids) - 1], img=[p for b in range(N) for p in range(vs[b], ve[b] + 1)],
                  owner=[b for b in range(N) for _ in range(vs[b], ve[b] + 1)], cap={}, gate={})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        cap, gate = ST["cap"], ST["gate"]; ST.update(on=False)
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return dict(word=lab, pos=labs.index(lab) + 1 if lab in labs else 0, ll=[float(lg[tid(l)]) for l in labs + extra]), cap, gate

    QS = [("E", k) for k in range(2, 7)] + [("S", N + 1 - k) for k in range(2, 7)]
    print(f"[{a.tag}] N {N} · CAPL {CAPL} · questions {[f'{fr}{k}' for fr, k in QS]}", flush=True)
    rng = random.Random(a.seed)
    meta, arrays = [], {}
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, labs=labs, extra=extra, ans={})
        for fr, k in QS:
            q = f"{fr}{k}"
            rec["ans"][q], cap, gate = run(fr, k, labs, pic, menu, extra)
            for li in CAPL:
                arrays[f"{si}|{q}|{li}|s"] = cap[li]; arrays[f"{si}|{q}|{li}|g"] = gate[li]
        meta.append(rec)
        gc.collect(); torch.cuda.empty_cache()
        if (si + 1) % 10 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n} · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, N=N, CAPL=CAPL, rows=["qm", "tail1", "ans"], QS=[f"{fr}{k}" for fr, k in QS], meta=meta),
              open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--N", type=int, required=True); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    a.seed = a.seed if a.seed is not None else {7: 76400, 8: 76401}.get(a.N, 76400 + a.N)
    main(a)
