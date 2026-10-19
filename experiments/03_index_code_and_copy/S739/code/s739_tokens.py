"""S739: does the index code count image blocks or tokens?  Pre-registration in knowledge/DECISIONS.md (2026-10-10 20:48) before any S739 data;
analysis exp/s739_analyze.py (frozen with it).  The S730 harness with one change in the scenes: the same scene and pictures are shown in two
size regimes -- UNI (all seven images 224 px) and MIX (each image 224, 448 or 672 px, drawn independently; block token counts recorded).
Scenes: procedural, N = 7, seed 73900, k in {2, ..., 6} (target j = N + 1 - k), the S730 question form and menu.
Arms per regime: START (forward wording of the same target), END (end-relative, unedited), R3 (END with the model's final R³).
Captured at the answer row: the matcher heads' block shares (target, literal) and the restricted answer -> landed image.
Usage: python s739_tokens.py MODEL TAG --base B --rc RCTAG --heads JSON [--reader_override R] [--n 30] [--device_map balanced]
"""
from __future__ import annotations

import argparse, os, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half
import r3_select

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    rep = r3_select.build(v, a.base, a.rc, reader=a.reader_override)
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    def aid(s):                                          # answer readout (S789): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return tid(s)
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}
    HL = {}

    def hook(li):
        def h(mod, args, kwargs, out):
            if not ST.get("on") or (ST.get("layers") is not None and li not in ST["layers"]):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos.float(), sin.float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads
            rows = ST["rows"]
            qp = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
            gated = qp.shape[-1] == 2 * nh * hd                                               # Qwen3.5 full attention carries an output gate
            q = (qp.view(len(rows), nh, 2 * hd)[..., :hd] if gated else qp.view(len(rows), nh, hd))
            qn, kn = getattr(mod, "q_norm", None), getattr(mod, "k_norm", None)                 # Qwen2-based models have no q / k norm
            q = qn(q.to(qn.weight.dtype)).float() if qn is not None else q
            kp = torch.nn.functional.linear(hs, mod.k_proj.weight.float(), None if mod.k_proj.bias is None else mod.k_proj.bias.float())
            k = kp.view(S, -1, hd)
            k = kn(k.to(kn.weight.dtype)).float() if kn is not None else k
            M_ = sys.modules[mod.__class__.__module__]
            if cos.dim() == 4 and hasattr(M_, "apply_multimodal_rotary_pos_emb"):           # Qwen2.5-VL: 3-axis M-RoPE, the module's own rotation
                sec = (getattr(mod.config, "rope_parameters", None) or mod.config.rope_scaling)["mrope_section"]
                qe = M_.apply_multimodal_rotary_pos_emb(q.transpose(0, 1)[None], q.transpose(0, 1)[None], cos[:, :, rows], sin[:, :, rows], sec)[0][0].transpose(0, 1)
                ke = M_.apply_multimodal_rotary_pos_emb(k.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin, sec)[1][0].transpose(0, 1)
            else:
                cos, sin = cos[0], sin[0]; rd = cos.shape[-1]
                qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
                ke = torch.cat([k[..., :rd] * cos[:, None] + rotate_half(k[..., :rd]) * sin[:, None], k[..., rd:]], -1)
            heads = torch.arange(nh, device=hs.device) if ST.get("allheads") else torch.tensor(HL[li], device=hs.device)
            kh = heads // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling
            sc = torch.stack([torch.logsumexp(lg[:, :, b0:b1 + 1], -1) for b0, b1 in ST["blocks"]], -1)   # [rows, heads, N]
            ST["cap"][li] = dict(sc=sc.cpu(), q=(None if ST.get("allheads") else q[:, heads].cpu()))
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in FA_L]

    def prompt(k, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(text, pics, menu, labs, writes=None, allheads=False, layers=None, mod=None):
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        B = v.image_blocks(inp["input_ids"], len(pics))                                     # the image tokens of each block (all families)
        ST.clear(); ST.update(on=True, rows=[len(ids) - 1], blocks=[(int(b[0]), int(b[-1])) for b in B], cap={}, allheads=allheads, layers=layers)
        mod = mod or rep
        mod.ST = dict(writes=writes) if writes else {}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        mod.ST = {}
        cap = ST["cap"]; ST.clear()
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return cap, (labs.index(lab) + 1 if lab in labs else 0)

    def scene(rng):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        raw = [proc_image(rng.choice(pool[l]))[0] for l in labs]
        sizes = [rng.choice([224, 448, 672]) for _ in labs]
        pics = {"UNI": [im.resize((224, 224)) for im in raw], "MIX": [im.resize((s, s)) for im, s in zip(raw, sizes)]}
        menu = labs + extra; rng.shuffle(menu)
        return labs, pics, menu, sizes

    for l, h in json.load(open(a.heads))["matchers"]:
        HL.setdefault(l, []).append(h)
    HL = {l: sorted(h) for l, h in HL.items()}
    lays = sorted(HL)
    rng = random.Random(73900)
    out = []
    for si in range(a.n):
        labs, PICS, menu, sizes = scene(rng)
        for reg in ("UNI", "MIX"):
            pics = PICS[reg]
            for k in range(2, 7):
                j = N + 1 - k
                rec = dict(si=si, k=k, j=j, regime=reg, sizes=sizes if reg == "MIX" else [224] * N)
                cS, rec["land_START"] = run(prompt(j, "first", menu), pics, menu, labs, layers=lays)
                tE = prompt(k, "last", menu)
                cE, rec["land_END"] = run(tE, pics, menu, labs, layers=lays)
                inp, writes = rep.plan(tE, pics)
                rec["n_writes"] = len(writes)
                rec["block_tokens"] = [int(b[-1]) - int(b[0]) + 1 for b in v.image_blocks(inp["input_ids"], len(pics))]
                cR, rec["land_R3"] = run(tE, pics, menu, labs, writes=writes, layers=lays)
                for c, cp in (("START", cS), ("END", cE), ("R3", cR)):
                    sh = torch.cat([torch.softmax(cp[l]["sc"][0], -1) for l in lays])
                    rec[f"tgt_{c}"] = float(sh[:, j - 1].mean()); rec[f"lit_{c}"] = float(sh[:, k - 1].mean())
                out.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, heads=[[l, h] for l in lays for h in HL[l]], rc=a.rc, choice=r3_select.choice(a.rc), rows=out),
              open(f"/vlm/results/r3/s739_{a.tag}_tokens.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s739_{a.tag}_tokens.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--heads", required=True); ap.add_argument("--reader_override", default=None)
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
