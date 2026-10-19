"""S791 (pre-registered in knowledge/DECISIONS.md before any S791 data): one causal chain in the same scenes -- address / frame writes
-> the lookup query -> the picture read -> the answer.  The S730 / S786 design (same code for writes, captures and readout; R3_READ=bare)
plus a content test for every arm: the picture at the target j, and separately at the literal image k, is replaced by a new object whose
name is already in the menu (one of the two distractors); the plan computed on the original scene is reused unchanged.
Scenes: procedural images (160 px), N = 7, seed 79100 (new); k in {2, ..., 6}, j = N + 1 - k.  Arms: START (the forward wording of j),
END, ADDR (R3's ordinal write only), FRAME (R3's frame write only), R3 (both).  Recorded per arm: the landed image, target / literal block
shares and query cosine to START at the matcher heads' answer row, and the answers with the picture at j (swapj) or at k (swapk) replaced.
Usage: python s791_chain.py MODEL TAG --base B --rc RCTAG --heads JSON [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, os, random, sys
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
    assert os.environ.get("R3_READ") in ("bare", "max"), "S791 must read the generated form (FINDINGS §846, §851); set R3_READ=max"
    dev = os.environ.get("S791_DEVICE", "cuda:0")                                        # smoke tests on CPU while the cards are busy
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    rep = r3_select.build(v, a.base, a.rc, reader=a.reader_override)
    repw = None
    if a.wl is not None:                                                                   # S731: the same READ, writes at the rule layer
        import s731_wcal
        s731_wcal.ensure(v, a.base, a.wl)
        repw = r3_select.build(v, a.base, a.rc, wl=a.wl, reader=a.reader_override)
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(c):                                          # answer readout (S786): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return ids1(c)[0]
        t = v.tok(c, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, c
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    T1 = []                                              # is the vocabulary-wide top-1 a bare menu token?
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
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return cap, (labs.index(lab) + 1 if lab in labs else 0)

    def scene(rng):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        return labs, pics, menu

    if a.discover:                                                                       # dense models: the S705a rule on discovery scenes
        drng = random.Random(73001); diag = {}
        for _ in range(15):
            labs, pics, menu = scene(drng)
            for t in range(1, N + 1):
                cap, _ = run(prompt(t, "first", menu), pics, menu, labs, allheads=True)
                for li, c in cap.items():
                    sh = torch.softmax(c["sc"][0], -1)                                     # [heads, N]
                    diag.setdefault(li, []).append(sh[:, t - 1])
        for li, xs in diag.items():
            m = torch.stack(xs).mean(0)
            HL[li] = [int(h) for h in (m >= .5).nonzero().flatten().tolist()]
        HL = {l: h for l, h in HL.items() if h}
        json.dump(dict(model=a.model, rule="mean START diagonal share >= .5 at the answer row, discovery scenes seed 73001 (15)",
                       matchers=[[l, h] for l in sorted(HL) for h in HL[l]]), open(f"/vlm/results/r3/s730_{a.tag}_heads.json", "w"))
        print(f"[{a.tag}] 정합 머리 {sum(len(h) for h in HL.values())} · 층 {sorted(HL)}", flush=True)
    else:
        for l, h in json.load(open(a.heads))["matchers"]:
            HL.setdefault(l, []).append(h)
        HL = {l: sorted(h) for l, h in HL.items()}
    lays = sorted(HL)
    rng = random.Random(a.seed)
    out = []
    for si in range(a.n):
        labs, pics, menu = scene(rng)
        nw = [l for l in menu if l not in labs][0]                                          # the new object: a menu distractor absent from the scene
        npic = proc_image(rng.choice(pool[nw]))[0].resize((a.pixels, a.pixels))
        def swap(pos):
            q = list(pics); q[pos - 1] = npic
            return q
        for k in range(2, 7):
            j = N + 1 - k
            rec = dict(si=si, k=k, j=j, new=nw)
            tS, tE = prompt(j, "first", menu), prompt(k, "last", menu)
            cS, rec["land_START"] = run(tS, pics, menu, labs, layers=lays)
            cE, rec["land_END"] = run(tE, pics, menu, labs, layers=lays)
            inp, writes = rep.plan(tE, pics)
            opos = {int(d["pos"]) for d in rep.last_detail if d.get("act")}
            W = {"R3": writes, "ADDR": [w for w in writes if int(w[0]) in opos], "FRAME": [w for w in writes if int(w[0]) not in opos]}
            rec["n_writes"] = {c: len(w) for c, w in W.items()}
            rec["plan"] = [(d["tok"], int(d["kh"]), int(d["t"]), bool(d["act"]), bool(d.get("fonly", False))) for d in rep.last_detail]
            caps = {"START": cS, "END": cE}
            for c, w in W.items():
                caps[c], rec[f"land_{c}"] = run(tE, pics, menu, labs, writes=w, layers=lays)
            labs_j = list(labs); labs_j[j - 1] = nw                                         # landed position = the slot that now shows nw
            labs_k = list(labs); labs_k[k - 1] = nw
            for c in ("START", "END", "ADDR", "FRAME", "R3"):
                t = tS if c == "START" else tE
                w = None if c in ("START", "END") else W[c]
                _, pj = run(t, swap(j), menu, labs_j, writes=w, layers=[])
                rec[f"swapj_{c}"] = int(pj == j)                                            # the answer names the new object placed at j
                if k != j:
                    _, pk = run(t, swap(k), menu, labs_k, writes=w, layers=[])
                    rec[f"swapk_{c}"] = int(pk == k)                                        # the answer names the new object placed at k
            qS = torch.cat([caps["START"][l]["q"][0] for l in lays])
            for c, cp in caps.items():
                sh = torch.cat([torch.softmax(cp[l]["sc"][0], -1) for l in lays])
                q = torch.cat([cp[l]["q"][0] for l in lays])
                rec[f"tgt_{c}"] = float(sh[:, j - 1].mean()); rec[f"lit_{c}"] = float(sh[:, k - 1].mean())
                rec[f"cos_{c}"] = float(torch.nn.functional.cosine_similarity(q, qS, dim=-1).mean())
            out.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), heads=[[l, h] for l in lays for h in HL[l]], rc=a.rc, choice=r3_select.choice(a.rc), rows=out),
              open(f"/vlm/results/r3/s791_{a.tag}_chain.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s791_{a.tag}_chain.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--heads", default=None); ap.add_argument("--discover", action="store_true")
    ap.add_argument("--wl", type=int, default=None); ap.add_argument("--reader_override", default=None)                                       # S731: the rule write layer, when it differs
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=79100)
    main(ap.parse_args())
