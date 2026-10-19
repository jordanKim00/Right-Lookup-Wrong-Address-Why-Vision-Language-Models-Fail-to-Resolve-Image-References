"""S581: Latent Reference Transport v1 -- the transport lives in the model's OWN address coordinates, only the shift is
committed, relations are learned jointly and tied by an algebraic constraint.

Changes from v0 (S580): (1) the pointer is read as coordinates in the model's own address subspace U (DAS, §463b: 4B L16,
IIA .90) -- values of the pointer-read head are U^T h; (2) Transport outputs a SHIFT dz in those coordinates and Commit
adds W_c dz to the retrieval query -- no relation, no shift, no change (forward and position-free inputs untouched by
construction); (3) four relations are trained together from answers only: identity (forward), reverse ("from the
last/end"), after ("the image right after the k-th image"), before; (4) reverse is constrained to be an involution in
address coordinates: T_rev(T_rev(z)) = z. Held out: two reverse phrasings, N = 9, and a never-trained composition
("the image right before the k-th image from the last").

S580 docstring follows.
S580: Latent Reference Transport (LRT) v0 -- restore only the missing reference transformation, one pass, causal.

Diagnosis this rests on: small models already have the native index system (block codes written by their own counter,
§585-§588) and an exact forward lookup by code; in a composed reference they never form the target code (§539) and the
answer is decided by the attention ratio of one retrieval layer (readout law §536: 4B L21). 27B forms the target code
causally after the reference -- end read from the blocks' codes (L47), transformation (L48-50), native lookup (L51).
LRT installs only that interface in the frozen small model:
    Read       at layer `lin` (4B L16, the end of the ordinal's causal window): every text position t after the images
               reads (a) the text before it (its own reference phrase: pointer + relation) and (b) the image blocks'
               START markers only -- a block's start marker precedes its own picture, so the module sees index
               structure, never picture content.
    Transport  a tiny MLP maps [reference read, structure read] to a low-dimensional reference shift z_t (d_c = 8).
    Commit     z_t enters ONLY the retrieval layer's query (layer `lq`, 4B L21; post-q_norm, pre-RoPE) through a
               zero-initialised rank-8 map: q_t <- q_t + W z_t. The model's own keys, values and lookup do the rest.
Loss: answer cross-entropy (the target index is never given) + locality KL on position-free content questions.
Same synthetic data, held-out sets and checks as S579 (the generic local-LoRA baseline).
"""
from __future__ import annotations

import argparse, json, math, pickle, random, sys, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s581_lrt1_{}.json"
CKPT = "/vlm/results/s581_lrt1_{}.pt"
CIFAR = "/vlm/assets/cifar10"
REV_TRAIN = [" from the last", " from the end"]
REV_HELD = [" counting from the end", " counting backwards"]
TEXT_PROMPTS = [
    "Explain why the sky looks blue during the day.", "Write a haiku about autumn leaves.",
    "What is the capital of Australia?", "Summarize the plot of Romeo and Juliet in one sentence.",
    "List three uses of copper.", "How do vaccines train the immune system?", "Translate 'good morning' into French.",
    "What is 17 multiplied by 6?", "Give a synonym for 'rapid'.", "Describe the water cycle briefly.",
    "Who wrote 'Pride and Prejudice'?", "What does a compiler do?", "Name a mammal that lays eggs.",
    "Why do we have seasons on Earth?", "Suggest a name for a friendly robot.", "What is photosynthesis?",
    "Convert 5 kilometres to miles.", "What is the boiling point of water at sea level in Celsius?",
    "Write one sentence encouraging a student before an exam.", "What is the difference between weather and climate?",
    "Name the largest planet in the solar system.", "What is a prime number?", "How many legs does a spider have?",
    "Give an example of a renewable energy source.", "What is the third letter of the English alphabet?",
    "Count backwards from five to one.", "Which comes first, the second or the fourth chapter?",
    "What is the last day of the week in most calendars?", "Name the first element of the periodic table.",
    "If a list has eight items, which item is in the middle?",
]


def load_cifar(pixels):
    with open(f"{CIFAR}/batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]
    with open(f"{CIFAR}/test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    by = {i: [] for i in range(len(names))}
    for arr, y in zip(data, d[b"labels"]):
        if len(by[y]) < 400:
            by[y].append(arr)
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC) for a in v_] for i, v_ in by.items()}


class LRT1(nn.Module):
    def __init__(self, d, n_heads, head_dim, U, dr=16, de=8, dk=64, dh=64):
        super().__init__()
        self.register_buffer("U", U)                                            # [d, a] orthonormal native address basis
        a = U.shape[1]
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.qp = nn.Linear(d, dk, bias=False); self.kp = nn.Linear(d, dk, bias=False)
        self.qr = nn.Linear(d, dk, bias=False); self.kr = nn.Linear(d, dk, bias=False); self.vr = nn.Linear(d, dr, bias=False)
        self.qe = nn.Linear(d, dk, bias=False); self.ke = nn.Linear(d, dk, bias=False); self.ve = nn.Linear(d, de, bias=False)
        self.T = nn.Sequential(nn.Linear(a + dr + de, dh), nn.GELU(), nn.Linear(dh, a))
        self.Wc = nn.Linear(a, n_heads * head_dim, bias=False)
        nn.init.zeros_(self.Wc.weight)
        self.H, self.D, self.dk = n_heads, head_dim, dk
        self.att_struct = None; self.last = {}

    def reads(self, h_txt, h_mark):
        a = self.n1(h_txt); b = self.n2(h_mark)
        T1 = h_txt.shape[0]
        causal = torch.ones(T1, T1, dtype=torch.bool, device=h_txt.device).tril()
        sp = (self.qp(a) @ self.kp(a).T / math.sqrt(self.dk)).masked_fill(~causal, float("-inf"))
        zc = (h_txt / h_txt.norm(dim=-1, keepdim=True).clamp_min(1e-6)) @ self.U * math.sqrt(h_txt.shape[-1])
        z = torch.softmax(sp, -1) @ zc                                           # pointer in native address coordinates
        sr = (self.qr(a) @ self.kr(a).T / math.sqrt(self.dk)).masked_fill(~causal, float("-inf"))
        r = torch.softmax(sr, -1) @ self.vr(a)                                   # relation
        Ae = torch.softmax(self.qe(a) @ self.ke(b).T / math.sqrt(self.dk), -1)
        e = Ae @ self.ve(b)                                                      # structure (block boundaries only)
        self.att_struct = Ae.detach()
        return z, r, e

    def shift(self, z, r, e):
        return self.T(torch.cat([z, r, e], -1))

    def forward(self, h_txt, h_mark):
        z, r, e = self.reads(h_txt, h_mark)
        dz = self.shift(z, r, e)
        dz2 = self.shift(z + dz, r, e)
        self.last = dict(dz=dz, inv=dz + dz2)                                    # involution residual: T(T(z)) - z
        return self.Wc(dz).view(-1, self.H, self.D)


def run(v, ufile, tag="job", lin=16, lq=21, steps=4000, lr=1e-3, lam=1.0, lam_inv=0.1, pixels=224, seed=58100, neval=30):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    A = v.layers[lq].self_attn
    H = A.q_proj.out_features // A.head_dim
    if v.fam == "qwen35":
        H = A.q_proj.out_features // (2 * A.head_dim)
    d = v.layers[0].input_layernorm.weight.shape[0]
    dev = A.q_proj.weight.device
    U = torch.linalg.qr(torch.load(ufile, weights_only=False)["Ua"].float())[0]
    mod = LRT1(d, H, A.head_dim, U.to(dev)).to(dev).float()
    params = list(mod.parameters())
    print(f"[{tag}] 읽기 L{lin} → 검색 질의 L{lq} ({H} 머리 × {A.head_dim}), 학습 파라미터 {sum(p.numel() for p in params)/1e6:.3f}M", flush=True)
    ST = {"on": False, "h": None}

    def cap_hook(m, i, o):
        if ST["on"]:
            t = o[0] if isinstance(o, tuple) else o
            ST["h"] = t[0]
        return o

    def q_hook(m, i, o):
        if not ST["on"] or ST["h"] is None:
            return o
        P, marks, h = ST["P"], ST["marks"], ST["h"]
        dq = mod(h[P:].float(), h[marks].float()).to(o.dtype)                  # [T1, H, D]
        o2 = torch.cat([o[:, :P], o[:, P:] + dq[None]], 1)
        return o2
    hks = [v.layers[lin].register_forward_hook(cap_hook), A.q_norm.register_forward_hook(q_hook)]

    def lora(on):                                                                # kept name: switches the module on/off
        ST["on"] = on

    def make(rng, N=None, fmt=None, rel=None, phr=None):
        N = N or rng.randint(3, 8)
        ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        fmt = fmt or rng.choice(["question", "statement", "options"])
        rel = rel or rng.choice(["F", "R", "R", "A", "B"])
        def ref(k):                                                              # (phrase, target) for pointer k
            if rel == "F":
                return f"the {ORD[k-1]} image", k
            if rel == "R":
                p = phr if phr is not None else rng.choice(REV_TRAIN)
                return f"the {ORD[k-1]} image{p}", N + 1 - k
            if rel == "A":
                return f"the image right after the {ORD[k-1]} image", k + 1
            if rel == "B":
                return f"the image right before the {ORD[k-1]} image", k - 1
            if rel == "C":                                                       # held-out composition: before(reverse(k))
                return f"the image right before the {ORD[k-1]} image from the last", N - k
        lo, hi = {"F": (1, N), "R": (1, N), "A": (1, N - 1), "B": (2, N), "C": (1, N - 1)}[rel]
        if fmt == "options":
            m = min(4, hi - lo + 1); ks = rng.sample(range(lo, hi + 1), m); target = rng.randrange(m)
            refs = [ref(kk) for kk in ks]
            tgt = refs[target][1]
            lines = [f"{chr(65 + i)}: {p[0].upper() + p[1:]}" for i, (p, _) in enumerate(refs)]
            text = (f"{v.IMG*N}\nWhich image shows a {cl[tgt - 1]}?\nSelect from the following choices.\n" + "\n".join(lines) +
                    "\nAnswer with the option's letter from the given choices directly.")
            cand = [tok1(chr(65 + i)) for i in range(m)]; gold = target
        else:
            k = rng.randint(lo, hi); p, tgt = ref(k)
            opts = cl[:]; rng.shuffle(opts)
            if fmt == "question":
                text = f"{v.IMG*N}\nWhat is in {p}? Answer with only one word from: {', '.join(opts)}."
            else:
                text = f"{v.IMG*N}\nLook at {p}. What is in it? Answer with only one word from: {', '.join(opts)}."
            cand = [tok1(" " + c) for c in cl]; gold = tgt - 1
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=cand, gold=gold, rel=rel)

    def make_content(rng):
        N = rng.randint(2, 8); ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        absent = rng.choice([names[x] for x in range(len(names)) if x not in ci])
        ch = [rng.choice(cl), absent]; rng.shuffle(ch)
        text = f"{v.IMG*N}\nWhich of these is shown in one of the images? Answer with only one word from: {', '.join(ch)}."
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=[tok1(" " + c) for c in ch],
                    gold=ch.index(next(c for c in ch if c in cl)))

    def make_text(rng):
        return dict(text=v.chat([("user", rng.choice(TEXT_PROMPTS))], prefix_assistant=THINK), pics=[], cand=None, gold=None)

    def logits(ex):
        inp = v.encode(ex["text"], ex["pics"]) if ex["pics"] else v.encode(ex["text"], [])
        if ex["pics"]:
            B = v.image_blocks(inp["input_ids"], len(ex["pics"]))
            ST.update(P=int(B[-1][-1]) + 2, marks=torch.tensor([int(b[0]) - 1 for b in B], device=dev), h=None)
            on = ST["on"]
        else:
            on = ST["on"]; ST["on"] = False                                      # no images -> nothing to read
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["on"] = on; ST["h"] = None
        return out.logits[0, -1].float()

    @torch.no_grad()
    def evaluate(sets):
        res = {}
        for nm, exs in sets.items():
            res[nm] = float(np.mean([int(logits(ex)[ex["cand"]].argmax()) == ex["gold"] for ex in exs]))
        return res

    @torch.no_grad()
    def text_kl(exs):
        kls = []
        for ex in exs:
            lora(False); p = torch.log_softmax(logits(ex), -1)
            lora(True); q = torch.log_softmax(logits(ex), -1)
            kls.append(float((p.exp() * (p - q)).sum()))
        return float(np.mean(kls))

    erng = random.Random(seed + 1)
    sets = {}
    for fmt in ("question", "statement", "options"):
        for rel in ("F", "R", "A", "B"):
            sets[f"{fmt}|{rel}"] = [make(erng, fmt=fmt, rel=rel) for _ in range(neval)]
        sets[f"{fmt}|R|held-phr"] = [make(erng, fmt=fmt, rel="R", phr=erng.choice(REV_HELD)) for _ in range(neval)]
        sets[f"{fmt}|R|N9"] = [make(erng, N=9, fmt=fmt, rel="R") for _ in range(neval // 2)]
        sets[f"{fmt}|F|N9"] = [make(erng, N=9, fmt=fmt, rel="F") for _ in range(neval // 2)]
    sets["question|C|held-comp"] = [make(erng, fmt="question", rel="C") for _ in range(neval)]
    sets["content|2AFC"] = [make_content(erng) for _ in range(neval)]
    texts_eval = [dict(text=v.chat([("user", p)], prefix_assistant=THINK), pics=[], cand=None, gold=None) for p in TEXT_PROMPTS]
    lora(False); before = evaluate(sets); lora(True)
    print(f"[{tag}] 학습 전: " + " · ".join(f"{k} {x:.2f}" for k, x in before.items()), flush=True)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed); t0 = time.time(); hist = {"ce": [], "kl": []}
    for step in range(steps):
        ex = make(rng)
        with torch.enable_grad():
            lg = logits(ex)
            ce = torch.nn.functional.cross_entropy(lg[ex["cand"]][None], torch.tensor([ex["gold"]], device=lg.device))
            inv = (mod.last["inv"] ** 2).sum(-1).mean() if ex["rel"] == "R" else torch.zeros((), device=lg.device)
            loc = make_content(rng) if rng.random() < .5 else make_text(rng)
            with torch.no_grad():
                lora(False); p = torch.log_softmax(logits(loc), -1); lora(True)
            q = torch.log_softmax(logits(loc), -1)
            kl = (p.exp() * (p - q)).sum()
            loss = ce + lam * kl + lam_inv * inv
            loss.backward()
        hist["ce"].append(float(ce)); hist["kl"].append(float(kl))
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 250 == 0:
            print(f"[{tag}] 걸음 {step+1}/{steps}  CE {np.mean(hist['ce'][-250:]):.3f}  KL {np.mean(hist['kl'][-250:]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    after = evaluate(sets)
    kl_text = text_kl(texts_eval)
    print(f"[{tag}] 학습 뒤: " + " · ".join(f"{k} {x:.2f}" for k, x in after.items()), flush=True)
    print(f"[{tag}] 글만 입력 KL(원 모델 || 조정 모델) 평균 {kl_text:.4f}", flush=True)
    res = dict(before=before, after=after, kl_text=kl_text, lin=lin, lq=lq, steps=steps, lam=lam)
    def avg(keys):
        return float(np.mean([after[k_] for k_ in keys])), float(np.mean([before[k_] for k_ in keys]))
    fm = ("question", "statement", "options")
    summ = {nm: avg([f"{f}|{nm}" for f in fm]) for nm in ("F", "R", "A", "B")}
    summ["R|held-phr"] = avg([f"{f}|R|held-phr" for f in fm]); summ["R|N9"] = avg([f"{f}|R|N9" for f in fm])
    summ["F|N9"] = avg([f"{f}|F|N9" for f in fm]); summ["C|held-comp"] = avg(["question|C|held-comp"]); summ["content"] = avg(["content|2AFC"])
    res["summary"] = {k_: dict(after=a_, before=b_) for k_, (a_, b_) in summ.items()}
    print("\n  요약(학습 전 → 뒤): " + " · ".join(f"{k_} {b_:.3f}→{a_:.3f}" for k_, (a_, b_) in summ.items()))
    dF = max(abs(after[f"{f}|F"] - before[f"{f}|F"]) for f in fm)
    print(f"  가름 (사전 등록 S581) (v1) 역순(세 형식) >= .85 · 직후/직전 >= .85. (v2) 보류 표현 >= .75, N9 >= .70, 학습 안 한 합성(C) 학습 전보다 +.20 이상.")
    print(f"    (v3) 전방 형식별 변화 <= .03(최대 {dF:.3f}), 내용 2AFC 변화 <= .03. (v4, 보고) 구조 읽기 주의, 전방 Δq 크기.")
    # where does the structure read go, and how large is the commit on forward vs reverse
    att = {"struct_last": [], "struct_first": [], "dq_F": [], "dq_R": []}
    with torch.no_grad():
        for rel_, key_ in (("R", "question|R"), ("F", "question|F")):
            for ex in sets[key_][:20]:
                logits(ex)
                if rel_ == "R":
                    a2 = mod.att_struct[-1].cpu(); att["struct_last"].append(float(a2[-1])); att["struct_first"].append(float(a2[0]))
                att[f"dq_{rel_}"].append(float(mod.last["dz"][-1].norm()))
    res["att"] = {k_: float(np.mean(x)) for k_, x in att.items()}
    print(f"  구조 읽기(역순, 마지막 글 자리): 마지막 블록 표지 {res['att']['struct_last']:.3f} · 첫 표지 {res['att']['struct_first']:.3f};  이동 |dz|: 역순 {res['att']['dq_R']:.3f} · 전방 {res['att']['dq_F']:.3f}")
    torch.save(mod.state_dict() | {"_meta": dict(lin=lin, lq=lq)}, CKPT.format(tag))
    json.dump(res, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}  {CKPT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ufile", required=True)
    ap.add_argument("--lin", type=int, default=16)
    ap.add_argument("--lq", type=int, default=21)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lam_inv", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--neval", type=int, default=30)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, a.ufile, tag=a.tag, lin=a.lin, lq=a.lq, steps=a.steps, lr=a.lr, lam=a.lam, lam_inv=a.lam_inv, neval=a.neval)
