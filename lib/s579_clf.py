"""S579: circuit-localized fine-tuning -- open, in the small model, the band of layers where 27B computes the reverse
target, and nothing else.

Why these layers (findings): small models never invoke the target computation in a composed reference (§539; memory
composed-run-never-computes-the-index) although their forward lookup is intact; their answer is set by the attention
ratio of one retrieval layer (readout law, 4B L21 / 8B L29, §536); 27B computes the target over five layers ending at
its retrieval layer -- L47 reads the end from the blocks' codes, L48-50 turn it into the target code, L51 retrieves it
(§597-§600). So the small model gets the same five-layer band ending at its own retrieval layer (4B L17-21), and only
there: low-rank adapters on the attention Q/K/V/O and MLP projections of those layers. Nothing is added, no token is
written, no value is computed outside the model.
Objective: answer cross-entropy on synthetic CIFAR scenes (question / statement / option-list formats, forward and
reverse; the target index is never given) + a locality term: KL(base || tuned) of the next-token distribution on
inputs the edit must not touch (position-free content questions on the same images, text-only prompts).
Built-in checks (benchmarks and the mechanism tests are run by a separate script on the saved adapters): synthetic
held-out sets before/after (held-out phrasings, N = 9), content 2AFC, text-only KL.
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
OUT = "/vlm/results/s579_clf_{}.json"
CKPT = "/vlm/results/s579_adapters_{}.pt"
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


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * (1.0 / math.sqrt(base.in_features)))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r
        self.on = True

    def forward(self, x):
        y = self.base(x)
        if not self.on:
            return y
        return y + ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)


def attach(v, layers, r, alpha):
    mods = []
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "out_proj")
    for l in layers:
        L = v.layers[l]
        for parent in (getattr(L, "self_attn", None), getattr(L, "linear_attn", None), getattr(L, "mlp", None)):
            if parent is None:
                continue
            for nm in names:
                lin = getattr(parent, nm, None)
                if isinstance(lin, nn.Linear):
                    lo = LoRALinear(lin, r, alpha); setattr(parent, nm, lo); mods.append((l, nm, lo))
    return mods


def run(v, tag="job", band=(17, 21), r=8, alpha=16, steps=3000, lr=2e-4, lam=1.0, pixels=224, seed=57900, neval=40):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    layers = list(range(band[0], band[1] + 1))
    mods = attach(v, layers, r, alpha)
    params = [p for _, _, m in mods for p in (m.A, m.B)]
    print(f"[{tag}] 열린 층 {layers}, 어댑터 {len(mods)} 개, 학습 파라미터 {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)

    def lora(on):
        for _, _, m in mods:
            m.on = on

    def make(rng, N=None, fmt=None, rev=None, phr=None):
        N = N or rng.randint(3, 8)
        ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        fmt = fmt or rng.choice(["question", "statement", "options"])
        rev = rng.random() < .5 if rev is None else rev
        phr = phr if phr is not None else (rng.choice(REV_TRAIN) if rev else "")
        if fmt == "options":
            m = min(4, N); ks = rng.sample(range(1, N + 1), m); target = rng.randrange(m)
            tgt = (N + 1 - ks[target]) if rev else ks[target]
            lines = [f"{chr(65 + i)}: The {ORD[kk-1]} image{phr}" for i, kk in enumerate(ks)]
            text = (f"{v.IMG*N}\nWhich image shows a {cl[tgt - 1]}?\nSelect from the following choices.\n" + "\n".join(lines) +
                    "\nAnswer with the option's letter from the given choices directly.")
            cand = [tok1(chr(65 + i)) for i in range(m)]; gold = target
        else:
            k = rng.randint(1, N); tgt = (N + 1 - k) if rev else k
            opts = cl[:]; rng.shuffle(opts)
            if fmt == "question":
                text = f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{phr}? Answer with only one word from: {', '.join(opts)}."
            else:
                text = f"{v.IMG*N}\nLook at the {ORD[k-1]} image{phr}. What is in it? Answer with only one word from: {', '.join(opts)}."
            cand = [tok1(" " + c) for c in cl]; gold = tgt - 1
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=cand, gold=gold)

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
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
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
        for rev in (False, True):
            sets[f"{fmt}|{'R' if rev else 'F'}|train-phr"] = [make(erng, fmt=fmt, rev=rev) for _ in range(neval)]
        sets[f"{fmt}|R|held-phr"] = [make(erng, fmt=fmt, rev=True, phr=erng.choice(REV_HELD)) for _ in range(neval)]
        sets[f"{fmt}|R|N9"] = [make(erng, N=9, fmt=fmt, rev=True) for _ in range(neval // 2)]
        sets[f"{fmt}|F|N9"] = [make(erng, N=9, fmt=fmt, rev=False) for _ in range(neval // 2)]
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
            loc = make_content(rng) if rng.random() < .5 else make_text(rng)
            with torch.no_grad():
                lora(False); p = torch.log_softmax(logits(loc), -1); lora(True)
            q = torch.log_softmax(logits(loc), -1)
            kl = (p.exp() * (p - q)).sum()
            loss = ce + lam * kl
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
    res = dict(before=before, after=after, kl_text=kl_text, band=layers, r=r, steps=steps, lam=lam)
    R_tr = np.mean([after[f"{f}|R|train-phr"] for f in ("question", "statement", "options")])
    R_held = np.mean([after[f"{f}|R|held-phr"] for f in ("question", "statement", "options")])
    R_n9 = np.mean([after[f"{f}|R|N9"] for f in ("question", "statement", "options")])
    dF = max(abs(after[f"{f}|F|train-phr"] - before[f"{f}|F|train-phr"]) for f in ("question", "statement", "options"))
    print(f"\n  요약: 역순(학습 표현) {np.mean([before[f'{f}|R|train-phr'] for f in ('question','statement','options')]):.3f} → {R_tr:.3f} · 보류 표현 → {R_held:.3f} · N9 → {R_n9:.3f}"
          f" · 전방 최대 변화 {dF:.3f} · 내용 2AFC {before['content|2AFC']:.3f} → {after['content|2AFC']:.3f} · 글 KL {kl_text:.4f}")
    print(f"  가름 (사전 등록 S579) (e1) 역순(학습 표현, 세 형식 평균) >= .85. (e2) 보류 표현 >= .75, N9 >= .70. (e3) 전방 변화 <= .03, 내용 2AFC 변화 <= .03, 글 KL <= .05.")
    torch.save({f"{l}.{nm}": (m.A.detach().cpu(), m.B.detach().cpu()) for l, nm, m in mods} | {"_meta": dict(band=layers, r=r, alpha=alpha)}, CKPT.format(tag))
    json.dump(res, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}  {CKPT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--band", type=int, nargs=2, default=[17, 21])
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--neval", type=int, default=40)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, band=tuple(a.band), r=a.r, steps=a.steps, lr=a.lr, lam=a.lam, neval=a.neval)
