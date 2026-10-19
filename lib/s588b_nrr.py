"""S588b: Native Reference Rebinding (NRR) -- learn only the missing index-axis map, from answers alone
(pre-registered in exp/QUEUE.md before data).

Findings this builds on: the operative reference is the source ordinal, re-read by the whole readout window (§553 §608
§609); rebinding that one token between the model's own forward ordinal prototypes fixes reverse (§611: .88 / .80);
the native ordinal code is an ordered but curved chart, so the reflection lives on the index axis (§610); with the
model's own two reads and a fixed reflection, the rebinding reaches the prototype ceiling whenever the end is read
correctly and carries 27B's two signatures (§612). Here nothing about the reflection is given:
  frozen   the model; ordinal prototypes M (forward questions, three formats); pointer k_hat = nearest prototype;
           end distribution e = image-block index reader (S588c, structural calibration on forward scenes only).
  learned  (answers only, no target index) a relation reader at the source layer -- one attention head over the next w
           tokens of the question -- that sets the coefficients of the index map t = c0 + c_e * E + c_p * k and a gate.
  write    h_src += g * (alpha - beta) @ M, alpha = pushforward of the pointer through the map (Gaussian on the index
           axis), beta = the same kernel at k_hat (identity coefficients write exactly zero).
Sources are found natively: question tokens after the images whose state lies near the prototype chart (threshold
calibrated on forward ordinal tokens). The image prefix is complete before the question, so the end reader may use
block states at any depth (two passes here; one pass with a cached image prefix).
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
from s588c_reader import probs as reader_probs

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s588b_nrr_{}.json"
CKPT = "/vlm/results/s588b_nrr_{}.pt"
CIFAR = "/vlm/assets/cifar10"
C = 10
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


class RelationReader(nn.Module):
    """One attention head from the source over the next w question tokens -> index-map coefficients and a gate."""
    def __init__(self, d, dh=16, dr=16):
        super().__init__()
        self.n = nn.LayerNorm(d, elementwise_affine=False)
        self.q = nn.Linear(d, dh, bias=False); self.k = nn.Linear(d, dh, bias=False); self.v = nn.Linear(d, dr, bias=False)
        self.A = nn.Linear(dr, 3); self.g = nn.Linear(dr, 1)
        nn.init.zeros_(self.A.weight); nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)
        self.dh = dh

    def forward(self, hs, hw):                                                    # [d], [w, d] (float32)
        a = torch.softmax((self.k(self.n(hw)) @ self.q(self.n(hs))) / math.sqrt(self.dh), 0)
        r = a @ self.v(self.n(hw))
        c = 3.0 * torch.tanh(self.A(r)) + torch.tensor([0.0, 0.0, 1.0], device=r.device)   # identity at init, bounded
        return c, torch.sigmoid(self.g(r))[0], a


class MicroCircuit(nn.Module):
    """S588d: READ (end anchor over the blocks' structural states) -> COMPOSE (two gated residual steps, no index scalar)
    -> RETARGET (attention over the blocks picks the target block; 27B L47 / L48-50 / L51 in miniature)."""
    def __init__(self, d, fdim, C, dh=16, dr=16, da=32, du=64):
        super().__init__()
        self.n = nn.LayerNorm(d, elementwise_affine=False); self.nf = nn.LayerNorm(fdim, elementwise_affine=False)
        self.rq = nn.Linear(d, dh, bias=False); self.rk = nn.Linear(d, dh, bias=False); self.rv = nn.Linear(d, dr, bias=False)
        zin = (C - 1) + dr
        self.aq = nn.Linear(zin, da); self.ak = nn.Linear(fdim, da, bias=False); self.av = nn.Linear(fdim, da, bias=False)
        self.inp = nn.Linear(zin + da, du)
        self.f1 = nn.Linear(du + dr + da, 2 * du); self.o1 = nn.Linear(du, du)
        self.f2 = nn.Linear(du + dr + da, 2 * du); self.o2 = nn.Linear(du, du)
        self.tq = nn.Linear(du, da); self.tk = nn.Linear(fdim, da, bias=False)
        self.g = nn.Linear(du, 1)
        nn.init.zeros_(self.g.weight); nn.init.constant_(self.g.bias, 3.0)        # open at init (a closed gate starves RETARGET)
        self.dh, self.da = dh, da

    def forward(self, hs, hw, z, X):
        r = torch.softmax((self.rk(self.n(hw)) @ self.rq(self.n(hs))) / math.sqrt(self.dh), 0) @ self.rv(self.n(hw))
        zr = torch.cat([z, r])
        Xn = self.nf(X)
        a_e = torch.softmax(self.ak(Xn) @ self.aq(zr) / math.sqrt(self.da), 0)  # READ
        a = a_e @ self.av(Xn)
        u = self.inp(torch.cat([zr, a]))                                          # COMPOSE
        for f, o_ in ((self.f1, self.o1), (self.f2, self.o2)):
            gg, vv = f(torch.cat([u, r, a])).chunk(2)
            u = u + o_(torch.nn.functional.silu(gg) * vv)
        pi = torch.softmax(self.tk(Xn) @ self.tq(u) / math.sqrt(self.da), 0)    # RETARGET
        return pi, torch.sigmoid(self.g(u))[0], a_e


MAPS = [(c0, ce, cp) for c0 in (-1, 0, 1) for ce in (0, 1) for cp in (-1, 0, 1)]   # integer affine maps t = c0 + ce*E + cp*k
MAP_ID = MAPS.index((0, 0, 1))


class MapSelector(nn.Module):
    """v1b: one attention head from the source over the next w question tokens -> a distribution over the integer affine
    maps of the index axis (starts at the identity map, which writes nothing)."""
    def __init__(self, d, dh=16, dr=16, id_logit=5.0):
        super().__init__()
        self.n = nn.LayerNorm(d, elementwise_affine=False)
        self.q = nn.Linear(d, dh, bias=False); self.k = nn.Linear(d, dh, bias=False); self.v = nn.Linear(d, dr, bias=False)
        self.out = nn.Linear(dr, len(MAPS))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.bias[MAP_ID] = id_logit
        self.dh = dh

    def forward(self, hs, hw):
        a = torch.softmax((self.k(self.n(hw)) @ self.q(self.n(hs))) / math.sqrt(self.dh), 0)
        r = a @ self.v(self.n(hw))
        self.last_r = r.detach()
        return torch.softmax(self.out(r), -1), a


class Retargeter(nn.Module):
    """v2b: READ = the model's own end distribution (frozen block-index read, 27B L47); COMPOSE = two gated residual steps over
    [source chart coordinates, relation, end] (27B L48-50, no index scalar); RETARGET = attention over the blocks' own codes
    (27B L51), starting at the identity (the block that carries the source's code)."""
    def __init__(self, d, C, dh=16, dr=16, du=64, de=32, lam=5.0):
        super().__init__()
        self.n = nn.LayerNorm(d, elementwise_affine=False)
        self.rq = nn.Linear(d, dh, bias=False); self.rk = nn.Linear(d, dh, bias=False); self.rv = nn.Linear(d, dr, bias=False)
        self.emb = nn.Linear(C, de)
        zin = (C - 1) + dr + de
        self.inp = nn.Linear(zin, du)
        self.f1 = nn.Linear(du + dr + de, 2 * du); self.o1 = nn.Linear(du, du)
        self.f2 = nn.Linear(du + dr + de, 2 * du); self.o2 = nn.Linear(du, du)
        self.tq = nn.Linear(du, de)
        self.lam = nn.Parameter(torch.tensor(lam))
        self.dh, self.de = dh, de

    def forward(self, hs, hw, z, e, PR, kh):
        r = torch.softmax((self.rk(self.n(hw)) @ self.rq(self.n(hs))) / math.sqrt(self.dh), 0) @ self.rv(self.n(hw))
        ee = self.emb(e)
        u = self.inp(torch.cat([z, r, ee]))
        for f, o_ in ((self.f1, self.o1), (self.f2, self.o2)):
            gg, vv = f(torch.cat([u, r, ee])).chunk(2)
            u = u + o_(torch.nn.functional.silu(gg) * vv)
        logits = self.emb(PR) @ self.tq(u) / math.sqrt(self.de) + self.lam * PR[:, int(round(kh)) - 1]
        return torch.softmax(logits, 0)


class Scanner(nn.Module):
    """v2r: 27B's L48-50 transform (half of it linear attention, §600) in miniature -- a learned gated linear recurrence over
    the image blocks in sequence order, from the start and from the end (the counter family of §588); the relation gates the
    direction; the target is the block whose scanned count matches the pointer. Content-free: blocks enter only as 'a block'."""
    def __init__(self, d, dh=16, dr=16, ds=8, warm=False, mix="score"):
        super().__init__()
        self.mix = mix                                                           # S597c "prob": the direction is a latent choice
        self.n = nn.LayerNorm(d, elementwise_affine=False)
        self.rq = nn.Linear(d, dh, bias=False); self.rk = nn.Linear(d, dh, bias=False); self.rv = nn.Linear(d, dr, bias=False)
        self.vin = nn.Parameter(torch.randn(ds) * 0.3)                           # per-block input of the recurrence
        self.lam = nn.Parameter(torch.tensor(1.0))                               # recurrence decay
        self.wp = nn.Parameter(torch.randn(ds) * 0.3)                            # pointer direction: p = k * wp
        if warm:                                                                 # S597b: start as the exact counter (identity)
            with torch.no_grad():
                u = torch.randn(ds); u = u / u.norm() * math.sqrt(ds)            # |vin|^2 = ds: one block = a clear step
                self.vin.copy_(u); self.wp.copy_(u)                              # count m*vin vs pointer k*vin -> block k
        self.g = nn.Linear(dr, 1)
        nn.init.zeros_(self.g.weight); nn.init.constant_(self.g.bias, -3.0)      # starts as the forward scan (identity)
        self.logtau = nn.Parameter(torch.tensor(0.0))
        self.dh = dh

    def forward(self, hs, hw, N, kh, prior=None):
        lg = (self.rk(self.n(hw)) @ self.rq(self.n(hs))) / math.sqrt(self.dh)
        if prior is not None:                                                    # S609: the model's own binding of each token
            lg = lg + torch.log(prior.to(lg.device).float().clamp(min=1e-3))     # to this ordinal (attention, r3_bind)
        a_ = torch.softmax(lg, 0)
        self.last_a = a_.detach()                                                # S600a: where the relation head looks
        r = a_ @ self.rv(self.n(hw))
        fw, bw = [], []
        h = torch.zeros_like(self.vin)
        for _ in range(N):
            h = self.lam * h + self.vin; fw.append(h)
        h = torch.zeros_like(self.vin)
        for _ in range(N):
            h = self.lam * h + self.vin; bw.append(h)
        fw = torch.stack(fw); bw = torch.stack(bw[::-1])                        # bw[b] = count from the end
        p = float(kh) * self.wp
        tau = torch.exp(self.logtau)
        s_fw = -((fw - p) ** 2).sum(1) / tau; s_bw = -((bw - p) ** 2).sum(1) / tau
        self.last_glogit = self.g(r)[0]                                          # S609c: logit-BCE for the gate
        g = torch.sigmoid(self.last_glogit)
        self.last_pf, self.last_pb = torch.softmax(s_fw, 0), torch.softmax(s_bw, 0)   # S618: the two scans separately
        if self.mix == "prob":                                                   # S597c: mixture of the two scans' targets --
            return g * torch.softmax(s_bw, 0) + (1 - g) * torch.softmax(s_fw, 0), g   # an unsure gate gives an unsure target
        return torch.softmax(g * s_bw + (1 - g) * s_fw, 0), g


def run(v, tag, Lb, reader_tag, w=8, steps=3000, lr=1e-3, accum=1, lam=1.0, sigma=0.5, neval=40, nproto=16,
        pixels=224, seed=58800, variant="index", pevery=250):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tok1 = lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]
    im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
    dev = v.device
    cfg = v.model.config
    d = cfg.get_text_config().hidden_size if hasattr(cfg, "get_text_config") else cfg.text_config.hidden_size
    R = torch.load(f"/vlm/results/s588c_reader_{reader_tag}.pt")
    RP, Rlayers, Rsites, SITES = R["P"], R["layers"], R["sites"], R["SITES"]
    ord_ids = {tok1(" " + o): i + 1 for i, o in enumerate(ORD[:C])}             # used ONLY to label calibration/diagnostics
    fdim = len(Rlayers) * len(Rsites) * d
    rel = {"index": lambda: RelationReader(d), "circuit": lambda: MicroCircuit(d, fdim, C),
           "map": lambda: MapSelector(d), "retarget": lambda: Retargeter(d, C)}[variant]().to(dev).float()
    print(f"[{tag}] 변형 {variant} · 학습 파라미터 {sum(p_.numel() for p_ in rel.parameters())/1e3:.1f}K · 끝 읽기 L{Rlayers} {Rsites}", flush=True)
    ST = {"on": False}

    # ---------------- generators (same family as S579) ----------------
    def make(rng, N=None, fmt=None, rev=None, phr=None, k=None):
        N = N or rng.randint(3, 8)
        ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        fmt = fmt or rng.choice(["question", "statement", "options"])
        rev = rng.random() < .5 if rev is None else rev
        phr = phr if phr is not None else (rng.choice(REV_TRAIN) if rev else "")
        if fmt == "options":
            m = min(4, N); ks = rng.sample(range(1, N + 1), m); target = rng.randrange(m)
            if k is not None and k not in ks:
                ks[rng.randrange(m)] = k
            tgt = (N + 1 - ks[target]) if rev else ks[target]
            lines = [f"{chr(65 + i)}: The {ORD[kk-1]} image{phr}" for i, kk in enumerate(ks)]
            text = (f"{v.IMG*N}\nWhich image shows a {cl[tgt - 1]}?\nSelect from the following choices.\n" + "\n".join(lines) +
                    "\nAnswer with the option's letter from the given choices directly.")
            cand = [tok1(chr(65 + i)) for i in range(m)]; gold = target
        else:
            k = k or rng.randint(1, N); tgt = (N + 1 - k) if rev else k
            opts = cl[:]; rng.shuffle(opts)
            if fmt == "question":
                text = f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{phr}? Answer with only one word from: {', '.join(opts)}."
            else:
                text = f"{v.IMG*N}\nLook at the {ORD[k-1]} image{phr}. What is in it? Answer with only one word from: {', '.join(opts)}."
            cand = [tok1(" " + c) for c in cl]; gold = tgt - 1
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=cand, gold=gold, N=N)

    def make_content(rng):
        N = rng.randint(2, 8); ci = rng.sample(range(len(names)), N)
        cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
        absent = rng.choice([names[x] for x in range(len(names)) if x not in ci])
        ch = [rng.choice(cl), absent]; rng.shuffle(ch)
        text = f"{v.IMG*N}\nWhich of these is shown in one of the images? Answer with only one word from: {', '.join(ch)}."
        return dict(text=v.chat([("user", text)], prefix_assistant=THINK), pics=pics, cand=[tok1(" " + c) for c in ch],
                    gold=ch.index(next(c for c in ch if c in cl)), N=N)

    def make_text(rng):
        return dict(text=v.chat([("user", rng.choice(TEXT_PROMPTS))], prefix_assistant=THINK), pics=[], cand=None, gold=None, N=0)

    # ---------------- hooks ----------------
    def hook_reader(l):
        def h(m, i, o):
            if ST.get("cap_blocks") and l in Rlayers:
                t = o[0] if isinstance(o, tuple) else o
                f = []
                for b in ST["B"]:
                    allsite = torch.stack([t[0, b].float().mean(0), t[0, int(b[0]) - 1].float(), t[0, int(b[-1]) + 1].float(),
                                           t[0, int(b[-1])].float()])
                    f.append(torch.cat([allsite[SITES.index(s_)] for s_ in Rsites]))
                ST["bf"][l] = torch.stack(f).cpu()
            return o
        return h

    def hook_bind(m, i, o):
        if not (ST["on"] or ST.get("cap_src")):
            return o
        t = o[0] if isinstance(o, tuple) else o
        if ST.get("cap_src"):
            ST["hsrc"] = t[0, ST["cand_pos"]].detach().float().cpu()
            return o
        rows = []
        tt = torch.arange(1, C + 1, device=t.device).float()
        blk = ST["blk"]
        for p, kh, zc in ST["src"]:
            hw = t[0, p + 1:min(p + 1 + w, ST["turn"])].float()
            if variant == "index":
                c, g, a = rel(t[0, p].float(), hw)
                ST["log"].append((float(c[0]), float(c[1]), float(c[2]), float(g)))
                ctr = c[0] + c[1] * tt + c[2] * kh                                 # [C]: map centre for each end value b
                K = torch.softmax(-(tt[None, :] - ctr[:, None]) ** 2 / (2 * sigma ** 2), 1)   # [b, t], stable
                alpha = (blk["e"][:, None] * K).sum(0)
            elif variant == "circuit":
                pi, g, a_e = rel(t[0, p].float(), hw, zc.to(t.device), blk["X"])
                ST["log"].append((int(pi.argmax()) + 1, float(g), float(pi.max()), int(a_e.argmax()) + 1))
                alpha = pi @ blk["PR"]
            elif variant == "map":
                wm, _ = rel(t[0, p].float(), hw)
                ST["log"].append(wm.detach().cpu().numpy())
                cm = torch.tensor(MAPS, device=t.device).float()                    # [m, 3]
                ctr = cm[:, 0:1] + cm[:, 1:2] * tt[None, :] + cm[:, 2:3] * kh       # [m, b]
                K = torch.softmax(-(tt[None, None, :] - ctr[:, :, None]) ** 2 / (2 * sigma ** 2), 2)   # [m, b, t]
                alpha = wm @ torch.einsum("b,mbt->mt", blk["e"], K)
                g = 1.0
            else:                                                                   # retarget
                pi = rel(t[0, p].float(), hw, zc.to(t.device), blk["e"], blk["PR"], kh)
                ST["log"].append((int(pi.argmax()) + 1, float(pi.max())))
                alpha = pi @ blk["PR"]
                g = 1.0
            if variant == "retarget":
                bk = int(torch.argmax(blk["PR"][:, int(round(kh)) - 1]))
                beta = blk["PR"][bk]                                                # identity = the block carrying k: zero write
            else:
                beta = torch.softmax(-(tt - kh) ** 2 / (2 * sigma ** 2), 0)
            rows.append((p, g * ((alpha - beta) @ Mdev)))
        if not rows:
            return o
        t2 = t.clone()
        for p, dlt in rows:
            t2[0, p] = (t[0, p].float() + dlt).to(t.dtype)
        return (t2,) + tuple(o[1:]) if isinstance(o, tuple) else t2
    hooks = [v.layers[l].register_forward_hook(hook_reader(l)) for l in Rlayers] + [v.layers[Lb].register_forward_hook(hook_bind)]

    def blocks_of(inp, N):
        return [torch.tensor(b, device=dev) for b in v.image_blocks(inp["input_ids"], N)] if N else []

    def question_span(ids, B):
        turn = max(i for i, x in enumerate(ids) if x == im_start)
        start = int(B[-1][-1]) + 2 if B else turn
        return start, turn

    # ---------------- calibration: prototypes (three forward formats) and the source detector ----------------
    rng = random.Random(seed + 1)
    t0 = time.time()

    @torch.no_grad()
    def cap_pass(ex, cap_blocks=True):
        inp = v.encode(ex["text"], ex["pics"])
        ids = inp["input_ids"][0].tolist()
        B = blocks_of(inp, ex["N"])
        s0, turn = question_span(ids, B)
        ST.update(on=False, cap_src=True, cand_pos=torch.arange(s0, turn, device=dev), cap_blocks=cap_blocks and bool(B), B=B, bf={})
        v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        hsrc, bf = ST.pop("hsrc"), ST.pop("bf")
        ST.update(cap_src=False, cap_blocks=False)
        return ids, B, s0, turn, hsrc, bf
    protos = {i: [] for i in range(1, C + 1)}
    other = []
    for i in range(1, C + 1):
        for _ in range(nproto):
            fmt = rng.choice(["question", "statement", "options"])
            ex = make(rng, N=rng.randint(max(3, i), 10), fmt=fmt, rev=False, k=i)
            ids, B, s0, turn, hsrc, _ = cap_pass(ex, cap_blocks=False)
            for jx, pos in enumerate(range(s0, turn)):
                if ord_ids.get(ids[pos]) == i:
                    protos[i].append(hsrc[jx])
                elif ids[pos] not in ord_ids:
                    other.append(hsrc[jx])
    M = torch.stack([torch.stack(protos[i]).mean(0) for i in range(1, C + 1)])     # [C, d]
    Pb = torch.linalg.svd(M - M.mean(0), full_matrices=False)[2][:C - 1].T
    MP = M @ Pb
    zmu, zsd = MP.mean(0), MP.std(0) + 1e-4
    Mdev = M.to(dev)

    def chart(h):                                                                 # [n, d] -> nearest prototype and its distance
        dd = torch.cdist(h @ Pb, MP)
        mn, am = dd.min(1)
        return am + 1, mn
    own = torch.cat([chart(torch.stack(protos[i]))[1] for i in range(1, C + 1)])
    oth = chart(torch.stack(other))[1]
    theta = float(torch.quantile(own, .99))
    fp = float((oth <= theta).float().mean())
    kacc = float(np.mean([float((chart(torch.stack(protos[i]))[0] == i).float().mean()) for i in range(1, C + 1)]))
    print(f"[{tag}] 프로토타입(세 형식) · 탐지 문턱 {theta:.2f}: 서수 재현 .99, 다른 질문 토큰 오탐 {fp:.4f} · 포인터 해독 {kacc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    def end_dist(bf):
        X = torch.cat([bf[l] for l in Rlayers], 1)
        pr = reader_probs(RP, X)                                                   # [N, C]
        E = pr @ torch.arange(1, C + 1).float()
        return pr[int(E.argmax())]

    def forward(ex, on, grad=False, log=None):
        src, blk, turn, B = [], None, None, []
        if on and ex["N"]:
            ids, B, s0, turn, hsrc, bf = cap_pass(ex, cap_blocks=True)
        if on and B:
            kh, dist = chart(hsrc)
            X = torch.cat([bf[l] for l in Rlayers], 1)
            PR = reader_probs(RP, X)
            E = PR @ torch.arange(1, C + 1).float()
            blk = dict(e=PR[int(E.argmax())].to(dev), X=X.to(dev), PR=PR.to(dev))
            zall = (hsrc @ Pb - zmu) / zsd
            for jx in (dist <= theta).nonzero().flatten().tolist():
                src.append((s0 + jx, float(kh[jx]), zall[jx]))
        inp = v.encode(ex["text"], ex["pics"])
        ST.update(on=bool(src), src=src, blk=blk, turn=turn, log=[] if log is None else log)
        with (torch.enable_grad() if grad else torch.no_grad()):
            out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["on"] = False
        return out.logits[0, -1].float(), len(src)

    @torch.no_grad()
    def evaluate(sets, on):
        res, nsrc = {}, {}
        for nm, exs in sets.items():
            acc, ns = [], []
            for ex in exs:
                lg, n_ = forward(ex, on)
                acc.append(int(lg[ex["cand"]].argmax()) == ex["gold"]); ns.append(n_)
            res[nm] = float(np.mean(acc)); nsrc[nm] = float(np.mean(ns))
        return res, nsrc

    @torch.no_grad()
    def text_kl(exs):
        kls = []
        for ex in exs:
            p = torch.log_softmax(forward(ex, False)[0], -1); q = torch.log_softmax(forward(ex, True)[0], -1)
            kls.append(float((p.exp() * (p - q)).sum()))
        return float(np.mean(kls))

    erng = random.Random(seed + 2)
    sets = {}
    for fmt in ("question", "statement", "options"):
        for rev in (False, True):
            sets[f"{fmt}|{'R' if rev else 'F'}|train-phr"] = [make(erng, fmt=fmt, rev=rev) for _ in range(neval)]
        sets[f"{fmt}|R|held-phr"] = [make(erng, fmt=fmt, rev=True, phr=erng.choice(REV_HELD)) for _ in range(neval)]
        sets[f"{fmt}|R|N9"] = [make(erng, N=9, fmt=fmt, rev=True) for _ in range(neval // 2)]
        sets[f"{fmt}|F|N9"] = [make(erng, N=9, fmt=fmt, rev=False) for _ in range(neval // 2)]
    sets["content|2AFC"] = [make_content(erng) for _ in range(neval)]
    texts_eval = [dict(text=v.chat([("user", p)], prefix_assistant=THINK), pics=[], cand=None, gold=None, N=0) for p in TEXT_PROMPTS]
    before, _ = evaluate(sets, on=False)
    print(f"[{tag}] 학습 전: " + " · ".join(f"{k} {x:.2f}" for k, x in before.items()), flush=True)

    opt = torch.optim.Adam(rel.parameters(), lr=lr)
    rng = random.Random(seed)
    hist = {"ce": [], "kl": []}
    for step in range(steps):
        ex = make(rng)
        lg, _ = forward(ex, True, grad=True)
        ce = torch.nn.functional.cross_entropy(lg[ex["cand"]][None], torch.tensor([ex["gold"]], device=lg.device))
        loc = make_content(rng) if rng.random() < .5 else make_text(rng)
        with torch.no_grad():
            p = torch.log_softmax(forward(loc, False)[0], -1)
        q = torch.log_softmax(forward(loc, True, grad=True)[0], -1)
        kl = (p.exp() * (p - q)).sum()
        loss = (ce + lam * kl) / accum
        if loss.requires_grad and torch.isfinite(loss):                            # no source and no image: nothing to learn here
            loss.backward()
        elif not torch.isfinite(loss):
            hist.setdefault("nonfinite", []).append(step)
        hist["ce"].append(float(ce)); hist["kl"].append(float(kl))
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(rel.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % pevery == 0:
            print(f"[{tag}] 걸음 {step+1}/{steps}  CE {np.mean(hist['ce'][-pevery:]):.3f}  KL {np.mean(hist['kl'][-pevery:]):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()

    after, nsrc = evaluate(sets, on=True)
    kl_text = text_kl(texts_eval)
    # learned coefficients by phrasing (question format, k 2-4, N 6-8)
    coef = {}
    for nm, phr in (("forward", ""), ("from the last", " from the last"), ("from the end", " from the end"),
                    ("counting from the end (held)", " counting from the end"), ("counting backwards (held)", " counting backwards")):
        logs = []
        crng = random.Random(seed + 7)
        for _ in range(12):
            ex = make(crng, N=crng.randint(6, 8), fmt="question", rev=bool(phr), phr=phr)
            lg_ = []
            forward(ex, True, log=lg_)
            if variant == "index":
                logs += lg_
            elif variant == "circuit":
                logs += [(float(x[0] == ex["gold"] + 1), x[1], x[2], float(x[3] == ex["N"])) for x in lg_]
            elif variant == "map":
                logs += [x for x in lg_]
            else:
                logs += [(float(x[0] == ex["gold"] + 1), x[1]) for x in lg_]
        if variant == "map" and logs:
            wbar = np.mean(np.array(logs), 0)
            top = np.argsort(-wbar)[:3]
            coef[nm] = [(MAPS[i], round(float(wbar[i]), 3)) for i in top]
        else:
            coef[nm] = np.mean(np.array(logs), 0).round(3).tolist() if logs else None
    print(f"[{tag}] 학습 뒤: " + " · ".join(f"{k} {x:.2f}" for k, x in after.items()), flush=True)
    print({"index": f"[{tag}] 학습된 색인 사상 (c0, c_e, c_p, 관문) — 역순 = (1, 1, −1):",
           "circuit": f"[{tag}] 미세회로 (RETARGET 이 정답 블록을 고름, 관문, 최대 π, READ 가 마지막 블록을 봄):",
           "map": f"[{tag}] 고른 정수 아핀 사상 상위 3 ((c0, c_e, c_p), 무게) — 역순 = (1, 1, −1), 울타리 N−k = (0, 1, −1):",
           "retarget": f"[{tag}] RETARGET 이 정답 블록을 고름, 최대 π:"}[variant])
    for nm, c in coef.items():
        print(f"    {nm:<30s} {c}")
    R_tr = np.mean([after[f"{f}|R|train-phr"] for f in ("question", "statement", "options")])
    R_held = np.mean([after[f"{f}|R|held-phr"] for f in ("question", "statement", "options")])
    R_n9 = np.mean([after[f"{f}|R|N9"] for f in ("question", "statement", "options")])
    dF = max(abs(after[f"{f}|F|train-phr"] - before[f"{f}|F|train-phr"]) for f in ("question", "statement", "options"))
    print(f"\n  요약: 역순(학습 표현) {np.mean([before[f'{f}|R|train-phr'] for f in ('question','statement','options')]):.3f} → {R_tr:.3f} · 보류 표현 → {R_held:.3f}"
          f" · N9 {np.mean([before[f'{f}|R|N9'] for f in ('question','statement','options')]):.3f} → {R_n9:.3f} · 전방 최대 변화 {dF:.3f}"
          f" · 내용 2AFC {before['content|2AFC']:.3f} → {after['content|2AFC']:.3f} · 글 KL {kl_text:.4f}", flush=True)
    print(f"  (비교 S579 LoRA 4B: 역순 .958 · 보류 .842 · N9 .433 · 전방 −.05)")
    verdict = dict(n1=bool(R_tr >= .80), n2=bool(R_held >= .70 and R_n9 >= .70), n3=bool(dF <= .03 and abs(after["content|2AFC"] - before["content|2AFC"]) <= .03 and kl_text <= .05))
    print(f"  가름 (사전 등록 S588b): " + " ".join(f"{k_} {'통과' if v_ else '기각'}" for k_, v_ in verdict.items()), flush=True)
    torch.save(dict(rel=rel.state_dict(), variant=variant, M=M, Pb=Pb, zmu=zmu, zsd=zsd, theta=theta, Lb=Lb, w=w, sigma=sigma, reader=reader_tag), CKPT.format(tag))
    json.dump(dict(before=before, after=after, nsrc=nsrc, kl_text=kl_text, coef=coef, theta=theta, fp=fp, kacc=kacc,
                   verdict=verdict, hist={k: [float(np.mean(x[i:i + 100])) for i in range(0, len(x), 100)] for k, x in hist.items()}),
              open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}  {CKPT.format(tag)}")
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--Lb", type=int, required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neval", type=int, default=40)
    ap.add_argument("--nproto", type=int, default=16)
    ap.add_argument("--variant", default="index", choices=["index", "circuit", "map", "retarget"])
    ap.add_argument("--pevery", type=int, default=250)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, a.tag, a.Lb, a.reader, steps=a.steps, lr=a.lr, neval=a.neval, nproto=a.nproto, variant=a.variant, pevery=a.pevery)
