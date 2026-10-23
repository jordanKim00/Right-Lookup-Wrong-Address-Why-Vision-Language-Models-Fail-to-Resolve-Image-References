"""S874 (exploratory, descriptive; knowledge/DECISIONS.md 10-21 14:0x plan item 'S874 = the original J-lens, AB vs AQ') - the vocabulary
Jacobian lens of Gurnee et al. 2026 (transformer-circuits.pub/2026/workspace; estimator of the public jacobian-lens code 581d398,
jlens/fitting.py, local copy research/ces/inputs/jacobian_lens_581d398) restricted to a candidate token set: is the address j* = N + 1 - k,
which the model computes without saying it when the answer is a picture's name (AB), readable as a number token, and is it when the question
asks for the number (AQ)?  In the paper the line-wrap character count was absent from the lens in the automatic task and surfaced only when
asked; the analogue predicts index tokens (j*, k, 'last') at the question / answer rows under AQ and not (or much less) under AB.
Part A (fit; text only: no images, no chat template, no BOS - Qwen has none): --fit (100) paragraphs of WikiText-2-raw train
  (research/ces/inputs/wikitext_b08601e, sha256 checked; the C28 J-lens corpus source), headings and paragraphs with task phrases excluded,
  WikiText's tokenised spacing undone (' @-@ ', ' , ', '( ' ...), the first 128 tokens of paragraphs with >= 128, in a --fit_seed order.
  Source points: the residual entering decoder layer l (= the output of block l - 1) for l in SRC (16 layers, 20..63), plus 'fin' = the final
  residual (input of the final RMSNorm, where J = I, i.e. the logit lens).  Candidates (44): digits '1'..'9' (' 1'..' 9' are two tokens in this
  tokenizer and are dropped by the single-token rule), number words ' one'..' nine', ordinals ' first'..' ninth', ' last' ' end' ' start'
  ' before' ' after' ' image' ' number', and ten control nouns.  Folded readout u_c = W_U[c] * (1 + w_norm) (Qwen3.5's RMSNorm scales by
  1 + weight; asserted from its source).  Row r_(l,c) = u_c^T J_l as in the official estimator: one forward with the input of layer SRC[0] as
  the graph leaf (parameters frozen), then per candidate one backward (retain_graph across candidates) with u_c as the cotangent at every
  valid target position of the final residual, the gradient at each source input averaged over the valid source positions (valid = 16..126:
  the first 16 and the last excluded); rows averaged over prompts (float64 sums; even / odd prompt halves kept for a split-half check).
  --cbatch B > 1 replicates the prompt B times and carries B candidates per backward (same estimator, fewer passes).  The lens logit of c for
  a state h is u_c . norm(J_l h) = r_(l,c) . h / rms(J_l h); the rms is common to all candidates at one (l, row), so ranks among candidates are
  exact, but nothing outside the candidate set is ranked (no full-vocabulary top-k).  Instrument check on --hold held-out paragraphs: at
  layers FD_L and for two contrasts ('7' - '3', ' last' - ' first') the prompt-local row and the averaged row predict the central finite
  difference of sum_t' (u_a - u_b) . h_fin,t' when +-.02 rms(h_l) z (z a seeded unit-rms Gaussian direction) is added at every valid
  position of the layer's input (the C28 admission design: corr, through-origin slope, relative RMSE).
Part B (apply; natural runs, no edits, no gradient): --n scenes (24) built as in S872 (nine sampled single-token labels, the first eight
  pictures shown, menu = all nine shuffled, 160 px).  Prompt: '{8 images}\\nThere are 8 images in a row. ' + question, THINK prefix.  Forms:
  AB_END(k) 'What is in the {ord} image from the last? Answer with only one word from: {menu}.' and AQ_END(k) 'Which image is the {ord} image
  from the last? Answer with only the image number (1-8).' for k 2..6 (target j* = 9 - k); AB_START(q), AQ_START(q) ('from the first') for
  q 3..5 (target q) as controls.  Equal-length forms with equal read rows run batched (--bs).  Read rows (15, asserted token by token): the
  question tail from the ordinal word to the frame word (5), '?', B0..B4 (<|im_end|> .. 'assistant\\n'), A0..A3 (the THINK tokens; A3 = the
  answer row).  Saved per run: J-lens scores r_(l,c) . h_(l,row) [44, 17 points, 15 rows] (the 17th point = fin with u_c), logit-lens scores
  u_c . h_(l,row) at the same points (reference), rms(h); the answer (AB: menu word by the two-form max readout, R3_READ=max; AQ: argmax over
  the digit tokens '1'..'9'), t1 (vocabulary top-1 is a menu word / a digit), the digit logits, the menu logits (AB), and the fin check (u_c .
  h_fin / rms vs the model's own logits at the answer row: bookkeeping of norm, rows and candidates).
Outputs: results/r3/s874_<tag>_rows.npz / .json (Part A; reusable with --rows), s874_<tag>_lens.npz and s874_<tag>_vocablens.json (Part B).
Analysis: exp/s874_analyze.py.  Absence of a number in this readout bounds what the average Jacobian transports to the output vocabulary at
those rows; it does not show that no address is computed.
Usage: python s874_vocablens.py MODEL TAG [--n 24] [--seed 87400] [--fit 100] [--hold 4] [--cbatch 1] [--rows PATH] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, hashlib, inspect, json, math, os, random, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image, proc_label

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
NUMW = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TASKW = ["last", "end", "start", "before", "after", "image", "number"]
CTRLW = ["apple", "river", "table", "green", "music", "paper", "horse", "window", "bread", "cloud"]
CAND = ([("digit", str(d)) for d in range(1, 10)] + [("digit_sp", f" {d}") for d in range(1, 10)] + [("numw", " " + w) for w in NUMW]
        + [("ordw", " " + w) for w in ORDW] + [("task", " " + w) for w in TASKW] + [("ctrl", " " + w) for w in CTRLW])
SRC = [20, 24, 27, 31, 35, 39, 43, 46, 47, 48, 49, 50, 51, 55, 59, 63]
PTS = [f"L{l}" for l in SRC] + ["fin"]
SEQ, SKIP = 128, 16
VAL = slice(SKIP, SEQ - 1); NV = SEQ - 1 - SKIP                                              # valid source / target positions 16..126
CORPUS = "/vlm/research/ces/inputs/wikitext_b08601e/wikitext-2-raw-v1/train-00000-of-00001.blob.parquet"
CORPUS_SHA = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
BANNED = ("image from the", "images in a row", "answer with only", "image number")
FD_L = [20, 47, 59]; FD_C = [("7", "3"), (" last", " first")]; FD_EPS = .02; FD_DIRS = 2
N = 8; KS = (2, 3, 4, 5, 6); QS = (3, 4, 5)
FORMS = [("AB", "END", k) for k in KS] + [("AQ", "END", k) for k in KS] + [("AB", "START", q) for q in QS] + [("AQ", "START", q) for q in QS]
ROWN = ["ord", "img", "from", "the", "frame", "qm", "B0", "B1", "B2", "B3", "B4", "A0", "A1", "A2", "A3"]
THINK = "<think>\n\n</think>\n\n"
OUT = "/vlm/results/r3/s874_{}_{}"


def detok(s):
    """WikiText's tokenised spacing back to running text: number separators, no space before closing punctuation / after opening brackets."""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?)\]]|'s\b|n't\b)", r"\1", s)
    return re.sub(r"([(\[]) ", r"\1", s)


def corpus(tok, n, seed):
    """n paragraphs of WikiText-2-raw train (no heading, no task phrase, >= SEQ tokens after detok), the first SEQ token ids each, seeded order."""
    import pyarrow.parquet as pq
    with open(CORPUS, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == CORPUS_SHA, "WikiText source changed"
    raw = pq.read_table(CORPUS, columns=["text"]).column("text").to_pylist()
    el = [(i, detok(t.strip())) for i, t in enumerate(raw) if t.strip() and not t.strip().startswith("=")]
    el = [(i, s) for i, s in el if not any(p in s.lower() for p in BANNED)]
    random.Random(seed).shuffle(el)
    out = []
    for i, s in el:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) >= SEQ:
            out.append(dict(row=i, ids=ids[:SEQ]))
            if len(out) == n:
                break
    assert len(out) == n, f"only {len(out)} paragraphs with >= {SEQ} tokens"
    return out, dict(raw_rows=len(raw), eligible_paragraphs=len(el))


def prompt_text(v, ans, frame, q, menu):
    fw = "last" if frame == "END" else "first"
    if ans == "AB":
        body = f"What is in the {ORDW[q - 1]} image from the {fw}? Answer with only one word from: {', '.join(menu)}."
    else:
        body = f"Which image is the {ORDW[q - 1]} image from the {fw}? Answer with only the image number (1-{N})."
    return v.chat([("user", f"{v.IMG * N}\nThere are {N} images in a row. {body}")], prefix_assistant=THINK)


def read_rows(ids, tok, q, frame):
    """the 15 read rows (ROWN) of one encoded prompt, asserted token by token."""
    T = tok.convert_tokens_to_ids
    one = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    ve = max(i for i, x in enumerate(ids) if x == T("<|vision_end|>"))
    qm = next(i for i in range(ve + 1, len(ids)) if "?" in tok.decode([ids[i]]))
    uend = max(i for i, x in enumerate(ids) if x == T("<|im_end|>"))
    o = [i for i in range(ve + 1, qm) if ids[i] == one(" " + ORDW[q - 1])[0]]
    assert len(o) == 1, "ordinal word not unique"
    tail = list(range(o[0], qm))
    fw = " last" if frame == "END" else " first"
    assert len(tail) == 5 and ids[tail[1]] == one(" image")[0] and ids[qm - 1] == one(fw)[0], "question tail"
    assert ids[uend + 2] == T("<|im_start|>") and tok.decode([ids[uend + 5]]) == "<think>" and len(ids) - uend - 5 == 4, "B / A rows"
    rows = tail + [qm] + list(range(uend, len(ids)))
    assert len(rows) == len(ROWN)
    return rows


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def fdstats(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    ok = len(pred) > 2 and pred.std() > 0 and obs.std() > 0
    return dict(n=int(len(pred)), corr=float(np.corrcoef(pred, obs)[0, 1]) if ok else None,
                slope=float((pred * obs).sum() / max((pred * pred).sum(), 1e-30)),                 # obs ~ slope * pred through the origin
                rel_rmse=float(np.sqrt(((obs - pred) ** 2).mean() / max((obs ** 2).mean(), 1e-30))))


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S874_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35" and v.n_layer == 64
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    tok = v.tok
    lm = v.model.language_model if hasattr(v.model, "language_model") else v.model.model.language_model
    norm, head = lm.norm, v.model.lm_head
    assert lm.layers is v.layers
    assert "1.0 + self.weight" in inspect.getsource(type(norm).forward), "final norm is not the (1 + weight) RMSNorm"
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    D = int(head.weight.shape[1])
    one = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    cands = [(g, s) for g, s in CAND if len(one(s)) == 1]
    dropped = [s for g, s in CAND if len(one(s)) != 1]
    names = [s for g, s in cands]; cid = [one(s)[0] for g, s in cands]; C = len(cands)
    assert len(set(cid)) == C and all(str(d) in names for d in range(1, 10))
    ci = names.index
    with torch.no_grad():
        U = (head.weight[torch.tensor(cid, device=head.weight.device)].float() * (1.0 + norm.weight.float().to(head.weight.device))).cpu()
    GR = dict(mode=None, t={}, fin=None, st={}, rows=None, fd=None)
    hooks = []

    def pre_src(l):
        def h(mod, args, kwargs):
            m = GR["mode"]
            if m is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            if m == "cap":
                GR["st"][l] = hs[:, GR["rows"]].detach().float().cpu()
                return None
            if m == "fd":
                if GR["fd"][0] != l:
                    return None
                x = hs.clone()
                x[:, VAL] += GR["fd"][1].to(x.device, x.dtype)
            else:                                                                             # "fit"
                if l != SRC[0]:
                    GR["t"][l] = hs
                    return None
                x = hs.detach().clone().requires_grad_(True)                                  # the graph leaf
                GR["t"][l] = x
            if args:
                return (x,) + tuple(args[1:]), kwargs
            kw = dict(kwargs); kw["hidden_states"] = x
            return args, kw
        return h

    def pre_norm(mod, args):
        if GR["mode"] == "cap":
            GR["st"]["fin"] = args[0][:, GR["rows"]].detach().float().cpu()
        elif GR["mode"] in ("fit", "fd"):
            GR["fin"] = args[0]
        return None

    def jrows(tokens, dirs, cb):
        """[len(dirs), len(SRC), D] float64 (cpu): per cotangent direction (rows of dirs, folded readouts) the mean over valid source positions
        of the gradient at each source input of sum_(valid t') dir . h_fin,t'; and rms(h_l) over the valid rows of each source input."""
        ids = torch.tensor([tokens], device=v.device).expand(cb, -1)
        out = torch.zeros(len(dirs), len(SRC), D, dtype=torch.float64)
        GR.update(mode="fit", t={}, fin=None)
        try:
            with torch.enable_grad():
                v.model(input_ids=ids, use_cache=False, return_dict=True, **v.forward_extra)
                fin = GR["fin"]; ins = [GR["t"][l] for l in SRC]
                assert fin is not None and fin.shape[1] == SEQ and fin.requires_grad and all(x.requires_grad for x in ins)
                rms = [float(x[0, VAL].detach().float().pow(2).mean().sqrt()) for x in ins]
                dd = dirs.to(fin.device, fin.dtype)
                nb = math.ceil(len(dirs) / cb)
                for bi in range(nb):
                    c0, c1 = bi * cb, min(len(dirs), (bi + 1) * cb)
                    cot = torch.zeros_like(fin)
                    cot[: c1 - c0, VAL] = dd[c0:c1, None, :]
                    gs = torch.autograd.grad(fin, ins, grad_outputs=cot, retain_graph=bi < nb - 1)
                    for li, g in enumerate(gs):                                               # each on its own layer's card
                        out[c0:c1, li] = g[: c1 - c0, VAL].float().mean(1).double().cpu()
                    del gs, cot
                del fin, ins
        finally:
            GR.update(mode=None, t={}, fin=None)
        return out, rms

    @torch.no_grad()
    def fd_score(tokens, l, delta, dirs):
        """sum over valid t' of dirs . h_fin,t' (float64, cpu) with delta added at the valid positions of layer l's input."""
        GR.update(mode="fd", fd=(l, delta), fin=None)
        try:
            v.model(input_ids=torch.tensor([tokens], device=v.device), use_cache=False, return_dict=True, **v.forward_extra)
            f = GR["fin"][0, VAL].float()
            return (f @ dirs.to(f.device).T).sum(0).double().cpu()
        finally:
            GR.update(mode=None, fd=None, fin=None)

    @torch.no_grad()
    def capture(inps, rw):
        GR.update(mode="cap", rows=rw, st={})
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float().cpu()
            st = dict(GR["st"])
        finally:
            GR.update(mode=None, rows=None, st={})
        return lg, st

    for l in SRC:
        hooks.append(v.layers[l].register_forward_pre_hook(pre_src(l), with_kwargs=True))
    hooks.append(norm.register_forward_pre_hook(pre_norm))
    t_start = time.time()
    print(f"[{a.tag}] candidates {C} (dropped as multi-token: {dropped}) · source layers {SRC} + fin · D {D} · eps {eps}", flush=True)
    try:
        # ---------------- Part A: fit (or reuse) the candidate rows
        if a.rows:
            Z = np.load(a.rows)
            assert Z["cand_ids"].tolist() == cid and Z["src"].tolist() == SRC, "rows file does not match the candidates / layers"
            R = torch.from_numpy(Z["rows"]).float()
            rows_json = a.rows[:-4] + ".json"
            print(f"[{a.tag}] rows reused from {a.rows}", flush=True)
        else:
            assert a.fit >= 2
            prompts, cstat = corpus(tok, a.fit + a.hold, a.fit_seed)
            fitp, holdp = prompts[:a.fit], prompts[a.fit:]
            SUM = torch.zeros(C, len(SRC), D, dtype=torch.float64); HALF = [torch.zeros_like(SUM), torch.zeros_like(SUM)]
            rel = []; t0 = time.time()
            for pi, pr in enumerate(fitp):
                rp, _ = jrows(pr["ids"], U, a.cbatch)
                if not torch.isfinite(rp).all():
                    raise RuntimeError(f"non-finite rows at prompt {pi}")
                if pi:
                    m_ = SUM / pi
                    r_ = ((rp - m_).norm(dim=-1) / ((pi + 1) * m_.norm(dim=-1).clamp_min(1e-30)))
                    rel.append(dict(max=float(r_.max()), median=float(r_.median())))
                SUM += rp; HALF[pi % 2] += rp
                if (pi + 1) % 10 == 0 or pi + 1 == len(fitp):
                    el = time.time() - t0
                    print(f"[{a.tag}] fit {pi + 1}/{len(fitp)} · {el:.0f} s ({el / (pi + 1):.1f} s / prompt) · running-mean shift "
                          f"max {rel[-1]['max'] if rel else float('nan'):.2e}", flush=True)
            R = (SUM / len(fitp)).float()
            H0 = HALF[0] / ((len(fitp) + 1) // 2); H1 = HALF[1] / (len(fitp) // 2)
            cos = F.cosine_similarity(H0, H1, dim=-1)                                           # [C, len(SRC)]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # instrument check on held-out paragraphs: local and averaged rows vs central finite differences
            gen = torch.Generator().manual_seed(a.fit_seed)
            Dc = torch.stack([U[ci(x)] - U[ci(y)] for x, y in FD_C])                            # [2, D]
            Rc = torch.stack([R[ci(x)] - R[ci(y)] for x, y in FD_C]).double()                   # [2, L, D]
            fd = []
            for hp in holdp:
                loc, rms = jrows(hp["ids"], Dc, 1)
                for l in FD_L:
                    li = SRC.index(l)
                    for di in range(FD_DIRS):
                        z = torch.randn(D, generator=gen); z = z / z.pow(2).mean().sqrt()
                        dl = rms[li] * z
                        obs = (fd_score(hp["ids"], l, FD_EPS * dl, Dc) - fd_score(hp["ids"], l, -FD_EPS * dl, Dc)) / (2 * FD_EPS)
                        fd.append(dict(row=hp["row"], layer=l, dir=di, rms=rms[li], obs=obs.tolist(),
                                       pred_local=(NV * loc[:, li] @ dl.double()).tolist(), pred_avg=(NV * Rc[:, li] @ dl.double()).tolist(),
                                       cos_local_avg=F.cosine_similarity(loc[:, li], Rc[:, li], dim=-1).tolist()))
            fds = {}
            f3 = lambda x: "-" if x is None else f"{x:.3f}"
            for l in FD_L if fd else []:
                for j, (x, y) in enumerate(FD_C):
                    q_ = [r for r in fd if r["layer"] == l]
                    obs = [r["obs"][j] for r in q_]
                    fds[f"L{l} {x.strip()}-{y.strip()}"] = s_ = dict(local=fdstats([r["pred_local"][j] for r in q_], obs),
                                                                      avg=fdstats([r["pred_avg"][j] for r in q_], obs),
                                                                      cos_local_avg=float(np.mean([r["cos_local_avg"][j] for r in q_])))
                    print(f"[{a.tag}] FD L{l} {x.strip()}-{y.strip()} (n {s_['avg']['n']}): local corr {f3(s_['local']['corr'])} slope "
                          f"{f3(s_['local']['slope'])} · avg corr {f3(s_['avg']['corr'])} slope {f3(s_['avg']['slope'])} rel RMSE "
                          f"{f3(s_['avg']['rel_rmse'])} · cos(local, avg) {f3(s_['cos_local_avg'])}", flush=True)
            print(f"[{a.tag}] split-half cos median per layer: " + " ".join(f"L{l} {float(cos[:, li].median()):.3f}" for li, l in enumerate(SRC)),
                  flush=True)
            np.savez(OUT.format(a.tag, "rows.npz"), rows=R.numpy(), fin=U.numpy(), half0=H0.float().numpy(), half1=H1.float().numpy(),
                     cand_ids=np.array(cid), src=np.array(SRC))
            rows_json = OUT.format(a.tag, "rows.json")
            json.dump(dict(tag=a.tag, model=a.model, estimator="official jlens/fitting.py reduction restricted to candidate cotangents: cotangent "
                           "u_c * (1 + w_norm) at every valid target position of the final residual, gradient at the input of each source layer "
                           "averaged over valid source positions, mean over prompts", seq=SEQ, skip_first=SKIP, valid=[SKIP, SEQ - 2],
                           src=SRC, cands=[dict(group=g, str=s, id=i) for (g, s), i in zip(cands, cid)], dropped=dropped,
                           corpus=dict(path=CORPUS, sha256=CORPUS_SHA, detok=True, banned=list(BANNED), fit_seed=a.fit_seed, **cstat),
                           fit_rows=[p["row"] for p in fitp], hold_rows=[p["row"] for p in holdp], n_fit=len(fitp), cbatch=a.cbatch,
                           running_mean_shift=rel, split_half_cos=cos.tolist(), fd_eps=FD_EPS, fd_layers=FD_L, fd_contrasts=FD_C, fd_dirs=FD_DIRS,
                           fd=fd, fd_summary=fds, fit_seconds=time.time() - t0),
                      open(rows_json, "w"))
            print(f"[{a.tag}] Part A done in {time.time() - t0:.0f} s -> {OUT.format(a.tag, 'rows.npz')}", flush=True)

        # ---------------- Part B: apply to AB / AQ runs
        DIG = [one(str(d))[0] for d in range(1, 10)]
        pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
        pool = {}
        for pid in pids[:4000]:
            lab = proc_label(pid)[0]                                                            # = proc_image(pid)[1], without rendering
            if len(one(" " + lab)) == 1:
                pool.setdefault(lab, []).append(pid)
        labels = sorted(l for l, x in pool.items() if len(x) >= 20)
        assert len(labels) >= 9
        tv = lambda s: (one(s)[0], one(" " + s)[0])
        sc = lambda lg, s: torch.maximum(lg[tv(s)[0]], lg[tv(s)[1]])
        t1ids = lambda ws: [tv(w)[0] for w in ws] + [tv(w)[1] for w in ws]
        P, RN = len(PTS), len(ROWN)
        JL = np.zeros((a.n, len(FORMS), C, P, RN), np.float32); LL = np.zeros_like(JL); RMS = np.zeros((a.n, len(FORMS), P, RN), np.float32)
        rng = random.Random(a.seed); runs = []; scenes = []; t0 = time.time()
        for si in range(a.n):
            labs = rng.sample(labels, 9)
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = list(labs); rng.shuffle(menu)
            enc, rws, groups = {}, {}, {}
            for fi, (ans, frame, q) in enumerate(FORMS):
                enc[fi] = v.encode(prompt_text(v, ans, frame, q, menu), pics[:N])
                rws[fi] = read_rows(enc[fi]["input_ids"][0].tolist(), tok, q, frame)
                groups.setdefault((int(enc[fi]["input_ids"].shape[1]), tuple(rws[fi])), []).append(fi)
            scenes.append(dict(si=si, labs=labs, menu=menu, groups=[dict(S=S_, rows=list(rw), forms=fis) for (S_, rw), fis in groups.items()]))
            for (S_, rw), fis in groups.items():
                for c0 in range(0, len(fis), a.bs):
                    chunk = fis[c0:c0 + a.bs]
                    lg, st = capture([enc[f] for f in chunk], list(rw))
                    for b, fi in enumerate(chunk):
                        ans, frame, q = FORMS[fi]
                        Hs = torch.stack([st[l][b] for l in SRC] + [st["fin"][b]])                # [P, RN, D] float32
                        JL[si, fi, :, :-1] = torch.einsum("cld,lrd->clr", R, Hs[:-1]).numpy()
                        JL[si, fi, :, -1] = (U @ Hs[-1].T).numpy()
                        LL[si, fi] = torch.einsum("cd,prd->cpr", U, Hs).numpy()
                        RMS[si, fi] = Hs.pow(2).mean(-1).sqrt().numpy()
                        lgb = lg[b]
                        hf = Hs[-1, -1]
                        fin_err = float((U @ hf / torch.sqrt(hf.pow(2).mean() + eps) - lgb[cid]).abs().max())
                        target = N + 1 - q if frame == "END" else q
                        top = int(lgb.argmax())
                        rec = dict(si=si, fi=fi, form=f"{ans}_{frame}{q}", ans=ans, frame=frame, q=q, target=target, top=tok.decode([top]),
                                   dig=[float(lgb[t]) for t in DIG], fin_err=fin_err)
                        if ans == "AB":
                            w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                            rec.update(answer=w, addr=labs.index(w) + 1, t1=int(top in t1ids(menu)), mlog=[float(sc(lgb, x)) for x in labs])
                        else:
                            d = int(lgb[DIG].argmax()) + 1
                            rec.update(answer=str(d), addr=d, t1=int(top in DIG))
                        rec["correct"] = int(rec["addr"] == target)
                        runs.append(rec)
                    del st
            rs = [r for r in runs if r["si"] == si]
            print(f"[{a.tag}] scene {si + 1}/{a.n} · AB_END {sum(r['correct'] for r in rs if r['form'].startswith('AB_END'))}/{len(KS)} · "
                  f"AQ_END {sum(r['correct'] for r in rs if r['form'].startswith('AQ_END'))}/{len(KS)} · START "
                  f"{sum(r['correct'] for r in rs if 'START' in r['form'])}/{2 * len(QS)} · t1 {sum(r['t1'] for r in rs)}/{len(rs)} · "
                  f"fin err {max(r['fin_err'] for r in rs):.3f} · {time.time() - t0:.0f} s", flush=True)
            gc.collect()
    finally:
        for h_ in hooks:
            h_.remove()
        GR.update(mode=None, t={}, fin=None, st={})
    np.savez(OUT.format(a.tag, "lens.npz"), jl=JL, ll=LL, rms=RMS)
    t1 = lambda ab: [r["t1"] for r in runs if r["ans"] == ab]
    json.dump(dict(tag=a.tag, model=a.model, N=N, ks=list(KS), qs=list(QS), forms=[list(f) for f in FORMS], rown=ROWN, pts=PTS, src=SRC,
                   cands=[dict(group=g, str=s, id=i) for (g, s), i in zip(cands, cid)], dropped=dropped, rows_json=rows_json,
                   lens_npz=OUT.format(a.tag, "lens.npz"), seed=a.seed, pixels=a.pixels, readout=os.environ.get("R3_READ"),
                   t1_ab=float(np.mean(t1("AB"))), t1_aq=float(np.mean(t1("AQ"))), fin_err_max=max(r["fin_err"] for r in runs),
                   n_runs=len(runs), seconds=time.time() - t_start, scenes=scenes, runs=runs),
              open(OUT.format(a.tag, "vocablens.json"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'vocablens.json')} ({time.time() - t_start:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=87400)
    ap.add_argument("--fit", type=int, default=100); ap.add_argument("--hold", type=int, default=4); ap.add_argument("--fit_seed", type=int, default=874)
    ap.add_argument("--cbatch", type=int, default=1); ap.add_argument("--rows", default=None)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
