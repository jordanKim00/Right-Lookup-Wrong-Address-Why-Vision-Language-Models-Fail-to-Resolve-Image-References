"""S737: the option-line query on natural images (pre-registration knowledge/DECISIONS.md 2026-10-10, before any S737 data).
Items: the MMIU development split (exp/s582_evalrepair.run's selection, first 50 items per position task, N <= 9), prompts exactly as the
evaluator builds them.  Forms: START (the original options "X: The k-th image"), END (the evaluator's end-relative rewrite
"X: The (N+1-k)-th image from the end"), END + the model's R³ (exp/r3_select.build, plan file in force).
Rows: the ordinal token of every option line (the last n_options ordinal words of the prompt; the option block ends the prompt).
Capture (all five models, the S730 matcher sets): block shares (softmax over image blocks of the log-sum-exp of each head's query-key
logits over the block's tokens) at those rows.  For option L naming forward image k_L: START share on block k_L; END and END + R³ shares on
the literal block N+1-k_L and on the named block k_L.
Causal (hybrid Qwen3.5 models, exp/c_tools.FullAttn): on END items answered wrongly, the matcher heads' queries (Q_ONLY) or output gates
(G_ONLY) at the option ordinal rows are replaced by those of the START form at the same option's ordinal row.
Usage: python s737_natqk.py MODEL TAG --base B --rc RCTAG --heads JSON [--reader_override R] [--device_map balanced] [--causal]
"""
from __future__ import annotations

import argparse, json, re, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from c_tools import rotate_half
import r3_select
import s582_evalrepair as E

ORDW = E.ORDW


def dev_items():
    out = []
    for task in E.POS_TASKS:
        keep = []
        for it in json.load(open(f"{E.ROOT}/{task}.json")):
            ls = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
            if ls and all(E.PAT.match(l) for l in ls):
                keep.append((it, [(E.PAT.match(l).group(1), ORDW.index(E.PAT.match(l).group(2)) + 1) for l in ls]))
        for it, opts in keep[:50]:
            if len(it["input"]["input_image_path"]) <= 9:
                out.append((task, it, opts))
    return out


def ctx_of(it):
    c = str(it["input"].get("context", "")).split("\n")[0]
    return re.split(r"\bA\s*:", c)[0].replace("Select from the following choices.", "").strip()[:400]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rep = r3_select.build(v, a.base, a.rc, reader=a.reader_override)
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    HL = {}
    for l, h in json.load(open(a.heads))["matchers"]:
        HL.setdefault(l, []).append(h)
    HL = {l: sorted(h) for l, h in HL.items()}
    lays = sorted(HL)
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    ST = {}

    def hook(li):
        def h(mod, args, kwargs, out):
            if not ST.get("on") or li not in HL:
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos.float(), sin.float()
            S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rows = ST["rows"]
            qp = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
            gated = qp.shape[-1] == 2 * nh * hd
            q = qp.view(len(rows), nh, 2 * hd)[..., :hd] if gated else qp.view(len(rows), nh, hd)
            qn, kn = getattr(mod, "q_norm", None), getattr(mod, "k_norm", None)
            q = qn(q.to(qn.weight.dtype)).float() if qn is not None else q
            k = torch.nn.functional.linear(hs, mod.k_proj.weight.float(), None if mod.k_proj.bias is None else mod.k_proj.bias.float()).view(S, -1, hd)
            k = kn(k.to(kn.weight.dtype)).float() if kn is not None else k
            M_ = sys.modules[mod.__class__.__module__]
            if cos.dim() == 4 and hasattr(M_, "apply_multimodal_rotary_pos_emb"):
                sec = (getattr(mod.config, "rope_parameters", None) or mod.config.rope_scaling)["mrope_section"]
                qe = M_.apply_multimodal_rotary_pos_emb(q.transpose(0, 1)[None], q.transpose(0, 1)[None], cos[:, :, rows], sin[:, :, rows], sec)[0][0].transpose(0, 1)
                ke = M_.apply_multimodal_rotary_pos_emb(k.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin, sec)[1][0].transpose(0, 1)
            else:
                c0, s0 = cos[0], sin[0]; rd = c0.shape[-1]
                qe = torch.cat([q[..., :rd] * c0[rows][:, None] + rotate_half(q[..., :rd]) * s0[rows][:, None], q[..., rd:]], -1)
                ke = torch.cat([k[..., :rd] * c0[:, None] + rotate_half(k[..., :rd]) * s0[:, None], k[..., rd:]], -1)
            heads = torch.tensor(HL[li], device=hs.device); kh = heads // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling
            ST["cap"][li] = torch.stack([torch.logsumexp(lg[:, :, b0:b1 + 1], -1) for b0, b1 in ST["blocks"]], -1).cpu()
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in FA_L]
    FA = None
    if a.causal:
        from c_tools import Ctx, FullAttn
        cx = Ctx(); FA = FullAttn(v, lays, cx)

    def load(task, p):
        im = Image.open(f"{E.ROOT}/{task}/{p.split('/')[-1]}").convert("RGB")
        s = 448 / max(im.size)
        return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s < 1 else im

    def prompt(it, lines, N):
        return v.chat([("user", f"{v.IMG*N}\n{it['input']['question']}\n{ctx_of(it)}\nSelect from the following choices.\n" + "\n".join(lines) +
                                "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)

    def ord_rows(ids, n):
        ve = max(i for i, t in enumerate(ids) if t == ve_id)
        pos = [i for i in range(ve + 1, len(ids)) if v.tok.decode([ids[i]]).strip().lower() in ORDW]
        return pos[-n:]

    @torch.no_grad()
    def run(text, pics, n_opt, cand, writes=None, capture=True, edit=None, grab=False):
        inp = v.encode(text, pics)
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        rows = ord_rows(ids, n_opt)
        ST.clear(); ST.update(on=capture, rows=rows, blocks=list(zip(vs, ve)), cap={})
        if FA is not None:
            cx.reset()
            if grab or edit:
                cx.rows = rows; cx.grab = grab; cx.edit = edit or {}
        rep.ST = dict(writes=writes) if writes else {}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        rep.ST = {}
        cap = ST["cap"]; ST.clear()
        pick = int(torch.stack([lg[c] for c in cand]).argmax())
        sh = torch.softmax(torch.cat([cap[l] for l in lays], 1), -1).numpy() if capture and cap else None    # [rows, heads, N]
        store = dict(cx.store) if (FA is not None and grab) else None
        if FA is not None:
            cx.reset()
        return pick, (sh.mean(1) if sh is not None else None), store

    out = []
    items = dev_items()[:a.cap] if a.cap else dev_items()
    for ii, (task, it, opts) in enumerate(items):
        paths = it["input"]["input_image_path"]; N = len(paths)
        pics = [load(task, p) for p in paths]
        letters = [L for L, _ in opts]; cand = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in letters]
        gold = str(it["output"]["output_text"]).strip()[:1]
        lS = [f"{L}: The {ORDW[k-1]} image" for L, k in opts]
        lE = [f"{L}: {E.rev_phrase(N + 1 - k, cap=True)}" for L, k in opts]
        tS, tE = prompt(it, lS, N), prompt(it, lE, N)
        rec = dict(task=task, N=N, gold=gold, opts=opts)
        pS, shS, stS = run(tS, pics, len(opts), cand, grab=FA is not None)
        pE, shE, _ = run(tE, pics, len(opts), cand)
        inp, writes = rep.plan(tE, pics)
        pR, shR, _ = run(tE, pics, len(opts), cand, writes=writes)
        rec.update(pick_START=letters[pS], pick_END=letters[pE], pick_R3=letters[pR], n_writes=len(writes))
        rec["START"] = [dict(named=float(shS[i, k - 1])) for i, (L, k) in enumerate(opts)] if shS is not None else None
        for nm, sh in (("END", shE), ("R3", shR)):
            rec[nm] = [dict(literal=float(sh[i, N - k]), named=float(sh[i, k - 1])) for i, (L, k) in enumerate(opts)] if sh is not None else None
        if FA is not None and letters[pE] != gold and stS:
            for arm, part in (("Q_ONLY", "q"), ("G_ONLY", "gate")):
                ed = {("F", li): {"heads": HL[li], part: stS[("F", li, part)]} for li in lays}
                pX, _, _ = run(tE, pics, len(opts), cand, capture=False, edit=ed)
                rec[f"pick_{arm}"] = letters[pX]
        out.append(rec)
        torch.cuda.empty_cache()
        if (ii + 1) % 50 == 0:
            print(f"[{a.tag}] {ii + 1}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    if FA is not None:
        FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, base=a.base, rc=a.rc, choice=r3_select.choice(a.rc), heads=[[l, h] for l in lays for h in HL[l]], rows=out),
              open(f"/vlm/results/r3/s737_{a.tag}_natqk.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s737_{a.tag}_natqk.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--heads", required=True); ap.add_argument("--reader_override", default=None)
    ap.add_argument("--device_map", default=None); ap.add_argument("--causal", action="store_true"); ap.add_argument("--cap", type=int, default=None)
    main(ap.parse_args())
