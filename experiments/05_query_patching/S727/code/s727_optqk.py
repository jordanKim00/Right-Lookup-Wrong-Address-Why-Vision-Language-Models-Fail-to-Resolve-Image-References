"""S727: where do 27B's index-matching queries aim when the end-relative reference sits in an option line (the MMIU form)?
§750 found that 81% of 27B's MMIU errors pick the lure (the option whose literal ordinal names the target's forward position); this run
asks whether the matcher heads' queries at the option lines align with the literal block, at the circuit level.
Scenes: procedural images, N = 7 (seed 72700), one label per image; the question "Which image shows a {label}?" with four options
"X: The {k}-th image from the {end|start}", the correct option, its lure (k_lure = N + 1 - k_correct, end-relative form only) and two
other ordinals from {1, 2, 3, 5, 6, 7}; letters shuffled; answer = the option letter (restricted argmax over the shown letters).
Conditions per scene: OPT_END (options from the end), OPT_START (the same ordinals from the start; the label of the correct START option),
OPT_END_R3 (OPT_END with the model's final R³ of S724 -- exp/r3_select.build(base r327b, rc_27b)), Q_END k in {2, 3} (the canonical
question, S705a / S719 form).
Capture (exp/s719_qkanatomy.py's hook): at the S705a capture layers, the fixed matcher set of S705a's discovery scenes (73 heads,
results/r3/s705a_qa_27b_matchers.json), rows = each option line's frame-word token (" end" / " start") and the answer row; question form:
the answer row.  Per (row, head): block scores = log-sum-exp of the head's logits over each image block's tokens -> share over blocks.
Pre-registration in knowledge/DECISIONS.md before any S727 data; analysis exp/s727_analyze.py (frozen with it).
Amendment (before the 27B data, after the 9B run): --extra_rows also captures each option line's ordinal token ("ord") and the line end ("nl");
reported only -- the registered rules stay on the frame-word rows ("opt").
Usage: python s727_optqk.py MODEL TAG [--n 30] [--device_map balanced] [--heads F --base B --rc RCTAG]
"""
from __future__ import annotations

import argparse, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    M = sorted(tuple(x) for x in json.load(open(a.heads))["matchers"])
    HL = {}
    for l, h in M:
        HL.setdefault(l, []).append(h)
    import r3_select
    rep = r3_select.build(v, a.base, a.rc)
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
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
            rows = ST["rows"]; heads = torch.tensor(HL[li], device=hs.device)
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
            k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[:, None] + rotate_half(k[..., :rd]) * sin[:, None], k[..., rd:]], -1)
            kh = heads // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe[:, heads], ke[:, kh]) * mod.scaling           # [rows, matcher heads of this layer, S]
            ST["cap"][li] = torch.stack([torch.logsumexp(lg[:, :, b0:b1 + 1], -1) for b0, b1 in ST["blocks"]], -1).cpu()   # [rows, heads, N]
            return out
        return h
    hooks = [v.layers[li].self_attn.register_forward_hook(hook(li), with_kwargs=True) for li in sorted(HL)]

    def opt_prompt(lab, opts, frame):
        lines = "\n".join(f"{L}: The {ORDW[k - 1]} image from the {frame}" for L, k in opts)
        return v.chat([("user", f"{v.IMG*N}\nWhich image shows a {lab}?\nSelect from the following choices.\n{lines}\n"
                                f"Answer with the option's letter from the given choices directly.")], prefix_assistant=THINK)

    def q_prompt(k, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the last? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(text, pics, frame_word=None, repair=False, cand=None):
        if repair:
            inp, writes = rep.plan(text, pics)
        else:
            inp, writes = v.encode(text, pics), []
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        blocks = list(zip(vs, ve))
        rows, lab = [], []
        if frame_word is not None:
            dec = [v.tok.decode([ids[i]]) for i in range(len(ids))]
            for i in range(ve[-1] + 1, len(ids)):
                if dec[i].strip() == frame_word:
                    rows.append(i); lab.append("opt")
                    if a.extra_rows:                                                  # amendment (before 27B data): reported rows
                        o = max(j for j in range(ve[-1] + 1, i) if dec[j].strip().lower() in ORDW)
                        rows.append(o); lab.append("ord")
                        if i + 1 < len(ids) and "\n" in dec[i + 1]:
                            rows.append(i + 1); lab.append("nl")
        rows.append(len(ids) - 1); lab.append("ans")
        ST.clear(); ST.update(on=True, rows=rows, blocks=blocks, cap={})
        rep.ST = dict(writes=writes) if writes else {}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        rep.ST = {}
        cap = ST["cap"]; ST.clear()
        sc = torch.cat([cap[l] for l in sorted(HL)], 1)                                   # [rows, 73, N] in the order of sorted layers
        pick = int(torch.stack([lg[c] for c in cand]).argmax()) if cand is not None else None
        return dict(scores=sc.numpy().round(3).tolist(), rowlab=lab, pick=pick, nwrites=len(writes))

    rng = random.Random(72700)
    out = []
    for si in range(a.n):
        labs = rng.sample(labels, N)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        kc = rng.choice([1, 2, 3, 5, 6, 7]); kl = N + 1 - kc
        others = rng.sample([k for k in (1, 2, 3, 5, 6, 7) if k not in (kc, kl)], 2)
        ks = [kc, kl] + others; rng.shuffle(ks)
        letters = ["A", "B", "C", "D"]; opts = list(zip(letters, ks))
        cand = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in letters]
        rec = dict(si=si, labs=labs, ks=ks, kc=kc, kl=kl)
        lab_end = labs[(N + 1 - kc) - 1]                                                  # END: the correct option names image N+1-kc
        lab_start = labs[kc - 1]                                                          # START: the same option names image kc
        rec["OPT_END"] = run(opt_prompt(lab_end, opts, "end"), pics, "end", cand=cand)
        rec["OPT_START"] = run(opt_prompt(lab_start, opts, "start"), pics, "start", cand=cand)
        rec["OPT_END_R3"] = run(opt_prompt(lab_end, opts, "end"), pics, "end", repair=True, cand=cand)
        menu = labs + [l for l in labels if l not in labs][:2]; rng.shuffle(menu)
        mc = [ids1(l)[0] for l in menu]
        for k in (2, 3):
            r = run(q_prompt(k, menu), pics, cand=mc)
            r["land"] = labs.index(menu[r["pick"]]) + 1 if menu[r["pick"]] in labs else 0
            rec[f"Q_END{k}"] = r
        rec["gold_letter"] = letters[ks.index(kc)]; rec["lure_letter"] = letters[ks.index(kl)]
        out.append(rec)
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, N=N, heads=[list(x) for x in M], layers=sorted(HL), base=a.base, rc=a.rc,
                   choice=r3_select.choice(a.rc), rec=out),
              open(f"/vlm/results/r3/s727_{a.tag}_optqk.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s727_{a.tag}_optqk.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--heads", default="/vlm/results/r3/s705a_qa_27b_matchers.json")
    ap.add_argument("--base", default="r327b"); ap.add_argument("--rc", default="rc_27b")
    ap.add_argument("--extra_rows", action="store_true")                              # also each option's ordinal token and line end
    main(ap.parse_args())
