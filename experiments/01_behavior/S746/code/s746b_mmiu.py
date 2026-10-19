"""S746b: does reasoning repair the benchmark failure?  Qwen3-VL Thinking checkpoints on the later MMIU split (the evaluator's "fresh"
selection, first --per_task items of each task), start- and end-relative option forms, reasoning skipped (OFF: restricted letter argmax)
and reasoning on (ON: greedy generation up to --max_new tokens, the first option letter after "</think>").  Pre-registration in
knowledge/DECISIONS.md before any S746b data; analysis exp/s746b_analyze.py.
Usage: python s746b_mmiu.py MODEL TAG [--per_task 10] [--batch 8] [--max_new 2048]
"""
from __future__ import annotations

import argparse, json, re, sys
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni
import s582_evalrepair as E


def items(per_task):
    out = []
    for task in E.POS_TASKS:
        keep = []
        for it in json.load(open(f"{E.ROOT}/{task}.json")):
            ls = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
            if ls and all(E.PAT.match(l) for l in ls):
                keep.append((it, [(E.PAT.match(l).group(1), E.ORDW.index(E.PAT.match(l).group(2)) + 1) for l in ls]))
        used = {p for it, _ in keep[:150] for p in it["input"]["input_image_path"]}
        sel = [(it, o) for it, o in keep[150:250] if not used & set(it["input"]["input_image_path"])]
        sel = [(it, o) for it, o in sel if len(it["input"]["input_image_path"]) <= 9][:per_task]
        out += [(task, it, o) for it, o in sel]
    return out


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    OFF, ON = "<think>\n\n</think>\n\n", "<think>\n"
    tok = v.processor.tokenizer; tok.padding_side = "left"

    def load(task, p):
        im = Image.open(f"{E.ROOT}/{task}/{p.split('/')[-1]}").convert("RGB")
        s = 448 / max(im.size)
        return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s < 1 else im

    def ctx_of(it):
        c = str(it["input"].get("context", "")).split("\n")[0]
        return re.split(r"\bA\s*:", c)[0].replace("Select from the following choices.", "").strip()[:400]

    def mc(it, lines, N, prefix):
        return v.chat([("user", f"{v.IMG*N}\n{it['input']['question']}\n{ctx_of(it)}\nSelect from the following choices.\n" + "\n".join(lines) +
                                "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=prefix)

    rows, queue = [], []
    for task, it, opts in items(a.per_task):
        paths = it["input"]["input_image_path"]; N = len(paths)
        pics = [load(task, p) for p in paths]
        letters = [L for L, _ in opts]; cand = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in letters]
        gold = str(it["output"]["output_text"]).strip()[:1]
        for form in ("F", "R"):
            lines = [f"{L}: The {E.ORDW[k-1]} image" if form == "F" else f"{L}: {E.rev_phrase(N + 1 - k, cap=True)}" for L, k in opts]
            with torch.no_grad():
                inp = v.encode(mc(it, lines, N, OFF), pics)
                lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
            r = dict(task=task, gold=gold, form=form, N=N, opts=opts, off=letters[int(torch.stack([lg[c] for c in cand]).argmax())])
            rows.append(r); queue.append((r, mc(it, lines, N, ON), pics, letters))
        torch.cuda.empty_cache()
    print(f"[{a.tag}] OFF done, {len(rows)} prompts", flush=True)
    for b0 in range(0, len(queue), a.batch):
        chunk = queue[b0:b0 + a.batch]
        with torch.no_grad():
            inp = v.processor(text=[c[1] for c in chunk], images=[im for c in chunk for im in c[2]], return_tensors="pt", padding=True).to(v.model.device)
            out = v.model.generate(**inp, max_new_tokens=a.max_new, do_sample=False, temperature=None, top_p=None, top_k=None)
        for (r, _, _, letters), o_ in zip(chunk, out):
            text = tok.decode(o_[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            ans = None
            if "</think>" in text:
                m = re.search(r"\b([A-H])\b", text.split("</think>", 1)[1])
                ans = m.group(1) if m and m.group(1) in letters else None
            r.update(on=ans, n_chars=len(text), text=text[-3000:])
        torch.cuda.empty_cache()
        print(f"[{a.tag}] {min(b0 + a.batch, len(queue))}/{len(queue)}", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, rows=rows), open(f"/vlm/results/r3/s746b_{a.tag}_mmiu.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s746b_{a.tag}_mmiu.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--per_task", type=int, default=10); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max_new", type=int, default=2048)
    main(ap.parse_args())
