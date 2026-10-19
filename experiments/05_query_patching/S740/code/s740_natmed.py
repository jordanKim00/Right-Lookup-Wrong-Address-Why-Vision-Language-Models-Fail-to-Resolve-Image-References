"""S740: does R³'s repair on photographs travel through the full-attention queries?  (pre-registration knowledge/DECISIONS.md 2026-10-11,
before any S740 data; analysis exp/s740_analyze.py, frozen with it).  The S722 design on natural items.
Items: the MMIU development split (exp/s737_natqk.dev_items: first 50 items per position task, N <= 9), prompts as the evaluator builds them,
end-relative options ("X: The m-th image from the end").  R³ = the model's final configuration (exp/r3_select.build, plan file in force).
R³ writes only at option tokens after the images, so image keys and values are identical in the unedited and the repaired run; the repaired
answer can reach image content only through the queries of the later tokens (full attention) or their recurrent reads.
Repaired items: R³ correct, unedited wrong.  Rows = every position from the first written token to the answer.  Arms on repaired items:
    NOOP          repaired run, every full-attention query at the rows replaced by its own value (instrument)
    R3_QBACK_ALL  repaired run, every full-attention query at the rows put back to the unedited run's
    END_QALL      unedited run given every full-attention query of the repaired run at the rows
    R3_QBACK_M    repaired run, the index-matching heads' queries put back
    END_QM        unedited run given the index-matching heads' repaired queries
Hybrid Qwen3.5 models only (exp/c_tools.FullAttn).  Usage: python s740_natmed.py MODEL TAG --base B --rc RCTAG --heads JSON
       [--reader_override R] [--max_rep 150] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, sys
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from c_tools import Ctx, FullAttn
import r3_select
import s737_natqk as Q737

E = Q737.E


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rep = r3_select.build(v, a.base, a.rc, reader=a.reader_override)
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    nh = v.layers[FA_L[0]].self_attn.config.num_attention_heads
    ALLL = {l: list(range(nh)) for l in FA_L}
    MATL = {}
    for l, h in json.load(open(a.heads))["matchers"]:
        MATL.setdefault(l, []).append(h)
    cx = Ctx(); FA = FullAttn(v, FA_L, cx)

    def load(task, p):
        im = Image.open(f"{E.ROOT}/{task}/{p.split('/')[-1]}").convert("RGB")
        s = 448 / max(im.size)
        return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s < 1 else im

    def prompt(it, lines, N):
        return v.chat([("user", f"{v.IMG*N}\n{it['input']['question']}\n{Q737.ctx_of(it)}\nSelect from the following choices.\n" + "\n".join(lines) +
                                "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run(inp, cand, writes=None, rows=None, grab=False, edit=None):
        cx.reset()
        if rows is not None:
            cx.rows = rows; cx.grab = grab; cx.edit = edit or {}
        rep.ST = dict(writes=writes) if writes else {}
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        rep.ST = {}
        store = dict(cx.store) if grab else None
        cx.reset()
        return int(torch.stack([lg[c] for c in cand]).argmax()), store

    def ed_from(store, heads):
        return {("F", li): {"heads": hs, "q": store[("F", li, "q")]} for li, hs in heads.items()}

    out = []; n_rep = 0
    items = Q737.dev_items()[:a.cap] if a.cap else Q737.dev_items()
    for ii, (task, it, opts) in enumerate(items):
        paths = it["input"]["input_image_path"]; N = len(paths)
        pics = [load(task, p) for p in paths]
        letters = [L for L, _ in opts]; cand = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in letters]
        gold = str(it["output"]["output_text"]).strip()[:1]
        lE = [f"{L}: {E.rev_phrase(N + 1 - k, cap=True)}" for L, k in opts]
        tE = prompt(it, lE, N)
        inp = v.encode(tE, pics)
        pE, _ = run(inp, cand)
        _, writes = rep.plan(tE, pics)
        rec = dict(task=task, N=N, gold=gold, pick_END=letters[pE], n_writes=len(writes))
        if writes:
            pR, _ = run(inp, cand, writes=writes)
            rec["pick_R3"] = letters[pR]
            if letters[pR] == gold and letters[pE] != gold and n_rep < a.max_rep:
                n_rep += 1
                ids = inp["input_ids"][0].tolist()
                rows = list(range(min(int(p) for p, _ in writes), len(ids)))
                _, sE = run(inp, cand, rows=rows, grab=True)
                _, sR = run(inp, cand, writes=writes, rows=rows, grab=True)
                arms = {"NOOP": (writes, ed_from(sR, ALLL)), "R3_QBACK_ALL": (writes, ed_from(sE, ALLL)), "END_QALL": (None, ed_from(sR, ALLL)),
                        "R3_QBACK_M": (writes, ed_from(sE, MATL)), "END_QM": (None, ed_from(sR, MATL))}
                for name, (w_, ed) in arms.items():
                    pX, _ = run(inp, cand, writes=w_, rows=rows, edit=ed)
                    rec[f"pick_{name}"] = letters[pX]
                rec["rows"] = len(rows)
                del sE, sR
        out.append(rec)
        torch.cuda.empty_cache()
        if (ii + 1) % 50 == 0:
            print(f"[{a.tag}] {ii + 1}/{len(items)} · repaired so far {n_rep}", flush=True)
    FA.remove()
    json.dump(dict(tag=a.tag, model=a.model, base=a.base, rc=a.rc, choice=r3_select.choice(a.rc), rows=out),
              open(f"/vlm/results/r3/s740_{a.tag}_natmed.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s740_{a.tag}_natmed.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--heads", required=True); ap.add_argument("--reader_override", default=None)
    ap.add_argument("--max_rep", type=int, default=150); ap.add_argument("--cap", type=int, default=None); ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
