"""S582: one evaluation for every repair -- the base model and the repaired model on the SAME items, paired.

Repairs: lora (S579 circuit-band adapters), lrt0 (S580), lrt1 (S581). Suites:
  efficacy    mmiu_pos  -- MMIU's 11 option-list position tasks, forward form and reverse form ("... from the end")
              cloze     -- MMIU visual_cloze, NATURALLY reverse options ("The k-th image from the end"), and its forward rewrite
              muir      -- MUIRBench position items (S491 selection), forward and reverse forms
  locality    mmiu_other-- MMIU tasks whose options are not image references (correspondence, NLVR2, MaRVL, quality, ...)
              imagenette-- single-image recognition
Nothing is tuned here; the repair is loaded as trained on synthetic CIFAR scenes.
"""
from __future__ import annotations

import argparse, io, json, random, re, sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni

OUT = "/vlm/results/s582_evalrepair_{}.json"
ROOT = "/vlm/assets/mmiu"
MAN = "/vlm/assets/benchmark_manifests/muir_source721.jsonl"
SNAP = "/vlm/assets/hf/hub/datasets--MUIRBENCH--MUIRBENCH/snapshots/4c393cffc985c77d28de3b9045e2e5186920df80/data"
INET = "/vlm/assets/imagenette2-160/val"
POS_TASKS = ["face_retrieval", "handwritten_retrieval", "image2image_retrieval", "person_reid", "vehicle_retrieval",
             "sketch2image_retrieval", "text2image_retrieval", "forensic_detection_blink", "forensic_detection_forgerynet",
             "visual_coherence", "Icon_Question_Answering_with_Spatial_Context"]
REV_STYLE = "end"                                                                   # S628: the reverse form's wording


def rev_phrase(K, cap=False):
    """the K-th image counted from the end, in the wording REV_STYLE (S628 phrase-shift; "end" = the original rewrite)."""
    o = ORDW[K - 1]
    s_ = {"end": f"the {o} image from the end",
          "tolast": ("the last image" if K == 1 else f"the {o}-to-last image"),
          "reverse": f"the {o} image in reverse order",
          "backwards": f"the {o} image counting backwards"}[REV_STYLE]
    return s_[0].upper() + s_[1:] if cap else s_


OTHER_TASKS = ["spot_the_similarity", "visually_grounded_reasoning_nlvr2", "visually_grounded_reasoning_marvl",
               "visual_quality_assessment_q_bench+", "functional_correspondence_blink", "semantic_correspondence_blink",
               "visual_correspondence_blink", "jigsaw_puzzle_solving", "visual_ordering", "textual_cloze", "point_tracking",
               "spot_the_diff"]
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
PAT = re.compile(r"^\s*([A-H])\s*:\s*[Tt]he (" + "|".join(ORDW) + r") image\s*$")
PAT_END = re.compile(r"^\s*([A-H])\s*:\s*[Tt]he (" + "|".join(ORDW) + r") image from the end\s*$")
INET_NAMES = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette", "n03000684": "chainsaw",
              "n03028079": "church", "n03394916": "horn", "n03417042": "truck", "n03425413": "pump", "n03445777": "golf",
              "n03888257": "parachute"}


class Repair:
    def __init__(self, v, kind, ckpt, ufile=None):
        self.v, self.kind, self.on = v, kind, False
        self.ctx = {}
        if kind == "none":
            return
        st = torch.load(ckpt, weights_only=False)
        if kind == "lora":
            from s579_clf import attach
            meta = st["_meta"]
            self.mods = attach(v, meta["band"], meta["r"], meta["alpha"])
            for l, nm, m in self.mods:
                A, B = st[f"{l}.{nm}"]; m.A.data.copy_(A.to(m.A.device)); m.B.data.copy_(B.to(m.B.device))
            return
        meta = st.pop("_meta")
        self.lin, self.lq = meta["lin"], meta["lq"]
        A = v.layers[self.lq].self_attn
        H = A.q_proj.out_features // A.head_dim
        if v.fam == "qwen35":
            H = A.q_proj.out_features // (2 * A.head_dim)
        d = v.layers[0].input_layernorm.weight.shape[0]
        dev = A.q_proj.weight.device
        if kind == "lrt0":
            from s580_lrt import LRT
            self.mod = LRT(d, H, A.head_dim).to(dev).float()
        else:
            from s581_lrt1 import LRT1
            self.mod = LRT1(d, H, A.head_dim, st["U"].float()).to(dev).float()
        self.mod.load_state_dict(st)
        self.dev = dev

        def cap(m, i, o):
            if self.on and self.ctx.get("img"):
                t = o[0] if isinstance(o, tuple) else o
                self.ctx["h"] = t[0]
            return o

        def qh(m, i, o):
            if not (self.on and self.ctx.get("img")) or self.ctx.get("h") is None:
                return o
            P, marks, h = self.ctx["P"], self.ctx["marks"], self.ctx["h"]
            with torch.no_grad():
                dq = self.mod(h[P:].float(), h[marks].float()).to(o.dtype)
            return torch.cat([o[:, :P], o[:, P:] + dq[None]], 1)
        v.layers[self.lin].register_forward_hook(cap)
        A.q_norm.register_forward_hook(qh)

    def set(self, on):
        self.on = on
        if self.kind == "lora":
            for _, _, m in self.mods:
                m.on = on

    @torch.no_grad()
    def logits(self, text, pics):
        v = self.v
        inp = v.encode(text, pics)
        if self.kind in ("lrt0", "lrt1"):
            if pics:
                B = v.image_blocks(inp["input_ids"], len(pics))
                self.ctx = dict(img=True, P=int(B[-1][-1]) + 2, marks=torch.tensor([int(b[0]) - 1 for b in B], device=self.dev), h=None)
            else:
                self.ctx = dict(img=False)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        self.ctx["h"] = None
        return out.logits[0, -1].float()

    def both(self, text, pics, cand):
        self.set(False); a = int(self.logits(text, pics)[cand].argmax())
        self.set(True); b = int(self.logits(text, pics)[cand].argmax())
        self.set(False)
        return a, b


def run(v, rep, tag, suites, per_task=50, per_other=25, n_inet=200, maxside=448, seed=58200, cap=None, sealed=False, fresh=False, prompt_mode="none"):
    # S601 sealed test: only items never used before (MMIU after the first 50 per task, MUIR outside the S582 sample,
    # position-free MMIU after the first 25 qualifying, MUIRBench outside the S596 sample)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    letter_id = lambda L: v.tok(L, add_special_tokens=False)["input_ids"][0]
    rows = []

    def load_mmiu(task, p):
        im = Image.open(f"{ROOT}/{task}/{p.split('/')[-1]}").convert("RGB")
        s = maxside / max(im.size)
        return im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s < 1 else im

    def mc(question, ctx, lines, N):
        return v.chat([("user", f"{v.IMG*N}\n{question}\n{ctx}\nSelect from the following choices.\n" + "\n".join(lines) +
                                "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)

    NOTE = (" Note: 'the k-th image from the end' counts backwards from the last image: the first image from the end is the last "
            "image, the second image from the end is the one just before it, and so on.")

    def twopass_k(pics, K):
        """S640 two-pass baseline: the model first states which image (from the start) is the K-th from the end."""
        N_ = len(pics)
        q_ = f"{v.IMG*N_}\nCounting from the start, which image is the {ORDW[K-1]} image from the end? Answer with a number only."
        inp_ = v.encode(v.chat([("user", q_)], prefix_assistant=THINK), pics)
        with torch.no_grad():
            out_ = v.model.generate(**inp_, max_new_tokens=6, do_sample=False)
        txt_ = v.tok.decode(out_[0, inp_["input_ids"].shape[1]:], skip_special_tokens=True)
        m_ = re.search(r"\d+", txt_)
        if m_ and 1 <= int(m_.group()) <= min(N_, 10):
            return int(m_.group())
        w_ = [i + 1 for i, o_ in enumerate(ORDW) if re.search(r"\b" + o_ + r"\b", txt_.lower())]
        return w_[0] if w_ and w_[0] <= N_ else None

    def ctx_of(it):
        c = str(it["input"].get("context", "")).split("\n")[0]
        return re.split(r"\bA\s*:", c)[0].replace("Select from the following choices.", "").strip()[:400]

    if "mmiu_pos" in suites:
        for task in POS_TASKS:
            items = json.load(open(f"{ROOT}/{task}.json"))
            keep = []
            for it in items:
                ls = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
                if ls and all(PAT.match(l) for l in ls):
                    keep.append((it, [(PAT.match(l).group(1), ORDW.index(PAT.match(l).group(2)) + 1) for l in ls]))
            if fresh:                                                              # S639: never-used items, no image shared with used ones
                used_imgs = {p_ for it_, _ in keep[:150] for p_ in it_["input"]["input_image_path"]}
                sel = [(it_, o_) for it_, o_ in keep[150:250] if not used_imgs & set(it_["input"]["input_image_path"])]
            else:
                sel = keep[50:150] if sealed else keep[:per_task]
            for it, opts in sel:
                paths = it["input"]["input_image_path"]; N = len(paths)
                if N > 9:
                    continue
                pics = [load_mmiu(task, p) for p in paths]
                letters = [L for L, _ in opts]; cand = [letter_id(L) for L in letters]
                gold = str(it["output"]["output_text"]).strip()[:1]
                r = dict(suite="mmiu_pos", task=task, gold=gold)
                for form in ("F", "R"):
                    lines = [f"{L}: The {ORDW[k-1]} image" if form == "F" else f"{L}: {rev_phrase(N + 1 - k, cap=True)}" for L, k in opts]
                    qtext = it["input"]["question"]
                    if form == "R" and prompt_mode == "explain":
                        qtext = qtext + NOTE
                    elif form == "R" and prompt_mode == "twopass":
                        new, tp = [], []
                        for (L, k) in opts:
                            m_ = twopass_k(pics, N + 1 - k)
                            tp.append((k, m_))                                     # S640: (true start index, model's stated index)
                            new.append(f"{L}: The {ORDW[m_-1]} image" if m_ else f"{L}: {rev_phrase(N + 1 - k, cap=True)}")
                        lines = new; r["R|tp"] = tp
                    a, b = rep.both(mc(qtext, ctx_of(it), lines, N), pics, cand)
                    r[f"{form}|base"] = letters[a]; r[f"{form}|rep"] = letters[b]
                    if hasattr(rep, "last_detail"):                                  # S616 diagnosis: per-source decisions
                        r[f"{form}|det"] = [(d["tok"], d["kh"], d["t"], d["map"][1], round(d["wmax"], 3), int(d["act"]), d["frame"], d.get("attn"))
                                            for d in rep.last_detail]
                        r[f"{form}|codes"] = getattr(rep, "last_codes", None)
                rows.append(r)
            torch.cuda.empty_cache(); print(f"[{tag}] mmiu_pos {task} 끝, 행 {len(rows)}", flush=True)

    if "cloze" in suites:
        items = json.load(open(f"{ROOT}/visual_cloze.json"))
        for it in items[:cap]:
            ls = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
            if not ls or not all(PAT_END.match(l) for l in ls):
                continue
            paths = it["input"]["input_image_path"]; N = len(paths)
            pics = [load_mmiu("visual_cloze", p) for p in paths]
            opts = [(PAT_END.match(l).group(1), ORDW.index(PAT_END.match(l).group(2)) + 1) for l in ls]   # k counted from the end
            letters = [L for L, _ in opts]; cand = [letter_id(L) for L in letters]
            gold = str(it["output"]["output_text"]).strip()[:1]
            r = dict(suite="cloze", task="visual_cloze", gold=gold)
            for form in ("F", "R"):
                lines = [f"{L}: The {ORDW[k-1]} image from the end" if form == "R" else f"{L}: The {ORDW[N-k]} image" for L, k in opts]
                a, b = rep.both(mc(it["input"]["question"], ctx_of(it), lines, N), pics, cand)
                r[f"{form}|base"] = letters[a]; r[f"{form}|rep"] = letters[b]
            rows.append(r)
        torch.cuda.empty_cache(); print(f"[{tag}] cloze 끝, 행 {len(rows)}", flush=True)

    if "muir" in suites:
        import pandas as pd
        man = [json.loads(l) for l in open(MAN)]
        man = [m for m in man if m["answerable"] and m["nsource"] <= 9]
        rsel = random.Random(49100); by = defaultdict(list)
        for m in man:
            by[m["task"]].append(m)
        items = []
        for task in sorted(by):
            L_ = by[task][:]; rsel.shuffle(L_); items += (L_[60:] if sealed else L_[:60])
        shards = {}
        for m in items[:cap]:
            if m["shard"] not in shards:
                shards[m["shard"]] = pd.read_parquet(f"{SNAP}/{m['shard']}")
            rr = shards[m["shard"]].iloc[m["row_in_shard"]]
            pics = []
            for bb in rr["image_list"]:
                im = Image.open(io.BytesIO(bb["bytes"])).convert("RGB"); sc = maxside / max(im.size)
                pics.append(im.resize((max(28, int(im.width * sc)), max(28, int(im.height * sc)))) if sc < 1 else im)
            N = len(pics); optsx = list(rr["options"]); src = m["option_source_indices"]; qtxt = str(rr["question"])
            if (qtxt.count("<image>") != len(m["question_source_indices"]) or any(len(s_) > 1 for s_ in src)
                    or any(str(o_).count("<image>") != len(s_) for o_, s_ in zip(optsx, src))):
                continue
            qsrc = iter(m["question_source_indices"])
            question = re.sub(r"<image>", lambda _: f"Image {next(qsrc) + 1}", qtxt)
            letters = [chr(ord("A") + i) for i in range(len(optsx))]; cand = [letter_id(L) for L in letters]
            r = dict(suite="muir", task=m["task"], gold=m["answer_letter"])
            for form in ("F", "R"):
                lines = []
                for i, (Lt, o_) in enumerate(zip(letters, optsx)):
                    if not src[i]:
                        lines.append(f"{Lt}: {o_}"); continue
                    k = src[i][0] + 1
                    ph = f"the {ORDW[k-1]} image" if form == "F" else rev_phrase(N + 1 - k)
                    if form == "R" and prompt_mode == "twopass":
                        m_ = twopass_k(pics, N + 1 - k)
                        r.setdefault("R|tp", []).append((k, m_))                   # S640: (true start index, model's stated index)
                        ph = f"the {ORDW[m_-1]} image" if m_ else ph
                    lines.append(f"{Lt}: {re.sub(r'<image>', ph, str(o_))}")
                text = v.chat([("user", f"{v.IMG*N}\n{question}{NOTE if (form == 'R' and prompt_mode == 'explain') else ''}\n" + "\n".join(lines) +
                                        "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)
                a, b = rep.both(text, pics, cand)
                r[f"{form}|base"] = letters[a]; r[f"{form}|rep"] = letters[b]
                if hasattr(rep, "last_detail"):                                      # S624e: per-source decisions on MUIR too
                    r[f"{form}|det"] = [(d["tok"], d["kh"], d["t"], d["map"][1], round(d["wmax"], 3), int(d["act"]), d["frame"], d.get("attn"))
                                        for d in rep.last_detail]
                    r[f"{form}|codes"] = getattr(rep, "last_codes", None)
            rows.append(r)
        torch.cuda.empty_cache(); print(f"[{tag}] muir 끝, 행 {len(rows)}", flush=True)

    if "mmiu_other" in suites:
        for task in OTHER_TASKS:
            try:
                items = json.load(open(f"{ROOT}/{task}.json"))
            except FileNotFoundError:
                continue
            n = 0; seen = 0
            for it in items:
                if n >= per_other:
                    break
                paths = it["input"]["input_image_path"]; N = len(paths)
                ls = [l for l in str(it.get("options", "")).split("\n") if l.strip() and re.match(r"^\s*[A-H]\s*:", l)]
                if N > 8 or not ls:
                    continue
                try:
                    pics = [load_mmiu(task, p) for p in paths]
                except Exception:
                    continue
                seen += 1
                if sealed and seen <= per_other:                                   # S601: skip the items S582 used
                    continue
                letters = [re.match(r"^\s*([A-H])\s*:", l).group(1) for l in ls]; cand = [letter_id(L) for L in letters]
                gold = str(it["output"]["output_text"]).strip()[:1]
                a, b = rep.both(mc(it["input"]["question"], ctx_of(it), ls, N), pics, cand)
                rows.append(dict(suite="mmiu_other", task=task, gold=gold, **{"O|base": letters[a], "O|rep": letters[b]}))
                n += 1
            torch.cuda.empty_cache(); print(f"[{tag}] mmiu_other {task} 끝 ({n})", flush=True)

    if "muir_all" in suites:                                                       # S596: all 12 MUIRBench tasks, native questions
        import glob as _glob, pandas as pd
        rsel = random.Random(59600); by = defaultdict(list)
        for f in sorted(_glob.glob(f"{SNAP}/*.parquet")):
            d_ = pd.read_parquet(f, columns=["task"])
            for i_, t_ in enumerate(d_["task"].tolist()):
                by[t_].append((f, i_))
        shards = {}
        for task in sorted(by):
            L_ = by[task][:]; rsel.shuffle(L_); n = 0
            if sealed:                                                             # S601: drop the S596 sample, reshuffle the rest
                used = []
                for f, i_ in L_:
                    if len(used) >= per_other:
                        break
                    if f not in shards:
                        shards[f] = pd.read_parquet(f)
                    if 2 <= len(shards[f].iloc[i_]["image_list"]) <= 9:
                        used.append((f, i_))
                L_ = [x for x in L_ if x not in set(used)]; random.Random(60100).shuffle(L_)
            for f, i_ in L_:
                if n >= per_other:
                    break
                if f not in shards:
                    shards[f] = pd.read_parquet(f)
                rr = shards[f].iloc[i_]
                N = len(rr["image_list"])
                if N > 9 or N < 2:
                    continue
                pics = []
                for bb in rr["image_list"]:
                    im = Image.open(io.BytesIO(bb["bytes"])).convert("RGB"); sc = maxside / max(im.size)
                    pics.append(im.resize((max(28, int(im.width * sc)), max(28, int(im.height * sc)))) if sc < 1 else im)
                cnt = iter(range(1, 100))
                sub = lambda t: re.sub(r"<image>", lambda _: f"Image {next(cnt)}", str(t))
                question = sub(rr["question"]); optsx = [sub(o_) for o_ in rr["options"]]
                letters = [chr(ord("A") + i) for i in range(len(optsx))]; cand = [letter_id(L) for L in letters]
                text = v.chat([("user", f"{v.IMG*N}\n{question}\n" + "\n".join(f"{L}: {o_}" for L, o_ in zip(letters, optsx)) +
                                        "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)
                a, b = rep.both(text, pics, cand)
                rows.append(dict(suite="muir_all", task=task, gold=str(rr["answer"]).strip()[:1], **{"O|base": letters[a], "O|rep": letters[b]}))
                n += 1
            torch.cuda.empty_cache(); print(f"[{tag}] muir_all {task} 끝 ({n})", flush=True)

    if "imagenette" in suites:
        rng = random.Random(seed); names = list(INET_NAMES.values()); per = n_inet // len(INET_NAMES)
        import os
        for wn, nm in INET_NAMES.items():
            fs = sorted(os.listdir(f"{INET}/{wn}")); rng.shuffle(fs)
            for f in fs[:(per if cap is None else 1)]:
                im = Image.open(f"{INET}/{wn}/{f}").convert("RGB")
                opts = names[:]; rng.shuffle(opts)
                text = v.chat([("user", f"{v.IMG}\nWhat is in this image? Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)
                cand = [v.tok(" " + o, add_special_tokens=False)["input_ids"][0] for o in opts]
                a, b = rep.both(text, [im], cand)
                rows.append(dict(suite="imagenette", task="imagenette", gold=nm, **{"O|base": opts[a], "O|rep": opts[b]}))
        torch.cuda.empty_cache(); print(f"[{tag}] imagenette 끝", flush=True)

    res = {}
    rng_ = np.random.default_rng(0)
    print(f"\n  묶음 / 형식          n     원 모델   수리      짝지은 차 [95%]")
    for suite in ("mmiu_pos", "cloze", "muir", "mmiu_other", "muir_all", "imagenette"):
        S = [r for r in rows if r["suite"] == suite]
        if not S:
            continue
        for form in (("F", "R") if suite in ("mmiu_pos", "cloze", "muir") else ("O",)):
            x0 = np.array([r[f"{form}|base"] == r["gold"] for r in S], float)
            x1 = np.array([r[f"{form}|rep"] == r["gold"] for r in S], float)
            dd = x1 - x0
            bs = [dd[rng_.integers(0, len(dd), len(dd))].mean() for _ in range(4000)]
            res[f"{suite}|{form}"] = dict(n=len(S), base=float(x0.mean()), rep=float(x1.mean()), d=[float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
            d = res[f"{suite}|{form}"]
            print(f"  {suite:>10s} / {form}      {d['n']:4d}    {d['base']:.3f}    {d['rep']:.3f}    {d['d'][0]:+.3f} [{d['d'][1]:+.3f}, {d['d'][2]:+.3f}]", flush=True)
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--kind", default="none", choices=["none", "lora", "lrt0", "lrt1", "nrr", "r3"])
    ap.add_argument("--reader", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--suites", default="mmiu_pos,cloze,muir,mmiu_other,imagenette")
    ap.add_argument("--per_task", type=int, default=50)
    ap.add_argument("--per_other", type=int, default=25)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--rev_style", default="end", choices=["end", "tolast", "reverse", "backwards"])   # S628
    ap.add_argument("--fresh", action="store_true")                                                   # S639
    ap.add_argument("--prompt_mode", default="none", choices=["none", "explain", "twopass"])         # S640
    ap.add_argument("--ablate", default=None, choices=[None, "no_frame", "no_ord", "rand_target"])   # S641
    ap.add_argument("--min_mode", action="store_true")                                                # S644 R3-min
    ap.add_argument("--codes_physical", action="store_true")                                          # S644 arm: block b = input order
    a = ap.parse_args()
    REV_STYLE = a.rev_style
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    if a.kind == "r3":                                                             # R³-Ref-trained module (r3_build.py)
        from r3_repair import R3Repair
        rep = R3Repair(v, a.ckpt, a.reader)
    elif a.kind == "nrr":
        import torch as _t
        from nrr_eval import NRRRepair, NRRMLRepair
        rep = NRRMLRepair(v, a.ckpt) if "mod" in _t.load(a.ckpt, weights_only=False) else NRRRepair(v, a.ckpt)
    else:
        rep = Repair(v, a.kind, a.ckpt)
    if a.ablate:
        rep.ablate = a.ablate
    if a.min_mode:
        rep.min_mode = True
    if a.codes_physical:
        rep.codes_physical = True
    run(v, rep, a.tag, set(a.suites.split(",")), per_task=a.per_task, per_other=a.per_other, cap=a.cap, sealed=a.sealed,
        fresh=a.fresh, prompt_mode=a.prompt_mode)
