"""S489: the mechanism-derived fix on a natural benchmark (MMIU positional tasks, the S462 items).

S487 (§499): computing the index in the question-first (QIQ) layout, writing it, and reading by the written index
lifts synthetic reverse retrieval .26-.35 -> .96-.99, exactly as the multiplicative law predicts. Here the same fix is
applied to the twelve MMIU tasks whose options name images by forward ordinals; the options are rewritten as reverse
ordinals ("The m-th image from the end") as in S462. Arms on the same items:
    F    original forward ordinals          E    explicit "Image k" (reference ceiling)
    R    reverse ordinals, one pass         R2Q  each reverse coordinate converted to a number in the QIQ layout,
                                                 then the options are shown as "Image j_hat" (two passes, no training)
Items with more than nine images are skipped (single-token digit readout) and counted.
"""
from __future__ import annotations

import argparse, json, os, re, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni

OUT = "/vlm/results/s489_mmiu_fix_{}.json"
ROOT = "/vlm/assets/mmiu"
TASKS = ["face_retrieval", "handwritten_retrieval", "image2image_retrieval", "person_reid",
         "vehicle_retrieval", "sketch2image_retrieval", "text2image_retrieval",
         "forensic_detection_blink", "forensic_detection_forgerynet", "visual_coherence",
         "Icon_Question_Answering_with_Spatial_Context"]
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
PAT = re.compile(r"^\s*([A-H])\s*:\s*[Tt]he (" + "|".join(ORDW) + r") image\s*$")


def run(model, tag="job", per_task=100, maxside=448):
    v = model
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rows, skipped = [], {}
    dig = [v.tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(1, 10)]

    def load(task, p):
        im = Image.open(f"{ROOT}/{task}/{p.split('/')[-1]}").convert("RGB")
        s = maxside / max(im.size)
        if s < 1:
            im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))
        return im

    for task in TASKS:
        items = json.load(open(f"{ROOT}/{task}.json"))
        keep = []
        for it in items:
            lines = [l for l in str(it.get("options", "")).split("\n") if l.strip()]
            if lines and all(PAT.match(l) for l in lines):
                keep.append((it, [(PAT.match(l).group(1), ORDW.index(PAT.match(l).group(2)) + 1)
                                  for l in lines]))
        keep = keep[:per_task]
        print(f"  {task}: {len(keep)} 문항", flush=True)
        for n, (it, opts) in enumerate(keep):
            paths = it["input"]["input_image_path"]
            N = len(paths)
            if N > 9:
                skipped[task] = skipped.get(task, 0) + 1
                continue
            pics = [load(task, p) for p in paths]
            letters = [L for L, _ in opts]
            lid = [v.tok(L, add_special_tokens=False)["input_ids"][0] for L in letters]
            gold = str(it["output"]["output_text"]).strip()[:1]
            # some tasks repeat the original options inside the context; cut them so that no variant
            # sees the forward ordinals, and drop the duplicated "Select from the following choices."
            ctx = str(it["input"].get("context", "")).split("\n")[0]
            ctx = re.split(r"\bA\s*:", ctx)[0].replace("Select from the following choices.", "").strip()[:400]
            forms = {
                "F": [f"{L}: The {ORDW[k-1]} image" for L, k in opts],
                "E": [f"{L}: Image {k}" for L, k in opts],
                "R": [f"{L}: The {ORDW[N-k]} image from the end" for L, k in opts],
            }
            r = dict(task=task, i=n, gold=gold, N=N, ks=[k for _, k in opts])

            def body(opt_lines, tail):
                return (f"{v.IMG*N}\n{it['input']['question']}\n{ctx}\nSelect from the following "
                        f"choices.\n" + "\n".join(opt_lines) + f"\n{tail}")

            @torch.no_grad()
            def restricted(text, prefix=""):
                txt = v.chat([("user", text)], prefix_assistant=THINK + prefix)
                out, _ = v.forward(txt, pics)
                lg = out.logits[0, -1].float()
                return letters[int(np.argmax([float(lg[i]) for i in lid]))]

            direct = "Answer with the option's letter from the given choices directly."
            for f, lines in forms.items():
                r[f] = restricted(body(lines, direct))
            # the mechanism-derived fix (S487): each reverse coordinate is converted to a number in the QIQ layout
            # (question, images, question -- no visual anchoring of the arithmetic), written, then read as "Image j"
            jh = []
            for L, k in opts:
                m = N + 1 - k
                q1 = (f"There are {N} images in a row. Which image number, counting from the start, is the "
                      f"{ORDW[m-1]} image from the end? Answer with only the number.")
                with torch.no_grad():
                    out, _ = v.forward(v.chat([("user", f"{q1}\n{v.IMG*N}\n{q1}")], prefix_assistant=THINK), pics)
                lg = out.logits[0, -1].float()
                jh.append(int(np.argmax([float(lg[d]) for d in dig[:N]])) + 1)
            r["jhat"] = jh
            r["conv_ok"] = all(a == k for a, (_, k) in zip(jh, opts))
            r["R2Q"] = restricted(body([f"{L}: Image {a}" for (L, _), a in zip(opts, jh)], direct))
            rows.append(r)
            if (n + 1) % 25 == 0:
                torch.cuda.empty_cache()
                print(f"[{tag}] {task} {n+1}/{len(keep)}", flush=True)

    res = {"skipped_N_gt_9": skipped}
    print(f"\n  {'과제':>36s} | {'F':>5s} {'E':>5s} {'R':>5s} {'R2Q':>5s} | P(변환 전부 정답) | n")
    for task in TASKS + ["전체"]:
        S = [r for r in rows if task == "전체" or r["task"] == task]
        if not S:
            continue
        acc = {f: float(np.mean([r[f] == r["gold"] for r in S])) for f in ("F", "E", "R", "R2Q")}
        pc = float(np.mean([r["conv_ok"] for r in S]))
        res[task] = dict(acc=acc, p_conv=pc, n=len(S))
        print(f"  {task[:36]:>36s} | " + " ".join(f"{acc[f]:5.3f}" for f in ("F", "E", "R", "R2Q")) + f" | {pc:5.3f} | {len(S)}")
    print(f"  제외(N>9): {skipped}")
    print(f"\n  가름 (사전 등록 S489) (m1) 전체 P(변환 전부 정답) >= .90. (m2) 전체 R2Q >= E − .05. (m3) 전체 R2Q − R >= +.15.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--per_task", type=int, default=100)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, per_task=a.per_task)
