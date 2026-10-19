"""S835 (descriptive re-measure, registered in knowledge/DECISIONS.md before any S835 data): which signature do Qwen3.5-27B's wrong answers on
end-relative MMIU options carry, under a clean readout?  §928 (exp/p2_w3_mmiu_signature.py) measured it on S489's saved answers (single-form
restricted option-letter argmax, no free-generation check): literal .599 vs chance .256, step .142 vs chance .243.
Items, prompt and pictures are S489's (exp/s489_mmiu_fix.py, the S462 items): the eleven MMIU tasks whose options all read 'X: The m-th image',
at most 100 per task, N <= 9, longer side <= 448 px; the F (forward ordinals) and R (end-relative ordinals) option forms of S489, one pass each,
same question, context cut and answer instruction.  Readout: per option letter the larger of its two tokenizations ('A', ' A'); t1 = the free
next token is one of the option letters in either form.  Saved per item: task, index, N, option positions ks, gold, both answers, letter logits,
t1 and the free top token.
Usage: python s835_mmiusig.py MODEL TAG [--per_task 100] [--device_map balanced]"""
from __future__ import annotations

import argparse, json, re, sys
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from uni import Uni

OUT = "/vlm/results/s835_{}_mmiusig.json"
ROOT = "/vlm/assets/mmiu"
TASKS = ["face_retrieval", "handwritten_retrieval", "image2image_retrieval", "person_reid",
         "vehicle_retrieval", "sketch2image_retrieval", "text2image_retrieval",
         "forensic_detection_blink", "forensic_detection_forgerynet", "visual_coherence",
         "Icon_Question_Answering_with_Spatial_Context"]
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
PAT = re.compile(r"^\s*([A-H])\s*:\s*[Tt]he (" + "|".join(ORDW) + r") image\s*$")


def run(v, tag, per_task=100, maxside=448):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    rows, skipped, nrun = [], {}, 0

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
        print(f"  {task}: {len(keep)} items", flush=True)
        for n, (it, opts) in enumerate(keep):
            paths = it["input"]["input_image_path"]
            N = len(paths)
            if N > 9:
                skipped[task] = skipped.get(task, 0) + 1
                continue
            pics = [load(task, p) for p in paths]
            letters = [L for L, _ in opts]
            lid = [(v.tok(L, add_special_tokens=False)["input_ids"][0], v.tok(" " + L, add_special_tokens=False)["input_ids"][0]) for L in letters]
            gold = str(it["output"]["output_text"]).strip()[:1]
            ctx = str(it["input"].get("context", "")).split("\n")[0]
            ctx = re.split(r"\bA\s*:", ctx)[0].replace("Select from the following choices.", "").strip()[:400]
            forms = {
                "F": [f"{L}: The {ORDW[k-1]} image" for L, k in opts],
                "R": [f"{L}: The {ORDW[N-k]} image from the end" for L, k in opts],
            }
            r = dict(task=task, i=n, gold=gold, N=N, ks=[k for _, k in opts], letters=letters)

            def body(opt_lines, tail):
                return (f"{v.IMG*N}\n{it['input']['question']}\n{ctx}\nSelect from the following "
                        f"choices.\n" + "\n".join(opt_lines) + f"\n{tail}")

            direct = "Answer with the option's letter from the given choices directly."
            for f, lines in forms.items():
                txt = v.chat([("user", body(lines, direct))], prefix_assistant=THINK)
                with torch.no_grad():
                    out, _ = v.forward(txt, pics)
                lg = out.logits[0, -1].float()
                sc = [float(torch.maximum(lg[a], lg[b])) for a, b in lid]
                top = int(lg.argmax())
                r[f] = letters[max(range(len(letters)), key=lambda i: sc[i])]
                r[f + "|logits"] = sc
                r[f + "|t1"] = int(any(top in ab for ab in lid))
                r[f + "|top"] = v.tok.decode([top])
                nrun += 1
            rows.append(r)
            if (n + 1) % 25 == 0:
                torch.cuda.empty_cache()
                print(f"[{tag}] {task} {n+1}/{len(keep)} · {nrun} runs", flush=True)
    acc = {f: sum(r[f] == r["gold"] for r in rows) / max(len(rows), 1) for f in ("F", "R")}
    t1 = {f: sum(r[f + "|t1"] for r in rows) / max(len(rows), 1) for f in ("F", "R")}
    json.dump(dict(tag=tag, model=v.path if hasattr(v, "path") else None, n=len(rows), skipped_N_gt_9=skipped, acc=acc, t1=t1, rows=rows),
              open(OUT.format(tag), "w"), indent=1)
    print(f"[{tag}] acc {acc} t1 {t1} -> {OUT.format(tag)}", flush=True)


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
