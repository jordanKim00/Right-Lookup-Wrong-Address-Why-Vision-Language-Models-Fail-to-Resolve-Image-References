"""Single-image recognition control for the S730 procedural pictures (post hoc, no verdict; 2026-10-11): can the model name one picture
of the pool at the S730 size?  105 trials (seed 73002), one picture, menu = its label plus eight other labels, restricted argmax; and the
same with seven pictures where the question names the image start-relatively (the S730 START question) for reference.
Usage: python s730r_recog.py MODEL TAG [--pixels P]"""
import argparse, json, random, sys
import numpy as np, torch
sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(73002); one = []; seven = []

    @torch.no_grad()
    def ask(n, q, pics, menu):
        text = v.chat([("user", f"{v.IMG*n}\n{q} Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)
        lg = v.model(**v.encode(text, pics), use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        return menu[int(torch.stack([lg[ids1(c)[0]] for c in menu]).argmax())]
    for _ in range(105):
        labs = rng.sample(labels, 9); pic = proc_image(rng.choice(pool[labs[0]]))[0].resize((a.pixels, a.pixels))
        menu = labs[:]; rng.shuffle(menu)
        one.append(ask(1, "What is in the image?", [pic], menu) == labs[0])
    for _ in range(15):
        labs = rng.sample(labels, 9); scene = labs[:7]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in scene]
        menu = labs[:]; rng.shuffle(menu)
        for t in range(1, 8):
            seven.append(ask(7, f"There are 7 images in a row. What is in the {ORDW[t-1]} image from the first?", pics, menu) == scene[t - 1])
    print(f"[{a.tag}] one picture {np.mean(one):.3f} (n={len(one)}) · seven pictures, start-relative {np.mean(seven):.3f} (n={len(seven)})")
    json.dump(dict(tag=a.tag, model=a.model, one=float(np.mean(one)), seven=float(np.mean(seven))), open(f"/vlm/results/r3/s730r_{a.tag}_recog.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s730r_{a.tag}_recog.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--pixels", type=int, default=160)
    main(ap.parse_args())
