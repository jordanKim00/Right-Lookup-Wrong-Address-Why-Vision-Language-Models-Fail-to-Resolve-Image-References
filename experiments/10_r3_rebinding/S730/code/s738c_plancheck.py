"""S738c: do the mechanistic runs made before the S738 switch (S730, S736, S737 -- R³ with the registered S724 Read) write the same plans as
the final S738 Read?  First passes only, no answers: the same items as those runs (same seeds and scene code), plans compared as the set of
(position, write vector) pairs.  Only models whose Read changed are checked (Qwen3.5-27B: margin .5 -> .45; Qwen3-VL-8B: margin .4 x1.5 ->
margin .30 x1.3); for the smaller models the S730 runs also used the calibration file's own reader and, in 8B and 9B, the base without the three
extra frame words, so --old_base / --old_reader give the configuration of the earlier run.
Usage: python s738c_plancheck.py MODEL TAG --base B --rc RC [--reader_override R] [--old_base B0] [--old_reader R0|none] [--sets S730,S736,S737]
       [--device_map balanced]"""
from __future__ import annotations

import argparse, json, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
import r3_select
import s737_natqk as Q737

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def same(w1, w2):
    if len(w1) != len(w2):
        return False
    a = sorted(w1, key=lambda x: int(x[0])); b = sorted(w2, key=lambda x: int(x[0]))
    return all(int(p) == int(q) and torch.allclose(d.float().cpu(), e.float().cpu(), atol=1e-5) for (p, d), (q, e) in zip(a, b))


def main(a):
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa")
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    # the configuration each earlier run actually used (base, reader, S724 Read) against the final one (S738 Read, final base and reader)
    old = r3_select.build(v, a.old_base or a.base, a.rc, path=r3_select.S724_PLAN, reader=(None if a.old_reader == "none" else (a.old_reader or a.reader_override)))
    new = r3_select.build(v, a.base, a.rc, path="/vlm/exp/s738_plan.json", reader=a.reader_override)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)

    def prompt(N, k, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[k-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    def scene(rng, N):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((160, 160)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        return pics, menu

    res = {}
    def check(name, items):
        eq = 0; n = 0; diff = []
        for text, pics in items:
            _, w1 = old.plan(text, pics); _, w2 = new.plan(text, pics)
            n += 1; s_ = same(w1, w2); eq += s_
            if not s_ and len(diff) < 5:
                diff.append(dict(old=len(w1), new=len(w2)))
        res[name] = dict(n=n, identical=eq, diff_examples=diff)
        print(f"[{a.tag}] {name}: identical plans {eq}/{n}", flush=True)

    sets = a.sets.split(",")
    rng = random.Random(73000); its = []                                              # S730: N = 7, 30 scenes, k = 2..6
    for si in range(30):
        pics, menu = scene(rng, 7)
        its += [(prompt(7, k, "last", menu), pics) for k in range(2, 7)]
    if "S730" in sets:
        check("S730", its)
    its = []
    for N in (5, 7, 9):                                                               # S736: 20 scenes per N, every k but the middle
        rng = random.Random(73600 + N)
        for si in range(20):
            pics, menu = scene(rng, N)
            its += [(prompt(N, k, "last", menu), pics) for k in range(1, N + 1) if 2 * k != N + 1]
    if "S736" in sets:
        check("S736", its)
    its = []                                                                          # S737: MMIU development items, end-relative options
    E = Q737.E
    for task, it, opts in (Q737.dev_items() if "S737" in sets else []):
        paths = it["input"]["input_image_path"]; N = len(paths)
        pics = []
        for p in paths:
            from PIL import Image
            im = Image.open(f"{E.ROOT}/{task}/{p.split('/')[-1]}").convert("RGB"); s = 448 / max(im.size)
            pics.append(im.resize((max(28, int(im.width * s)), max(28, int(im.height * s)))) if s < 1 else im)
        lE = [f"{L}: {E.rev_phrase(N + 1 - k, cap=True)}" for L, k in opts]
        text = v.chat([("user", f"{v.IMG*N}\n{it['input']['question']}\n{Q737.ctx_of(it)}\nSelect from the following choices.\n" + "\n".join(lE) +
                        "\nAnswer with the option's letter from the given choices directly.")], prefix_assistant=THINK)
        its.append((text, pics))
    if "S737" in sets:
        check("S737", its)
    if "S722" in sets:                                                               # S722 mediation items, against the R³-v3 plan they used
        from r3_min_md import R3MinMD
        from r3ref import ORD as ORD2, load_image
        old_v3 = R3MinMD(v, f"/vlm/results/r3/s709_{a.old_base or a.base}_cal.pt", v2=True, det_rule="margin", det_param=.5, anchor_scale=1.2, n_mode="max")
        keys = sorted({k for l in open("/vlm/assets/r3cal_a3/dev.jsonl") for k in json.loads(l)["images"]})
        by_lab = {}
        for kk in keys[:3000]:
            lb = proc_image(int(kk.split(":")[1]))[1]
            if len(ids1(lb)) == 1:
                by_lab.setdefault(lb, []).append(kk)
        labs2 = sorted(l for l, x in by_lab.items() if len(x) >= 10)
        rng = random.Random(a.s722_seed); its = []
        for N, ks in ((7, (3, 4, 5, 6)), (5, (3, 4))):
            for si in range(25):
                labs = rng.sample(labs2, N); extra = [l for l in labs2 if l not in labs][:2]
                pics = [load_image(rng.choice(by_lab[l])) for l in labs]
                menu = labs + extra; rng.shuffle(menu)
                for k in ks:
                    its.append((v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORD2[k-1]} image from the last? "
                                                 f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK), pics))
        old_keep = old; old = old_v3
        check("S722", its)
        old = old_keep
    json.dump(dict(tag=a.tag, model=a.model, old=dict(read=r3_select.choice(a.rc, r3_select.S724_PLAN), base=a.old_base or a.base, reader=a.old_reader or a.reader_override),
                   new=dict(read=r3_select.choice(a.rc, "/vlm/exp/s738_plan.json"), base=a.base, reader=a.reader_override), res=res),
              open(f"/vlm/results/r3/s738c_{a.tag}_plancheck.json", "w"), indent=1)
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s738c_{a.tag}_plancheck.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--reader_override", default=None); ap.add_argument("--device_map", default=None)
    ap.add_argument("--old_base", default=None); ap.add_argument("--old_reader", default=None)        # "none": the calibration file's own reader
    ap.add_argument("--sets", default="S730,S736,S737"); ap.add_argument("--s722_seed", type=int, default=72200)   # 72300: the 9B mediation scenes (S723, S728)
    main(ap.parse_args())
