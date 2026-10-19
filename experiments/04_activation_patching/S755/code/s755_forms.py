"""S755: is the composed END address a phrase-specific association or an abstract count-back code?
Pre-registration in knowledge/DECISIONS.md (2026-10-13 02:xx) before any S755 data; analysis exp/s755_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), menu = 7 labels + 2.  The same addresses (d steps before the last image, target 7 - d) asked as
    F1 "the {second..fifth} image from the last"      F2 "the {second..fifth}-to-last image"      F3 "the image {one..four} places before the last image"
    F4 "the {penultimate | antepenultimate} image" (d = 1, 2)      F5 "the {2nd..5th} image from the last"
controls: START "the {sixth..third} image from the first" and "the last image".  Cross-form transplant: the residual of the "?" row after layer
L in {30, 34} (27B) from a d = 1 question in form F4, F2, F3 or F1 (same-form control) into the F1 d = 4 question of the same scene.
Readout: the menu-restricted answer (image position).
Usage: python s755_forms.py MODEL TAG [--n 40] [--device_map balanced]
"""
from __future__ import annotations

import argparse, json, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N = 7
OUT = "/vlm/results/r3/s755_{}_forms.json"
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
NUM = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
CNT = ["zero", "one", "two", "three", "four"]
LAY27 = (30, 34)


def questions():
    q = {}
    for d in (1, 2, 3, 4):
        k = d + 1
        q[f"F1|{d}"] = f"What is in the {ORD[k-1]} image from the last?"
        q[f"F2|{d}"] = f"What is in the {ORD[k-1]}-to-last image?"
        q[f"F3|{d}"] = f"What is in the image {CNT[d]} place{'s' if d > 1 else ''} before the last image?"
        q[f"F5|{d}"] = f"What is in the {NUM[k-1]} image from the last?"
        q[f"S|{d}"] = f"What is in the {ORD[N - d - 1]} image from the first?"
    q["F4|1"] = "What is in the penultimate image?"
    q["F4|2"] = "What is in the antepenultimate image?"
    q["LAST|0"] = "What is in the last image?"
    return q


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    kind = lambda i: "F" if i in FA_L else "D"

    def mapl(l27):
        if nl == 64:
            return l27
        want = round(l27 / 63 * (nl - 1)); typ = "F" if (l27 + 1) % 4 == 0 else "D"
        return min([i for i in range(nl) if kind(i) == typ], key=lambda i: abs(i - want))
    LAY = [mapl(l) for l in LAY27]
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    T = v.tok.convert_tokens_to_ids
    ve_id = T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    MODE = {"row": None, "grab": False, "store": {}, "rep": {}}

    def hook(li):
        def h(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            r = MODE["row"]
            if r is None:
                return out
            if MODE["grab"]:
                MODE["store"][li] = t[0, r].detach().float().cpu().clone()
            if li in MODE["rep"]:
                t = t.clone(); t[0, r] = MODE["rep"][li].to(t.device, t.dtype)
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            return out
        return h
    hooks = [v.layers[li].register_forward_hook(hook(li)) for li in sorted(set(LAY))]
    Q = questions()

    def prompt(qtext, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. {qtext} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(qtext, labs, pic, menu, grab=False, rep=None):
        inp = v.encode(prompt(qtext, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        ve = max(i for i, t in enumerate(ids) if t == ve_id)
        qm = next(i for i in range(ve + 1, len(ids)) if "?" in v.tok.decode([ids[i]]))
        MODE.update(row=qm, grab=grab, store={}, rep=rep or {})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        store = MODE["store"]; MODE.update(row=None, grab=False, store={}, rep={})
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), store

    print(f"[{a.tag}] layers {LAY} · {len(Q)} questions", flush=True)
    rng = random.Random(75500)
    meta = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        rec = dict(si=si, land={})
        don = {}
        for key, qt in Q.items():
            land, store = run(qt, labs, pic, menu, grab=key in ("F1|1", "F2|1", "F3|1", "F4|1"))
            rec["land"][key] = land
            if store:
                don[key] = store
        for src in ("F1|1", "F2|1", "F3|1", "F4|1"):
            for L in LAY:
                rec["land"][f"X{L}|{src}>F1|4"] = run(Q["F1|4"], labs, pic, menu, rep={L: don[src][L]})[0]
        meta.append(rec)
        if (si + 1) % 10 == 0:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, LAY=LAY, LAY27=LAY27, questions=Q, meta=meta), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
