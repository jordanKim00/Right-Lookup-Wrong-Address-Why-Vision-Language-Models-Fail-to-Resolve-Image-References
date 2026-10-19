"""S654: content-first vs reference-first yes/no with END and START frames.
Per scene (procedural, N 5-7, k 2-4, j = N+1-k != k fixed before outcomes): three contents of the SAME scene
  T = image j (END target), L = image k (start-relative lure), U = image u, u not in {j, k} (neutral).
Twelve paired prompts = frame {END, START} x order {CF "Is the {X} in the {k-th} image[ from the end]?", RF "Does the {k-th} image[ from the end] show the {X}?"}
x role {T, L, U}, each ending "Answer yes or no." The ordinal word is always k (END target j; START target k).
Predicted yes-profiles (T, L, U): CF_END  H_hold (yes,no,no) / H_match (no,yes,no); RF_END both (yes,no,no); START correct (no,yes,no).
Endpoint: next-token logits of " Yes"/"Yes"/"yes" vs " No"/"No"/"no" variants after the exact assistant prefix (max over each set), plus the unrestricted
argmax token (validity flag: the argmax is one of those variants). Rows keep PROC ids, image and input hashes.
Usage: python s654_yn.py MODEL TAG [--n 80]"""
from __future__ import annotations

import argparse, hashlib, json, random, sys, time
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    one = lambda w: v.tok(w, add_special_tokens=False)["input_ids"]
    YES = sorted({one(w)[0] for w in ("Yes", " Yes", "yes", " yes") if len(one(w)) == 1})
    NO = sorted({one(w)[0] for w in ("No", " No", "no", " no") if len(one(w)) == 1})
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    rng = random.Random(65400)
    designs = [(N, k) for N in (5, 6, 7) for k in (2, 3, 4) if N + 1 - k != k]   # j = k excluded before outcomes (N5 k3, N7 k4)
    rows = []; t0 = time.time()
    for s_i in range(a.n):
        N, k = designs[s_i % len(designs)]
        j = N + 1 - k
        scene = rng.sample(labels, N)
        sp = [rng.choice(pool[l]) for l in scene]
        pics = [proc_image(p)[0].resize((a.pixels, a.pixels)) for p in sp]
        u = rng.choice([x for x in range(1, N + 1) if x not in (j, k)])
        role = {"T": j, "L": k, "U": u}
        r = dict(scene=s_i, N=N, k=k, j=j, u=u, proc_ids=sp, labels=scene,
                 image_sha256=[hashlib.sha256(p.tobytes()).hexdigest()[:16] for p in pics], ans={})
        for frame in ("END", "START"):
            ref = f"the {ORDW[k-1]} image" + (" from the end" if frame == "END" else "")
            for order in ("CF", "RF"):
                for rn, pos in role.items():
                    X = scene[pos - 1]
                    q = f"Is the {X} in {ref}?" if order == "CF" else f"Does {ref} show the {X}?"
                    q = q[0].upper() + q[1:] + " Answer yes or no."
                    text = v.chat([("user", f"{v.IMG*N}\n{q}")], prefix_assistant=THINK)
                    inp = v.encode(text, pics)
                    with torch.no_grad():
                        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
                    y, n_ = float(lg[YES].max()), float(lg[NO].max())
                    am = int(lg.argmax())
                    r["ans"][f"{frame}|{order}|{rn}"] = dict(yes=y > n_, margin=round(y - n_, 3), argmax=v.tok.decode([am]),
                                                            valid=am in YES or am in NO,
                                                            input_sha=hashlib.sha256(inp["input_ids"].cpu().numpy().tobytes()).hexdigest()[:16])
        rows.append(r)
        if (s_i + 1) % 20 == 0:
            prof = lambda f, o: "/".join(f"{sum(x['ans'][f'{f}|{o}|{rn}']['yes'] for x in rows)/len(rows):.2f}" for rn in "TLU")
            print(f"[{a.tag}] {s_i+1}: CF_END {prof('END','CF')} · RF_END {prof('END','RF')} · CF_START {prof('START','CF')} · RF_START {prof('START','RF')}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(dict(tag=a.tag, model=a.model, yes_ids=YES, no_ids=NO, rows=rows), open(f"/vlm/results/r3/s654_{a.tag}_yn.json", "w"))
    print(f"[{a.tag}] 끝 -> /vlm/results/r3/s654_{a.tag}_yn.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=84); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
