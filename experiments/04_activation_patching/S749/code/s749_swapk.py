"""S749: does the END address travel with the states that END(k) writes after its ordinal, from which layer, and does the blur of far
references travel with it?  Pre-registration in knowledge/DECISIONS.md (2026-10-12 20:1x) before any S749 data; analysis exp/s749_analyze.py.
Qwen3.5-27B, seven procedural pictures (160 px), the S703 / S748 prompt ("What is in the {ordinal} image from the last?").
Natural runs per scene: END(k), k in {2, 3, 4, 5} (target 8 - k), and START(t), t in {3, 4, 5, 6}.  The END prompts differ only in the
ordinal word (one token), so every row after it lines up; the ordinal token itself always keeps the recipient's state.
Transplants, 12 ordered pairs (recipient k <- donor k', k != k'), at the rows after the ordinal:
    SEND    the outputs of the six sender sublayers of S690 / S693 (attention at L31, L33, L34, L35, L42, L46 of 27B)
    CTRL    the MLP outputs of the same six layers
    RES@L   the residual stream after layer L, L in {34, 38, 42, 46}
    NOOP    SEND taken from the recipient's own natural run
Readouts: the menu-restricted answer, and the fixed-bank block scores of every head at L35, 39, 43, 47, 51, 55 for the rows "?", tail1
and the answer (as S748).  Layers are given for 27B (64 layers) and mapped by relative depth (same block type) for other models.
Usage: python s749_swapk.py MODEL TAG [--start 0] [--n 40] [--device_map balanced]
(--start / --n run scenes start .. start+n-1 of the same seeded sequence; parts are merged by exp/s749_merge.py)
"""
from __future__ import annotations

import argparse, gc, json, random, resource, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s749_{}_swapk"
KS, TS = (2, 3, 4, 5), (3, 4, 5, 6)
SEND27, RES27 = (31, 33, 34, 35, 42, 46), (34, 38, 42, 46)


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    kind = lambda i: "F" if i in FA_L else "D"

    def mapl(l27):                                   # same layer for 27B; relative depth, same block type, otherwise
        if nl == 64:
            return l27
        want = round(l27 / 63 * (nl - 1)); typ = kind(l27) if nl == 64 else ("F" if (l27 + 1) % 4 == 0 else "D")
        cands = [i for i in range(nl) if kind(i) == typ]
        return min(cands, key=lambda i: abs(i - want))
    SEND = sorted({mapl(l) for l in SEND27}); RES = sorted({mapl(l) for l in RES27})
    R = lambda r: int(round(r * (nl - 1)))
    CAP = sorted({min(FA_L, key=lambda l: abs(l - R(x / 63))) for x in (35, 39, 43, 47, 51, 55)})
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)

    ST = {}                                          # capture state: rows, image positions, owners, block scores
    MODE = {"rows": None, "grab": False, "store": {}, "rep": {}}

    def cap_hook(li):
        def h(mod, args, kwargs, out):
            if not ST.get("on"):
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
            rows = ST["rows"]; img = torch.tensor(ST["img"], device=hs.device)
            qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(),
                                            None if mod.q_proj.bias is None else mod.q_proj.bias.float()).view(len(rows), nh, 2 * hd)
            q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
            k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
            qe = torch.cat([q[..., :rd] * cos[rows][:, None] + rotate_half(q[..., :rd]) * sin[rows][:, None], q[..., rd:]], -1)
            ke = torch.cat([k[..., :rd] * cos[img][:, None] + rotate_half(k[..., :rd]) * sin[img][:, None], k[..., rd:]], -1)
            kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
            lg = torch.einsum("thd,shd->ths", qe, ke[:, kh]) * mod.scaling
            own = torch.tensor(ST["owner"], device=q.device)
            ST["cap"][li] = torch.stack([torch.logsumexp(lg[:, :, own == b], -1) for b in range(N)], -1).cpu().numpy().astype(np.float32)
            return out
        return h

    def swap(key, t):
        """grab or replace rows of tensor t [1, S, D] for key; returns t (possibly a modified clone)."""
        rows = MODE["rows"]
        if rows is None:
            return t
        if MODE["grab"]:
            MODE["store"][key] = t[0, rows].detach().clone()
        rep = MODE["rep"].get(key)
        if rep is not None:
            t = t.clone(); t[0, rows] = rep.to(t.device, t.dtype)
        return t

    def res_hook(li):
        def h(mod, args, out):
            if isinstance(out, tuple):
                return (swap(("res", li), out[0]),) + tuple(out[1:])
            return swap(("res", li), out)
        return h

    def att_hook(li):
        def h(mod, args, out):
            if isinstance(out, tuple):
                return (swap(("att", li), out[0]),) + tuple(out[1:])
            return swap(("att", li), out)
        return h

    def mlp_hook(li):
        def h(mod, args, out):
            return swap(("mlp", li), out)
        return h

    hooks = [v.layers[li].self_attn.register_forward_hook(cap_hook(li), with_kwargs=True) for li in CAP]
    hooks += [v.layers[li].register_forward_hook(res_hook(li)) for li in RES]
    for li in SEND:
        mod = v.layers[li].self_attn if kind(li) == "F" else v.layers[li].linear_attn
        hooks.append(mod.register_forward_hook(att_hook(li)))
        hooks.append(v.layers[li].mlp.register_forward_hook(mlp_hook(li)))

    def prompt(fr, q, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the "
                                f"{'last' if fr == 'E' else 'first'}? Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(fr, q, labs, pic, menu, rows=None, grab=False, rep=None):
        inp = v.encode(prompt(fr, q, menu), [pic[l] for l in labs])
        ids = inp["input_ids"][0].tolist()
        vs = [i for i, t in enumerate(ids) if t == vs_id]; ve = [i for i, t in enumerate(ids) if t == ve_id]
        trows = list(range(ve[-1] + 1, len(ids)))
        qm = next(i for i in trows if "?" in v.tok.decode([ids[i]]))
        uend = max(i for i, t in enumerate(ids) if t == T("<|im_end|>"))
        ST.clear(); ST.update(on=True, rows=[qm, uend + 1, len(ids) - 1], img=[p for b in range(N) for p in range(vs[b], ve[b] + 1)],
                              owner=[b for b in range(N) for _ in range(vs[b], ve[b] + 1)], cap={})
        MODE.update(rows=rows, grab=grab, store={}, rep=rep or {})
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        cap = ST["cap"]; store = MODE["store"]
        ST.clear(); MODE.update(rows=None, grab=False, store={}, rep={})
        lab = menu[int(torch.stack([lg[tid(c)] for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), cap, store, ids

    print(f"[{a.tag}] SEND {SEND} · RES {RES} · CAP {CAP}", flush=True)
    rng = random.Random(74900)
    meta, arrays = [], {}
    for si in range(a.start + a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs}
        menu = labs + extra; rng.shuffle(menu)
        if si < a.start:
            continue
        rec = dict(si=si, land={})
        don, idsk = {}, {}
        # rows after the ordinal: END prompts differ in exactly one token (processor only, no forward pass)
        i2 = v.encode(prompt("E", 2, menu), [pic[l] for l in labs])["input_ids"][0].tolist()
        i5 = v.encode(prompt("E", 5, menu), [pic[l] for l in labs])["input_ids"][0].tolist()
        diff = [i for i, (x, y) in enumerate(zip(i2, i5)) if x != y]
        assert len(i2) == len(i5) and len(diff) == 1, (len(i2), len(i5), diff)
        rows = list(range(diff[0] + 1, len(i2)))
        for k in KS:
            land, cap, store, ids = run("E", k, labs, pic, menu, rows=rows, grab=True)
            assert len(ids) == len(i2)
            rec["land"][f"E{k}"] = land; don[k] = {kk: t.cpu() for kk, t in store.items()}
            for li, sc in cap.items():
                arrays[f"{si}|E{k}|{li}"] = sc
        for t in TS:
            land, cap, _, _ = run("S", t, labs, pic, menu)
            rec["land"][f"S{t}"] = land
            for li, sc in cap.items():
                arrays[f"{si}|S{t}|{li}"] = sc
        arms = {"SEND": [("att", l) for l in SEND], "CTRL": [("mlp", l) for l in SEND]}
        arms.update({f"RES{l}": [("res", l)] for l in RES})
        for k in KS:
            land, cap, _, _ = run("E", k, labs, pic, menu, rows=rows, rep={key: don[k][key] for key in arms["SEND"]})
            rec["land"][f"NOOP|{k}"] = land
            for kp in KS:
                if kp == k:
                    continue
                for arm, keys in arms.items():
                    land, cap, _, _ = run("E", k, labs, pic, menu, rows=rows, rep={key: don[kp][key] for key in keys})
                    rec["land"][f"{arm}|{k}<{kp}"] = land
                    for li, sc in cap.items():
                        arrays[f"{si}|{arm}|{k}<{kp}|{li}"] = sc
        meta.append(rec)
        del don
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{a.tag}] scene {si} done · peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for h in hooks:
        h.remove()
    np.savez_compressed(OUT.format(a.tag) + ".npz", **arrays)
    json.dump(dict(tag=a.tag, model=a.model, SEND=SEND, RES=RES, CAP=CAP, rows=["qm", "tail1", "ans"], KS=KS, TS=TS, meta=meta),
              open(OUT.format(a.tag) + ".json", "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=0); ap.add_argument("--pixels", type=int, default=160)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
