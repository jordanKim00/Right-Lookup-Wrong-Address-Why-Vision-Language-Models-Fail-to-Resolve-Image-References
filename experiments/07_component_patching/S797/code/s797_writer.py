"""S797 (pre-registered in knowledge/DECISIONS.md before any S797 data): which component writes the address?
S795 placed the formation of the lookup-readable address in the answer row's stream at layers 33-35 of Qwen3.5-27B (FINDINGS §855): through the
index-matching heads' own query map, the start-relative target share rises from chance after layer 32 to .80-.96 after layer 35, while far
end-relative references receive a weakened copy of the ordinal.  Here the outputs of single components, at the rows from the question mark to the
answer, are exchanged between the natural END(k) run and the START(j) run of the same scene (j = N + 1 - k; the prompts differ only in the
ordinal and frame words, so every position aligns, as in S793): the Gated DeltaNet mixers of layers 32-34 (DN), the MLPs of layers 32-35 (MLP),
the full-attention mixer of layer 35 (FA), all of these (ALL), and, as a control band, every mixer of the frame-read band 27-31 (FB).
Layers are mapped by relative depth for other Qwen3.5 sizes (smoke tests only).
Scenes: procedural pictures (160 px), N = 7, seed 79700 (new), 40 scenes, k in {2, 3, 5, 6}.
Arms per (scene, k): START, END (capture the outputs), NOOP (END given its own ALL outputs), E<-S for DN, MLP, FA, ALL, FB (END given START's
outputs), S<-E for DN, MLP, ALL (START given END's outputs).  Readout: the two-form menu readout (S794, R3_READ in {bare, max} asserted), t1;
block shares at the answer row at two reader layers (39 and 47 in 27B), all heads and the S786 index-matching heads of those layers.
Usage: python s797_writer.py MODEL TAG [--n 40] [--ks 2 3 5 6] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath import QPath

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s797_{}_writer.json"


def main(a):
    assert os.environ.get("R3_READ") in ("bare", "max"), "S797 must read the generated form (FINDINGS §846, §851)"
    dev = os.environ.get("S797_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    mixer = lambda li: v.layers[li].self_attn if li in FA_L else v.layers[li].linear_attn
    W_LO, W_HI = mp(32), mp(35)
    DN = [l for l in range(W_LO, W_HI + 1) if l not in FA_L]
    FA = [l for l in range(W_LO, W_HI + 1) if l in FA_L]
    MLP = list(range(W_LO, W_HI + 1))
    FB = list(range(mp(27), mp(31) + 1))
    SETS = {"DN": [("mix", l) for l in DN], "MLP": [("mlp", l) for l in MLP], "FA": [("mix", l) for l in FA],
            "ALL": [("mix", l) for l in range(W_LO, W_HI + 1)] + [("mlp", l) for l in MLP], "FB": [("mix", l) for l in FB]}
    READ = sorted({min(FA_L, key=lambda l: abs(l - mp(x))) for x in (39, 47)})
    MH = {}
    heads = a.heads or {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
                        "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads:
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(s):
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794 two-form readout
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
        if os.environ.get("R3_READ") != "max":
            return [aid(w) for w in ws]
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id = T("<|vision_start|>"), T("<|vision_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)

    X = dict(mode=None, rows=None, put=set(), store={}, donor={})

    def mk(kind, li):
        def h(mod, args, out):
            if X["mode"] is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            if X["mode"] == "grab":
                X["store"][(kind, li)] = t[0, X["rows"]].detach().clone()
                return None
            if (kind, li) in X["put"]:
                t = t.clone(); t[0, X["rows"]] = X["donor"][(kind, li)].to(t.device, t.dtype)
                return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
            return None
        return h
    used = sorted({x for st in SETS.values() for x in st})
    hooks = [(mixer(li) if kind == "mix" else v.layers[li].mlp).register_forward_hook(mk(kind, li)) for kind, li in used]
    TOOL = QPath(v, READ, [])
    T1 = []

    def prompt(q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    def share(st):
        o = {}
        for li in READ:
            if ("cap", li) not in st:
                continue
            p = torch.softmax(st[("cap", li)][0].float(), -1)
            o[str(li)] = dict(all=p.mean(0).tolist())
            if li in MH:
                o[str(li)]["mh"] = p[MH[li]].mean(0).tolist()
        return o

    @torch.no_grad()
    def run(inp, labs, menu, cap, rows, mode=None, put=(), donor=None):
        X.update(mode=mode, rows=rows, put=set(put), donor=donor or {}); X["store"] = {} if mode == "grab" else X["store"]
        TOOL.set(dict(rows=rows, cap=cap))
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); TOOL.set(None)
        grabbed = dict(X["store"]) if mode == "grab" else None
        X.update(mode=None, put=set(), donor={})
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), share(st), grabbed

    print(f"[{a.tag}] DN {DN} FA {FA} MLP {MLP} FB {FB} READ {READ} matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        for k in a.ks:
            j = N + 1 - k
            iS, iE = v.encode(prompt(j, "first", menu), pics), v.encode(prompt(k, "last", menu), pics)
            idS, idE = iS["input_ids"][0].tolist(), iE["input_ids"][0].tolist()
            assert len(idS) == len(idE), "START and END prompts must align"
            S_ = len(idE)
            vs = [i for i, t in enumerate(idE) if t == vs_id]; ve = [i for i, t in enumerate(idE) if t == ve_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([idE[i]]))
            assert max(i for i, (x, y) in enumerate(zip(idS, idE)) if x != y) < qm
            rows = list(range(qm, S_))
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            cap = dict(rows=[len(rows) - 1], owner=owner, nb=N)
            rec = dict(si=si, k=k, j=j, land={}, share={})
            rec["land"]["START"], rec["share"]["START"], gS = run(iS, labs, menu, cap, rows, mode="grab")
            rec["land"]["END"], rec["share"]["END"], gE = run(iE, labs, menu, cap, rows, mode="grab")
            rec["land"]["NOOP"], _, _ = run(iE, labs, menu, cap, rows, mode="put", put=SETS["ALL"], donor=gE)
            for nm in ("DN", "MLP", "FA", "ALL", "FB"):
                if SETS[nm]:
                    rec["land"][f"E<-S:{nm}"], rec["share"][f"E<-S:{nm}"], _ = run(iE, labs, menu, cap, rows, mode="put", put=SETS[nm], donor=gS)
            for nm in ("DN", "MLP", "ALL"):
                if SETS[nm]:
                    rec["land"][f"S<-E:{nm}"], rec["share"][f"S<-E:{nm}"], _ = run(iS, labs, menu, cap, rows, mode="put", put=SETS[nm], donor=gE)
            out.append(rec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    TOOL.remove()
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), sets={k: [list(x) for x in v_] for k, v_ in SETS.items()}, READ=READ,
                   matcher_layers=sorted(MH), t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=79700); ap.add_argument("--heads", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 5, 6])
    main(ap.parse_args())
