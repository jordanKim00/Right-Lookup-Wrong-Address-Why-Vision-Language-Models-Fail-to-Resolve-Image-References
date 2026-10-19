"""S827 (pre-registered in knowledge/DECISIONS.md before any S827 data): why does the end-relative reading corrupt a valid positional reference
when no transformation is needed?  Identity references (target j = k = (N + 1) / 2): N = 7 END(4) (natural accuracy ~.45) and N = 9 END(5)
(S825: 4 .50 · 5 .10 · 6 .325).  START(k) of the same pictures, ordinal and menu answers k; the two prompts differ only at the frame word.
FINDINGS §915 (S823): the question tail absorbs the frame at layers 31-34; the boundary rows' late computation (from 39-43) makes END(4)'s errors and
END(7)'s right answer; §902 / §904: the answer side takes the question side's state over at 43-51.
Per scene (procedural pictures, 160 px; nine labels: N = 9 shows all nine, N = 7 the first seven, the other two are menu-only; the same nine-word
menu in both): cases N7 (k = 4) and N9 (k = 5); in each, END(k) and START(k), component outputs (every mixer and MLP) grabbed at every row from the
frame word on.  Rows: QT = frame word to '?', M = menu, B = the five boundary rows, A = think markers and the answer row.
  NOOP_E / NOOP_S        END / START <- its own outputs (exactness)
  X_QT31 [_mix | _mlp]   END <- START at the QT rows, layers 31-34 (all / mixers / MLPs)
  X_B39 [_mix | _mlp]    END <- START at the B rows, layers 39-50
  X_QTB                  both;  X_Bc39  B rows 39-63;  X_A47c  A rows 47-63 (the decision itself, downstream control)
  R_QT31 / R_B39 / R_QTB START <- END at the same rows and layers (does the END component produce the error in START?)
Readout: two-form menu readout (S794), t1, the chosen word and the two-form logits of the nine menu words; internal signal: at the question mark, the
five boundary rows and the answer row, per look layer, the index-matching heads' block scores (log-sum-exp over each picture's tokens).
Seed 82700 (new), 40 scenes.
Usage: python s827_identity.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_b import QPathB

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s827_{}_identity.json"
CASES = [(7, 4), (9, 5)]


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S827_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    LOOK = [l for l in FA_L if .555 <= l / (nl - 1) <= .875]
    mixer = lambda li: v.layers[li].self_attn if li in FA_L else v.layers[li].linear_attn
    ALL = [("mix", l) for l in range(nl)] + [("mlp", l) for l in range(nl)]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    def sc(lg, s):
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
        return [v.tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] + [v.tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in ws]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id, end_id = T("<|vision_start|>"), T("<|vision_end|>"), T("<|im_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    assert len(labels) >= 9

    TOOL = QPathB(v, LOOK)
    XB = dict(active=False, specs=None)

    def mk(kind, li):
        def h(mod, args, out):
            if not XB["active"]:
                return None
            t = out[0] if isinstance(out, tuple) else out
            new = None
            for b, sp in enumerate(XB["specs"]):
                if sp is None:
                    continue
                if sp["mode"] == "grab":
                    sp["store"][(kind, li)] = t[b, sp["rows"]].detach().clone()
                elif sp["mode"] == "putrows":                                    # per (kind, layer, row) keys (two-part arms)
                    sel = [i for i, r in enumerate(sp["rows"]) if (kind, li, r) in sp["keys"]]
                    if sel:
                        if new is None:
                            new = t.clone()
                        new[b, [sp["rows"][i] for i in sel]] = sp["donor"][(kind, li)][[sp["idx"][i] for i in sel]].to(t.device, t.dtype)
                elif (kind, li) in sp["put"]:
                    if new is None:
                        new = t.clone()
                    new[b, sp["rows"]] = sp["donor"][(kind, li)][sp["idx"]].to(t.device, t.dtype)
            if new is None:
                return None
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return h
    hooks = [(mixer(li) if kind == "mix" else v.layers[li].mlp).register_forward_hook(mk(kind, li)) for kind, li in ALL]
    T1 = []

    def prompt(N, q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def runb(inps, xs, qs):
        XB.update(active=any(x is not None for x in xs), specs=xs); TOOL.set(qs)
        lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
        st = TOOL.store; TOOL.set(None); XB.update(active=False, specs=None)
        return lg, st

    print(f"[{a.tag}] look {LOOK} · matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0
    for si in range(a.n):
        labs9 = rng.sample(labels, 9)
        pics9 = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs9]
        menu = list(labs9); rng.shuffle(menu)
        for N, k in CASES:
            labs = labs9[:N]; pics = pics9[:N]; words = labs9
            j = N + 1 - k
            enc = dict(E=v.encode(prompt(N, k, "last", menu), pics), S=v.encode(prompt(N, k, "first", menu), pics))
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1
            S_ = len(ids["E"])
            ve = [i for i, x in enumerate(ids["E"]) if x == ve_id]; vs = [i for i, x in enumerate(ids["E"]) if x == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E"][i]]))
            uend = max(i for i, x in enumerate(ids["E"]) if x == end_id)
            assert v.tok.decode([ids["E"][uend + 3]]).strip() == "assistant"
            Fw = [i for i, (x, y) in enumerate(zip(ids["S"], ids["E"])) if x != y]
            assert Fw and max(Fw) < qm
            G = dict(QT=list(range(min(Fw), qm + 1)), M=list(range(qm + 1, uend)), B=list(range(uend, uend + 5)), A=list(range(uend + 5, S_)))
            TX = list(range(min(Fw), S_))
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            rowsC = [qm] + G["B"] + [S_ - 1]
            capq = lambda: dict(rows=rowsC, cap=dict(rows=list(range(len(rowsC))), owner=owner, nb=N))
            x1 = [dict(mode="grab", rows=TX, store={}), dict(mode="grab", rows=TX, store={})]
            lg, st = runb([enc["E"], enc["S"]], x1, [capq(), capq()]); nrun += 2
            rec = dict(si=si, N=N, k=k, j=j, labs=labs, words=words, menu=menu, land={}, mlog={}, qk={})
            def keep(nm, lgb, stb):
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = labs.index(w) + 1 if w in labs else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in words]
                o = {}
                for li in LOOK:
                    if ("cap", li) in stb and MH.get(li):
                        o[str(li)] = stb[("cap", li)][:, MH[li]].float().mean(1).tolist()   # [rows, N] matcher-head mean block scores
                rec["qk"][nm] = o
            keep("END", lg[0], st[0]); keep("START", lg[1], st[1])
            DE, DS = x1[0]["store"], x1[1]["store"]
            arms, ax, ae = [], [], []
            def put(rows, keys, donor):
                return dict(mode="put", rows=rows, idx=[TX.index(r) for r in rows], put=set(keys), donor=donor)
            K = lambda lo, hi, kinds=("mix", "mlp"): {(kd, l) for kd, l in ALL if kd in kinds and mp(lo) <= l <= mp(hi)}
            def add(nm, x, e):
                arms.append(nm); ax.append(x); ae.append(e)
            add("NOOP_E", put(TX, set(ALL), DE), "E"); add("NOOP_S", put(TX, set(ALL), DS), "S")
            for kd, suf in ((("mix", "mlp"), ""), (("mix",), "_mix"), (("mlp",), "_mlp")):
                add(f"X_QT31{suf}", put(G["QT"], K(31, 34, kd), DS), "E")
                add(f"X_B39{suf}", put(G["B"], K(39, 50, kd), DS), "E")
            add("X_QTB", dict(mode="put2"), "E")
            add("X_Bc39", put(G["B"], K(39, 63), DS), "E"); add("X_A47c", put(G["A"], K(47, 63), DS), "E")
            add("R_QT31", put(G["QT"], K(31, 34), DE), "S"); add("R_B39", put(G["B"], K(39, 50), DE), "S")
            add("R_QTB", dict(mode="put2"), "S")
            # the two-part arms: QT 31-34 and B 39-50 in one spec (disjoint rows)
            for i, nm in enumerate(arms):
                if nm in ("X_QTB", "R_QTB"):
                    don = DS if nm == "X_QTB" else DE
                    rows_ = G["QT"] + G["B"]
                    keys = {(kd, l, r) for kd, l in ALL for r in rows_ if (r in G["QT"] and mp(31) <= l <= mp(34)) or (r in G["B"] and mp(39) <= l <= mp(50))}
                    ax[i] = dict(mode="putrows", donor=don, rows=rows_, idx=[TX.index(r) for r in rows_], keys=keys)
            lg, st = runb([enc[e] for e in ae], ax, [capq() for _ in arms]); nrun += len(arms)
            for b, nm in enumerate(arms):
                keep(nm, lg[b], st[b])
            out.append(rec)
            del DE, DS
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n} · {nrun} runs", flush=True)
    TOOL.remove()
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), look=LOOK, matcher_layers=sorted(MH),
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=82700)
    main(ap.parse_args())
