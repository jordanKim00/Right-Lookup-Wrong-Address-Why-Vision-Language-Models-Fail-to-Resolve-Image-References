"""S826 (pre-registered in knowledge/DECISIONS.md before any S826 data): where is the stated count read, and which rows hold it?  FINDINGS §916
(S824): the end-relative answer follows the stated count ('There are N images in a row.') more than the pictures shown - with 8 pictures, END(7)
answers the first picture when told '7' (.85) and not when told '8' (0).  Minimal pair (the S815 design with the count token in place of the frame
word): recipient END(k) told 8, donor END(k) told 7, the same 8 pictures; the two prompts differ only at the count token, so a state swap at it is
exact from that layer on.
Per (scene, k), k in {5, 6, 7}, 8 pictures shown:
  natural  END8(k) (told 8), END7(k) (told 7) (residuals at the inputs of the swap layers captured at every row from the count token on)
  NOOP     END8 <- its own count-token residual at the input of layer 27 (exactness)
  SRC_l    the count token's residual from END7 at the input of layer l, l in {0, 7, 15, 23, 27, 31, 35, 39, 43, 47}
  QT_l     every row from the count token through the question mark from END7, l in {27, 35, 43}
  QMB_l    every row from the count token through the five boundary rows from END7, l in {27, 35, 43, 51}
  ANS_l    the answer-side rows from END7, l in {35, 43, 51};  BD_l  the five boundary rows from END7, l in {35, 43}
  KO_l     the answer-side rows cannot attend to the count token at full-attention layers >= l, l in {27, 35}
Readout: two-form menu readout (S794), t1, the two-form logits of the menu words.
Scenes: procedural pictures (160 px), 8 pictures + 1 extra word, seed 82600 (new), 40 scenes.
Usage: python s826_count_window.py MODEL TAG [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_b import QPathB

N = 8
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s826b_{}_count.json"
STEP = {3: 4, 4: 3, 5: 4, 6: 5, 7: 6}


def pick_d(k):
    j = N + 1 - k
    return next(d for d in (6, 5, 3, 4) if d not in (k, j, 2, STEP[k]))


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max", "S826 reads the generated form in either tokenization (S794)"
    dev = os.environ.get("S826_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam in ("qwen35", "qwen3vl")
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""            # Qwen3-VL Instruct: no think markers
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    dep = lambda l: l / (nl - 1)
    LOOK = [l for l in FA_L if .555 <= dep(l) <= .875 and (v.fam == "qwen35" or l % 4 == 3)]   # dense: the 27B's grid
    SRC_L = sorted({0} | {mp(l) for l in (7, 15, 23, 27, 31, 35, 39, 43, 47)})
    ANS_L = sorted({mp(l) for l in (35, 43, 51)})
    QMB_L = sorted({mp(l) for l in (27, 35, 43, 51)})
    PART_L = sorted({mp(l) for l in (27, 35, 43)})
    ANSD_L = []
    KO_L = sorted({mp(l) for l in (27, 35)})
    CAPL = sorted(set(SRC_L) | set(ANS_L) | set(QMB_L) | set(PART_L) | set(ANSD_L))
    READERS = sorted(set(LOOK) | {l for l in FA_L if l >= min(KO_L)})
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json", "Qwen3-VL-32B-Instruct": "/vlm/exp/s786_heads/q3vl_32b.json",
             "Qwen3-VL-8B-Instruct": "/vlm/exp/s786_heads/q3vl_8b.json", "Qwen3-VL-4B-Instruct": "/vlm/exp/s786_heads/q3vl_4b.json"}.get(
        os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    MH = {l: h for l, h in MH.items() if l in LOOK}
    MFB = not MH                                                               # no matcher heads at the look layers: all heads (flagged)
    if MFB:
        nh_ = v.layers[LOOK[0]].self_attn.config.num_attention_heads
        MH = {l: list(range(nh_)) for l in LOOK}
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

    RB = dict(active=False, specs=None)                                       # batched residual capture / swap at layer inputs

    def pre(li):
        def h(mod, args, kwargs):
            if not RB["active"]:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            new = None
            for b, sp in enumerate(RB["specs"]):
                if sp is None:
                    continue
                for r in sp.get("cap", {}).get(li, ()):
                    sp["store"][(li, r)] = hs[b, r].detach().clone()
                for r, val in sp.get("put", {}).get(li, {}).items():
                    if new is None:
                        new = hs.clone()
                    new[b, r] = val.to(hs.device, hs.dtype)
            if new is None:
                return None
            if args:
                return (new,) + tuple(args[1:]), kwargs
            kw = dict(kwargs); kw["hidden_states"] = new
            return args, kw
        return h
    hooks = [v.layers[li].register_forward_pre_hook(pre(li), with_kwargs=True) for li in CAPL]
    TOOL = QPathB(v, READERS)
    T1 = []

    def prompt(q, frame, menu, ns=N):
        return v.chat([("user", f"{v.IMG*N}\nThere are {ns} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def runb(inps, rs, qs):
        RB.update(active=any(r is not None for r in rs), specs=rs); TOOL.set(qs)
        lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
        st = TOOL.store; TOOL.set(None); RB.update(active=False, specs=None)
        return lg, st

    def readout(stb):
        o = {}
        for li in LOOK:
            if ("cap", li) not in stb or not MH.get(li):
                continue
            hh = MH[li]; scr = stb[("cap", li)].float()                     # [3 cap rows, H, nb]
            blk = torch.softmax(scr, -1)
            o[str(li)] = dict(blk=blk[:, hh].mean(1).tolist(), score=scr[:, hh].mean(1).tolist())
        return o

    def fidx(stb, qE, qS):
        o = {}
        for li in LOOK:
            if ("q", li) not in stb or not MH.get(li):
                continue
            hh = MH[li]; q = stb[("q", li)][:, hh].double(); e = qE[li][:, hh].double(); s = qS[li][:, hh].double()
            d = s - e
            o[str(li)] = [float(((q[r] - e[r2]) * d[r2]).sum() / max(float((d[r2] * d[r2]).sum()), 1e-12)) for r, r2 in ((0, 0), (-1, -1))]
        return o

    print(f"[{a.tag}] look {LOOK} · SRC {SRC_L} · ANS {ANS_L} · QMB {QMB_L} · parts {PART_L} · ANSd {ANSD_L} · KO {KO_L} (readers {READERS}) · "
          f"matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        words = labs + extra
        for k in a.ks:
            j = N + 1 - k
            texts = dict(E=prompt(k, "last", menu, 8), S=prompt(k, "last", menu, 7))
            enc = {nm: v.encode(tx, pics) for nm, tx in texts.items()}
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1, "the prompts must align"
            S_ = len(ids["E"])
            ve = [i for i, t in enumerate(ids["E"]) if t == ve_id]; vs = [i for i, t in enumerate(ids["E"]) if t == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["E"][i]]))
            uend = max(i for i, t in enumerate(ids["E"]) if t == end_id)
            assert v.tok.decode([ids["E"][uend + 3]]).strip() == "assistant"
            F = [i for i, (x, y) in enumerate(zip(ids["S"], ids["E"])) if x != y]
            assert F and len(F) == 1 and min(F) > ve[-1] and max(F) < qm, "the count token must sit between the pictures and the question mark"
            f0 = F[0]
            G = (dict(A=list(range(uend + 5, S_)), B=list(range(uend, uend + 5))) if THINK      # Qwen3.5: think markers + answer row after B
                 else dict(A=[S_ - 1], B=list(range(uend, S_ - 1))))                    # Qwen3-VL: the answer row ends the boundary rows
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            rows3 = [qm, uend, S_ - 1]
            opt = [next(i for i in range(qm, uend) if ids["E"][i] == tid(w)) for w in words]
            capcfg = lambda: dict(rows=rows3, cap=dict(rows=[0, 1, 2], owner=owner, nb=N, opt=opt), grabq=set(LOOK))
            rowsK = [qm, uend] + G["A"]                                       # knockout arms: the answer-side rows are recomputed
            kocfg = lambda l: dict(rows=rowsK, cap=dict(rows=[0, 1, len(rowsK) - 1], owner=owner, nb=N, opt=opt), grabq=set(LOOK),
                                   ko=dict(layers={x for x in READERS if x >= l}, rows=list(range(2, len(rowsK))), pos=F))
            caprows = list(range(f0, S_))
            # batch 1: natural runs
            rs = [dict(cap={L_: caprows for L_ in CAPL}, store={}), dict(cap={L_: caprows for L_ in CAPL}, store={})]
            lg, st = runb([enc["E"], enc["S"]], rs, [capcfg(), capcfg()]); nrun += 2
            rE, rS = rs[0]["store"], rs[1]["store"]
            qE = {li: st[0][("q", li)] for li in LOOK if ("q", li) in st[0]}; qS = {li: st[1][("q", li)] for li in LOOK if ("q", li) in st[1]}
            rec = dict(si=si, k=k, j=j, count_row=F, land={}, mlog={}, read={}, fidx={})
            def keep(nm, lgb, stb):
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                word = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = labs.index(word) + 1 if word in labs else 0
                rec["mlog"][nm] = [float(sc(lgb, w)) for w in words]; rec["read"][nm] = readout(stb); rec["fidx"][nm] = fidx(stb, qE, qS)
            keep("END", lg[0], st[0]); keep("START", lg[1], st[1])
            # batch 2: swaps on END(k)
            arms, ar, aq = [], [], []
            def add(nm, put, q=None):
                arms.append(nm); ar.append(None if put is None else dict(put=put)); aq.append(q or capcfg())
            pt = lambda l, rws, src: {l: {r: src[(l, r)] for r in rws}}
            add("NOOP", pt(mp(27), F, rE))
            for l in SRC_L:
                add(f"SRC_{l}", pt(l, F, rS))
            for l in ANS_L:
                add(f"ANS_{l}", pt(l, G["A"], rS))
            for l in QMB_L:
                add(f"QMB_{l}", pt(l, list(range(f0, uend + 5)), rS))
            for l in PART_L:
                add(f"QT_{l}", pt(l, list(range(f0, qm + 1)), rS))
                if l != mp(27):
                    add(f"BD_{l}", pt(l, G["B"], rS))
            for l in KO_L:
                add(f"KO_{l}", None, kocfg(l))
            lg2, st2 = runb([enc["E"]] * len(arms), ar, aq); nrun += len(arms)
            for b, nm in enumerate(arms):
                keep(nm, lg2[b], st2[b])
            out.append(rec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n} · {nrun} runs", flush=True)
    TOOL.remove()
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), n_layer=nl, look=LOOK, src_layers=SRC_L, ans_layers=ANS_L, qmb_layers=QMB_L,
                   part_layers=PART_L, ansd_layers=ANSD_L, ko_layers=KO_L, readers=READERS, matcher_layers=sorted(MH), matcher_fallback=MFB, cap_rows=["question_mark", "boundary0", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=82610); ap.add_argument("--ks", type=int, nargs="+", default=[5, 6, 7])
    main(ap.parse_args())
