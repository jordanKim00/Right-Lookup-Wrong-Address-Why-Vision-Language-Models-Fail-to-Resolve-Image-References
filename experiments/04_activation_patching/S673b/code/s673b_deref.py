"""S673b: at which layer does the ordinal token's state turn from an ADDRESS into CONTENT (the dereferencing step)?
Pre-registered 2026-10-08 before any data (knowledge/DECISIONS.md).  S673: everything the answer token carries is content; the
remaining candidate for a position-addressed lookup is the ordinal token's own lookup (S670b, L13-21 in 4B).
Permuted-scene interchange (as S673): base B and donor D show the same pictures in different orders, same menu.
Base question on B: END_k "What is in the k-th image from the last?" (k in {2,3,5,6}, j = 8 - k).
Donor question on D: FWD_j "What is in the j-th image?" -- a natural pointer j, no write; its prefix up to the ordinal is identical
to the base's, so the ordinal sits at the same position.  D[j] is placed in B at y_j not in {j, k}.
Patches into the base run (one per forward):
    ORD@L   the donor's whole ordinal-token residual (layer output) at layer L replaces the base's, L = 9 relative depths .06-.80
    ORD_Q   the donor's ordinal-token queries (q_norm, all heads) in the query layers above L_port up to relative .60   (descriptive)
    ORD_O   the outputs of the S670b ORD TOP8 heads at the ordinal (dense only)                                  (descriptive)
    SAME    same-scene instrument: the ordinal residual of FWD_j on B at L_port
Landing in B: j (untransformed address), k (base's own surface = also the 27B-transformed address N+1-j), y_j (donor content), other.
Registered rules (analysis exp/s673b_analyze.py), dense Qwen3-VL-4B and 8B:
    D0 instrument:    P(j | SAME) >= .50
    D1 EARLY_ADDRESS: at L_port, P(j) - P(y_j) >= .30
    D2 LATE_CONTENT:  some layer with P(y_j) - P(j) >= .15
    D3 DEREF_WINDOW:  the shallowest layer with P(y_j) > P(j) lies in the ordinal self-lookup window: 4B [13, 22], 8B [14, 22]
    DEREF_AT_ORDINAL = D1 and D2 and D3.  Predicted for 4B and 8B.  Qwen3.5-27B: described (expected: an early transplanted address is
    transformed, i.e. lands on k, and the content switch at the ordinal comes later or not at all).
Usage: python s673b_deref.py MODEL TAG [--qc QC_TAG] [--n 60] [--port P] [--device_map balanced]
"""
from __future__ import annotations

import argparse, os, json, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
OUT = "/vlm/results/r3/s673b_{}_deref.json"
PORT_DENSE = {"Qwen3-VL-4B-Instruct": 8, "Qwen3-VL-8B-Instruct": 13}
REL = (.06, .14, .22, .30, .38, .46, .54, .62, .80)


def main(a, v=None):
    v = v or (Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map) if a.device_map
              else Uni(a.model, device="cuda:0", attn="sdpa"))
    for p in v.model.parameters():
        p.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    nl = v.n_layer
    base = a.model.rstrip("/").split("/")[-1]
    L_port = a.port if a.port is not None else PORT_DENSE.get(base, int(round(.22 * (nl - 1))))
    LS = sorted({int(round(r * (nl - 1))) for r in REL} | {L_port})
    attn_layers = {li: L.self_attn for li, L in enumerate(v.layers) if hasattr(L, "self_attn") and hasattr(L.self_attn, "q_norm")}
    QW = [li for li in attn_layers if L_port < li <= int(round(.60 * (nl - 1)))]
    OH = [tuple(x) for x in json.load(open(f"/vlm/results/r3/s670b_{a.qc}_qcircuit.json"))["TOP8"]["ORD"]] if a.qc else []
    H = v.n_head
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tid = lambda s: ids1(s)[0]
    def aid(s):                                          # answer readout (S789): R3_READ=bare reads the bare token the model generates
        if os.environ.get("R3_READ") != "bare":
            return tid(s)
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    im_start = v.tok.convert_tokens_to_ids("<|im_start|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    ST = {}

    def mk_layer(li):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            if ST.get("grab"):
                ST["st"][li] = t[0, ST["ord"]].detach().clone()
            if ST.get("rep") is not None and ST["rep"][0] == li:
                t = t.clone(); t[0, ST["ord"]] = ST["rep"][1].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return h

    def mk_q(li):
        def h(m, i, o):
            if ST.get("grab"):
                ST["q"][li] = o[0, ST["ord"]].detach().clone()
            if li in ST.get("qrep", {}):
                o = o.clone(); o[0, ST["ord"]] = ST["qrep"][li].to(o.device, o.dtype)
            return o
        return h

    def mk_o(li):
        def pre(m, args):
            x = args[0]
            if ST.get("grab"):
                ST["o"][li] = x[0, ST["ord"]].detach().clone()
            rep = ST.get("orep", {}).get(li)
            if rep is not None:
                x = x.clone(); D = x.shape[-1] // H
                for hh, vec in rep:
                    x[0, ST["ord"], hh * D:(hh + 1) * D] = vec.to(x.device, x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return pre
    hooks = [v.layers[li].register_forward_hook(mk_layer(li)) for li in LS]
    hooks += [attn_layers[li].q_norm.register_forward_hook(mk_q(li)) for li in QW]
    olayers = sorted({li for li, _ in OH})
    hooks += [attn_layers[li].o_proj.register_forward_pre_hook(mk_o(li)) for li in olayers]

    def prompt(kind, q, menu):
        s = {"FWD": f"What is in the {ORDW[q-1]} image?", "AB": f"What is in the {ORDW[q-1]} image from the last?"}[kind]
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. {s} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run(kind, q, order, pic, menu, grab=False, rep=None, qrep=None, orep=None):
        inp = v.encode(prompt(kind, q, menu), [pic[l] for l in order])
        ids = inp["input_ids"][0].tolist()
        turn = max(i for i, t in enumerate(ids) if t == im_start)
        ST.clear(); ST["ord"] = max(i for i, t in enumerate(ids[:turn]) if t == tid(ORDW[q - 1]))
        if grab:
            ST["grab"] = True; ST["st"] = {}; ST["q"] = {}; ST["o"] = {}
        ST["rep"] = rep
        if qrep:
            ST["qrep"] = qrep
        if orep:
            ST["orep"] = orep
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        res = dict(lab=menu[int(torch.stack([sc(lg, x) for x in menu]).argmax())], pos=ST["ord"])
        if grab:
            res.update(st=ST["st"], q=ST["q"], o=ST["o"])
        ST.clear()
        return res

    def scene(rng):
        labs = rng.sample(labels, N + 2)
        pic = {l: proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs[:N]}
        menu = list(labs); rng.shuffle(menu)
        return labs[:N], pic, menu

    rng = random.Random(67350)
    rows = []
    print(f"[{a.tag}] L_port {L_port} · 서수 잔차 층 {LS} · 서수 쿼리 층 {QW[0] if QW else None}..{QW[-1] if QW else None} · ORD 헤드 {OH}", flush=True)
    while len(rows) < a.n:
        B, pic, menu = scene(rng)
        k = rng.choice((2, 3, 5, 6)); j = N + 1 - k
        for _ in range(200):
            Dm = B[:]; rng.shuffle(Dm)
            if B.index(Dm[j - 1]) + 1 not in (j, k):
                break
        else:
            continue
        posB = lambda lab: B.index(lab) + 1 if lab in B else 0
        rb = run("AB", k, B, pic, menu)
        rd = run("FWD", j, Dm, pic, menu, grab=True)
        rs = run("FWD", j, B, pic, menu, grab=True)
        assert rd["pos"] == rb["pos"] == rs["pos"]
        rec = dict(k=k, j=j, yj=posB(Dm[j - 1]), B=B, D=Dm, donor_ok=int(rd["lab"] == Dm[j - 1]), land=dict(NONE=posB(rb["lab"])))
        rec["land"]["SAME"] = posB(run("AB", k, B, pic, menu, rep=(L_port, rs["st"][L_port]))["lab"])
        for L in LS:
            rec["land"][f"ORD@{L}"] = posB(run("AB", k, B, pic, menu, rep=(L, rd["st"][L]))["lab"])
        if QW:
            rec["land"]["ORD_Q"] = posB(run("AB", k, B, pic, menu, qrep={li: rd["q"][li] for li in QW})["lab"])
        if OH:
            orep = {}
            D_ = rd["o"][olayers[0]].shape[-1] // H
            for li, hh in OH:
                orep.setdefault(li, []).append((hh, rd["o"][li][hh * D_:(hh + 1) * D_]))
            rec["land"]["ORD_O"] = posB(run("AB", k, B, pic, menu, orep=orep)["lab"])
        rows.append(rec)
        torch.cuda.empty_cache()
        if len(rows) % 20 == 0:
            print(f"[{a.tag}] {len(rows)}/{a.n}", flush=True)
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, L_port=int(L_port), LS=[int(x) for x in LS], QW=[int(x) for x in QW],
                   OH=[[int(a_), int(b_)] for a_, b_ in OH], rows=rows), open(OUT.format(a.tag), "w"), indent=1)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--qc", default=None); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device_map", default=None)
    main(ap.parse_args())
