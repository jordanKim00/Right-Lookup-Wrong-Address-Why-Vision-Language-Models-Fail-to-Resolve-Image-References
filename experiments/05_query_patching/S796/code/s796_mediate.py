"""S796 (pre-registered in knowledge/DECISIONS.md before any S796 data): does the frame read reach the answer through the lookup queries?
Frame -> query -> answer mediation on the END(4) / END(5) instrument of S769, with content swaps for the retrieval step.  Qwen3.5 only (gated full
attention recomputed by exp/c_qpath.py; complete-route read interchange of the frame word as exp/s769_end4.py, without its convolution patch).
Scenes: procedural pictures (160 px), N = 7, seed 79600 (new), 40 scenes; q in {4, 5}.  END(q) = "the q-th image from the last" (target N + 1 - q:
END(4) targets the literal image 4, END(5) targets image 3 with literal image 5) and START(q) = "... from the first" (target q), same pictures and
menu; the two prompts differ only in the frame word, so every position aligns.
Bands (mp maps 27B layers to other depths): P1 = L27-34, the frame-read band of S769; READ = every full-attention layer after P1 (L35, 39, .., 63).
Arms per (scene, q), rows = "?" .. answer:
  E, S    natural END(q) / START(q); capture the queries of the rows at every READ layer and the frame word's mixer input / output at every layer;
  NOOPQ   E given its own queries at every READ layer (faithfulness of the float32 recompute);
  NOOPF   E reads its own frame word at every layer (faithfulness of the interchange);
  FR      E reads S's frame word at P1 (S769 F4:P1);
  FRNQ    FR with the rows' attention recomputed at every READ layer from their own queries (FR under the recompute of FRQB);
  FRQB    FR with the rows' queries at every READ layer put back to E's (the frame read with its query route blocked);
  QS      E given S's queries at every READ layer (no frame read).
Content swaps: the picture of block p becomes a picture of the menu word absent from the scene (p = 4 for q = 4; p = 3 and p = 5 for q = 5); E, S,
FR, FRQB and QS rerun on the swapped scene with captures from the swapped scene.
Readout: the two-form menu readout of S794 (R3_READ in {bare, max} asserted), the vocabulary top-1 check, the landed block and the answered word;
block shares at the answer row at every READ layer, all heads and (with --heads) the index-matching heads of that layer.
Usage: python s796_mediate.py MODEL TAG [--n 40] [--heads JSON] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath import QPath

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s796_{}_mediate.json"


def main(a):
    assert os.environ.get("R3_READ") in ("bare", "max"), "S796 must read the generated form (FINDINGS §846, §851)"
    dev = os.environ.get("S796_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35", "QPath recomputes the gated full attention of Qwen3.5"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    mix = [L.self_attn if i in FA_L else L.linear_attn for i, L in enumerate(v.layers)]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    P1 = list(range(mp(26) + 1, mp(34) + 1))
    READ = [l for l in FA_L if l > P1[-1]]
    MH = {}
    if a.heads is None:                                  # the S786 index-matching heads of the model, when none are given
        a.heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
                   "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if a.heads:
        for l, h in json.load(open(a.heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def aid(s):                                          # answer readout: the bare token the model generates
        t = v.tok(s, add_special_tokens=False)["input_ids"]
        assert len(t) == 1, s
        return t[0]
    def sc(lg, s):                                       # S794: R3_READ=max scores a word by the larger of its bare and space-prefixed first tokens
        if os.environ.get("R3_READ") != "max":
            return lg[aid(s)]
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):                                       # S794: the first tokens greedy decoding could emit for the menu words, in either form
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

    X = dict(grab=False, f=None, swap={}, reset={}, g={})       # read interchange of the frame word (S758 / S769 instrument)

    def pre(li):
        def h(mod, args, kwargs):
            hs = kwargs["hidden_states"]
            if X["grab"]:
                X["g"][("in", li)] = hs[0, X["f"]].detach().clone()
            d = X["swap"].get(li)
            if d is None:
                return None
            hs = hs.clone(); hs[0, X["f"]] = d.to(hs.device, hs.dtype)
            kw = dict(kwargs); kw["hidden_states"] = hs
            return args, kw
        return h

    def post(li):
        def h(mod, args, kwargs, out):
            t = out[0] if isinstance(out, tuple) else out
            if X["grab"]:
                X["g"][("out", li)] = t[0, X["f"]].detach().clone()
            r = X["reset"].get(li)
            if r is None:
                return None
            t = t.clone(); t[0, X["f"]] = r.to(t.device, t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return h

    hooks = [mix[li].register_forward_pre_hook(pre(li), with_kwargs=True) for li in range(nl)]
    hooks += [mix[li].register_forward_hook(post(li), with_kwargs=True) for li in range(nl)]
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
            p = torch.softmax(st[("cap", li)][0].float(), -1)                 # [H, N]
            o[str(li)] = dict(all=p.mean(0).tolist())
            if li in MH:
                o[str(li)]["mh"] = p[MH[li]].mean(0).tolist()
        return o

    @torch.no_grad()
    def run(inp, labs_now, menu, qc, xs=None):
        X.update(grab=False, swap={}, reset={}, g={})
        if xs is not None:
            xs()
        TOOL.set(qc)
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); TOOL.set(None)
        g = X["g"]; X.update(grab=False, swap={}, reset={}, g={})
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return dict(land=(labs_now.index(lab) + 1 if lab in labs_now else 0), ans=lab), st, g

    def cell(pics_now, labs_now, menu, q, natural):
        iE, iS = v.encode(prompt(q, "last", menu), pics_now), v.encode(prompt(q, "first", menu), pics_now)
        idE, idS = iE["input_ids"][0].tolist(), iS["input_ids"][0].tolist()
        assert len(idE) == len(idS), "END and START prompts must align"
        d = [i for i, (x, y) in enumerate(zip(idE, idS)) if x != y]
        assert len(d) == 1, d
        f = d[0]
        assert v.tok.decode([idE[f]]).strip() == "last" and v.tok.decode([idS[f]]).strip() == "first", (v.tok.decode([idE[f]]), v.tok.decode([idS[f]]))
        S_ = len(idE)
        vs = [i for i, t in enumerate(idE) if t == vs_id]; ve = [i for i, t in enumerate(idE) if t == ve_id]
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([idE[i]]))
        assert qm > f
        rows = list(range(qm, S_))
        owner = [-1] * S_
        for b in range(N):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        cap = dict(rows=[len(rows) - 1], owner=owner, nb=N)
        X["f"] = f
        def grab():
            X["grab"] = True
        def fr(rg, dg, layers):                                         # recipient reads the donor's frame word at these layers
            def s():
                for li in layers:
                    X["swap"][li] = dg[("in", li)]; X["reset"][li] = rg[("out", li)]
            return s
        res, shr = {}, {}
        res["E"], stE, gE = run(iE, labs_now, menu, dict(rows=rows, cap=cap, grabq=set(READ)), grab)
        res["S"], stS, gS = run(iS, labs_now, menu, dict(rows=rows, cap=cap, grabq=set(READ)), grab)
        qE = {li: stE[("q", li)] for li in READ}; qS = {li: stS[("q", li)] for li in READ}
        shr["E"], shr["S"] = share(stE), share(stS)
        if natural:
            res["NOOPQ"], _, _ = run(iE, labs_now, menu, dict(rows=rows, edit={li: dict(qd=qE[li]) for li in READ}))
            res["NOOPF"], _, _ = run(iE, labs_now, menu, dict(rows=rows), fr(gE, gE, range(nl)))
            res["FRNQ"], _, _ = run(iE, labs_now, menu, dict(rows=rows, noop=set(READ)), fr(gE, gS, P1))
        res["FR"], st, _ = run(iE, labs_now, menu, dict(rows=rows, cap=cap), fr(gE, gS, P1)); shr["FR"] = share(st)
        res["FRQB"], st, _ = run(iE, labs_now, menu, dict(rows=rows, cap=cap, edit={li: dict(qd=qE[li]) for li in READ}), fr(gE, gS, P1))
        shr["FRQB"] = share(st)
        res["QS"], st, _ = run(iE, labs_now, menu, dict(rows=rows, cap=cap, edit={li: dict(qd=qS[li]) for li in READ})); shr["QS"] = share(st)
        return dict(pos=dict(f=f, qm=qm, S=S_), res=res, share=shr)

    print(f"[{a.tag}] P1 {P1[0]}-{P1[-1]} · READ {READ} · matcher layers {sorted(MH)}", flush=True)
    rng = random.Random(a.seed); out = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        nw = extra[0]
        nwpic = proc_image(random.Random(a.seed * 1000 + si).choice(pool[nw]))[0].resize((a.pixels, a.pixels))
        for q in a.qs:
            tgt = N + 1 - q
            rec = dict(si=si, q=q, tgt=tgt, lit=q, nw=nw, labs=labs, nat=cell(pics, labs, menu, q, True), swap={})
            for p in sorted({tgt, q}):
                pics_sw = list(pics); pics_sw[p - 1] = nwpic
                labs_sw = list(labs); labs_sw[p - 1] = nw
                rec["swap"][str(p)] = cell(pics_sw, labs_sw, menu, q, False)
            out.append(rec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (si + 1) % 5 == 0 or a.n < 5:
            print(f"[{a.tag}] {si + 1}/{a.n}", flush=True)
    TOOL.remove()
    for h in hooks:
        h.remove()
    json.dump(dict(tag=a.tag, model=a.model, readout=os.environ.get("R3_READ"), N=N, P1=P1, READ=READ, matcher_layers=sorted(MH),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=79600); ap.add_argument("--heads", default=None)
    ap.add_argument("--qs", type=int, nargs="+", default=[4, 5])
    main(ap.parse_args())
