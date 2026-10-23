"""S868 (registered in knowledge/DECISIONS.md before its data; user 10-21 'the complete algorithmic circuit of END'; the post hoc reading of S860's
k3 end-pin cells, §972 supplement) - near-end END answers come from the early end route, and the writer's k-switched step breaks an exact target.
One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (N 8 and 9, stated = shown).
  Natural END(k), k in {1, 2, 3} (targets N, N - 1, N - 2), and per k:
    PIN_NM1   the read rows' early queries (U part at 35-47) <- the natural START(N - 1) (a sharp N - 1 anchor, no end-route offset);
    PIN_TGT   U part <- the natural START(j*) (the exact target address);
    TGT_BANDR as PIN_TGT with the band's read-row outputs <- R(j*) = START(j*) pinned at its own address (the writer's END part removed);
    EMASK     the read rows may not read the last picture's positions at 35-47 (all heads; S705a / S707 moved the end with such masks);
    LMASK     the same mask at 51-63.
  References: natural END(k); R(N), R(N - 1), R(N - 2) (START pinned at their own addresses: the band donors).
Readouts per run: answer, menu-word logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores of
B1, B4, answer.  Usage: python s868_nearend.py MODEL TAG [--n 36] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK
from c_headpatch import HeadPatch, FAHeadPatchPinned, MLPInputPath

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s868_{}_{}.json"
ANTI = [0, 2, 5, 13, 15]; H16 = [16]; SELFADDR = [1, 4, 12, 17, 20]
_rest = [3, 6, 7, 8, 9, 10, 11, 14, 18, 19, 21, 22, 23]
_rr = random.Random(860)
RAND5 = [sorted(_rr.sample(_rest, 5)) for _ in range(2)]; RAND1 = [[h] for h in _rr.sample(_rest, 2)]   # the S860 draws
LAYS = (8, 9); KS = (1, 2, 3)


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S863_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    band = a.band
    FA_W = [l for l in band if l in FA_L]
    F47 = FA_W[0]
    DNL = [l for l in band if l not in FA_L]
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    F43 = [l for l in EARLY if l < F47][-1]
    COMPS = {"BAND": ALLC}
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    assert NH == 24
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
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
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=())
    FHP = FAHeadPatchPinned(v, [F47])                                                     # after TOOL: corrects the rows its pins wrote
    MIP = MLPInputPath(v, [F47])                                                          # after FHP: MLP47 may read a donor's attention
    T1 = []

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?",
             "BEF": lambda q: f"What is in the image just before the {ORDW[q-1]}?", "AFT": lambda q: f"What is in the image just after the {ORDW[q-1]}?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, fhs, mps):
        TOOL.set(qs); FHP.set(fhs); MIP.set(mps)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); FHP.set(None); MIP.set(None)
        return lg, st

    rng = random.Random(a.seed); nrun = 0; recs = []
    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · layouts {LAYS} · k {KS}", flush=True)
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        rec = dict(si=si, labs=labs, menu=menu, land={}, mlog={}, qk={}, zl={}, subpass={})
        for N in LAYS:
            names = {**{f"END{k}": ("END", k) for k in KS}, **{f"START{q}": ("START", q) for q in (N, N - 1, N - 2)}}
            enc = {nm: v.encode(prompt(f, q, menu, N), pics[:N]) for nm, (f, q) in names.items()}
            ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
            assert len({len(x) for x in ids.values()}) == 1, "prompts must align"
            ref = next(iter(ids.values())); S_ = len(ref)
            ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
            uend = max(i for i, x in enumerate(ref) if x == end_id)
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            A4 = list(range(uend + 5, S_)); assert len(A4) == 4
            opt = []
            for w_ in menu:
                pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
                assert len(pos_) == 1
                opt.append(pos_[0])
            rowsC = [qm] + list(range(uend, uend + 5)) + A4
            last = [i for i, o in enumerate(owner) if o == N - 1]
            capq = lambda **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=owner, nb=N, opt=opt), **kw)
            # donors: natural START(N), START(N - 1), START(N - 2)
            qn = [N, N - 1, N - 2]
            inps = [enc[f"START{q}"] for q in qn]
            lg0, st0 = run1(inps, [capq(grabq=set(EARLY)) for _ in qn], [None] * 3, [None] * 3)
            qd = {}
            for b, q in enumerate(qn):
                qd[q] = {li: st0[b][("q", li)] for li in EARLY}
                w = menu[int(torch.stack([sc(lg0[b], c) for c in menu]).argmax())]
                rec["land"][f"N{N}_START{q}"] = labs.index(w) + 1
                T1.append(int(int(lg0[b].argmax()) in t1ids(menu)))
            nrun += 3
            del st0; gc.collect()
            pin = lambda q: {li: dict(qsub=dict(owner=owner, nb=N, part="U", donors=[(1.0, qd[q][li])])) for li in EARLY}
            inps, qs, fhs, mps, tag = [], [], [], [], []
            def add(nm, encname, q):
                inps.append(enc[encname]); qs.append(q); fhs.append(None); mps.append(None); tag.append(nm)
                return len(inps) - 1
            idx = {}
            for q in qn:                                                          # band donors R(q): START(q) pinned at its own address
                idx[f"R{q}"] = add(f"R{q}", f"START{q}", capq(edit=pin(q)))
            for k in KS:
                jt = N + 1 - k
                add(f"END{k}", f"END{k}", capq(grabq=set(EARLY)))
                add(f"END{k}_PIN_NM1", f"END{k}", capq(edit=pin(N - 1)))
                add(f"END{k}_PIN_TGT", f"END{k}", capq(edit=pin(jt)))
                q_ = capq(edit=pin(jt)); q_["cpatch"] = dict(src=idx[f"R{jt}"], comps=ALLC)
                add(f"END{k}_TGT_BANDR", f"END{k}", q_)
                add(f"END{k}_EMASK", f"END{k}", capq(ko=dict(layers=set(EARLY), rows=list(range(10)), pos=last)))
                add(f"END{k}_LMASK", f"END{k}", capq(ko=dict(layers=set(LATE), rows=list(range(10)), pos=last)))
            lg, st = run1(inps, qs, [None] * len(inps), [None] * len(inps))
            for b, nm in enumerate(tag):
                lgb, stb = lg[b], st[b]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                key = f"N{N}_{nm}"
                rec["land"][key] = labs.index(w) + 1
                rec["mlog"][key] = [float(sc(lgb, x)) for x in labs[:N]]
                rec["qk"][key] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                rec["zl"][key] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
                nrun += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        recs.append(rec)
        print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs", flush=True)
    TOOL.remove(); FHP.remove(); MIP.remove()
    json.dump(dict(tag=a.tag, model=a.model, lays=list(LAYS), ks=list(KS), band=band,
                   read=READ, early=EARLY, late=LATE, late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"),
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, rows=recs),
              open(OUT.format(a.tag, "nearend"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'nearend')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=36); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=86800); ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    ap.add_argument("--size", type=int, default=10)
    main(ap.parse_args())
