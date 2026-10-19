"""S855 (Gap 2, second stage; pre-registered in knowledge/DECISIONS.md before any S855 data).  Does the actual-count dependence of the late lookup
come from positional geometry (where the question sits relative to the pictures under multimodal RoPE) or from the extra pictures themselves?
Pictures 1-6 occupy the same positions and keys whether 6 or 8 pictures are shown; with 8 shown, every text token after the pictures sits D
positions later (D = the position advance of two pictures).  Arms (stated count 8 in every prompt, so the text is identical; early lookup pinned
to p as in S854-1):
  E6       6 pictures, natural positions                  E8       8 pictures, natural positions (the reference)
  E6S      6 pictures, text after the pictures moved +D at every layer (the 8-picture geometry with the 6-picture content)
  E6SL     6 pictures, +D only in the rotary embedding of the late full-attention layers (51, 55, 59, 63)
  E8S      8 pictures, text moved -D at every layer (the 6-picture geometry with pictures 7-8 still present)
  E8SL     8 pictures, -D only at the late full-attention layers
Pin donors START(p) are computed with the same position shift as their recipients (E6S / E8S: shifted donors; E6SL / E8SL: natural donors, as
the shift starts after the early layers).  k in {3, 5}, p in {3, 4}; 30 new scenes (seed 85600).  Exactness: E6Z / E8Z = explicit natural position ids in
the same batch; E6_k5p3N / E8_k5p3N = the same pinned run alone with internal position ids,
E.._k5p3NX = alone with explicit natural ids (NX must reproduce N; the batched arms differ from both by batch-size numerics only).  Readouts: answer, menu-word logits, positional scores of every read row, picture mass; late queries
(51-63) and writer outputs (layers 46-49) at rows B1, B4, answer.
Usage: python s855_geom.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_j import QPathJ

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s855_{}_{}.json"
NS = 8; KS = (3, 5); PS = (3, 4)


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S855_DEVICE", "cuda:0")
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
    band = a.band if nl == 64 else sorted({int(round(b * (nl - 1) / 63)) for b in a.band} - set(LATE))
    KROWS = [2, 5, 9]
    COMPS = [(l_, kd) for l_ in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
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
    TOOL = QPathJ(v, READ, res_layers=(), comp_layers=band, patch_comps=())
    base = v.model.model                                                    # Qwen3_5Model: get_rope_index, language_model.rotary_emb
    rot = base.language_model.rotary_emb
    T1 = []
    LATE_SHIFT = {}                                                         # batch index -> (cos, sin) rows to use at the late FA layers
    def late_pre(li):
        def h(mod, args, kwargs):
            if not LATE_SHIFT or "position_embeddings" not in kwargs:
                return None
            cos, sin = kwargs["position_embeddings"]; cos2, sin2 = cos.clone(), sin.clone()
            for b, (cb, sb) in LATE_SHIFT.items():
                cos2[b] = cb.to(cos2.device, cos2.dtype); sin2[b] = sb.to(sin2.device, sin2.dtype)
            kwargs = dict(kwargs); kwargs["position_embeddings"] = (cos2, sin2)
            return args, kwargs
        return h
    hooks = [v.layers[li].self_attn.register_forward_pre_hook(late_pre(li), with_kwargs=True, prepend=True) for li in LATE]   # same kwargs as the readers

    def prompt(na, q, frame, menu):
        return v.chat([("user", f"{v.IMG*na}\nThere are {NS} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    def posids(bt):
        pid, _ = base.get_rope_index(bt["input_ids"], mm_token_type_ids=bt["mm_token_type_ids"], image_grid_thw=bt.get("image_grid_thw"),
                                     attention_mask=bt.get("attention_mask"))
        return pid

    @torch.no_grad()
    def run1(inps, qs, shifts):
        """shifts[b] = (kind, D, t0): kind in {None, 'all', 'late', 'zero'}; text positions >= t0 move by D (all layers or late FA layers)."""
        bt = batch_inputs(inps)
        pid = posids(bt); pid2 = pid.clone()
        for b, (kind, D, t0) in enumerate(shifts):
            if kind in ("all", "late"):
                pid2[:, b, t0:] += D
        extra = {}
        if any(k_ in ("all", "zero") for k_, _, _ in shifts):
            pall = pid.clone()
            for b, (kind, D, t0) in enumerate(shifts):
                if kind == "all":
                    pall[:, b, t0:] += D
            extra["position_ids"] = pall
        LATE_SHIFT.clear()
        if any(k_ == "late" for k_, _, _ in shifts):
            dummy = torch.zeros(1, device=rot.inv_freq.device, dtype=torch.float32)
            cos_s, sin_s = rot(dummy, pid2.to(dummy.device))
            for b, (kind, D, t0) in enumerate(shifts):
                if kind == "late":
                    LATE_SHIFT[b] = (cos_s[b], sin_s[b])
        TOOL.set(qs)
        try:
            lg = v.model(**bt, **extra, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); LATE_SHIFT.clear()
        return lg, st

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band}", flush=True)
    rng = random.Random(a.seed); nrun = 0; npz = {}; keyrel = []
    scenes = []
    for si in range(a.n):
        labs = rng.sample(labels, 8); extra_w = [l for l in labels if l not in labs][:1]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra_w; rng.shuffle(menu)
        scenes.append(dict(si=si, labs=labs, words=labs + extra_w, menu=menu, pics=pics))
    recs = {s["si"]: dict(si=s["si"], labs=s["labs"], words=s["words"], menu=s["menu"], land={}, mlog={}, qk={}, mass={}) for s in scenes}

    def measure(rec, lgb, stb):
        menu = rec["menu"]
        T1.append(int(int(lgb.argmax()) in t1ids(menu)))
        w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
        hs_ = lambda li, H: MH[li] if MH.get(li) else list(range(H))
        return dict(land=rec["labs"].index(w) + 1 if w in rec["labs"] else 0, mlog=[float(sc(lgb, x)) for x in rec["words"]],
                    qk={str(li): stb[("cap", li)][:, hs_(li, stb[("cap", li)].shape[1])].float().mean(1).tolist() for li in READ if ("cap", li) in stb},
                    mass={str(li): stb[("img", li)][:, hs_(li, stb[("img", li)].shape[1])].float().mean(1).tolist() for li in READ if ("img", li) in stb})

    def put(rec, nm, m):
        for key in ("land", "mlog", "qk", "mass"):
            rec[key][nm] = m[key]

    def build_scene(s, na):
        nms = {**{f"END_k{k}": (k, "last") for k in KS}, **{f"START_q{q}": (q, "first") for q in (3, 4, 5)}}
        enc = {nm: v.encode(prompt(na, q, fr, s["menu"]), s["pics"][:na]) for nm, (q, fr) in nms.items()}
        ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
        assert len({len(x) for x in ids.values()}) == 1
        ref = ids["END_k5"]; S_ = len(ref)
        ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
        assert len(ve) == na
        qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
        uend = max(i for i, x in enumerate(ref) if x == end_id)
        owner = [-1] * S_
        for b in range(na):
            for p_ in range(vs[b], ve[b] + 1):
                owner[p_] = b
        A4 = list(range(uend + 5, S_)); assert len(A4) == 4
        opt = []
        for w_ in s["menu"]:
            pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
            assert len(pos_) == 1
            opt.append(pos_[0])
        return dict(enc=enc, S=S_, owner=owner, t0=ve[-1] + 1, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, na=na)
    # the position advance D of pictures 7-8 (first text position after the pictures), measured once
    s0 = scenes[0]; p6 = build_scene(s0, 6); p8 = build_scene(s0, 8)
    pid6 = posids(batch_inputs([p6["enc"]["END_k5"]])); pid8 = posids(batch_inputs([p8["enc"]["END_k5"]]))
    D = int(pid8[0, 0, p8["t0"]] - pid6[0, 0, p6["t0"]]); assert D > 0
    print(f"[{a.tag}] position advance of pictures 7-8: D = {D} (t0: 6 pictures {p6['t0']}, 8 pictures {p8['t0']})", flush=True)
    BA = list(range(10)); CB = set(band); LQ = set(LATE)
    for gi, s in enumerate(scenes):
        rec = recs[s["si"]]
        for na, sgn in ((6, +1), (8, -1)):
            p = build_scene(s, na)
            capq = lambda **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=na, opt=p["opt"]), **kw)
            # stage 1: END(k) and START(q) natural and with the all-layer shift (pin donors)
            inps, qs, shifts, tag = [], [], [], []
            for kind in (None, "all"):
                for nm in ("END_k3", "END_k5", "START_q3", "START_q4", "START_q5"):
                    inps.append(p["enc"][nm]); qs.append(capq(grabq=set(EARLY), grabkb=set(LOOK)))
                    shifts.append((kind, sgn * D, p["t0"])); tag.append(f"{nm}_a{na}" + ("" if kind is None else "_S"))
            lg, st = run1(inps, qs, shifts); don = {}
            for b, nm in enumerate(tag):
                put(rec, nm, measure(rec, lg[b], st[b])); nrun += 1
                don[nm] = dict(q={li: st[b][("q", li)] for li in EARLY}, kb={li: st[b][("kb", li)] for li in LOOK})
            kr = max(max(float((don[nm]["kb"][li] - don[f"END_k5_a{na}"]["kb"][li]).abs().max() / don[f"END_k5_a{na}"]["kb"][li].abs().max()) for li in LOOK)
                     for nm in tag if not nm.endswith("_S"))
            keyrel.append(kr)
            def ed(p_, shifted):
                dq = don[f"START_q{p_}_a{na}" + ("_S" if shifted else "")]["q"]
                return {li: [dict(qsub=dict(owner=p["owner"], nb=na, part="U", donors=[(1.0, dq[li][BA])]), erows=BA)] for li in EARLY}
            # stage 2: pinned arms
            inps, qs, shifts, tag = [], [], [], []
            inps.append(p["enc"]["END_k5"]); qs.append(capq(noop=set(LOOK))); shifts.append((None, 0, p["t0"])); tag.append(f"QNOOP_a{na}")
            for k in KS:
                for p_ in PS:
                    base_nm = f"E{na}_k{k}p{p_}"
                    for kind, nm, shifted in ((None, base_nm, False), ("zero", base_nm + "Z", False), ("all", base_nm + "S", True), ("late", base_nm + "SL", False)):
                        inps.append(p["enc"][f"END_k{k}"])
                        qs.append(capq(edit=ed(p_, shifted), keepcomp=CB, grabcomp=CB, grabq=LQ)); shifts.append((kind, sgn * D, p["t0"])); tag.append(nm)
            lg, st = run1(inps, qs, shifts)
            # exactness of explicit position ids, free of batch-composition effects: the same pinned run alone with internal position ids (N)
            # and alone with explicit natural position ids (NX); NX must reproduce N (the batched E arms differ from both by batch numerics only)
            for kind_, sfx in ((None, "N"), ("zero", "NX")):
                lgN, stN = run1([p["enc"]["END_k5"]], [capq(edit=ed(3, False))], [(kind_, 0, p["t0"])])
                put(rec, f"E{na}_k5p3{sfx}", measure(rec, lgN[0], stN[0])); nrun += 1
            for b, nm in enumerate(tag):
                put(rec, nm, measure(rec, lg[b], st[b])); nrun += 1
                if ("mix", band[0]) in st[b]:
                    npz[f"s{s['si']}_{nm}_comp"] = np.stack([st[b][(kd, l_)][KROWS].float().numpy() for l_, kd in COMPS]).astype(np.float16)
                if ("q", LATE[0]) in st[b]:
                    npz[f"s{s['si']}_{nm}_q"] = np.stack([st[b][("q", li)][KROWS].float().numpy() for li in LATE]).astype(np.float16)
            del don
        if gi % 5 == 4 or gi == len(scenes) - 1:
            print(f"[{a.tag}] {gi + 1}/{len(scenes)} scenes · {nrun} runs · key diff max {max(keyrel):.2e}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    TOOL.remove()
    nm_ = "geom"
    json.dump(dict(tag=a.tag, model=a.model, D=D, ks=KS, ps=PS, band=band, comps=COMPS, readout=os.environ.get("R3_READ"), read=READ, look=LOOK,
                   early=EARLY, late=LATE, krows=KROWS, matcher_layers=sorted(MH), n_layer=nl,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"],
                   t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), key_rel_diff_max=max(keyrel), n_runs=nrun,
                   rows=[recs[k] for k in sorted(recs)]), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=85600)
    ap.add_argument("--band", type=int, nargs="+", default=[46, 47, 48, 49])
    main(ap.parse_args())
