"""S833 (pre-registered in knowledge/DECISIONS.md before any S833 data): does R3's frame rebind work by turning off END's late re-lookup?
Phase-1 mechanism (§921-§929): END re-reads pictures late (L51-55) and moves the early-found position a fixed step toward its end-relative
target, even at the target.  Predictions: writing the frame END -> START (R3's Rebind of the frame) turns the late re-reading off (late picture
mass START-like) so the early lookup's position stands; writing only the ordinal k -> j leaves the END frame on, the early lookup moves to j and
the late step moves it toward the end-relative target of the written ordinal (N + 1 - j = k), i.e. toward k; both writes give j.
Cells (N, k, j): identity N = 7 END(4), N = 9 END(5); non-identity N = 7 END(5) (j = 3), N = 9 END(7) (j = 3).  Settings exactly as the main
series; seed 83400 (new), 30 scenes per cell.  Writes as R3 / S711's oracle (27B calibration results/r3/s709_r327b_cal.pt, write layer Lb):
d_ord = Mb[j] - Mb[k] added at the ordinal token, d_fr = mu_start - mu_last at the frame word, both to the residual output of layer Lb.
Runs per scene (one batch, no donors): END (natural), START (natural START(k), mass reference), WNOOP (zero writes at the same positions;
exactness), O_FR (frame write), and in non-identity cells also O_ORD (ordinal write) and O_ORD_FR (both).
Read: landing (two-form readout), every head's block scores, gated output, picture attention mass and menu-word attention at the read rows.
Usage: python s833_r3mode.py MODEL TAG [--n 30] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_b import QPathB
from pipe_run import pipelined

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s833_{}_{}.json"
CASES = [(7, 4, 4), (9, 5, 5), (7, 5, 3), (9, 7, 3)]
NAT1 = ["END", "START"]
RUNS2 = 14


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S833_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    LOOK = [l for l in FA_L if .555 <= l / (nl - 1) <= .875]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    scaling = float(v.layers[LOOK[0]].self_attn.scaling)
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
    TOOL = QPathB(v, LOOK)
    st_ = torch.load("/vlm/results/r3/s709_r327b_cal.pt", weights_only=False); Lb = int(st_["Lb"]); calM = st_["cal"]["Mb"]; frc = st_["cal"]["frame"]
    d_fr = (frc["mu_start"] - frc["mu"]["last"]).float()
    WR = dict(specs=None)
    def wr_hook(mod, args, out):
        if WR["specs"] is None:
            return None
        t = out[0] if isinstance(out, tuple) else out
        t2 = None
        for b, sp in enumerate(WR["specs"]):
            if not sp:
                continue
            if t2 is None:
                t2 = t.clone()
            for pos, dlt in sp:
                t2[b, pos] = (t[b, pos].float() + dlt.to(t.device)).to(t.dtype)
        if t2 is None:
            return None
        return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
    whook = v.layers[Lb].register_forward_hook(wr_hook)
    T1 = []

    def prompt(N, q, frame, menu):
        return v.chat([("user", f"{v.IMG*N}\nThere are {N} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
                                f"Answer with only one word from: {', '.join(menu)}.")], prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs, ws=None):
        TOOL.set(qs); WR["specs"] = ws
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
            st = TOOL.store
        finally:
            TOOL.set(None); WR["specs"] = None
        return lg, st

    def runb(inps, qs, ws):
        """One batch; on out-of-memory split in halves (each sample's computation does not depend on its batch neighbours).  The retry runs
        outside the except block, so the failed attempt's activations (held by the traceback) are freed first."""
        try:
            return run1(inps, qs, ws)
        except torch.cuda.OutOfMemoryError:
            if len(inps) == 1:
                raise
        gc.collect(); torch.cuda.empty_cache()
        h_ = len(inps) // 2
        print(f"[{a.tag}] OOM at batch {len(inps)} -> {h_} + {len(inps) - h_}", flush=True)
        l1, s1 = runb(inps[:h_], qs[:h_], ws[:h_]); l2, s2 = runb(inps[h_:], qs[h_:], ws[h_:])
        return torch.cat([l1, l2], 0), {**s1, **{b + h_: x for b, x in s2.items()}}

    print(f"[{a.tag}] look {LOOK} · matcher layers {sorted(MH)} · scaling {scaling}", flush=True)
    rng = random.Random(a.seed); out = []; nrun = 0; npz = {}
    scenes = []
    for si in range(a.n):                                                   # the main-series scene recipe, per case
        for N, k, j in CASES:
            labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:9 - N]
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = labs + extra; rng.shuffle(menu)
            scenes.append(dict(si=si, N=N, k=k, j=j, labs=labs, words=labs + extra, menu=menu, pics=pics))
    keyrel = []
    for N, k, j in CASES:
        SC = [s for s in scenes if s["N"] == N]
        SC = [s_ for s_ in SC if s_["k"] == k]
        G = a.group or max(1, a.max_images // (6 * N))   # scenes per batch: pictures per batch <= max_images
        print(f"[{a.tag}] N{N}: {G} scenes per batch", flush=True)
        for g0 in range(0, len(SC), G):
            grp = SC[g0:g0 + G]
            P = []
            for s in grp:
                qk_ = dict(END=(k, "last"), START=(k, "first"))
                enc = {nm: v.encode(prompt(N, q, fr, s["menu"]), s["pics"]) for nm, (q, fr) in qk_.items()}
                ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
                assert len({len(x) for x in ids.values()}) == 1, "the prompts of a scene must align"
                S_ = len(ids["END"])
                ve = [i for i, x in enumerate(ids["END"]) if x == ve_id]; vs = [i for i, x in enumerate(ids["END"]) if x == vs_id]
                qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids["END"][i]]))
                uend = max(i for i, x in enumerate(ids["END"]) if x == end_id)
                assert v.tok.decode([ids["END"][uend + 3]]).strip() == "assistant"
                Fw = [i for i, (x, y) in enumerate(zip(ids["START"], ids["END"])) if x != y]
                assert Fw and max(Fw) < qm
                owner = [-1] * S_
                for b in range(N):
                    for p_ in range(vs[b], ve[b] + 1):
                        owner[p_] = b
                A4 = list(range(uend + 5, S_)); assert len(A4) == 4 and A4[-1] == S_ - 1
                opt = []
                for w_ in s["menu"]:
                    pos_ = [i for i in range(qm + 1, uend) if ids["END"][i] == ids1(w_)[0]]
                    assert len(pos_) == 1, f"menu word {w_} must be one token once"
                    opt.append(pos_[0])
                o_pos = max(i for i in range(ve[-1] + 1, qm) if ids["END"][i] == ids1(ORDW[k - 1])[0])
                f_pos = max(i for i in range(o_pos + 1, qm + 1) if ids["END"][i] == ids1("last")[0])
                P.append(dict(s=s, enc=enc, S=S_, owner=owner, qm=qm, opt=opt, rowsC=[qm] + list(range(uend, uend + 5)) + A4, o_pos=o_pos, f_pos=f_pos))
            assert len({p["S"] for p in P}) == 1, "scenes of one case must align for batching"
            capq = lambda p, **kw: dict(rows=p["rowsC"], cap=dict(rows=list(range(10)), owner=p["owner"], nb=N, headout=True, opt=p["opt"]), **kw)
            recs = [dict(si=p["s"]["si"], N=N, k=k, j=j, labs=p["s"]["labs"], words=p["s"]["words"], menu=p["s"]["menu"], land={}, mlog={}, qk={},
                         menu_pos=[(p["s"]["labs"].index(w_) + 1 if w_ in p["s"]["labs"] else 0) for w_ in p["s"]["menu"]]) for p in P]
            def keep(gi, nm, lgb, stb):
                rec = recs[gi]; menu = rec["menu"]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                rec["land"][nm] = rec["labs"].index(w) + 1 if w in rec["labs"] else 0
                rec["mlog"][nm] = [float(sc(lgb, x)) for x in rec["words"]]
                rec["qk"][nm] = {str(li): stb[("cap", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if ("cap", li) in stb and MH.get(li)}
                rec.setdefault("mass", {})[nm] = {str(li): stb[("img", li)][:, MH[li]].float().mean(1).tolist() for li in LOOK if MH.get(li)}
                key = f"N{N}k{k}_s{rec['si']}_{nm}"
                npz[key + "_cap"] = np.stack([stb[("cap", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H, nb]
                npz[key + "_hout"] = np.stack([stb[("hout", li)].float().numpy() for li in LOOK]).astype(np.float16)  # [L, 3, rows, H]
                npz[key + "_img"] = np.stack([stb[("img", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H] picture mass
                npz[key + "_opt"] = np.stack([stb[("opt", li)].float().numpy() for li in LOOK]).astype(np.float16)    # [L, rows, H, 9] menu words
            inps, qs, ws, tag = [], [], [], []
            for gi, p in enumerate(P):
                d_ord = (calM[j - 1] - calM[k - 1]).float()
                zero = torch.zeros_like(d_fr)
                runs = [("END", "END", None), ("START", "START", None), ("WNOOP", "END", [(p["o_pos"], zero), (p["f_pos"], zero)]),
                        ("O_FR", "END", [(p["f_pos"], d_fr)])]
                if j != k:
                    runs += [("O_ORD", "END", [(p["o_pos"], d_ord)]), ("O_ORD_FR", "END", [(p["o_pos"], d_ord), (p["f_pos"], d_fr)])]
                for nm, rcp, w_ in runs:
                    inps.append(p["enc"][rcp]); qs.append(capq(p)); ws.append(w_); tag.append((gi, nm))
            lg, st = runb(inps, qs, ws); nrun += len(inps)
            for b, (gi, nm) in enumerate(tag):
                keep(gi, nm, lg[b], st[b])
            out.extend(recs)
            del P
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{a.tag}] N{N} {min(g0 + G, len(SC))}/{len(SC)} · {nrun} runs · key diff max {(max(keyrel) if keyrel else float('nan')):.2e}", flush=True)
    TOOL.remove(); whook.remove()
    nm_ = "r3mode"
    json.dump(dict(tag=a.tag, model=a.model, Lb=Lb, readout=os.environ.get("R3_READ"), look=LOOK, matcher_layers=sorted(MH),
                   matchers={str(l): h for l, h in MH.items()}, n_layer=nl, scaling=scaling,
                   qk_rows=["question_mark", "B0", "B1", "B2", "B3", "B4", "think", "nl", "think_end", "answer"], t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1),
                   key_rel_diff_max=(max(keyrel) if keyrel else None), rows=out), open(OUT.format(a.tag, nm_), "w"))
    np.savez_compressed(OUT.format(a.tag, nm_).replace(".json", ".npz"), **npz)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, nm_)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=83400); ap.add_argument("--group", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=330)              # 27B: GPU 0 holds the vision tower; N = 7 x 48 runs (336 pictures) fit
    main(ap.parse_args())
