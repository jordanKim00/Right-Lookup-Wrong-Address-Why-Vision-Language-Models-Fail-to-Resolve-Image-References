"""S809 (pre-registered in knowledge/DECISIONS.md before any S809 data): close the circuit's open links in Qwen3.5-27B.
S804 (FINDINGS §883) placed the completion of the end-relative address at layers 36-46 (B1) and a frame-specific reading mode at layers 47-50 (B2),
each sufficient from START's side; S807 (§879) placed the sources (near: last image's code; far: the stated count).  Open: whether END's B1 / B2
are necessary for its errors, which band carries each source's effect, which component of B2 is the reading mode, and whether R3 repairs through
these bands.  Instrument: S797 / S804 component-output exchange at the rows from the question mark to the answer (same prompt and pairs), with
the donors' outputs grabbed in their own runs:
  START(j), END(k)                    natural runs
  R3                                  END(k) with R3's writes (r3_select, plan from the model's own first pass, as S798)
  CNT8                                END(k) with "There are 8 images" (7 shown; S807)
  CODE6                               END(k) whose last picture block carries the states of a donor where it is the 6th picture (S807 transplant,
                                      own state restored from layer 51: the lookup's key bank stays unedited)
Bands: B1 = every mixer and MLP output of layers 36-46; B2 = layers 47-50; B2 split into B2_DN (Gated DeltaNet mixers 48-50), B2_FA (full
attention 47), B2_MLP (MLPs 47-50).  Arms per (scene, k):
  NOOP E<-E:B1B2;  E<-S:B1, E<-S:B2 (S804 replication);  S<-E:B1, S<-E:B2, S<-E:B1B2 (necessity);
  E<-R:B1, E<-R:B2, R<-E:B1, R<-E:B2 (R3's route);  E<-CNT8:B1, E<-CNT8:B2 (count path);  E<-CODE6:B1, E<-CODE6:B2 (code path);
  E<-S:B2_DN, E<-S:B2_FA, E<-S:B2_MLP, S<-E:B2_DN, S<-E:B2_FA, S<-E:B2_MLP (reading-mode component).
Readout: two-form menu readout (S794), t1; answer-row block shares and the cosine of the answer-row query with START's at the lookup layers.
Scenes: procedural pictures (160 px), N = 7, seed 80900 (new), 40 scenes, k in {2, 3, 5, 6}.
Usage: python s809_circuit.py MODEL TAG --base BASE --rc RC [--n 40] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath import QPath
import r3_select

N = 7
ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
OUT = "/vlm/results/r3/s809_{}_circuit.json"


def main(a):
    assert os.environ.get("R3_READ") == "max", "S809 reads the generated form in either tokenization (S794)"
    dev = os.environ.get("S809_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    mp = lambda l: l if nl == 64 else int(round(l / 63 * (nl - 1)))
    mixer = lambda li: v.layers[li].self_attn if li in FA_L else v.layers[li].linear_attn
    span = lambda lo, hi: [("mix", l) for l in range(mp(lo), mp(hi) + 1)] + [("mlp", l) for l in range(mp(lo), mp(hi) + 1)]
    B2L = list(range(mp(47), mp(50) + 1))
    SETS = {"B1": span(36, 46), "B2": span(47, 50)}
    SETS["B1B2"] = SETS["B1"] + SETS["B2"]
    SETS["B2_DN"] = [("mix", l) for l in B2L if l not in FA_L]
    SETS["B2_FA"] = [("mix", l) for l in B2L if l in FA_L]
    SETS["B2_MLP"] = [("mlp", l) for l in B2L]
    assert SETS["B2_DN"] and SETS["B2_FA"], ("band 47-50 must hold both mixer kinds", B2L, FA_L)
    LFIX = mp(51)
    READ = sorted({min(FA_L, key=lambda l: abs(l - mp(x))) for x in (39, 43, 47, 51, 55)})
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json", "Qwen3.5-9B": "/vlm/exp/s786_heads/35_9b.json",
             "Qwen3.5-4B": "/vlm/exp/s786_heads/35_4b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads:
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    rep = r3_select.build(v, a.base, a.rc)
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    def sc(lg, s):
        return torch.maximum(lg[v.tok(s, add_special_tokens=False)["input_ids"][0]], lg[v.tok(" " + s, add_special_tokens=False)["input_ids"][0]])
    def t1ids(ws):
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

    X = dict(mode=None, rows=None, put=set(), store={}, donor={})          # component-output exchange (S797 / S804)

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

    TR = dict(mode=None)                                                      # last-block transplant (S807)

    def tk(l):
        def hook(m, i, o):
            if TR["mode"] is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if TR["mode"] == "cap":
                TR["cap"][l] = t[0, TR["span"]].detach().clone()
                return o
            if TR["mode"] == "put" and l <= LFIX - 1:
                src = TR["don"][l] if l < LFIX - 1 else TR["orig"]
                t = t.clone(); t[0, TR["dst"]] = src.to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return hook
    hooks += [v.layers[l].register_forward_hook(tk(l)) for l in range(nl)]
    TOOL = QPath(v, READ, [])
    T1 = []; CHK = []

    def text(nimg, q, frame, menu, n_said):
        return v.chat([("user", f"{v.IMG*nimg}\nThere are {n_said} images in a row. What is in the {ORDW[q-1]} image from the {frame}? "
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

    def qcos(st, ref):
        o = {}
        for li in READ:
            if ("q", li) in st and ref is not None and li in MH:
                c = torch.nn.functional.cosine_similarity(st[("q", li)][-1].float(), ref[("q", li)][-1].float(), dim=-1)[MH[li]]
                o[str(li)] = float(c.median())
        return o

    @torch.no_grad()
    def run(inp, labs, menu, cap, rows, mode=None, put=(), donor=None, writes=None, tr=None):
        X.update(mode=mode, rows=rows, put=set(put), donor=donor or {}); X["store"] = {} if mode == "grab" else X["store"]
        TR.update(mode=None)
        if tr is not None:
            TR.update(**tr)
        rep.ST = dict(writes=writes) if writes else {}
        TOOL.set(dict(rows=rows, cap=cap, grabq=set(READ)))
        lg = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra).logits[0, -1].float()
        st = dict(TOOL.store); TOOL.set(None); rep.ST = {}
        grabbed = dict(X["store"]) if mode == "grab" else None
        X.update(mode=None, put=set(), donor={}); TR.update(mode=None)
        T1.append(int(int(lg.argmax()) in t1ids(menu)))
        lab = menu[int(torch.stack([sc(lg, c) for c in menu]).argmax())]
        return (labs.index(lab) + 1 if lab in labs else 0), st, grabbed

    print(f"[{a.tag}] sets " + " ".join(f"{k}:{v_[0][1]}-{v_[-1][1]}({len(v_)})" for k, v_ in SETS.items()) + f" READ {READ} L_fix {LFIX}", flush=True)
    rng = random.Random(a.seed); out = []
    for si in range(a.n):
        labs = rng.sample(labels, N); extra = [l for l in labels if l not in labs][:2]
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = labs + extra; rng.shuffle(menu)
        dp = pics[:N - 2] + [pics[N - 1]]                                    # donor where the last picture is the 6th (S807 CODE6)
        inp_d = v.encode(text(len(dp), 1, "first", menu, N), dp); idd = inp_d["input_ids"][0].tolist()
        vsd = [i for i, t in enumerate(idd) if t == vs_id]; ved = [i for i, t in enumerate(idd) if t == ve_id]
        TR.update(mode="cap", cap={}, span=list(range(vsd[-1], ved[-1] + 1)))
        with torch.no_grad():
            v.model(**inp_d, use_cache=False, return_dict=True, **v.forward_extra)
        don6 = {l: TR["cap"][l] for l in range(LFIX - 1)}; TR.update(mode=None)
        for k in a.ks:
            j = N + 1 - k
            tS, tE = text(N, j, "first", menu, N), text(N, k, "last", menu, N)
            iS, iE, iC = v.encode(tS, pics), v.encode(tE, pics), v.encode(text(N, k, "last", menu, 8), pics)
            idS, idE, idC = iS["input_ids"][0].tolist(), iE["input_ids"][0].tolist(), iC["input_ids"][0].tolist()
            assert len(idS) == len(idE) == len(idC), "prompts must align"
            S_ = len(idE)
            vs = [i for i, t in enumerate(idE) if t == vs_id]; ve = [i for i, t in enumerate(idE) if t == ve_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([idE[i]]))
            assert max(i for i, (x, y) in enumerate(zip(idS, idE)) if x != y) < qm and max(i for i, (x, y) in enumerate(zip(idC, idE)) if x != y) < qm
            rows = list(range(qm, S_))
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            cap = dict(rows=[len(rows) - 1], owner=owner, nb=N)
            dst = list(range(vs[N - 1], ve[N - 1] + 1)); assert len(dst) == len(don6[0]), "donor and recipient blocks must match"
            _, writes = rep.plan(tE, pics)
            plan = [(d["tok"], int(d["kh"]), int(d["t"]), bool(d["act"])) for d in rep.last_detail]
            rec = dict(si=si, k=k, j=j, plan=plan, plan_ok=any(p[3] and p[2] == j for p in plan), land={}, share={}, qcos={})
            # natural runs (with the transplant capture for the restore state inside END)
            rec["land"]["START"], stS, gS = run(iS, labs, menu, cap, rows, mode="grab")
            TRcap = dict(mode="cap", cap={}, span=dst)
            rec["land"]["END"], stE, gE = run(iE, labs, menu, cap, rows, mode="grab", tr=TRcap)
            orig, ref_fix = TR["cap"][LFIX - 1], TR["cap"][LFIX]
            rec["land"]["R3"], stR, gR = run(iE, labs, menu, cap, rows, mode="grab", writes=writes)
            rec["land"]["CNT8"], stC, gC = run(iC, labs, menu, cap, rows, mode="grab")
            chk = dict(mode="put", don=don6, orig=orig, dst=dst)
            rec["land"]["CODE6"], stK, gK = run(iE, labs, menu, cap, rows, mode="grab", tr=dict(chk, cap={}))
            for nm_, st_ in (("START", stS), ("END", stE), ("R3", stR), ("CNT8", stC), ("CODE6", stK)):
                rec["share"][nm_] = share(st_); rec["qcos"][nm_] = qcos(st_, stS)
            if not CHK:                                                       # the key bank from L_fix on must be the unedited one
                TRc = dict(mode="put", don=don6, orig=orig, dst=dst)
                X.update(mode=None); TR.update(**TRc); rep.ST = {}
                grab_fix = {}
                def hfix(m, i, o, store=grab_fix):
                    t = o[0] if isinstance(o, tuple) else o
                    store["s"] = t[0, dst].detach().clone()
                hh = v.layers[LFIX].register_forward_hook(hfix)
                with torch.no_grad():
                    v.model(**iE, use_cache=False, return_dict=True, **v.forward_extra)
                hh.remove(); TR.update(mode=None)
                CHK.append(float((grab_fix["s"].float() - ref_fix.float()).abs().max()))
                print(f"[{a.tag}] key-bank check at layer {LFIX}: {CHK[0]:.3e}", flush=True)
                assert CHK[0] < 1e-2
            arms = [("NOOP", iE, "B1B2", gE, None, None),
                    ("E<-S:B1", iE, "B1", gS, None, None), ("E<-S:B2", iE, "B2", gS, None, None),
                    ("S<-E:B1", iS, "B1", gE, None, None), ("S<-E:B2", iS, "B2", gE, None, None), ("S<-E:B1B2", iS, "B1B2", gE, None, None),
                    ("E<-R:B1", iE, "B1", gR, None, None), ("E<-R:B2", iE, "B2", gR, None, None),
                    ("R<-E:B1", iE, "B1", gE, writes, None), ("R<-E:B2", iE, "B2", gE, writes, None),
                    ("E<-CNT8:B1", iE, "B1", gC, None, None), ("E<-CNT8:B2", iE, "B2", gC, None, None),
                    ("E<-CODE6:B1", iE, "B1", gK, None, None), ("E<-CODE6:B2", iE, "B2", gK, None, None)]
            for c in ("DN", "FA", "MLP"):
                arms += [(f"E<-S:B2_{c}", iE, f"B2_{c}", gS, None, None), (f"S<-E:B2_{c}", iS, f"B2_{c}", gE, None, None)]
            for nm_, inp_, set_, don_, wr_, tr_ in arms:
                rec["land"][nm_], st_, _ = run(inp_, labs, menu, cap, rows, mode="put", put=SETS[set_], donor=don_, writes=wr_, tr=tr_)
                rec["share"][nm_] = share(st_); rec["qcos"][nm_] = qcos(st_, stS)
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
                   lfix=LFIX, keybank_check=(CHK[0] if CHK else None), matcher_layers=sorted(MH), t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), rows=out), open(OUT.format(a.tag), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--base", required=True); ap.add_argument("--rc", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=80900); ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 5, 6])
    main(ap.parse_args())
