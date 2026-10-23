"""S879 (registered in knowledge/DECISIONS.md before its data; advisor review 5 'boundary-route reinforcement': why N 5 END(3) is answered .88 while
N 7 END(4) breaks) - in the middle identity cells (target j* = N + 1 - k = k) the copy route (ordinal k) and the end route (N - (N - k)) point at the
same picture; does the end route's support protect the middle identity from the writer's END step, and is that protection specific to small N?
One scene = nine pictures with all nine names as the menu; layouts show the first N pictures (N 5, 7, 9; stated = shown); cells END(3) at N 5,
END(4) at N 7, END(5) at N 9 (all middle identities), natural (no pins).  Arms per cell (one sub-pass per layout, every arm with the same
prompt except the donor):
  NAT          natural END(k);
  EMASK        the read rows may not read the last picture's positions at the early look layers 35-47 (all heads; S868's end-route
               weakening, which moved near-end answers one picture start-ward);
  BANDS        the band's read-row outputs (mix / mlp 46-49) <- the natural START(k)'s (same ordinal = the target here; the writer's END step
               replaced by START's band);
  EMASK_BANDS  both;
  START(k)     the donor (natural; answers k).
Readouts per run: answer, menu logits, head-mean block scores of the ten read rows at every read layer, per-head late block scores.
Scenes run in --threads threads; no-grad forwards share --tokens memory tokens; the vision tower runs one thread at a time.
Usage: python s879_reinforce.py MODEL TAG [--n 48] [--seed 87900] [--device_map balanced] [--threads 3] [--tokens 4]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath_k import QPathK

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
OUT = "/vlm/results/r3/s879_{}_{}.json"
CELLS = ((5, 3), (7, 4), (9, 5))


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
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LOOK = [l for l in READ if l / (nl - 1) <= .875]
    EARLY = [l for l in LOOK if l / (nl - 1) <= .75]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    band = [46, 47, 48, 49]
    ALLC = [(l, kd) for l in band for kd in ("mix", "mlp")]
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    NH = v.layers[READ[0]].self_attn.config.num_attention_heads
    hm = lambda li: MH[li] if MH.get(li) else list(range(NH))
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tv = lambda s: (v.tok(s, add_special_tokens=False)["input_ids"][0], v.tok(" " + s, add_special_tokens=False)["input_ids"][0])
    def sc(lg, s):
        t0, t1 = tv(s); return torch.maximum(lg[t0], lg[t1])
    def t1ids(ws):
        return [tv(w)[0] for w in ws] + [tv(w)[1] for w in ws]
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id, end_id = T("<|vision_start|>"), T("<|vision_end|>"), T("<|im_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    TOOL = QPathK(v, READ, res_layers=(), comp_layers=(), patch_comps=band, rpatch_layers=[])

    class _Tokens:
        def __init__(self, n):
            self.n = n; self.c = threading.Condition()
        def take(self, k):
            with self.c:
                while self.n < k:
                    self.c.wait()
                self.n -= k
        def give(self, k):
            with self.c:
                self.n += k; self.c.notify_all()
    TOK = _Tokens(a.tokens)
    _vis = getattr(v.model.model, "visual", None); VLOCK = threading.Lock(); _vis_fwd = None
    if _vis is not None:
        _vis_fwd = _vis.forward
        def _vis_locked(*args_, **kw_):
            with VLOCK:
                return _vis_fwd(*args_, **kw_)
        _vis.forward = _vis_locked
    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps, qs):
        TOK.take(1)
        try:
            TOOL.set(qs)
            try:
                lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()
                st = TOOL.store
            finally:
                TOOL.set(None)
        finally:
            TOK.give(1)
        return lg, st

    rng = random.Random(a.seed); T1 = []; CNT = [0]; CLOCK = threading.Lock()
    SC = []
    for si in range(a.n):
        labs = rng.sample(labels, 9)
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        enc = {}
        for N, k in CELLS:
            enc[(N, "END")] = v.encode(prompt("END", k, menu, N), pics[:N]); enc[(N, "START")] = v.encode(prompt("START", k, menu, N), pics[:N])
        SC.append(dict(si=si, labs=labs, menu=menu, enc=enc))
    RECS = [dict(si=si, labs=SC[si]["labs"], menu=SC[si]["menu"], land={}, mlog={}, qk={}, zl={}) for si in range(a.n)]

    def task(si):
        sc_ = SC[si]; menu = sc_["menu"]; labs = sc_["labs"]
        for N, k in CELLS:
            e = sc_["enc"][(N, "END")]; ids = e["input_ids"][0].tolist(); ids2 = sc_["enc"][(N, "START")]["input_ids"][0].tolist()
            assert len(ids) == len(ids2), "prompts must align"
            S_ = len(ids)
            ve = [i for i, x in enumerate(ids) if x == ve_id]; vs = [i for i, x in enumerate(ids) if x == vs_id]
            qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ids[i]]))
            uend = max(i for i, x in enumerate(ids) if x == end_id)
            owner = [-1] * S_
            for b in range(N):
                for p_ in range(vs[b], ve[b] + 1):
                    owner[p_] = b
            A4 = list(range(uend + 5, S_)); assert len(A4) == 4
            opt = []
            for w_ in menu:
                pos_ = [i for i in range(qm + 1, uend) if ids[i] == ids1(w_)[0]]
                assert len(pos_) == 1
                opt.append(pos_[0])
            rowsC = [qm] + list(range(uend, uend + 5)) + A4
            last = [i for i, o in enumerate(owner) if o == N - 1]
            capq = lambda **kw: dict(rows=rowsC, cap=dict(rows=list(range(10)), owner=owner, nb=N, opt=opt), **kw)
            ko = dict(layers=set(EARLY), rows=list(range(10)), pos=last)
            inps, qs, tag = [], [], []
            def add(nm, ek, q):
                inps.append(sc_["enc"][(N, ek)]); qs.append(q); tag.append(nm); return len(inps) - 1
            iS = add(f"N{N}_START{k}", "START", capq())
            add(f"N{N}_NAT", "END", capq())
            add(f"N{N}_EMASK", "END", capq(ko=ko))
            q_ = capq(); q_["cpatch"] = dict(src=iS, comps=ALLC); add(f"N{N}_BANDS", "END", q_)
            q_ = capq(ko=ko); q_["cpatch"] = dict(src=iS, comps=ALLC); add(f"N{N}_EMASK_BANDS", "END", q_)
            lg, st = run1(inps, qs)
            for b, nm in enumerate(tag):
                lgb, stb = lg[b], st[b]
                T1.append(int(int(lgb.argmax()) in t1ids(menu)))
                w = menu[int(torch.stack([sc(lgb, c) for c in menu]).argmax())]
                RECS[si]["land"][nm] = labs.index(w) + 1
                RECS[si]["mlog"][nm] = [float(sc(lgb, x)) for x in labs[:N]]
                RECS[si]["qk"][nm] = {str(li): stb[("cap", li)][:, hm(li)].float().mean(1).tolist() for li in READ if ("cap", li) in stb}
                RECS[si]["zl"][nm] = {str(li): stb[("cap", li)][[2, 5, 9]].float().tolist() for li in LATE if ("cap", li) in stb}
            with CLOCK:
                CNT[0] += len(tag)
            del st; gc.collect()

    print(f"[{a.tag}] read {READ} · early {EARLY} · late {LATE} · band {band} · cells {CELLS}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for i_, _ in enumerate(ex.map(task, range(a.n))):
                print(f"[{a.tag}] scene {i_ + 1}/{a.n} · {CNT[0]} runs", flush=True)
    finally:
        TOOL.remove()
        if _vis is not None:
            _vis.forward = _vis_fwd
    json.dump(dict(tag=a.tag, model=a.model, cells=[list(c) for c in CELLS], band=band, read=READ, early=EARLY, late=LATE,
                   late_mean={str(li): hm(li) for li in LATE}, readout=os.environ.get("R3_READ"), t1_menu=(sum(T1) / len(T1) if T1 else None),
                   t1_n=len(T1), n_runs=CNT[0], rows=RECS), open(OUT.format(a.tag, "reinforce"), "w"))
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag, 'reinforce')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87900); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--tokens", type=int, default=4)
    main(ap.parse_args())
