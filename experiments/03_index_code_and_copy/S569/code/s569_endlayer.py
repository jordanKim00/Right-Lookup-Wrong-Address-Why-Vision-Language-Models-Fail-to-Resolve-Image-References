"""S569: at which layers does the question read the END? The S567 full-attention route (keys + values of the last
block lowered to index N-1) applied only in windows of full-attention layers (27B: full layers 3, 7, ..., 63).
§596: the whole effect of lowering the last block (k = 2: j-1 18/24) comes through the full-attention keys/values.
Windows: early L3-23, mid L27-39, L43, L47, L51, L55-63, and L43+47. Reverse questions without a count sentence, k = 2.

S567 docstring follows.
S567: through which route does the question read the END? Route-specific transplants into the LAST block (27B).

§593: lowering the last block's code to N-1 (residual at every layer) moves the reverse target to j-1 (k = 2: 27/36,
base 9/36), raising it is ignored. Here the same lowering (donor = the scene without picture N-1, whose block N-1 is
picture N at index N-1) enters the last block through one route at a time, at every layer of that kind:
    res     the residual stream                                    (= S564 M2; instrument)
    lin     the input of every linear-attention layer               (what the block writes into the recurrent state)
    fakv    full-attention keys (non-rotary dims) and values
    fak     full-attention keys only (what the question's queries match)
Reverse questions WITHOUT a count sentence, k = 2, 3.

S556 docstring follows.
S556: through which route does the lookup read a block's index code? Route-specific transplants of "the same picture
at another index" into the target block (27B).

§545 §549: the image blocks' residual index codes move the forward and reverse landings. §570: the full-attention KEYS'
non-rotary part of the target block can come from another index with no effect on reverse; §583: the target's class
content is first written by the L51 full attention. So the index is read by some other route. Donor = the recipient
with blocks t and x swapped (donor block x shows the recipient's target picture at index x). Into the recipient's
target block t (all its tokens with the vision markers), at every layer of the given kind:
    res     the residual stream (every decoder layer's output)                 -- the block now claims index x everywhere
    fakv    full-attention keys (non-rotary dims) and values
    fav     full-attention values only
    lin     the input of every linear-attention (Gated DeltaNet) layer
Forward (target k) and reverse (target j) questions; landing on the target, on x.
"""
from __future__ import annotations

import argparse, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni
import transformers.models.qwen3_5.modeling_qwen3_5 as M

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s569_endlayer_{}.json"
CIFAR = "/vlm/assets/cifar10"
HOOK = {"layer": None, "fn": None}
_orig_rope = M.apply_rotary_pos_emb


def _wrap_rope(q, k, cos, sin, unsqueeze_dim=1):
    qe, ke = _orig_rope(q, k, cos, sin, unsqueeze_dim)
    if HOOK["layer"] is not None and HOOK["fn"] is not None:
        qe, ke = HOOK["fn"](HOOK["layer"], qe, ke, int(cos.shape[-1]))
    return qe, ke


M.apply_rotary_pos_emb = _wrap_rope


def load_cifar(pixels):
    with open(f"{CIFAR}/batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]
    with open(f"{CIFAR}/test_batch", "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    by = {i: [] for i in range(len(names))}
    for arr, y in zip(data, d[b"labels"]):
        if len(by[y]) < 200:
            by[y].append(arr)
    return names, {i: [Image.fromarray(a).resize((pixels, pixels), Image.BICUBIC)
                       for a in v_] for i, v_ in by.items()}


def run(v, tag="job", ntest=16, pixels=224, Ns=(6, 7, 8), ks=(2,), seed=56900):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    Lh = v.n_layer
    full = [i for i, L in enumerate(v.layers) if not hasattr(L, "linear_attn")]
    lin = [i for i, L in enumerate(v.layers) if hasattr(L, "linear_attn")]
    ST = {"op": None}
    D = {"res": {}, "k": {}, "v": {}, "lin": {}}

    def rows_of(blk):
        return blk.to("cpu")

    # residual stream
    def mk_res(l):
        def h(m, i, o):
            if ST["op"] is None:
                return o
            t = o[0] if isinstance(o, tuple) else o
            if ST["op"] == "cap":
                D["res"][l] = t[0, ST["src"].to(t.device)].detach().clone(); return o
            if ST["op"] == "res":
                t = t.clone(); t[0, ST["dst"].to(t.device)] = D["res"][l].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return h

    # full-attention keys (non-rotary) via the rope wrapper
    def fn(l, qe, ke, rd):
        if ST["op"] == "cap":
            D["k"][l] = ke[0][:, ST["src"].to(ke.device), rd:].detach().clone()
        elif ST["op"] in ("fakv", "fak") and l in ST.get("win", ()):
            ke = ke.clone(); ke[0][:, ST["dst"].to(ke.device), rd:] = D["k"][l].to(ke.device, ke.dtype)
        return qe, ke
    HOOK["fn"] = fn

    def on(l):
        def pre(m, args, kwargs):
            HOOK["layer"] = l
            return None
        return pre

    def off(m, i, o):
        HOOK["layer"] = None
        return o

    # full-attention values (v_proj output)
    def mk_v(l):
        def h(m, i, o):
            if ST["op"] == "cap":
                D["v"][l] = o[0, ST["src"].to(o.device)].detach().clone(); return o
            if ST["op"] in ("fakv", "fav") and l in ST.get("win", ()):
                o = o.clone(); o[0, ST["dst"].to(o.device)] = D["v"][l].to(o.device, o.dtype)
            return o
        return h

    # linear-attention inputs
    def mk_lin(l):
        def pre(m, args, kwargs):
            if ST["op"] is None:
                return None
            x = kwargs["hidden_states"]
            if ST["op"] == "cap":
                D["lin"][l] = x[0, ST["src"].to(x.device)].detach().clone(); return None
            if ST["op"] == "lin":
                x = x.clone(); x[0, ST["dst"].to(x.device)] = D["lin"][l].to(x.device, x.dtype)
                kwargs = dict(kwargs); kwargs["hidden_states"] = x
                return args, kwargs
            return None
        return pre

    hks = [v.layers[l].register_forward_hook(mk_res(l)) for l in range(Lh)]
    for l in full:
        A = v.layers[l].self_attn
        hks += [A.register_forward_pre_hook(on(l), with_kwargs=True), A.register_forward_hook(off), A.v_proj.register_forward_hook(mk_v(l))]
    for l in lin:
        hks.append(v.layers[l].linear_attn.register_forward_pre_hook(mk_lin(l), with_kwargs=True))

    def prompt(rev, k, N, opts):
        tail = " from the last" if rev else ""
        return v.chat([("user", f"{v.IMG*N}\nWhat is in the {ORD[k-1]} image{tail}? "
                                f"Answer with only one word from: {', '.join(opts)}.")], prefix_assistant=THINK)

    def blocks(inp, N):
        return [torch.arange(int(b[0]) - 1, int(b[-1]) + 2) for b in v.image_blocks(inp["input_ids"], N)]

    @torch.no_grad()
    def forward(rev, k, N, cl, pics, opts, op=None, **kw):
        inp = v.encode(prompt(rev, k, N, opts), pics)
        B = blocks(inp, N)
        ST.update(op=op, B=B, **kw)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["op"] = None
        return int(np.argmax([out.logits[0, -1, tid1(c)].item() for c in cl])) + 1, B

    WINS = {"all": [l for l in full], "early": [l for l in full if l <= 23], "mid": [l for l in full if 27 <= l <= 39],
            "L43": [43], "L47": [47], "L43+47": [43, 47], "L51": [51], "late": [l for l in full if l >= 55]}
    rng = random.Random(seed)
    rows = []
    for t_ in range(ntest):
        for N in Ns:
            ci = rng.sample(range(len(names)), N)
            cl = [names[x] for x in ci]; pics = [rng.choice(tiles[x]) for x in ci]
            opts = cl[:]; rng.shuffle(opts)
            dp = pics[:N - 2] + [pics[N - 1]]                                   # picture N at index N-1
            inp_d = v.encode(prompt(False, 1, N - 1, opts), dp); Bd = blocks(inp_d, N - 1)
            for key in D:
                D[key] = {}
            ST.update(op="cap", src=Bd[N - 2])
            with torch.no_grad():
                v.model(**inp_d, use_cache=False, return_dict=True, **v.forward_extra)
            ST["op"] = None
            for k in ks:
                j = N + 1 - k
                r = dict(t=t_, N=N, k=k, j=j)
                a0, B = forward(True, k, N, cl, pics, opts)
                r["none"] = a0
                if len(B[N - 1]) != len(Bd[N - 2]):
                    continue
                r["res"] = forward(True, k, N, cl, pics, opts, op="res", dst=B[N - 1])[0]
                for wn, wl in WINS.items():
                    r[wn] = forward(True, k, N, cl, pics, opts, op="fakv", dst=B[N - 1], win=set(wl))[0]
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()
    M.apply_rotary_pos_emb = _orig_rope
    res = {}
    rng_ = np.random.default_rng(0)
    print(f"\n  역순 정답률(없음, 개수 문장 없음) k2 {np.mean([r['none'] == r['j'] for r in rows]):.3f}, n {len(rows)}")
    print(f"  창(전 주의 키·값 낮춤)   j−1 착지(없음)   짝지은 평균 이동 [95%]     res 대비")
    base = None
    for op in ["res"] + list(WINS):
        sh = np.array([np.clip(r[op] - r["j"], -3, 3) - np.clip(r["none"] - r["j"], -3, 3) for r in rows], float)
        bs = [sh[rng_.integers(0, len(sh), len(sh))].mean() for _ in range(4000)]
        d = dict(jm1=float(np.mean([r[op] == r["j"] - 1 for r in rows])), jm1_none=float(np.mean([r["none"] == r["j"] - 1 for r in rows])),
                 shift=[float(sh.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        if op == "res":
            base = d["shift"][0]
        d["frac_of_res"] = float(d["shift"][0] / base) if base and abs(base) > 1e-9 else float("nan")
        res[op] = d
        print(f"  {op:>8s}                 {d['jm1']:.3f} ({d['jm1_none']:.3f})      {d['shift'][0]:+.2f} [{d['shift'][1]:+.2f}, {d['shift'][2]:+.2f}]        {d['frac_of_res']:+.2f}", flush=True)
    print(f"\n  가름 (사전 등록 S569) (l0, 계기) all 의 이동이 res 의 70% 이상(§596 재현). (l1) 한 창(early·mid·L43·L47·L43+47·L51·late)이 res 의 60% 이상을 혼자 내고 나머지 창은")
    print(f"    각각 30% 미만 → 끝은 그 층에서 읽힌다. 여러 창이 30-60% 씩이면 분산. (l2, 보고) L43+47 대 단독.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=16)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, ntest=a.ntest)
