"""S571: where is the end turned into the target code? Path decomposition between the L47 end reading and retrieval.

§599: the whole effect of lowering the end (k = 2: j -> j-1) enters through L47's attention output at the suffix
(P47 = 100%), and the L51 query carries about half of it. Between L47's attention and L51 lie L47's MLP and three
linear-attention layers (L48-50, each Gated DeltaNet + MLP). On top of P47, components at the suffix positions are
held at their clean-run values:
    P47_mlp     MLP outputs of L47-L50
    P47_lin     linear-attention outputs of L48-L50
    P47_both    both (the end information then reaches the retrieval only through the residual stream itself)
    P47_q       the L51 and L55 queries
If the end -> target transformation is done by those MLPs, P47_mlp loses most of the effect.

S570 docstring follows.
S570: the path from the L47 end reading to the L51 retrieval (27B), by path patching.

§598: lowering the last block's code only in the L47 full-attention keys/values gives 71% of the whole effect
(k = 2: target j -> j-1); §597: the L51 query then retrieves the block with the new target code. Here, per item
(reverse, k = 2, no count sentence):
    none          clean run; captures L47's gated attention output (o_proj input) and L51's query (q_norm output) at
                  the suffix positions ("?" to the end)
    M2_L47        the last block's L47 keys/values lowered (S569 arm); captures L47's o_proj input at the suffix
    P47           clean run with L47's o_proj input at the suffix taken from M2_L47 (all heads)
    P47_g0..g3    the same for one KV group's six query heads only
    P47_q51       P47 with L51's query at the suffix held at the clean run's value
If the end enters through L47's attention output, P47 reproduces M2_L47; if it acts on the retrieval through the L51
query, P47_q51 loses most of it.

S569 docstring follows.
S569: at which layers does the question read the END? The S567 full-attention route (keys + values of the last
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

import argparse, os, json, pickle, random, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/vlm/exp")
from d1_grammar import ORD as _ORD
from uni import Uni
import transformers.models.qwen3_5.modeling_qwen3_5 as M

ORD = list(_ORD) + ["eleventh", "twelfth"]
OUT = "/vlm/results/s571_arith_{}.json"
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


def run(v, tag="job", ntest=16, pixels=224, Ns=(6, 7, 8), ks=(2,), seed=57100, LR=47, LQ=51):
    THINK = "<think>\n\n</think>\n\n" if v.fam == "qwen35" else ""
    names, tiles = load_cifar(pixels)
    tid1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"][0]
    # answer readout (S789): R3_READ=bare reads the first bare token of the class word, the form the model generates (FINDINGS §846)
    aid1 = (lambda s: v.tok(s, add_special_tokens=False)["input_ids"][0]) if os.environ.get("R3_READ") == "bare" else tid1
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
    Dh = v.layers[LR].self_attn.head_dim
    Hq = v.layers[LR].self_attn.q_proj.out_features // (2 * Dh)                  # q_proj carries query + gate
    Hk = v.layers[LR].self_attn.k_proj.out_features // Dh
    PP = {"cap_o": None, "put_o": None, "heads": None, "cap_q": None, "put_q": None, "P": None}
    LOG = {"last": None}
    STORE = {}

    def o_pre(m, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        P = PP["P"]
        if PP["cap_o"] is not None:
            STORE[PP["cap_o"]] = x[0, -P:].detach().clone()
        if PP["put_o"] is not None:
            x = x.clone(); src = STORE[PP["put_o"]].to(x.device, x.dtype)
            if PP["heads"] is None:
                x[0, -P:] = src
            else:
                for h in PP["heads"]:
                    x[0, -P:, h * Dh:(h + 1) * Dh] = src[:, h * Dh:(h + 1) * Dh]
            return (x,) + tuple(args[1:]), kwargs
        return None
    hks.append(v.layers[LR].self_attn.o_proj.register_forward_pre_hook(o_pre, with_kwargs=True))

    def q_hook(m, i, o):
        P = PP["P"]
        if PP["cap_q"] is not None:
            STORE[PP["cap_q"]] = o[0, -P:].detach().clone()
        if PP["put_q"] is not None:
            o = o.clone(); o[0, -P:] = STORE[PP["put_q"]].to(o.device, o.dtype)
        return o
    hks.append(v.layers[LQ].self_attn.q_norm.register_forward_hook(q_hook))
    MIDL = list(range(LR, LQ))                                                   # L47..L50
    QLS = [l for l in (LQ, LQ + 4) if l < Lh and not hasattr(v.layers[l], "linear_attn")]
    FR = {"cap": False, "mlp": set(), "lin": set(), "q": set()}

    def mk_comp(kind, l):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            P = PP["P"]
            key = f"{kind}{l}"
            if FR["cap"]:
                STORE[key] = t[0, -P:].detach().clone()
                return o
            if l in FR[kind]:
                t = t.clone(); t[0, -P:] = STORE[key].to(t.device, t.dtype)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            return o
        return h
    for l in MIDL:
        hks.append(v.layers[l].mlp.register_forward_hook(mk_comp("mlp", l)))
        if hasattr(v.layers[l], "linear_attn"):
            hks.append(v.layers[l].linear_attn.register_forward_hook(mk_comp("lin", l)))
    for l in QLS:
        hks.append(v.layers[l].self_attn.q_norm.register_forward_hook(mk_comp("q", l)))
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

    def suffix_len(inp):
        ids = inp["input_ids"][0].tolist()
        q = max(i for i, t in enumerate(ids) if "?" in v.tok.decode([t]))
        return len(ids) - q

    @torch.no_grad()
    def forward(rev, k, N, cl, pics, opts, op=None, pp=None, **kw):
        inp = v.encode(prompt(rev, k, N, opts), pics)
        B = blocks(inp, N)
        PP.update(cap_o=None, put_o=None, heads=None, cap_q=None, put_q=None, P=suffix_len(inp))
        if pp:
            PP.update(pp)
        ST.update(op=op, B=B, **kw)
        out = v.model(**inp, use_cache=False, return_dict=True, **v.forward_extra)
        ST["op"] = None
        lg = [out.logits[0, -1, aid1(c)].item() for c in cl]
        LOG["last"] = lg
        return int(np.argmax(lg)) + 1, B

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
                if len(B[N - 1]) != len(Bd[N - 2]):
                    continue
                STORE.clear()
                FR.update(cap=True, mlp=set(), lin=set(), q=set())
                r["none"] = forward(True, k, N, cl, pics, opts, pp=dict(cap_o="o_none"))[0]; r["lg|none"] = LOG["last"]
                FR["cap"] = False
                r["M2_L47"] = forward(True, k, N, cl, pics, opts, op="fakv", dst=B[N - 1], win={LR}, pp=dict(cap_o="o_m2"))[0]
                r["P47"] = forward(True, k, N, cl, pics, opts, pp=dict(put_o="o_m2"))[0]; r["lg|P47"] = LOG["last"]
                for nm, fr in (("P47_mlp", dict(mlp=set(MIDL))), ("P47_lin", dict(lin=set(MIDL))),
                               ("P47_both", dict(mlp=set(MIDL), lin=set(MIDL))), ("P47_q", dict(q=set(QLS)))):
                    FR.update(mlp=set(), lin=set(), q=set()); FR.update(fr)
                    r[nm] = forward(True, k, N, cl, pics, opts, pp=dict(put_o="o_m2"))[0]
                FR.update(mlp=set(), lin=set(), q=set())
                rows.append(r)
        torch.cuda.empty_cache()
        print(f"[{tag}] 시행 {t_+1}/{ntest}  행 {len(rows)}", flush=True)
    for h in hks:
        h.remove()
    M.apply_rotary_pos_emb = _orig_rope
    res = {}
    rng_ = np.random.default_rng(0)
    arms = ["M2_L47", "P47", "P47_mlp", "P47_lin", "P47_both", "P47_q"]
    print(f"\n  역순 정답률(없음, 개수 문장 없음) k2 {np.mean([r['none'] == r['j'] for r in rows]):.3f}, n {len(rows)}")
    print(f"  고정 층: MLP {MIDL}, 질의 {QLS}")
    print(f"  팔          j−1 착지(없음)   짝지은 평균 이동 [95%]     M2_L47 대비")
    base = None
    for op in arms:
        sh = np.array([np.clip(r[op] - r["j"], -3, 3) - np.clip(r["none"] - r["j"], -3, 3) for r in rows], float)
        bs = [sh[rng_.integers(0, len(sh), len(sh))].mean() for _ in range(4000)]
        d = dict(jm1=float(np.mean([r[op] == r["j"] - 1 for r in rows])), jm1_none=float(np.mean([r["none"] == r["j"] - 1 for r in rows])),
                 shift=[float(sh.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        if op == "M2_L47":
            base = d["shift"][0]
        d["frac"] = float(d["shift"][0] / base) if base and abs(base) > 1e-9 else float("nan")
        res[op] = d
        print(f"  {op:>9s}     {d['jm1']:.3f} ({d['jm1_none']:.3f})      {d['shift'][0]:+.2f} [{d['shift'][1]:+.2f}, {d['shift'][2]:+.2f}]        {d['frac']:+.2f}", flush=True)
    dl = [float(np.max(np.abs(np.array(r["lg|P47"]) - np.array(r["lg|none"])))) for r in rows]
    print(f"  계기 점검: P47 대 none 로짓 최대 차 평균 {np.mean(dl):.3f}(0 이면 패치가 안 든 것)")
    print(f"\n  가름 (사전 등록 S571) (a1, 계기) P47 의 j−1 착지 >= .70. (a2) P47_mlp 의 이동이 P47 의 40% 이하 → 끝 → 목표 변환은 L47-50 MLP 를 지난다. (a3, 보고) P47_lin.")
    print(f"    (a4) P47_both 가 20% 이하 → 모든 효과가 L47-50 의 MLP·선형 층을 지난다; 60% 이상 → 끝 정보가 잔차로 곧장 검색 질의에 닿는다. (a5, 보고) P47_q.")
    json.dump({"res": res, "rows": rows}, open(OUT.format(tag), "w"), indent=1)
    print(f"  -> {OUT.format(tag)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--ntest", type=int, default=16)
    ap.add_argument("--lr", type=int, default=47)
    ap.add_argument("--lq", type=int, default=51)
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    v = Uni(a.model, device="cuda:0", attn="sdpa", device_map=a.device_map)
    for p in v.model.parameters():
        p.requires_grad_(False)
    run(v, tag=a.tag, ntest=a.ntest, LR=a.lr, LQ=a.lq)
