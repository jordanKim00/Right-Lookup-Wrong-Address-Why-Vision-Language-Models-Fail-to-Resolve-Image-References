"""S875 (advisor review 2, H1-H3: the QK-OV-logit bridge; design note research/ces/outputs/writer_circuit_2026-10-20/main/s875_design_note.md) -
does the late lookup's attention on picture b, carried through each head's output gate, value path and W_O, write picture b's menu word into
the answer logits (H1); do these direct writes predict the logit margin between the chosen and the runner-up picture (H2); is the answer
carried by several heads and layers additively (H3)?
Natural runs only (no pins, no patches, no gradients).  One scene = nine pictures with all nine names as the menu (shuffled, as S862 / S872);
layouts show its first N pictures (N 6-9, stated = shown); one batch per layout: END(k) for every k = 1..N and START(q) for q in {1, N}.
Per run, at the late full-attention layers (51, 55, 59, 63 on 27B), recomputed in float32 from the layer's own inputs (input-normed hidden
states, q_norm / k_norm, partial RoPE, GQA, sigmoid output gate; as exp/c_headpatch.py FAHeadPatchPinned._heads_out) for the rows B1, B4 and
the answer row (rowsC indices 2, 5, 9 of S862 / S872):
  key buckets    c = the N picture blocks (vision_start..vision_end of picture b, picture order), the other text (every non-image key that is
                 not a menu word), the menu word of each of the nine labels (label order); 'text' = other text + the nine menu words
  mass           a[L, row, h, c] = sum over t in c of head h's attention weight (rows B1, B4, answer)
  block score    z[L, row, h, b] = log-sum-exp over the keys of picture b of the scaled q.k (S869R's late scores; rows B1, B4, answer)
  bucket write   o[L, h, c] = sigmoid(gate_h) * sum_{t in c} att_t V_h,t (answer row; an exact split of the head's gated output: sum_c o = o_h),
                 its residual write W_O[:, h] o[L, h, c] (norm stored)
  DLA            D[L, h, c, i] = (W_U[id_i] * (1 + w_final)) . (W_O[:, h] o[L, h, c]) / rms_final  (answer row; i = the nine labels in
                 picture order, the first N shown; both spacing forms are projected and the form with the larger final logit in this run is
                 kept, as the readout's max), rms_final = sqrt(mean(x_final^2) + eps) of the run's own final residual at the answer position
                 (the input of the final norm, whose scale is (1 + weight) in Qwen3.5 - asserted on the module before the runs)
  module DLA     the same projection of the residual entering layer 51, of every token-mixer output (FA and Gated DeltaNet) and every MLP
                 output of layers 51..63, and of the final residual: the parts sum to the final residual, so the label logits decompose
                 exactly at the run's fixed rms
Controls per run: the recomputed attention output (heads through W_O) against the module's own output at the three rows (relative error)
and the bucket split against the recompute; the linearised logits against the model's logits; the parts against the final residual; t1.
Saved: results/r3/s875_<tag>_scenes/scene<si>.npz (per layout N: dla [R, 4, 24, N + 10, 9] f16, mass [R, 4, 3, 24, N + 10] f16, z [R, 4, 3, 24,
N] f32, wnorm [R, 4, 24, N + 10] f16, gate [R, 4, 24] f16 (answer-row mean sigmoid gate), hchk [R, 4, 4] (three rows + split closure), farec
[R, 4, 9] (DLA of the recomposed answer-row output), parts [R, 27, 9], final [R, 9], logit [R, 9], rms [R], var [R, 9]; R = N + 2 runs in the
order END1..ENDN, START1, STARTN) and results/r3/s875_<tag>_bridge.json (answers, menu logits, t1, checks, layouts).
Usage: python s875_bridge.py MODEL TAG [--n 24] [--seed 87500] [--device_map balanced]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
NS = (6, 7, 8, 9)
NLAB = 9                                                                   # every scene label is a menu word; the first N pictures are shown


def batch_inputs(inps):
    out = {}
    for key in inps[0]:
        vals = [x[key] for x in inps]
        out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
    return out


def rope(x, cos, sin, rd):
    return torch.cat([x[..., :rd] * cos[:, None, :] + rotate_half(x[..., :rd]) * sin[:, None, :], x[..., rd:]], -1)


def f16(t):
    return t.detach().cpu().to(torch.float16).numpy()


def f32(t):
    return t.detach().cpu().float().numpy()


def main(a):
    assert os.environ.get("R3_READ") == "max"
    dev = os.environ.get("S875_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    nl = v.n_layer
    FA_L = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
    READ = [l for l in FA_L if l / (nl - 1) >= .555]
    LATE = [l for l in READ if l / (nl - 1) > .75]
    L0 = LATE[0]; TAIL = list(range(L0, nl)); DNT = [l for l in TAIL if l not in FA_L]
    assert [l for l in TAIL if l in FA_L] == LATE and all(getattr(v.layers[l], "linear_attn", None) is not None for l in DNT)
    NH = v.layers[L0].self_attn.config.num_attention_heads
    if nl == 64:
        assert LATE == [51, 55, 59, 63] and NH == 24
    PARTS = [f"in{L0}"] + [f"{kd}{l}" for l in TAIL for kd in (("fa" if l in LATE else "dn"), "mlp")]
    # the final norm and the unembedding (Qwen3_5ForConditionalGeneration: .model = Qwen3_5Model, .model.language_model = Qwen3_5TextModel
    # whose .norm is the final Qwen3_5RMSNorm, output = x / rms * (1 + weight); .lm_head is untied)
    lm = v.model.model.language_model
    assert lm.layers is v.layers
    fnorm, W_U = lm.norm, v.model.lm_head.weight
    eps = float(getattr(fnorm, "eps", getattr(fnorm, "variance_epsilon", 1e-6)))
    with torch.no_grad():                                                 # the scale convention, checked on the module itself (before any hook)
        xx = torch.randn(2, fnorm.weight.shape[0], device=fnorm.weight.device, dtype=fnorm.weight.dtype)
        ref = xx.float() * torch.rsqrt(xx.float().pow(2).mean(-1, keepdim=True) + eps) * (1.0 + fnorm.weight.float())
        got = fnorm(xx).float().to(ref.device)
        nchk = float((got - ref).norm() / ref.norm())
        assert nchk < 1e-2, f"final norm is not x / rms * (1 + w) (relative difference {nchk:.3g})"
        del xx, ref, got
    MH = {}
    heads = {"Qwen3.5-27B": "/vlm/exp/s786_heads/27b.json"}.get(os.path.basename(a.model.rstrip("/")))
    if heads and os.path.exists(heads):
        for l, h in json.load(open(heads))["matchers"]:
            MH.setdefault(int(l), []).append(int(h))
    hm = lambda li: [h for h in MH.get(li, []) if h < NH] or list(range(NH))
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
    assert len(labels) >= NLAB

    # ------------------------------------------------------------------ hooks (results go to the CPU at once: robust to a balanced device map)
    X = dict(on=False)

    def Ud(d_):
        """the readout directions W_U[id] * (1 + w_final) [2 * NLAB, E] on device d_ (one copy per device and scene)."""
        k_ = str(d_)
        if k_ not in X["Ud"]:
            X["Ud"][k_] = X["U"].to(d_)
        return X["Ud"][k_]

    def fa_post(li):
        def h(mod, args, kwargs, out):
            if not X["on"]:
                return None
            t0 = out[0] if isinstance(out, tuple) else out                                   # the module's own output [B, S, E]
            hsb = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]          # input-normed hidden states
            cosb, sinb = kwargs["position_embeddings"]
            d_ = hsb.device; B_, S_ = hsb.shape[0], hsb.shape[1]
            hd = mod.head_dim; nh = mod.config.num_attention_heads; nb = X["n"]
            bias = lambda m: None if m.bias is None else m.bias.float()
            Wq, Wk, Wv = mod.q_proj.weight.float(), mod.k_proj.weight.float(), mod.v_proj.weight.float()
            Wo = mod.o_proj.weight.float().view(-1, nh, hd); bo = bias(mod.o_proj)                # [E, H, hd]
            rt = torch.tensor(X["rows"], device=d_); R = len(X["rows"])
            assert X["rows"][-1] == S_ - 1
            img = X["img"].to(d_); BK = X["BK"].to(d_)                                        # [S] picture of a key or -1; [S, nk] bucket one-hot
            U = Ud(d_)
            kh = torch.arange(nh, device=d_) // mod.num_key_value_groups
            causal = torch.arange(S_, device=d_)[None, :] > rt[:, None]
            res = dict(dla=[], mass=[], z=[], wn=[], gate=[], chk=[], rec=[])
            for b in range(B_):
                hs = hsb[b].float(); cos, sin = cosb[b].to(d_).float(), sinb[b].to(d_).float(); rd = cos.shape[-1]
                qg = F.linear(hs[rt], Wq, bias(mod.q_proj)).view(R, nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
                k = mod.k_norm(F.linear(hs, Wk, bias(mod.k_proj)).view(S_, -1, hd).to(mod.k_norm.weight.dtype)).float()
                V = F.linear(hs, Wv, bias(mod.v_proj)).view(S_, -1, hd)[:, kh]                 # [S, H, hd] per query head (GQA)
                qe = rope(q, cos[rt], sin[rt], rd); ke = rope(k[:, kh], cos, sin, rd)
                lg = torch.einsum("rhd,shd->rhs", qe, ke) * mod.scaling                        # [R, H, S]
                z = torch.stack([torch.logsumexp(lg[:, :, img == c], -1) for c in range(nb)], -1)   # [R, H, N] block scores (S869R)
                att = torch.softmax(lg.masked_fill(causal[:, None, :], float("-inf")), -1)
                sg = torch.sigmoid(gate)                                                        # [R, H, hd] output gate
                full = sg * torch.einsum("rhs,shd->rhd", att, V)
                rec = torch.einsum("rhd,ehd->re", full, Wo)
                if bo is not None:
                    rec = rec + bo
                act = t0[b, X["rows"]].to(d_).float()
                chk = (rec - act).norm(dim=-1) / act.norm(dim=-1).clamp_min(1e-12)              # [R] recompute vs the module's output
                # answer row (the last row): each head's gated output split exactly over the key buckets, its write and its DLA
                ob = sg[-1][:, None, :] * torch.bmm((att[-1][:, :, None] * BK[None]).transpose(1, 2), V.transpose(0, 1))   # [H, nk, hd]
                w = torch.einsum("hcd,ehd->hce", ob, Wo)                                       # [H, nk, E]
                ws = w.sum((0, 1)) + (bo if bo is not None else 0)
                split = (ws - rec[-1]).norm() / rec[-1].norm().clamp_min(1e-12)
                res["dla"].append(torch.einsum("hce,ie->hci", w, U).cpu())                     # [H, nk, 2 * NLAB]; / rms after the pass
                res["mass"].append(torch.einsum("rhs,sc->rhc", att, BK).cpu())                 # [R, H, nk]
                res["z"].append(z.cpu()); res["wn"].append(w.norm(dim=-1).cpu()); res["gate"].append(sg[-1].mean(-1).cpu())
                res["chk"].append(torch.cat([chk, split[None]]).cpu())
                res["rec"].append((U @ ws).cpu())                                               # DLA of the recomposed answer-row output
            X["fa"][li] = {k_: torch.stack(v_) for k_, v_ in res.items()}
            X["mix"][li] = t0[:, -1].detach().float().cpu()
            return None
        return h

    def last_row(key, l):
        def h(mod, args, out):
            if X["on"]:
                t = out[0] if isinstance(out, tuple) else out
                X[key][l] = t[:, -1].detach().float().cpu()
            return None
        return h

    def pre_in0(mod, args, kwargs):
        if X["on"]:
            hs = args[0] if args else kwargs["hidden_states"]
            X["in0"] = hs[:, -1].detach().float().cpu()
        return None

    def pre_fin(mod, args):
        if X["on"]:
            X["fin"] = args[0][:, -1].detach().float().cpu()
        return None

    hooks = [v.layers[l].self_attn.register_forward_hook(fa_post(l), with_kwargs=True) for l in LATE]
    hooks += [v.layers[l].linear_attn.register_forward_hook(last_row("mix", l)) for l in DNT]
    hooks += [v.layers[l].mlp.register_forward_hook(last_row("mlp", l)) for l in TAIL]
    hooks.append(v.layers[L0].register_forward_pre_hook(pre_in0, with_kwargs=True))
    hooks.append(fnorm.register_forward_pre_hook(pre_fin))

    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the last?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    @torch.no_grad()
    def run1(inps):
        X.update(on=True, fa={}, mix={}, mlp={}, in0=None, fin=None)
        try:
            lg = v.model(**batch_inputs(inps), use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float().cpu()
            got = dict(fa=X["fa"], mix=X["mix"], mlp=X["mlp"], in0=X["in0"], fin=X["fin"])
        finally:
            X.update(on=False, fa={}, mix={}, mlp={}, in0=None, fin=None)
        return lg, got

    rng = random.Random(a.seed); nrun = 0; recs = []; T1 = []
    OUTJ = f"{a.outdir}/s875_{a.tag}_bridge.json"; OUTD = f"{a.outdir}/s875_{a.tag}_scenes"
    os.makedirs(OUTD, exist_ok=True)

    def dump(complete):
        meta = dict(tag=a.tag, model=a.model, seed=a.seed, Ns=list(NS), pixels=a.pixels, late=LATE, tail=TAIL, dn=DNT, fa=FA_L, read=READ, nh=NH,
                    late_mean={str(li): hm(li) for li in LATE}, parts=PARTS, nlab=NLAB, eps=eps, norm_check=nchk,
                    rows3="B1 = im_end + 1, B4 = im_end + 4, answer = the last position (rowsC indices 2, 5, 9 of S862 / S872)",
                    buckets="0..N-1 picture blocks (picture order), N other text, N+1..N+9 the menu word of label 1..9 (label order)",
                    labels="the nine scene labels in picture order (labs); the first N pictures are shown",
                    runs_order="END1..ENDN, START1, STARTN per layout", npz_dir=OUTD, readout=os.environ.get("R3_READ"),
                    t1_menu=(sum(T1) / len(T1) if T1 else None), t1_n=len(T1), n_runs=nrun, complete=complete, rows=recs)
        tmp = OUTJ + ".tmp"
        json.dump(meta, open(tmp, "w")); os.replace(tmp, OUTJ)

    print(f"[{a.tag}] late {LATE} · tail {TAIL[0]}-{TAIL[-1]} (DN {DNT}) · heads {NH} · final norm check {nchk:.2e} · labels {len(labels)} · "
          f"matcher means {[(li, len(hm(li))) for li in LATE]}", flush=True)
    try:
        for si in range(a.n):
            labs = rng.sample(labels, NLAB)
            pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
            menu = list(labs); rng.shuffle(menu)
            IDS = t1ids(labs)                                                                   # [no-space form x 9, space form x 9]
            X["U"] = (W_U[IDS].float() * (1.0 + fnorm.weight.float().to(W_U.device))).cpu()     # [18, E]
            X["Ud"] = {}
            rec = dict(si=si, labs=labs, menu=menu, layout={}, land={}, mlog={}, t1={}, chk={})
            NPZ = {}; cmax = dict(attn=0., split=0., lin=0., closure=0., farec=0.)
            for n in NS:
                names = {**{f"END{k}": ("END", k) for k in range(1, n + 1)}, **{f"START{q}": ("START", q) for q in sorted({1, n})}}
                keys = list(names)
                enc = {nm: v.encode(prompt(f, q, menu, n), pics[:n]) for nm, (f, q) in names.items()}
                ids = {nm: e["input_ids"][0].tolist() for nm, e in enc.items()}
                assert len({len(x) for x in ids.values()}) == 1, "prompts of a layout must align"
                ref = ids[keys[0]]; S_ = len(ref)
                ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
                assert len(vs) == len(ve) == n
                qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
                uend = max(i for i, x in enumerate(ref) if x == end_id)
                diff = sorted({i for nm in ids for i in range(S_) if ids[nm][i] != ref[i]})
                assert diff and all(ve[-1] < i < qm for i in diff) and len(diff) <= 2, f"prompts differ outside the question: {diff}"
                assert v.tok.decode([ref[uend + 3]]).strip() == "assistant"
                A4 = list(range(uend + 5, S_)); assert len(A4) == 4
                rowsC = [qm] + list(range(uend, uend + 5)) + A4
                rows3 = [rowsC[2], rowsC[5], rowsC[9]]; assert rows3[-1] == S_ - 1
                img = [-1] * S_
                for b in range(n):
                    for p_ in range(vs[b], ve[b] + 1):
                        img[p_] = b
                opt = []
                for w_ in labs:                                                                 # the menu word of each label (label order)
                    pos_ = [i for i in range(qm + 1, uend) if ref[i] == ids1(w_)[0]]
                    assert len(pos_) == 1
                    opt.append(pos_[0])
                bucket = [b if b >= 0 else n for b in img]
                for j, p_ in enumerate(opt):
                    bucket[p_] = n + 1 + j
                nk = n + 1 + NLAB
                X.update(n=n, rows=rows3, img=torch.tensor(img), BK=F.one_hot(torch.tensor(bucket), nk).float())
                rec["layout"][str(n)] = dict(S=S_, vs=vs, ve=ve, qm=qm, uend=uend, rows3=rows3, opt=opt, diff=diff, keys=keys)

                lg, got = run1([enc[k_] for k_ in keys])
                R_ = len(keys)
                x_f = got["fin"]
                rms = torch.sqrt(x_f.pow(2).mean(-1) + eps)                                        # [R] the run's own final-norm scale
                Uc = X["U"]
                proj = lambda t: (t @ Uc.T) / rms[:, None]                                         # [R, 18]
                parts18 = torch.stack([proj(got["in0"])] + [proj(got[kd][l]) for l in TAIL for kd in ("mix", "mlp")], 1)   # [R, P, 18]
                var = (lg[:, IDS[NLAB:]] > lg[:, IDS[:NLAB]]).long()                               # 1 = the space form is larger (as s872 tokvar)
                col = torch.arange(NLAB)[None] + NLAB * var                                        # [R, 9]

                def pick(t):
                    return torch.gather(t, -1, col.view(R_, *([1] * (t.dim() - 2)), NLAB).expand(*t.shape[:-1], NLAB))
                mlog = torch.maximum(lg[:, IDS[:NLAB]], lg[:, IDS[NLAB:]])                         # [R, 9] the readout's max over the forms
                FA = got["fa"]
                dla = torch.stack([pick(FA[l]["dla"] / rms[:, None, None, None]) for l in LATE], 1)   # [R, nL, H, nk, 9]
                mass = torch.stack([FA[l]["mass"] for l in LATE], 1)                               # [R, nL, 3, H, nk]
                z = torch.stack([FA[l]["z"] for l in LATE], 1)                                     # [R, nL, 3, H, N]
                wn = torch.stack([FA[l]["wn"] for l in LATE], 1)                                   # [R, nL, H, nk]
                gt = torch.stack([FA[l]["gate"] for l in LATE], 1)                                 # [R, nL, H]
                hchk = torch.stack([FA[l]["chk"] for l in LATE], 1)                                # [R, nL, 4]
                farec = torch.stack([pick(FA[l]["rec"] / rms[:, None]) for l in LATE], 1)          # [R, nL, 9]
                parts = pick(parts18); final = pick(proj(x_f))                                     # [R, P, 9], [R, 9]
                for nm_, t_ in (("dla", dla), ("mass", mass), ("z", z), ("parts", parts), ("final", final)):
                    assert torch.isfinite(t_).all(), f"non-finite {nm_} in layout {n}"
                ifa = [PARTS.index(f"fa{l}") for l in LATE]
                ck = dict(attn=hchk[..., :3].amax((1, 2)), split=hchk[..., 3].amax(1), lin=(final - mlog).abs().amax(-1),
                          closure=(parts.sum(1) - final).abs().amax(-1), farec=(farec - parts[:, ifa]).abs().amax((1, 2)))
                for b, nm in enumerate(keys):
                    key = f"G{n}_{nm}"
                    w = menu[int(torch.stack([sc(lg[b], c) for c in menu]).argmax())]
                    rec["land"][key] = labs.index(w) + 1
                    rec["mlog"][key] = [float(x) for x in mlog[b]]
                    rec["t1"][key] = int(int(lg[b].argmax()) in t1ids(menu)); T1.append(rec["t1"][key])
                    rec["chk"][key] = {c_: float(t_[b]) for c_, t_ in ck.items()}
                    for c_ in cmax:
                        cmax[c_] = max(cmax[c_], rec["chk"][key][c_])
                    nrun += 1
                NPZ.update({f"N{n}_dla": f16(dla), f"N{n}_mass": f16(mass), f"N{n}_z": f32(z), f"N{n}_wnorm": f16(wn), f"N{n}_gate": f16(gt),
                            f"N{n}_hchk": f32(hchk), f"N{n}_farec": f32(farec), f"N{n}_parts": f32(parts), f"N{n}_final": f32(final),
                            f"N{n}_logit": f32(mlog), f"N{n}_rms": f32(rms), f"N{n}_var": var.numpy().astype(np.int8)})
                del lg, got, FA, dla, mass, z, wn, parts18, parts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            np.savez(f"{OUTD}/scene{si:02d}.npz", **NPZ)
            recs.append(rec); dump(False)
            print(f"[{a.tag}] scene {si + 1}/{a.n} · {nrun} runs · t1 {sum(T1) / len(T1):.3f} · attn recompute rel {cmax['attn']:.2e} · split "
                  f"{cmax['split']:.1e} · logit lin {cmax['lin']:.3f} · closure {cmax['closure']:.3f} · FA recompose {cmax['farec']:.3f}", flush=True)
            del NPZ; gc.collect()
    finally:
        for h_ in hooks:
            h_.remove()
    dump(True)
    print(f"[{a.tag}] 끝 -> {OUTJ}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--seed", type=int, default=87500); ap.add_argument("--outdir", default="/vlm/results/r3")
    main(ap.parse_args())
