"""S927 - parameter-free linear-response (LR) predictor of intervention outcomes from the model's own gradients (user 10-23: '모델 내부 그라디언트나
뭐 이것저것 잘 조합해서 해봐 수학적으로 우아하게').  Nothing is fitted to any outcome.

For a natural END(k) run let z be the menu logits (one per menu word, max over the word's two first tokens; in float32 from the final hidden state,
i.e. the bf16 logits without their output rounding; order = labs: positions 1..N are the pictures, the rest distractors).  The natural answer
x_nat is the collector's (bf16 logits, argmax in menu order).  An intervention I changes circuit sites s by Delta_s^I and its first-order prediction is
    z^I = z + sum_s < dz/dh_s , Delta_s^I >,
  sites: the read-row outputs (rows rowsC, after o_proj) of the early full-attention readers 35 / 39 / 43 / 47 (address pins) and of the band
  components 46-49 (mixer = DeltaNet or FA47, and MLP; band replacement);
  Delta for a pin to p (U part of the read-row queries <- START(p)'s, as QPath._qsub part 'U'): the read-row attention output recomputed in float32
  with the pinned query minus the same recomputation with the natural query, both from the natural inputs of that layer (exact local change);
  Delta for the band <- START(q): START(q)'s band read-row outputs minus END's;
  gradients: total (mode T) for pin-only interventions; with the band read-row outputs held fixed (mode D: detached at the read rows) for band-
  replacing interventions, so that the paths through the replaced outputs are cut and only the paths around them remain.
Arms (as in S924 / S926): PIN (p = a), PINm (a - 1), PINp (a + 1), PINt (t), PINm2 (a - 2), PINp2 (a + 2) [mode T]; B = E_BANDR (band <- START(a)),
PINB (PIN + band <- START(a)), PINtB (t, START(t)), PINmB (a - 1, START(a - 1)), PINpB (a + 1, START(a + 1)) [mode D]; a = the natural L47 anchor,
read from the outcome file (no outcome of any intervention is read), t = N + 1 - k.
Stored per scene and k as transfer matrices over every pinned position q = 1..N (rows) and menu word b (columns): T[q] (mode T, pin to q),
Dp[q] (mode D, the pin's change at 35 / 39 / 43 through the paths around the band outputs), BAND[q] (mode D, band <- START(q)); arms are sums,
e.g. PINm = T[a - 1], B = BAND[a], PINtB = Dp[t] + BAND[t]; the band-path part of a pin is T[q] - Dp[q].
--fd 1 (smoke): finite-difference checks of the gradients (z(e Delta) - z(0)) / e for e = .1, .25 and the full additive local patch (e = 1) of PINm
and E_BANDR at the first k of the first scene.
Usage: python s927_lr.py MODEL TAG --collect /vlm/results/r3/s924_s924_27b_collect.json --seed 92408 [--n 48] [--fd 0]"""
from __future__ import annotations

import argparse, gc, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, "/vlm/exp")
from uni import Uni
from r3_proc import proc_image
from c_qpath import QPath
from c_tools import rotate_half

ORDW = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
OUT = "/vlm/results/r3/s927_{}_lr.json"
EARLY = [35, 39, 43, 47]; BAND = [46, 47, 48, 49]


class LRHooks:
    """mode None: inactive.  'grab': per batch element store the read-row queries (after q_norm, before RoPE) at EARLY and the band read-row
    outputs.  'T' / 'D' (batch 1): compute the exact local pin changes for the donors in self.donors {name: {layer: q}} at EARLY, add zero probes
    (requires grad) to the read-row outputs (T: EARLY; D: EARLY minus 47, and the band sites after detaching their read-row outputs), and store
    the natural band read-row outputs.  'add' (batch 1): add self.add[site] to the read-row outputs (finite-difference checks)."""

    def __init__(self, v, rows, owner, nb):
        self.v = v; self.rows = rows; self.owner = owner; self.nb = nb; self.mode = None; self.hooks = []
        self.donors = {}; self.dlt = {}; self.probe = {}; self.nat = {}; self.grab = {}; self.add = {}; self.check = False; self.checks = []
        for li in sorted(set(EARLY) | set(BAND)):
            L = v.layers[li]
            mix = L.self_attn if getattr(L, "self_attn", None) is not None else L.linear_attn
            self.hooks.append(mix.register_forward_hook(self._mix(li), with_kwargs=True))
            if li in BAND:
                self.hooks.append(L.mlp.register_forward_hook(self._mlp(li)))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _prep(self, mod, kwargs, b):
        """float32 pieces of FA layer mod's read-row attention for batch element b (QPath._hook's computation): post-RoPE read-row queries,
        post-RoPE keys and values per query head, the output gate, the pre-RoPE queries, and per head the basis U of the centred block-mean
        picture keys (QPath._qsub part 'U')."""
        hs = kwargs["hidden_states"][b].float(); cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
        S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]; rows = self.rows; R = len(rows)
        lin = lambda x, m: torch.nn.functional.linear(x, m.weight.float(), None if m.bias is None else m.bias.float())
        qg = lin(hs[rows], mod.q_proj).view(R, nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
        k = mod.k_norm(lin(hs, mod.k_proj).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
        vv = lin(hs, mod.v_proj).view(S, -1, hd)
        kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
        K = k[:, kh]; V = vv[:, kh]
        rope = lambda x, c_, s_: torch.cat([x[..., :rd] * c_[:, None, :] + rotate_half(x[..., :rd]) * s_[:, None, :], x[..., rd:]], -1)
        qe = rope(q, cos[rows], sin[rows]); ke = rope(K, cos, sin)
        own = torch.tensor(self.owner, device=q.device)
        Kb = torch.stack([ke[own == b_].mean(0) for b_ in range(self.nb)]); Cc = Kb - Kb.mean(0, keepdim=True)
        Us = []
        for h_ in range(nh):
            U, s_, _ = torch.linalg.svd(Cc[:, h_].T, full_matrices=False); Us.append(U[:, s_ > 1e-6 * s_.max()])
        mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
        return dict(qe=qe, ke=ke, V=V, gate=gate, q=q, Us=Us, mask=mask, rope=lambda x: rope(x, cos[rows], sin[rows]), lin=lin)

    def _attend(self, mod, P, qe):
        lg = torch.einsum("thd,shd->ths", qe, P["ke"]) * mod.scaling
        att = torch.softmax(lg.masked_fill(P["mask"][:, None, :], float("-inf")), -1)
        o = torch.sigmoid(P["gate"]) * torch.einsum("ths,shd->thd", att, P["V"])
        return P["lin"](o.reshape(o.shape[0], -1), mod.o_proj)

    def _pin(self, P, dq):
        """QPath._qsub part 'U' with one donor (weight 1): per head the read-row query's part in U <- the donor's (post-RoPE)."""
        qe = P["qe"]; don = P["rope"](dq.to(qe.device).float()); out = qe.clone()
        for h_, U in enumerate(P["Us"]):
            out[:, h_] = qe[:, h_] - (qe[:, h_] @ U) @ U.T + (don[:, h_] @ U) @ U.T
        return out

    def _site(self, key, t):
        """apply the mode's edit to the read-row outputs of site key in output tensor t (batch 1 for T / D / add)."""
        rows = self.rows
        if self.mode in ("T", "D"):
            self.nat[key] = t[0, rows].detach().float()
            hold = self.mode == "D" and key[0] in BAND
            if self.mode == "T" and key[1] != "mix":
                return None
            if self.mode == "T" and key[0] not in EARLY:
                return None
            if self.mode == "D" and not hold and key[0] not in EARLY:
                return None
            p = torch.zeros(len(rows), t.shape[-1], device=t.device, dtype=torch.float32, requires_grad=True)
            self.probe[key] = p
            t2 = t.clone()
            base = t[0, rows].detach() if hold else t[0, rows]
            t2[0, rows] = base + p.to(t.dtype)
            return t2
        if self.mode == "add" and key in self.add:                                   # ('add', d): current + d; ('set', x): the read rows <- x
            kind, d = self.add[key]; t2 = t.clone()
            t2[0, rows] = ((t[0, rows].float() + d.to(t.device)) if kind == "add" else d.to(t.device)).to(t.dtype); return t2
        return None

    def _mix(self, li):
        def h(mod, args, kwargs, out):
            if self.mode is None:
                return None
            t = out[0] if isinstance(out, tuple) else out
            if self.mode == "grab":
                with torch.no_grad():
                    for b in range(t.shape[0]):
                        if li in EARLY:
                            lin = lambda x, m: torch.nn.functional.linear(x, m.weight.float(), None if m.bias is None else m.bias.float())
                            hs = kwargs["hidden_states"][b].float(); hd = mod.head_dim; nh = mod.config.num_attention_heads
                            qg = lin(hs[self.rows], mod.q_proj).view(len(self.rows), nh, 2 * hd)
                            self.grab.setdefault(b, {})[("q", li)] = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float().detach()
                        if li in BAND:
                            self.grab.setdefault(b, {})[(li, "mix")] = t[b, self.rows].detach().float()
                return None
            if self.mode == "T" and li in EARLY and self.donors:
                with torch.no_grad():
                    P = self._prep(mod, kwargs, 0); base = self._attend(mod, P, P["qe"])
                    for nm, dq in self.donors.items():
                        self.dlt[(nm, li)] = (self._attend(mod, P, self._pin(P, dq[li])) - base).detach()
                    if self.check:                                             # the vectorised pin equals QPath._qsub (first donor, this layer)
                        nm0 = next(iter(self.donors)); spec = dict(owner=self.owner, nb=self.nb, part="U", donors=[(1.0, self.donors[nm0][li])])
                        cos, sin = kwargs["position_embeddings"]; rd = cos.shape[-1]
                        ref = QPath._qsub(QPath, P["qe"], P["ke"], spec, cos[0].float()[self.rows], sin[0].float()[self.rows], rd)
                        self.checks.append(float((ref - self._pin(P, self.donors[nm0][li])).abs().max()))
            t2 = self._site((li, "mix"), t)
            if t2 is None:
                return None
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h

    def _mlp(self, li):
        def h(mod, args, out):
            if self.mode is None:
                return None
            if self.mode == "grab":
                for b in range(out.shape[0]):
                    self.grab.setdefault(b, {})[(li, "mlp")] = out[b, self.rows].detach().float()
                return None
            return self._site((li, "mlp"), out)
        return h


def main(a):
    assert os.environ.get("R3_READ") == "max"
    import c_tritonlock; c_tritonlock.install()
    C = json.load(open(a.collect)); N = C["N"]; KS = C["ks"]
    dev = os.environ.get("S863_DEVICE", "cuda:0")
    v = Uni(a.model, device=dev, attn="sdpa", device_map=a.device_map) if a.device_map else Uni(a.model, device=dev, attn="sdpa")
    assert v.fam == "qwen35"
    for p_ in v.model.parameters():
        p_.requires_grad_(False)
    THINK = "<think>\n\n</think>\n\n"
    ids1 = lambda s: v.tok(" " + s, add_special_tokens=False)["input_ids"]
    tv = lambda s: (v.tok(s, add_special_tokens=False)["input_ids"][0], v.tok(" " + s, add_special_tokens=False)["input_ids"][0])
    T = v.tok.convert_tokens_to_ids
    vs_id, ve_id, end_id = T("<|vision_start|>"), T("<|vision_end|>"), T("<|im_end|>")
    pids = sorted({int(k.split(":")[1]) for l in open("/vlm/assets/r3cal_a3/train.jsonl") for k in json.loads(l)["images"]})
    pool = {}
    for pid in pids[:4000]:
        lab = proc_image(pid)[1]
        if len(ids1(lab)) == 1:
            pool.setdefault(lab, []).append(pid)
    labels = sorted(l for l, x in pool.items() if len(x) >= 20)
    FRAME = C.get("phrase", "last")
    QUEST = {"END": lambda q: f"What is in the {ORDW[q-1]} image from the {FRAME}?", "START": lambda q: f"What is in the {ORDW[q-1]} image from the first?"}

    def prompt(form, q, menu, n):
        return v.chat([("user", f"{v.IMG*n}\nThere are {n} images in a row. {QUEST[form](q)} Answer with only one word from: {', '.join(menu)}.")],
                      prefix_assistant=THINK)

    rng = random.Random(a.seed); SC = []
    for si in range(a.n):                                                                        # the outcome collector's scenes, in its order
        labs = rng.sample(labels, max(9, N))
        pics = [proc_image(rng.choice(pool[l]))[0].resize((a.pixels, a.pixels)) for l in labs]
        menu = list(labs); rng.shuffle(menu)
        assert labs == C["rows"][si]["labs"] and menu == C["rows"][si]["menu"], f"scene {si} differs from the outcome file"
        SC.append(dict(labs=labs, menu=menu, pics=pics[:N]))
    enc = lambda si, f, q: v.encode(prompt(f, q, SC[si]["menu"], N), SC[si]["pics"])
    e0 = enc(0, "END", KS[0]); ref = e0["input_ids"][0].tolist(); S_ = len(ref)
    ve = [i for i, x in enumerate(ref) if x == ve_id]; vs = [i for i, x in enumerate(ref) if x == vs_id]
    qm = next(i for i in range(ve[-1] + 1, S_) if "?" in v.tok.decode([ref[i]]))
    uend = max(i for i, x in enumerate(ref) if x == end_id)
    owner = [-1] * S_
    for b in range(N):
        for p_ in range(vs[b], ve[b] + 1):
            owner[p_] = b
    rowsC = [qm] + list(range(uend, uend + 5)) + list(range(uend + 5, S_))
    assert rowsC == C["rowsC"], "read rows differ from the outcome file"
    H = LRHooks(v, rowsC, owner, N)
    LMH = v.model.lm_head; HL = {}
    def _lmh(mod, args):
        HL["h"] = args[0][:, -1]
        return None
    h_lmh = LMH.register_forward_pre_hook(_lmh)
    TOKS = {si: torch.tensor([[tv(w)[0], tv(w)[1]] for w in SC[si]["labs"]]) for si in range(a.n)}

    def z32(si):
        """menu logits in float32 from the final hidden state (the bf16 logits without their output rounding): max over the word's two tokens"""
        h = HL["h"][0]; W = LMH.weight[TOKS[si].to(LMH.weight.device).view(-1)].float()
        return (W @ h.float().to(W.device)).view(-1, 2).max(-1).values

    def xmenu(lg, si):
        """the collector's answer: argmax over the menu words in menu order of the bf16 logits, as a position (labs index + 1)"""
        m = SC[si]["menu"]
        return SC[si]["labs"].index(m[int(torch.stack([torch.maximum(lg[tv(w)[0]], lg[tv(w)[1]]) for w in m]).argmax())]) + 1

    def zvec(lg, si):
        return torch.stack([torch.maximum(lg[tv(w)[0]], lg[tv(w)[1]]) for w in SC[si]["labs"]])

    def fwd(e):
        bi = {k: (t.to(dev) if torch.is_tensor(t) else t) for k, t in e.items()}
        return v.model(**bi, use_cache=False, return_dict=True, **v.forward_extra).logits[:, -1].float()

    def batch(es):
        out = {}
        for key in es[0]:
            vals = [x[key] for x in es]
            out[key] = torch.cat(vals, 0) if torch.is_tensor(vals[0]) else vals[0]
        return out

    RES = []; t0 = time.time()
    try:
        for si in range(a.n):
            labs = SC[si]["labs"]; rec = dict(si=si, k={})
            # START(q) for every q: donor queries and band outputs (no grad)
            H.mode = "grab"; H.grab = {}
            with torch.no_grad():
                es = [enc(si, "START", q) for q in range(1, N + 1)]
                lgS = fwd(batch(es))
            G = {q: H.grab[q - 1] for q in range(1, N + 1)}; H.mode = None
            rec["start_answers"] = [int(labs.index(SC[si]["menu"][int(torch.stack([torch.maximum(lgS[q - 1][tv(w)[0]], lgS[q - 1][tv(w)[1]]) for w in SC[si]["menu"]]).argmax())]) + 1)
                                    for q in range(1, N + 1)]
            for k in KS:
                if f"B{k}_E" not in C["rows"][si]["land"]:
                    continue
                an = int(C["rows"][si]["anchor"][str(k)]); t_ = N + 1 - k; nL = len(labs); Q = list(range(1, N + 1))
                e = enc(si, "END", k); out = dict(anchor=an, t=t_)
                SITES_B = [(li, kd) for li in BAND for kd in ("mix", "mlp")]
                dot = lambda Jb, site, d: float((Jb[site] * d.to(Jb[site].device)).sum())
                for mode in ("T", "D"):
                    H.mode = mode; H.probe = {}; H.nat = {}
                    if mode == "T":
                        H.dlt = {}; H.donors = {q: {li: G[q][("q", li)] for li in EARLY} for q in Q}; H.check = (a.fd and si == 0); H.checks = []
                    else:
                        H.donors = {}
                    with torch.enable_grad():
                        lg = fwd(e)[0]; z = z32(si)
                        keys = list(H.probe); P = [H.probe[k_] for k_ in keys]
                        J = []
                        for b in range(nL):
                            g = torch.autograd.grad(z[b], P, retain_graph=(b < nL - 1), allow_unused=True)
                            J.append({k_: (gg.float() if gg is not None else torch.zeros_like(H.probe[k_])) for k_, gg in zip(keys, g)})
                    H.mode = None; H.check = False
                    zz = z.detach().float()
                    if mode == "T":
                        out["z"] = zz.tolist(); out["x_nat"] = xmenu(lg.detach(), si); out["z_bf16"] = zvec(lg.detach(), si).tolist()
                        out["T"] = [[sum(dot(J[b], (li, "mix"), H.dlt[(q, li)]) for li in EARLY) for b in range(nL)] for q in Q]
                        if H.checks:
                            out["pin_check_maxabs"] = max(H.checks)
                    else:
                        out["z_D"] = zz.tolist()
                        out["Dp"] = [[sum(dot(J[b], (li, "mix"), H.dlt[(q, li)]) for li in (35, 39, 43)) for b in range(nL)] for q in Q]
                        out["BAND"] = [[sum(dot(J[b], st, G[q][st] - H.nat[st]) for st in SITES_B) for b in range(nL)] for q in Q]
                    if a.fd and si == 0 and k == KS[0]:
                        out.setdefault("fd", {})
                        # PINm: the additive local patch eps x Delta at the early read-row outputs (eps = 1: the local changes summed, without the
                        # layer-to-layer interaction of the real pin); B: the band read-row outputs SET to natural + eps x (START(a) - END) with the
                        # other band outputs held (eps = 1 is exactly E_BANDR)
                        cases = ([("PINm", {(li, "mix"): ("add", H.dlt[(an - 1, li)]) for li in EARLY}, out["T"][an - 2])] if mode == "T" and an >= 2 else []) + \
                                ([("B", {st: ("set", (H.nat[st], G[an][st] - H.nat[st])) for st in SITES_B}, out["BAND"][an - 1])] if mode == "D" else [])
                        for nm, add, lin in cases:
                            for eps in (.1, .25, 1.0):
                                H.mode = "add"
                                H.add = {s_: ((kd_, eps * d_) if kd_ == "add" else (kd_, d_[0].to(d_[1].device) + eps * d_[1])) for s_, (kd_, d_) in add.items()}
                                with torch.no_grad():
                                    fwd(e); z2 = z32(si).float()
                                H.mode = None; H.add = {}
                                out["fd"][f"{nm}_eps{eps}"] = dict(fd=((z2 - zz) / eps).tolist(), lin=list(lin))
                    del J, P; gc.collect()
                rec["k"][str(k)] = out
            RES.append(rec)
            print(f"[{a.tag}] scene {si + 1}/{a.n} {time.time() - t0:.0f} s", flush=True)
            if (si + 1) % 8 == 0 or si + 1 == a.n:
                json.dump(dict(tag=a.tag, collect=a.collect, seed=a.seed, N=N, ks=KS, early=EARLY, band=BAND, rows=RES), open(OUT.format(a.tag), "w"))
    finally:
        H.remove(); h_lmh.remove()
        for p_ in v.model.parameters():
            p_.requires_grad_(False)
    print(f"[{a.tag}] 끝 -> {OUT.format(a.tag)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--collect", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--pixels", type=int, default=160); ap.add_argument("--device_map", default=None)
    ap.add_argument("--fd", type=int, default=0)
    main(ap.parse_args())
