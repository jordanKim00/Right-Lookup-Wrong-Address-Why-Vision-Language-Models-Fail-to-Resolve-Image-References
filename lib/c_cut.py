"""P2 tool (S857): cut a source span of tokens (pictures 7-8) from the text rows that follow it, per sample of a batch, through every route
of the Qwen3.5 hybrid, and transplant late queries within the same forward pass.

Routes from a source span [s, e] (picture 7's vision-start .. picture 8's vision-end) into the text rows t >= t0 = e + 1:
  full attention   the text rows' attention to positions s..e (every full-attention layer)
  DeltaNet state   the gated delta-rule state that the text rows read (every linear-attention layer)
  DeltaNet conv    the 4-tap causal convolution on q/k/v, whose windows at t0..t0+2 contain the last source tokens
A cut at layer l removes the route at that layer only (text rows may read the source again at uncut layers, from the source tokens' own states).

SourceCut(v).set(dict(s=s, e=e, cuts=[None | dict(fa=set(layers), dn=set(layers)) per sample])) switches the instrument on for one batch:
  - every full-attention layer gets an explicit boolean mask for every sample (causal; for a sample with l in fa, rows >= t0 may not attend to
    s..e), so all samples of the batch go through the same masked SDPA kernel whether or not they are cut;
  - every linear-attention layer runs the chunked delta rule in three segments [0, s), [s, t0), [t0, T) for every sample, chaining the state
    (exact; S561 showed that a finite decay leaks, so the source is skipped by giving segment 3 the state after segment 1 instead of the
    state after segment 2) - a cut sample (l in dn) gets the state after [0, s), the others the full chain;
  - for a cut sample the convolution outputs at t0..t0+K-2 are recomputed from the spliced window (inputs s-K+1..s-1 followed by t0..).
  With the instrument on and no cut, the computation equals the natural one up to the segment boundaries' float32 rounding.
Readers recompute the read rows' attention in float32 with their own causal mask (exp/c_qpath_b.py); the caller adds the source positions to
QPathB's 'ko' for the cut readers (see ko_for).

QFrom(v, layers, qtool): per-sample cfg key 'qfrom' {layer: src batch index} - at that full-attention reader the pre-RoPE query of the sample's
read rows is replaced by the src sample's query of the same rows, computed in this forward pass from src's own hidden states with QPathB's
float32 code (keys, values, the output gate and every other row stay the recipient's; same positions, so no re-rotation is needed).  It is
written as a QPathB 'all' edit, so a sample with qfrom = itself must reproduce its plain recompute bit for bit (S856b SELF)."""
from __future__ import annotations

import threading
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M


class SourceCut:
    cfg = property(lambda self: getattr(self._tl, "cfg", None), lambda self, x: setattr(self._tl, "cfg", x))
    cur = property(lambda self: getattr(self._tl, "cur", None), lambda self, x: setattr(self._tl, "cur", x))

    def __init__(self, v):
        self._tl = threading.local()
        self.v = v; self.hooks = []; self.calls = {"conv": 0, "chunk": 0, "mask": 0}
        self.FA = [i for i, L in enumerate(v.layers) if getattr(L, "self_attn", None) is not None]
        self.DN = [i for i, L in enumerate(v.layers) if getattr(L, "linear_attn", None) is not None]
        self._masks = {}
        for li in self.FA:
            self.hooks.append(v.layers[li].register_forward_pre_hook(self._fa_pre(li), with_kwargs=True))
        for li in self.DN:
            self.hooks.append(v.layers[li].linear_attn.register_forward_pre_hook(self._dn_pre(li), with_kwargs=True))
        if not hasattr(M, "_s857_orig_conv"):                                # the true originals, kept on the module across resident-worker jobs
            M._s857_orig_conv, M._s857_orig_chunk = M.causal_conv1d_fn, M.torch_chunk_gated_delta_rule
        self._orig_conv, self._orig_chunk = M._s857_orig_conv, M._s857_orig_chunk
        M.causal_conv1d_fn = self._conv
        M.torch_chunk_gated_delta_rule = self._chunk

    def set(self, cfg):
        self.cfg = cfg; self._masks = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        M.causal_conv1d_fn, M.torch_chunk_gated_delta_rule = M._s857_orig_conv, M._s857_orig_chunk

    @staticmethod
    def ko_for(s, e, fa_layers, readers, nrows):
        """QPathB 'ko' for a cut sample: the read rows may not attend to the source at the cut readers."""
        ly = set(fa_layers) & set(readers)
        return dict(layers=ly, rows=list(range(nrows)), pos=list(range(s, e + 1))) if ly else None

    # ---- full attention: explicit masks for every sample
    def _fa_pre(self, li):
        def h(mod, args, kwargs):
            c = self.cfg
            if c is None:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            B, S = hs.shape[0], hs.shape[1]; dev = hs.device
            key = (dev, li)
            if key not in self._masks:
                m = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))[None, None].repeat(B, 1, 1, 1)
                s, t0 = c["s"], c["e"] + 1
                for b, cu in enumerate(c["cuts"]):
                    if cu is not None and li in cu.get("fa", ()):
                        m[b, 0, t0:, s:t0] = False
                self._masks[key] = m
            kwargs = dict(kwargs); kwargs["attention_mask"] = self._masks[key]; self.calls["mask"] += 1
            return args, kwargs
        return h

    # ---- DeltaNet: segment the delta rule, splice the convolution
    def _dn_pre(self, li):
        def h(mod, args, kwargs):
            self.cur = li
            return None
        return h

    def _conv(self, hidden_states, weight, bias=None, activation=None, **kw):
        out = self._orig_conv(hidden_states, weight, bias, activation=activation, **kw)
        c = self.cfg
        if c is None:
            return out
        self.calls["conv"] += 1
        bs = [b for b, cu in enumerate(c["cuts"]) if cu is not None and self.cur in cu.get("dn", ())]
        if not bs:
            return out
        K = weight.shape[-1]; s, t0 = c["s"], c["e"] + 1
        idx = torch.tensor(list(range(s - (K - 1), s)) + list(range(t0, t0 + K - 1)), device=hidden_states.device)
        win = hidden_states[bs][:, :, idx].contiguous()                       # [nb, C, 2(K-1)]: spliced window
        seg = self._orig_conv(win, weight, bias, activation=activation)
        out = out.clone()
        out[bs, :, t0:t0 + K - 1] = seg[:, :, K - 1:]
        return out

    def _chunk(self, query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        c = self.cfg
        f = self._orig_chunk
        if c is None:
            return f(query, key, value, g=g, beta=beta, chunk_size=chunk_size, initial_state=initial_state, output_final_state=output_final_state,
                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        self.calls["chunk"] += 1
        s, t0 = c["s"], c["e"] + 1
        seg = lambda a, b_, init, fin: f(query[:, a:b_], key[:, a:b_], value[:, a:b_], g=g[:, a:b_], beta=beta[:, a:b_], chunk_size=chunk_size,
                                         initial_state=init, output_final_state=fin, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        T = query.shape[1]
        o1, s1 = seg(0, s, initial_state, True)
        o2, s2 = seg(s, t0, s1, True)
        init3 = s2.clone()
        bs = [b for b, cu in enumerate(c["cuts"]) if cu is not None and self.cur in cu.get("dn", ())]
        if bs:
            init3[bs] = s1[bs]
        o3, s3 = seg(t0, T, init3, output_final_state)
        return torch.cat([o1, o2, o3], 1), s3


class QFrom:
    def __init__(self, v, layers, qtool):
        self.v = v; self.q = qtool; self.hooks = []
        for li in layers:
            self.hooks.append(v.layers[li].self_attn.register_forward_pre_hook(self._pre(li), with_kwargs=True, prepend=True))

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _pre(self, li):
        def h(mod, args, kwargs):
            cfgs = self.q.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and li in (c.get("qfrom") or {})]
            if not todo:
                return None
            hsall = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            hd = mod.head_dim; nh = mod.config.num_attention_heads
            gated = mod.q_proj.weight.shape[0] == 2 * nh * hd
            for b, c in todo:
                src = c["qfrom"][li]; rows = c["rows"]; R = len(rows)
                hs = hsall[src].float()
                qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float()
                                                ).view(R, nh, 2 * hd if gated else hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
                ed = dict(qsub=dict(part="all", donors=[(1.0, q)]))
                c.setdefault("edit", {})
                prev = c["edit"].get(li)
                if prev is not None:
                    raise ValueError(f"qfrom and another edit at layer {li}")
                c["edit"][li] = ed
            return None
        return h
