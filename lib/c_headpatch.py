"""S860 tool (prepared 10-20 for the head-level causal tests that follow S859): exact head-output patching inside full-attention and Gated
DeltaNet layers of Qwen3.5.  A forward pre-hook on the layer's output projection (FA: o_proj, whose input is the gated per-head attention
output [B, S, nh * hd]; DN: out_proj, whose input is the normalised, z-gated per-head core [B, S, Hv * dv]) edits chosen heads' slices at
chosen rows of a sample, so the edit reaches the residual only through those heads' own output weights.
Per-sample cfg (list over the batch, None = untouched):
  dict(rows=[positions], edits={layer: [dict(heads=[...], op=..., src=batch index, coef=float, vec=tensor)]})
    op 'swap'   the heads' slices <- the src sample's slices of the same rows (same forward pass; src itself is untouched)
    op 'lin'    the heads' slices <- this + coef * (src - this)   (coef 1 = swap, 0.5 = halfway, 2 = double the difference)
    op 'zero'   the heads' slices <- 0
    op 'add'    the heads' slices += vec[row, head, :] (a fixed vector in the head's output space, [R, len(heads), d])
A self swap (src = the sample itself) returns the input bit for bit (the exactness control).  The edits of a layer are applied in the
order given, on the unpatched input of every sample (sources are read before any write).
Limitation: at a full-attention layer whose rows QPathB recomputes (pinned early look layers, NOOP arms) the recompute writes W_O o itself
without calling o_proj, so a pre-hook edit there is overwritten for those rows - such layers need a post-hook variant that re-applies the
pins (as exp/c_anat.py does) and corrects the written rows by W_O[:, h](o_src,h - o_this,h).  DeltaNet layers and unedited FA layers are exact."""
from __future__ import annotations

import threading
import torch


class HeadPatch:
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def __init__(self, v, layers):
        self._tl = threading.local(); self.v = v; self.cfgs = None; self.hooks = []; self.calls = 0
        for l in layers:
            L = v.layers[l]
            if getattr(L, "linear_attn", None) is not None:
                proj = L.linear_attn.out_proj; d = L.linear_attn.head_v_dim
            else:
                proj = L.self_attn.o_proj; d = L.self_attn.head_dim
            self.hooks.append(proj.register_forward_pre_hook(self._pre(l, d)))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _pre(self, l, d):
        def h(mod, args):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and l in c.get("edits", {})]
            if not todo:
                return None
            x = args[0]
            x0 = x.clone()                                                                   # sources: the unpatched input
            x2 = x.clone()
            S = x.shape[1]
            for b, c in todo:
                rows = torch.tensor(c["rows"], device=x.device)
                for e in c["edits"][l]:
                    for j, hh in enumerate(e["heads"]):
                        sl = slice(hh * d, (hh + 1) * d)
                        op = e["op"]
                        if op == "swap":
                            x2[b, rows, sl] = x0[e["src"], rows, sl]
                        elif op == "lin":
                            cur = x0[b, rows, sl].float(); src = x0[e["src"], rows, sl].float()
                            x2[b, rows, sl] = (cur + float(e["coef"]) * (src - cur)).to(x2.dtype)
                        elif op == "zero":
                            x2[b, rows, sl] = 0
                        elif op == "add":
                            x2[b, rows, sl] = (x2[b, rows, sl].float() + e["vec"][:, j].to(x.device).float()).to(x2.dtype)
                        else:
                            raise ValueError(op)
            self.calls += 1
            return (x2,) + tuple(args[1:])
        return h


class FAHeadPatchPinned:
    """Head-output edits at full-attention layers whose rows QPathB recomputes (pinned early look layers): a forward hook registered after the
    QPath tool's hooks re-applies each involved sample's pins (QPath._qsub, float32, as QPathB), recomputes the per-head gated outputs o_h of
    the sample's rows and of the source sample's rows, and corrects the written rows by sum_h W_O[:, h] (o_new,h - o_this,h), where o_new is
    the source's o_h ('swap'), this + coef (src - this) ('lin'), 0 ('zero'), sigmoid(gate_src,h) * (att V)_this,h ('gate': the source's
    output gate on this sample's own attention read), or sigmoid(gate_this,h) * (att V)_src,h ('read': the source's read under this gate).  A self swap / self gate adds exactly zero (bit-identical output).
    cfgs[b] = dict(rows=[the sample's QPath rows], edit=the sample's QPath 'edit' dict, prow=[positions to patch, a subset of rows],
                   hedits={layer: [dict(heads=[...], op='swap' | 'lin' | 'zero', src=batch index, coef=float)]})"""
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def __init__(self, v, layers):
        self._tl = threading.local(); self.v = v; self.cfgs = None; self.hooks = []; self.calls = 0
        for l in layers:
            self.hooks.append(v.layers[l].self_attn.register_forward_hook(self._post(l), with_kwargs=True))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _heads_out(self, mod, l, b, kwargs, prow, parts=False):
        """gated per-head outputs [len(prow), nh, hd] (float32) of sample b at positions prow, with b's pins at layer l re-applied
        (parts=True: also the ungated read and the gate's sigmoid)."""
        from c_tools import rotate_half
        from c_qpath import QPath
        c = self.cfgs[b]; rows = c["rows"]
        ai = [rows.index(t) for t in prow]
        hs = kwargs["hidden_states"][b].float()
        cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
        S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]; R = len(rows)
        qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
        k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
        vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
        kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
        K = k[:, kh]; V = vv[:, kh]
        qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
        ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
        ed = c.get("edit", {}).get(l)
        for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):
            if "qsub" not in e_:
                continue
            sub = e_.get("erows")
            if sub is None:
                qe = QPath._qsub(QPath, qe, ke, e_["qsub"], cos[rows], sin[rows], rd)
            else:
                qe = qe.clone(); cr_, sr_ = cos[rows][sub], sin[rows][sub]
                qe[sub] = QPath._qsub(QPath, qe[sub], ke, e_["qsub"], cr_, sr_, rd)
        qe = qe[ai]; gt = gate[ai]
        lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
        mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(prow, device=hs.device)[:, None]
        att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
        rd_ = torch.einsum("rhs,shd->rhd", att, V); sg = torch.sigmoid(gt)
        return (rd_ * sg, rd_, sg) if parts else rd_ * sg

    def _post(self, l):
        def h(mod, args, kwargs, out):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and l in c.get("hedits", {})]
            if not todo:
                return None
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            Wo = mod.o_proj.weight.float().view(-1, mod.config.num_attention_heads, mod.head_dim)
            cache = {}
            for b, c in todo:
                prow = c["prow"]
                if b not in cache:
                    cache[b] = self._heads_out(mod, l, b, kwargs, prow, parts=True)
                ob, rb, gb = cache[b]
                corr = torch.zeros(len(prow), Wo.shape[0], device=ob.device)
                for e in c["hedits"][l]:
                    src = e.get("src", b)
                    if e["op"] in ("swap", "lin", "gate", "read") and src not in cache:
                        assert cfgs[src] is not None and cfgs[src]["prow"] == prow, "the source must carry the same rows"
                        cache[src] = self._heads_out(mod, l, src, kwargs, prow, parts=True)
                    for hh in e["heads"]:
                        if e["op"] == "swap":
                            dlt = cache[src][0][:, hh] - ob[:, hh]
                        elif e["op"] == "lin":
                            dlt = float(e["coef"]) * (cache[src][0][:, hh] - ob[:, hh])
                        elif e["op"] == "gate":
                            dlt = rb[:, hh] * cache[src][2][:, hh] - ob[:, hh]
                        elif e["op"] == "read":                                              # the source's read under this sample's gate
                            dlt = cache[src][1][:, hh] * gb[:, hh] - ob[:, hh]
                        elif e["op"] == "zero":
                            dlt = -ob[:, hh]
                        else:
                            raise ValueError(e["op"])
                        corr = corr + dlt @ Wo[:, hh].T
                t2[b, prow] = (t[b, prow].float() + corr).to(t2.dtype)
            self.calls += 1
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h


class MLPInputPath:
    """Path patch from a full-attention layer's output into the same layer's MLP only (S860 PATH47): for a sample with cfg src, the MLP of layer l
    reads post_attention_layernorm(residual + a_src) at the rows instead of post_attention_layernorm(residual + a_this), where residual = the
    layer's input and a = the attention outputs (after every other tool's edits) at those rows; the residual stream keeps this sample's own
    attention output, so only the MLP sees the donor's attention.  The sum is formed in the model's dtype exactly as the decoder layer forms it,
    and the norm is applied to the whole batch tensor (the same shape as the layer's own call), so src = the sample itself is bit-identical.
    cfgs[b] = dict(rows=[positions], src=batch index) or None; register after the tools that edit the attention output."""
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def __init__(self, v, layers):
        self._tl = threading.local(); self.v = v; self.cfgs = None; self.hooks = []; self.calls = 0
        self.att = {}; self.res = {}; self.full = {}; self._inside = False
        for l in layers:
            L = v.layers[l]
            self.hooks.append(L.register_forward_pre_hook(self._res(l), with_kwargs=True))
            self.hooks.append(L.self_attn.register_forward_hook(self._att(l), with_kwargs=True))
            self.hooks.append(L.post_attention_layernorm.register_forward_pre_hook(self._mid(l)))
            self.hooks.append(L.mlp.register_forward_pre_hook(self._mlp(l)))

    def set(self, cfgs):
        self.cfgs = cfgs; self.att = {}; self.res = {}; self.full = {}

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _need(self):
        cf = self.cfgs
        if cf is None:
            return set(), None
        s_, rows = set(), None
        for b, c in enumerate(cf):
            if c is not None:
                s_.add(b); s_.add(c["src"]); rows = c["rows"]
        return s_, rows

    def _res(self, l):
        def h(mod, args, kwargs):
            need, rows = self._need()
            if not need:
                return None
            x = args[0] if args else kwargs["hidden_states"]
            for b in need:
                self.res[(l, b)] = x[b, rows].detach().clone()
            return None
        return h

    def _att(self, l):
        def h(mod, args, kwargs, out):
            need, rows = self._need()
            if not need:
                return None
            t = out[0] if isinstance(out, tuple) else out
            for b in need:
                self.att[(l, b)] = t[b, rows].detach().clone()
            return None
        return h

    def _mid(self, l):
        def h(mod, args):
            if self._inside:
                return None
            need, _ = self._need()
            if need:
                self.full[l] = args[0]
            return None
        return h

    def _mlp(self, l):
        def h(mod, args):
            cf = self.cfgs
            if cf is None:
                return None
            todo = [(b, c) for b, c in enumerate(cf) if c is not None]
            if not todo:
                return None
            full = self.full[l].clone()
            for b, c in todo:
                rows = torch.tensor(c["rows"], device=full.device)
                full[b, rows] = self.res[(l, b)].to(full.device) + self.att[(l, c["src"])].to(full.device)
            ln = self.v.layers[l].post_attention_layernorm
            self._inside = True
            try:
                normed = ln(full)
            finally:
                self._inside = False
            x = args[0]; x2 = x.clone()
            for b, c in todo:
                rows = torch.tensor(c["rows"], device=x.device)
                x2[b, rows] = normed[b, rows].to(x2.dtype)
            self.calls += 1
            return (x2,) + tuple(args[1:])
        return h
