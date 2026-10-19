"""Batched variant of exp/c_qpath.py (S814 on): the same per-sample computations as QPath, one configuration per batch element.
cfgs = list (length = batch) of None or a dict with the keys of QPath's cfg that S814 uses:
  rows      list of row indices (query rows) for this sample
  cap       dict(rows=[indices into rows], owner=[block of every position or -1], nb, opt=[menu-word positions]): block scores ('cap'),
            attention weights on the menu words ('opt') and on all image tokens ('img') of the cap rows (passive)
  grabq     set of readers: store the query (after q_norm, before RoPE) of every row, [R, H, hd]
  grabkb    set of readers: store the block-mean post-RoPE keys per query head, [nb, H, hd] (needs cap's owner / nb; S816 U-complement fidelity)
  grabqe    set of readers: store the post-RoPE query of every row before any edit, [R, H, hd] (S827)
  cap['headout']  True: also store per head, for the cap rows, the norm of the gated output written to the residual, of its picture part, and
            the mean output gate ('hout', [3, r, H]; passive; S827)
  edit      {reader: dict(qsub=dict(owner, nb, part='U'|'perp', donors=[(w, q)], self=False, uref=None), erows=None | [indices into rows])}:
            the query edit of QPath._qsub, applied only to this sample (and with erows only to those rows, the donors given for them); the
            attention of the sample's rows is recomputed in float32 (as QPath) and written back for this sample; the value may also be a list
            of such dicts on disjoint erows (several edits in one layer, S814 F arms)
  noop      set of readers: float32 recompute without edit (exactness reference)
  ko        dict(layers=set of readers, rows=[indices into rows], pos=[key positions], heads=None | {reader: [heads]}): attention knockout -
            those rows cannot attend to those positions at those readers (only the listed heads when heads is given; float32 recompute; without
            an edit or noop only the knocked-out rows are written back; S815, S821)
Store: self.store[b][(key, reader)].  Samples without a cfg are untouched.  Residual pre-hooks of QPath (grab / delta) are not used here.
Attention without an output gate (Qwen3-VL, dense) is handled as well (q_proj of width heads x head_dim; no sigmoid gate)."""
from __future__ import annotations

import threading
import torch
from c_tools import rotate_half
from c_qpath import QPath


class QPathB:
    # cfgs / store are per thread (10-16, exp/pipe_run.py runs micro-batches of one batch in threads so the cards of a balanced device map
    # work at once); a single-threaded caller sees exactly the old attributes.
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))
    store = property(lambda self: getattr(self._tl, "store", {}), lambda self, x: setattr(self._tl, "store", x))

    def __init__(self, v, readers):
        self._tl = threading.local()
        self.v = v; self.cfgs = None; self.store = {}; self.hooks = []; self.readers = set(readers)
        for li in readers:
            self.hooks.append(v.layers[li].self_attn.register_forward_hook(self._hook(li), with_kwargs=True))

    def set(self, cfgs):
        self.cfgs = cfgs; self.store = {} if cfgs is None else {b: {} for b in range(len(cfgs))}

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _hook(self, li):
        def h(mod, args, kwargs, out):
            if self.cfgs is None:
                return out
            t0 = None
            for b, c in enumerate(self.cfgs):
                if c is None:
                    continue
                ed = c.get("edit", {}).get(li); grabq = li in c.get("grabq", ()); noop = li in c.get("noop", ()); cap = c.get("cap")
                ko = c.get("ko"); kol = ko is not None and li in ko["layers"]
                grabkb = li in c.get("grabkb", ())
                if ed is None and not grabq and not noop and cap is None and not kol and not grabkb:
                    continue
                hs = kwargs["hidden_states"][b].float()
                cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
                S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
                rows = c["rows"]; R = len(rows)
                qproj = lambda x: torch.nn.functional.linear(x, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
                gated = mod.q_proj.weight.shape[0] == 2 * nh * hd                 # Qwen3.5: query + output gate; Qwen3-VL: query only
                qg = qproj(hs[rows]).view(R, nh, 2 * hd if gated else hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:] if gated else None
                if grabq:
                    self.store[b][("q", li)] = q.cpu()
                k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
                vv = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
                kh = torch.arange(nh, device=q.device) // mod.num_key_value_groups
                K = k[:, kh]; V = vv[:, kh]
                qe = torch.cat([q[..., :rd] * cos[rows][:, None, :] + rotate_half(q[..., :rd]) * sin[rows][:, None, :], q[..., rd:]], -1)
                ke = torch.cat([K[..., :rd] * cos[:, None, :] + rotate_half(K[..., :rd]) * sin[:, None, :], K[..., rd:]], -1)
                if li in c.get("grabqe", ()):                                       # S827: post-RoPE query before any edit
                    self.store[b][("qe", li)] = qe.cpu()
                for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):   # a list = several edits on disjoint row sets
                    if "qsub" not in e_:
                        continue
                    sub = e_.get("erows")                                       # edit only these rows (indices into rows); donors given for them
                    if sub is None:
                        qe = QPath._qsub(QPath, qe, ke, e_["qsub"], cos[rows], sin[rows], rd)
                    else:
                        qe = qe.clone(); cr_, sr_ = cos[rows][sub], sin[rows][sub]
                        qe[sub] = QPath._qsub(QPath, qe[sub], ke, e_["qsub"], cr_, sr_, rd)
                lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
                if grabkb:
                    own_ = torch.tensor(cap["owner"], device=q.device)
                    self.store[b][("kb", li)] = torch.stack([ke[own_ == bb].mean(0) for bb in range(cap["nb"])]).cpu()
                if kol:
                    kr = torch.tensor(ko["rows"], device=q.device, dtype=torch.long); kp = torch.tensor(ko["pos"], device=q.device, dtype=torch.long)
                    hk = ko.get("heads", {}).get(li) if ko.get("heads") is not None else None   # S821: only these heads (None = every head)
                    if len(kp) and len(kr) and hk is None:                      # an empty set = exactness of the recompute path
                        lg[kr[:, None], :, kp[None, :]] = float("-inf")
                    elif len(kp) and len(kr) and len(hk):
                        hh_ = torch.tensor(hk, device=q.device, dtype=torch.long)
                        lg[kr[:, None, None], hh_[None, :, None], kp[None, None, :]] = float("-inf")
                if cap is not None:
                    own = torch.tensor(cap["owner"], device=q.device); cr = cap["rows"]
                    self.store[b][("cap", li)] = torch.stack([torch.logsumexp(lg[cr][:, :, own == bb], -1) for bb in range(cap["nb"])], -1).cpu()
                    if cap.get("headout"):                                          # S827: per-head gated output of the cap rows (passive)
                        rr = torch.tensor([rows[i] for i in cr], device=q.device)
                        mk_ = torch.arange(S, device=q.device)[None, :] > rr[:, None]
                        ac = torch.softmax(lg[cr].masked_fill(mk_[:, None, :], float("-inf")), -1)             # [r, H, S]
                        g_ = torch.sigmoid(gate[cr]) if gate is not None else 1.0
                        Wo = mod.o_proj.weight.float().view(-1, nh, hd)                                       # [E, H, hd]
                        oa = torch.einsum("rhs,shd->rhd", ac, V) * g_
                        oi = torch.einsum("rhs,shd->rhd", ac * (own >= 0)[None, None, :], V) * g_
                        self.store[b][("hout", li)] = torch.stack([torch.einsum("rhd,ehd->rhe", oa, Wo).norm(dim=-1),
                                                                    torch.einsum("rhd,ehd->rhe", oi, Wo).norm(dim=-1),
                                                                    (g_.mean(-1) if gate is not None else torch.ones_like(oa[..., 0]))], 0).cpu()   # [3, r, H]
                    if cap.get("opt") is not None:
                        rr = torch.tensor([rows[i] for i in cr], device=q.device)
                        mk = torch.arange(S, device=q.device)[None, :] > rr[:, None]
                        ac = torch.softmax(lg[cr].masked_fill(mk[:, None, :], float("-inf")), -1)
                        self.store[b][("opt", li)] = ac[:, :, torch.tensor(cap["opt"], device=q.device)].cpu()
                        self.store[b][("img", li)] = ac[:, :, own >= 0].sum(-1).cpu()
                if ed is None and not noop and not kol:
                    continue
                mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)
                o = torch.einsum("ths,shd->thd", att, V)
                if gate is not None:
                    o = torch.sigmoid(gate) * o
                new = torch.nn.functional.linear(o.reshape(R, -1), mod.o_proj.weight.float(), None if mod.o_proj.bias is None else mod.o_proj.bias.float())
                if t0 is None:
                    t0 = (out[0] if isinstance(out, tuple) else out).clone()
                wr = list(range(R)) if (ed is not None or noop) else list(ko["rows"])
                t0[b, [rows[i] for i in wr]] = new[wr].to(t0.dtype)
            if t0 is None:
                return out
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h
