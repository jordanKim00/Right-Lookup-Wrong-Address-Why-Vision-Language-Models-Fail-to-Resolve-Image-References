"""S918 tool (address vs reading strength; advisor review 2, 10-22): QPathK with two extra keys in a reader layer's edit dict, applied in the
float32 recompute of the run's rows (QPathB):
  mass  dict(target=[R, H] picture attention mass per row (cfg rows order) and head, img=[picture token positions]): after the softmax each
        row and head's attention is rescaled so that the picture positions sum to the target, keeping the distribution within the pictures
        and within the other positions (the address - the relative picture scores - is untouched);
  gate  [R, H, hd] output-gate pre-activations (a donor's): the output is sigmoid(gate) * (A V) with this run's own read A V.
Extra cfg key grabg = set of readers: store the run's own gate pre-activations ('g', reader) [R, H, hd].  With cap, ('img2', reader) [R, H]
is the picture mass actually used after the edit (every row).  The reader hook below is QPathB._hook with these additions (generated from
exp/c_qpath_b.py; QPathB itself is unchanged)."""
from __future__ import annotations

import torch
from c_tools import rotate_half
from c_qpath import QPath
from c_qpath_k import QPathK


class QPathRM(QPathK):
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
                if li in c.get("grabg", ()) and gate is not None:
                    self.store[b][("g", li)] = gate.detach().cpu()
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
                eds = [] if ed is None else ed if isinstance(ed, list) else [ed]
                me = next((e_["mass"] for e_ in eds if isinstance(e_, dict) and "mass" in e_), None)
                ge = next((e_["gate"] for e_ in eds if isinstance(e_, dict) and "gate" in e_), None)
                if me is not None:                                                  # picture mass per row and head <- target (shape kept)
                    img = torch.zeros(S, dtype=torch.bool, device=q.device); img[torch.tensor(me["img"], device=q.device)] = True
                    tgt = me["target"].to(q.device).float().clamp(1e-6, 1 - 1e-6)                    # [R, H]
                    mI = att[:, :, img].sum(-1).clamp_min(1e-12); mO = (1 - mI).clamp_min(1e-12)
                    att = att.clone()
                    att[:, :, img] = att[:, :, img] * (tgt / mI)[..., None]
                    att[:, :, ~img] = att[:, :, ~img] * ((1 - tgt) / mO)[..., None]
                if cap is not None:
                    own2 = torch.tensor(cap["owner"], device=q.device)
                    self.store[b][("img2", li)] = att[:, :, own2 >= 0].sum(-1).cpu()                   # achieved picture mass, every row
                o = torch.einsum("ths,shd->thd", att, V)
                if ge is not None and gate is not None:                             # the donor's output gate on this run's read
                    o = torch.sigmoid(ge.to(q.device).float()) * o
                elif gate is not None:
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

    def _hook_d(self, li):
        """QPathD wraps the reader hooks with QPathB._hook called by name, which would bypass the hook above: use this class's reader hook
        unless a sample carries a 'delta' at this reader (then the inherited delta path, which never sees 'mass' / 'gate' / 'grabg')."""
        mine = self._hook(li)
        parent = super()._hook_d(li)

        def h(mod, args, kwargs, out):
            if self.cfgs is not None and any(c is not None and li in c.get("delta", {}) for c in self.cfgs):
                return parent(mod, args, kwargs, out)
            return mine(mod, args, kwargs, out)
        return h
