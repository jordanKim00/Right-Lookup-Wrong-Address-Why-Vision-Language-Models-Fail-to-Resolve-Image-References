"""Tools for dissecting the address-composition computation C inside Qwen3.5 layers (S687b and later).

FullAttn: per full-attention layer, recompute the output rows of chosen query positions with components exchanged:
    q (pre-RoPE, after q_norm; parts "all" / "rot" (first rotary_dim dims) / "pass"), the output gate, keys (content swap between
    two position sets), key positions (RoPE cos/sin swap between two position sets) and values (swap), for a chosen head set.
    Every head of the chosen rows is recomputed in float32 (NOOP = recompute without changes, the control for the recomputation).
DeltaNet: per Gated DeltaNet layer, recompute the core output rows o_t = sum_s w(t, s) v_s exactly,
    w(t, s) = beta_s k_s . A_{s+1} ... A_t q_t, A_j = exp(g_j)(I - beta_j k_j k_j^T), with the query of the rows and / or the memory
    (k, v, beta, g of every position before the tail) taken from a donor run, for a chosen value-head set.
Grab mode records what an edit needs from a donor run (q, gate per full layer; q, k, v, g, beta per DeltaNet layer).
"""
from __future__ import annotations

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


class Ctx:
    """shared state: rows (query positions), the run's mode, grabbed tensors, edits."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows = None; self.grab = False; self.grab_dn = False; self.grab_kv = None; self.store = {}; self.edit = {}; self.cur = None


class FullAttn:
    def __init__(self, v, layers, ctx):
        self.ctx = ctx; self.hooks = []
        for li in layers:
            mod = v.layers[li].self_attn
            self.hooks.append(mod.register_forward_hook(self._hook(li), with_kwargs=True))

    def _parts(self, mod, hs, cos, sin):
        S = hs.shape[0]; hd = mod.head_dim; nh = mod.config.num_attention_heads
        qg = torch.nn.functional.linear(hs, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
        qg = qg.view(S, nh, 2 * hd)
        q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float(); gate = qg[..., hd:]
        k = mod.k_norm(torch.nn.functional.linear(hs, mod.k_proj.weight.float()).view(S, -1, hd).to(mod.k_norm.weight.dtype)).float()
        v = torch.nn.functional.linear(hs, mod.v_proj.weight.float()).view(S, -1, hd)
        return q, gate, k, v

    def _hook(self, li):
        def h(mod, args, kwargs, out):
            c = self.ctx
            if c.rows is None:
                return out
            grab = c.grab; ed = c.edit.get(("F", li))
            if not grab and ed is None:
                return out
            hs = (kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0])[0].float()
            cos, sin = kwargs["position_embeddings"]; cos, sin = cos[0].float(), sin[0].float()
            q, gate, k, v = self._parts(mod, hs, cos, sin)
            rows = c.rows
            if grab:
                c.store[("F", li, "q")] = q[rows].cpu(); c.store[("F", li, "gate")] = gate[rows].cpu()
                if getattr(c, "grab_kv", None):
                    gp = c.grab_kv
                    c.store[("F", li, "k")] = k[gp].cpu(); c.store[("F", li, "v")] = v[gp].cpu()
            if ed is None:
                return out
            nh = q.shape[1]; grp = mod.num_key_value_groups; rd = cos.shape[-1]
            kh = torch.arange(nh, device=q.device) // grp
            K = k[:, kh].clone(); V = v[:, kh].clone()                                     # per query head
            Cs, Sn = cos[:, None, :].expand(-1, nh, -1).clone(), sin[:, None, :].expand(-1, nh, -1).clone()
            qr = q[rows].clone(); gr = gate[rows].clone()
            H = torch.tensor(ed.get("heads", list(range(nh))), device=q.device, dtype=torch.long)
            if "qin" in ed:                                                                # path-specific query input for the head set
                qi = ed["qin"].to(q.device).float()
                qg2 = torch.nn.functional.linear(qi, mod.q_proj.weight.float(), None if mod.q_proj.bias is None else mod.q_proj.bias.float())
                qg2 = qg2.view(len(rows), nh, 2 * mod.head_dim)
                q2 = mod.q_norm(qg2[..., :mod.head_dim].to(mod.q_norm.weight.dtype)).float()
                qr[:, H] = q2[:, H]; gr[:, H] = qg2[..., mod.head_dim:][:, H]
            if "q" in ed:                                                                  # donor q for the head set
                qd = ed["q"].to(q.device); part = ed.get("qpart", "all")
                sl = slice(0, rd) if part == "rot" else slice(rd, None) if part == "pass" else slice(None)
                qr[:, H, sl] = qd[:, H, sl]
            if "gate" in ed:
                gr[:, H] = ed["gate"].to(q.device)[:, H]
            if "kvpos" in ed:                                                              # donor keys / values at given positions
                pp, dk, dv = ed["kvpos"]
                P_ = torch.tensor(pp, device=q.device)
                K[P_[:, None], H[None, :]] = dk.to(q.device)[:, kh][:, H]
                V[P_[:, None], H[None, :]] = dv.to(q.device)[:, kh][:, H]
            for kind in ("kswap", "pswap", "vswap"):
                if kind in ed:
                    P1 = torch.tensor(ed[kind][0], device=q.device); P2 = torch.tensor(ed[kind][1], device=q.device)
                    part = ed.get("kpart", "all") if kind == "kswap" else "all"
                    sl = slice(0, rd) if part == "rot" else slice(rd, None) if part == "pass" else slice(None)
                    T = K if kind == "kswap" else V
                    if kind == "pswap":
                        for A in (Cs, Sn):
                            a1 = A[P1][:, H].clone(); a2 = A[P2][:, H].clone()
                            A[P1[:, None], H[None, :]] = a2; A[P2[:, None], H[None, :]] = a1
                    else:
                        a1 = T[P1][:, H, sl].clone(); a2 = T[P2][:, H, sl].clone()
                        idx1 = (P1[:, None], H[None, :]); idx2 = (P2[:, None], H[None, :])
                        tmp1 = T[idx1]; tmp1[..., sl] = a2; T[idx1] = tmp1
                        tmp2 = T[idx2]; tmp2[..., sl] = a1; T[idx2] = tmp2
            qpos_c, qpos_s = cos[rows][:, None, :], sin[rows][:, None, :]
            qe = torch.cat([qr[..., :rd] * qpos_c + rotate_half(qr[..., :rd]) * qpos_s, qr[..., rd:]], -1)
            ke = torch.cat([K[..., :rd] * Cs + rotate_half(K[..., :rd]) * Sn, K[..., rd:]], -1)
            lg = torch.einsum("thd,shd->ths", qe, ke) * mod.scaling
            S = hs.shape[0]
            mask = torch.arange(S, device=q.device)[None, :] > torch.tensor(rows, device=q.device)[:, None]
            lg = lg.masked_fill(mask[:, None, :], float("-inf"))
            if "ko" in ed:                                                                 # sources the head set may not read
                kom = torch.zeros(S, dtype=torch.bool, device=q.device); kom[torch.tensor(ed["ko"], device=q.device)] = True
                sel = torch.zeros(nh, dtype=torch.bool, device=q.device)
                sel[torch.tensor(ed["ko_heads"], device=q.device, dtype=torch.long) if "ko_heads" in ed else H] = True   # ko_heads: own head set (S706)
                lg = lg.masked_fill(kom[None, None, :] & sel[None, :, None], float("-inf"))
            if "mblocks" in ed:                                                            # head margins: log-mass(target) - log-mass(literal)
                Pt = torch.tensor(ed["mblocks"][0], device=q.device); Pl = torch.tensor(ed["mblocks"][1], device=q.device)
                mh = torch.logsumexp(lg[:, :, Pt], -1) - torch.logsumexp(lg[:, :, Pl], -1)
                c.store[("m", li)] = mh[:, H].detach().cpu()
                if len(ed["mblocks"]) > 2:
                    Px = torch.tensor(ed["mblocks"][2], device=q.device)
                    c.store[("mx", li)] = (torch.logsumexp(lg[:, :, Px], -1) - torch.logsumexp(lg[:, :, Pt], -1))[:, H].detach().cpu()
            att = torch.softmax(lg, -1)
            o = torch.sigmoid(gr) * torch.einsum("ths,shd->thd", att, V)
            new = torch.nn.functional.linear(o.reshape(len(rows), -1), mod.o_proj.weight.float(),
                                             None if mod.o_proj.bias is None else mod.o_proj.bias.float())
            t0 = out[0] if isinstance(out, tuple) else out
            t0 = t0.clone(); t0[0, rows] = new.to(t0.dtype)
            return (t0,) + tuple(out[1:]) if isinstance(out, tuple) else t0
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


class DeltaNet:
    """wraps torch_chunk_gated_delta_rule; ctx.cur is set by a pre-hook on each linear_attn."""
    def __init__(self, v, layers, ctx):
        self.ctx = ctx; self.layers = set(layers); self.orig = M.torch_chunk_gated_delta_rule; self.hooks = []
        M.torch_chunk_gated_delta_rule = self._rule
        for i, L in enumerate(v.layers):
            if getattr(L, "linear_attn", None) is not None:
                self.hooks.append(L.linear_attn.register_forward_pre_hook(lambda mod, args, i=i: setattr(self.ctx, "cur", i)))

    @staticmethod
    def norm_qk(q, k):
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / (q.shape[-1] ** .5)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        return q, k

    @staticmethod
    def rows_out(q, k, v, g, b, rows, ko=None, ko_heads=None):
        """exact o_t for t in rows (sorted), each [S, H, d] / [S, H]; ko = source positions whose value terms are dropped
        (for ko_heads, default all) -- the memory dynamics (decay and erasure through every position) are kept."""
        R = len(rows); H, dk = q.shape[1], q.shape[2]
        O = torch.zeros(R, H, v.shape[-1], device=q.device)
        r = torch.zeros(R, H, dk, device=q.device)
        idx = {t: i for i, t in enumerate(rows)}
        kos = set(ko) if ko is not None else set()
        hm = torch.ones(H, device=q.device)
        if ko_heads is not None:
            hm = torch.ones(H, device=q.device); hm[torch.tensor(ko_heads, device=q.device)] = 0.
        for s in range(rows[-1], -1, -1):
            if s in idx:
                r[idx[s]] = q[s]
            kr = (r * k[s][None]).sum(-1)                                                  # [R, H]
            term = (b[s][None] * kr)[..., None] * v[s][None]
            if s in kos:
                term = term * (hm if ko_heads is not None else torch.zeros(H, device=q.device))[None, :, None]
            O += term
            r = torch.exp(g[s])[None, :, None] * (r - (b[s][None, :, None] * kr[..., None]) * k[s][None])
        return O

    def _rule(self, query, key, value, g, beta, **kw):
        out = self.orig(query, key, value, g=g, beta=beta, **kw)
        c = self.ctx; li = c.cur
        if c.rows is None or li not in self.layers:
            return out
        ed = c.edit.get(("D", li))
        grab = c.grab and c.grab_dn
        if not grab and ed is None:
            return out
        q, k = self.norm_qk(query[0].float(), key[0].float()); v = value[0].float(); gg = g[0].float(); bb = beta[0].float()
        rows = c.rows
        if grab:
            c.store[("D", li)] = dict(q=q.cpu(), k=k.cpu(), v=v.cpu(), g=gg.cpu(), b=bb.cpu())
        if ed is None:
            return out
        H = torch.tensor(ed.get("heads", list(range(q.shape[1]))), device=q.device, dtype=torch.long)
        q2, k2, v2, g2, b2 = q.clone(), k.clone(), v.clone(), gg.clone(), bb.clone()
        if "mem" in ed:                                                                     # donor memory: positions before the tail
            dm = ed["mem"]; pre = slice(0, ed["tail0"])
            for T, key_ in ((k2, "k"), (v2, "v")):
                T[pre][:, H] = dm[key_][pre][:, H].to(q.device)
            for T, key_ in ((g2, "g"), (b2, "b")):
                T[pre][:, H] = dm[key_][pre][:, H].to(q.device)
        if "mempos" in ed:                                                                 # donor k, v, g, beta at given positions only
            pp, dm = ed["mempos"]
            P_ = torch.tensor(pp, device=q.device)
            for T_, key_ in ((k2, "k"), (v2, "v"), (g2, "g"), (b2, "b")):
                T_[P_] = dm[key_][pp].to(q.device)
        if "vswap" in ed:                                                                  # exchange two position sets' values in the readout
            P1 = torch.tensor(ed["vswap"][0], device=q.device); P2 = torch.tensor(ed["vswap"][1], device=q.device)
            t1 = v2[P1].clone(); v2[P1] = v2[P2]; v2[P2] = t1
        if "q" in ed:
            qd = ed["q"].to(q.device)
            for i, t in enumerate(rows):
                q2[t, H] = qd[t, H] if qd.shape[0] > max(rows) else qd[i, H]
        O = self.rows_out(q2, k2, v2, g2, b2, rows, ko=ed.get("ko"), ko_heads=ed.get("heads") if "ko" in ed and "heads" in ed else None)
        o0 = out[0].clone()
        o0[0, rows] = O.to(o0.dtype)
        return (o0,) + tuple(out[1:])

    def remove(self):
        M.torch_chunk_gated_delta_rule = self.orig
        for h in self.hooks:
            h.remove()
