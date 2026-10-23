"""S859 tool: passive anatomy of the writer band (27B: layers 46-49) and of its direct path into the late lookup (FA 51/55/59/63).
Nothing is written back: every hook only reads, so a pass with the anatomy on is the pass without it (checked by the caller: logits of the
two passes must be bit-identical).  All anatomy arithmetic is float32.

Two passes over the same batch (same inputs, same pins / edits of the experiment's other tools):
  pass 1 (mode 'stat'): at every late layer L, for each anatomy sample, keep the residual x_L at L's input (anatomy rows) and, per query head h
      and image block b, kbar[h, b] = sum_{i in b} softmax_{i in b}(scaling qe_h . ke_{i,h}) ke_{i,h} (post-RoPE) - the exact gradient of the
      block score z[h, b] = logsumexp_{i in b} scaling qe_h . ke_{i,h} with respect to the post-RoPE query - and the row's RoPE cos / sin.
  pass 2 (mode 'anat'): at the writer layers every unit's contribution to the residual at the rows is computed exactly and split by source
      group, and projected at once through the direct path into each late layer's block scores with the pass-1 statistics:
        FA layer l:  c[h, g] = W_O[:, h] (sigmoid(gate_h) * sum_{s in g} att_h(t, s) V_{s, kv(h)}); att from the query this pass used at l
                     (the sample's QPathB 'edit' pins re-applied with QPath._qsub exactly as QPathB does, float32);
        DN layer l:  c[h, g] = W_out[:, h] (w_n * silu(z_h) * (sum_{s in g} w_h(t, s) v_{s,h}) / rms(o_h(t))), with the exact core
                     o_h(t) = sum_s w_h(t, s) v_{s,h}, w(t, s) = beta_s k_s . A_{s+1} .. A_t q_t, A_j = exp(g_j)(I - beta_j k_j k_j^T), from the q,
                     k, v, g, beta that the layer's delta rule received (q, k l2-normalised, q / sqrt(d_k)) and z = in_proj_z(input);
        MLP layer l: its output rows (one unit, no source split).
      Direct path: dq = J_L(x_L) c with the analytic Jacobian of q(x) = q_norm(W_q input_layernorm(x)) (Qwen3.5 zero-centred RMSNorms),
      rotated by the row's RoPE; dz[h, b] = scaling * dq_h . kbar[h, b] - the first-order change of L's block scores if c is added at L's input.
      'pair' = batch index of another anatomy sample in the same pass: its contributions are also projected at this sample's statistics
      (for differences such as END - START at the same early position, linearised where the late lookup of this sample runs); 'pairs' =
      [(prefix, batch index), ...] gives several such references (stored as <prefix>_..., the gate / read split as <prefix>_gate_... and
      <prefix>_read_...; 'pair' alone is the same as pairs = [('pair', index)] with the gate / read arrays named gate_... / read_...).
cfg['anat'] = dict(rows=[positions], group=[group id per position, -1 = none], ng=int, owner=[image block per position, -1 = none], nb=int,
                   late_mean={L: [heads averaged]}, perhead=[late layers with per-head unit output], gz_late=[late layers with group split],
                   pair=None | batch index)
Stored (self.out[b], numpy float16 unless noted; R = len(rows); G = ng; nb blocks):
  fa_gz [nFA, H, G, len(gz_late), R, nb]   dn_gz [nDN, Hv, G, len(gz_late), R, nb]     (head mean over late_mean[L])
  u_z   [nU, len(late), R, nb]  (every unit, groups summed; units in the order FA heads, DN heads, MLP layers - see self.units)
  u_hz  [nU, len(perhead), H_L, R, nb]                                                   (every late head)
  pair_fa_gz, pair_dn_gz, pair_u_z, pair_u_hz                                            (with 'pair')
  with 'pair', the END - pair difference of every FA / DN head split exactly into a gate part and a read part:
      FA: W_O[(sig(gate_b) - sig(gate_pair)) * a_b] + W_O[sig(gate_pair) * (a_b - a_pair)], a = sum_{s in g} att V (ungated head output)
      DN: W_out[(gz_b - gz_pair) * m_b] + W_out[gz_pair * (m_b - m_pair)], gz = w_n * silu(z), m = the group's normalised core part
    gate_uz / read_uz [nFA + nDN heads, len(late), R, nb]; gate_fa_gz / read_fa_gz [nFA, H, G, len(gz_late), R, nb]
  fa_mass [nFA, H, G, R] attention mass;  dn_mass [nDN, Hv, G, R] norm of the group's normalised core part;  u_norm [nU, R] float32
  chk_fa [nFA, R], chk_dn [nDN, R]: max |sum of contributions - layer output| / max |layer output| (float32)
  chk_jac [len(late), R]: |J c - (q(x + e c) - q(x - e c)) / 2e| / |J c| for c = the sum of every unit (analytic Jacobian check)
  chk_lin [len(late), R]: |J c - (q(x) - q(x - c))| / |q(x) - q(x - c)| (how linear removing the whole band is at L's query)"""
from __future__ import annotations

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from c_tools import rotate_half
from c_qpath import QPath


def _rope(x, cos, sin, rd):
    return torch.cat([x[..., :rd] * cos + rotate_half(x[..., :rd]) * sin, x[..., rd:]], -1)


class Anat:
    def __init__(self, v, fa_layers, dn_layers, mlp_layers, late_layers):
        self.v = v; self.fa = list(fa_layers); self.dn = list(dn_layers); self.mlpL = list(mlp_layers); self.late = list(late_layers)
        self.cfgs = None; self.mode = None; self.stat = {}; self.out = {}; self.hooks = []; self.dncap = {}; self.cur_dn = None; self.dnin = {}
        self.Wq = {}
        L0 = v.layers[self.late[0]].self_attn
        self.H = L0.config.num_attention_heads
        self.units = [("fa", l, h) for l in self.fa for h in range(self.H)]
        Hv = None
        for l in self.dn:
            la = v.layers[l].linear_attn
            Hv = la.num_v_heads
            self.units += [("dn", l, h) for h in range(Hv)]
        self.Hv = Hv
        self.units += [("mlp", l, 0) for l in self.mlpL]
        self.nU = len(self.units)
        cur = M.torch_chunk_gated_delta_rule
        while hasattr(cur, "__self__") and type(cur.__self__).__name__ == "Anat":         # a stale wrapper left by a crashed job (by name:
            #                                                                                 a purged module's class is not this class)
            st_ = cur.__self__
            st_.out = {}; st_.stat = {}; st_.dncap = {}; st_.dnin = {}; st_.Wq = {}; st_.cfgs = None; st_.mode = None
            cur = st_._prev_chunk
        self._prev_chunk = cur
        M.torch_chunk_gated_delta_rule = self._chunk
        for i, Lr in enumerate(v.layers):                                                   # every DeltaNet layer marks itself
            if getattr(Lr, "linear_attn", None) is not None:
                self.hooks.append(Lr.linear_attn.register_forward_pre_hook(self._dn_pre(i), with_kwargs=True))
        for l in self.dn:
            self.hooks.append(v.layers[l].linear_attn.register_forward_hook(self._dn_post(l), with_kwargs=True))
        for l in self.fa:
            self.hooks.append(v.layers[l].self_attn.register_forward_hook(self._fa_post(l), with_kwargs=True))
        for l in self.mlpL:
            self.hooks.append(v.layers[l].mlp.register_forward_hook(self._mlp_post(l), with_kwargs=True))
        for L in self.late:
            self.hooks.append(v.layers[L].register_forward_pre_hook(self._late_pre(L), with_kwargs=True))
            self.hooks.append(v.layers[L].self_attn.register_forward_hook(self._late_attn(L), with_kwargs=True))

    # ---------------------------------------------------------------- control
    def set(self, cfgs, mode):
        """mode 'stat' (pass 1) keeps the late statistics; mode 'anat' (pass 2) computes and projects the contributions; None = off."""
        self.cfgs = cfgs; self.mode = mode; self.dncap = {}; self.dnin = {}
        if mode == "stat":
            self.stat = {}
        if mode == "anat":
            self.out = {b: {} for b, c in enumerate(cfgs) if c is not None and c.get("anat") is not None}

    def remove(self):
        for h in self.hooks:
            h.remove()
        M.torch_chunk_gated_delta_rule = self._prev_chunk
        self.stat = {}; self.out = {}; self.dncap = {}; self.Wq = {}

    @staticmethod
    def _pairs(a):
        if a.get("pairs"):
            return [(pre_, ix) for pre_, ix in a["pairs"] if ix is not None]
        return [("pair", a["pair"])] if a.get("pair") is not None else []

    def _samples(self):
        if self.cfgs is None:
            return []
        return [b for b, c in enumerate(self.cfgs) if c is not None and c.get("anat") is not None]

    # ---------------------------------------------------------------- pass 1: late statistics
    def _late_pre(self, L):
        def h(mod, args, kwargs):
            if self.mode != "stat":
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            for b in self._samples():
                a = self.cfgs[b]["anat"]
                self.stat[(b, L, "x")] = hs[b, a["rows"]].detach().float()
            return None
        return h

    def _late_attn(self, L):
        def h(mod, args, kwargs, out):
            if self.mode != "stat":
                return None
            sm = self._samples()
            if not sm:
                return None
            for b in sm:
                a = self.cfgs[b]["anat"]; rows = a["rows"]; R = len(rows)
                hs = kwargs["hidden_states"][b].float()
                cos, sin = kwargs["position_embeddings"]; cos, sin = cos[b].float(), sin[b].float()
                hd = mod.head_dim; nh = mod.config.num_attention_heads; rd = cos.shape[-1]
                own = torch.tensor(a["owner"], device=hs.device)
                img = (own >= 0).nonzero().flatten()
                qg = torch.nn.functional.linear(hs[rows], mod.q_proj.weight.float()).view(R, nh, 2 * hd)
                q = mod.q_norm(qg[..., :hd].to(mod.q_norm.weight.dtype)).float()
                k = mod.k_norm(torch.nn.functional.linear(hs[img], mod.k_proj.weight.float()).view(len(img), -1, hd).to(mod.k_norm.weight.dtype)).float()
                kh = torch.arange(nh, device=hs.device) // mod.num_key_value_groups
                K = k[:, kh]
                ke = _rope(K, cos[img][:, None, :], sin[img][:, None, :], rd)
                qe = _rope(q, cos[rows][:, None, :], sin[rows][:, None, :], rd)
                lg = torch.einsum("rhd,ihd->rhi", qe, ke) * mod.scaling
                ob = own[img]
                kbar = torch.zeros(R, nh, a["nb"], hd, device=hs.device)
                for bb in range(a["nb"]):
                    sel = (ob == bb).nonzero().flatten()
                    if len(sel) == 0:
                        continue
                    w = torch.softmax(lg[:, :, sel], -1)
                    kbar[:, :, bb] = torch.einsum("rhi,ihd->rhd", w, ke[sel])
                self.stat[(b, L, "kbar")] = kbar
                self.stat[(b, L, "cs")] = (cos[rows], sin[rows], rd, float(mod.scaling))
            return None
        return h

    # ---------------------------------------------------------------- projection through the direct path
    def _wq(self, L):
        if L not in self.Wq:
            mod = self.v.layers[L].self_attn; hd = mod.head_dim; nh = mod.config.num_attention_heads
            W = mod.q_proj.weight.float().view(nh, 2 * hd, -1)[:, :hd, :].reshape(nh * hd, -1).contiguous()
            ln = self.v.layers[L].input_layernorm
            self.Wq[L] = (W, (1.0 + ln.weight.float()), float(ln.eps), (1.0 + mod.q_norm.weight.float()), float(mod.q_norm.eps), nh, hd)
        return self.Wq[L]

    def _qfun(self, L, x):
        """q(x) = q_norm(W_q input_layernorm(x)) per head (float32), x [n, E] -> [n, nh, hd]."""
        W, wl, el, wq, eq, nh, hd = self._wq(L)
        xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + el) * wl
        u = (xn @ W.T).view(x.shape[0], nh, hd)
        return u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eq) * wq

    def _jvp(self, L, x, C):
        """analytic J_q(x) C; x [E], C [n, E] -> [n, nh, hd]."""
        W, wl, el, wq, eq, nh, hd = self._wq(L)
        E = x.shape[0]
        r = torch.rsqrt(x.pow(2).mean() + el)
        a = wl * r * (C - (C @ x)[:, None] * x[None] * (r * r / E))
        u = ((x * r * wl) @ W.T).view(nh, hd)
        ru = torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + eq)                           # [nh, 1]
        bq = (a @ W.T).view(C.shape[0], nh, hd)
        return wq * ru[None] * (bq - (bq * u[None]).sum(-1, keepdim=True) * u[None] * (ru[None] ** 2 / hd))

    def _project(self, b, C):
        """C [R, n, E] (contributions at the rows, any device) -> dict L -> dz [R, n, nh, nb] (float32, on L's device)."""
        out = {}
        for L in self.late:
            kbar = self.stat[(b, L, "kbar")]; x = self.stat[(b, L, "x")]
            cos, sin, rd, scal = self.stat[(b, L, "cs")]
            dev = kbar.device
            res = []
            for r in range(C.shape[0]):
                Cr = C[r].to(dev)
                parts = []
                for i0 in range(0, Cr.shape[0], 512):
                    dq = self._jvp(L, x[r], Cr[i0:i0 + 512])                                  # [m, nh, hd]
                    dqe = _rope(dq, cos[r][None, None, :], sin[r][None, None, :], rd)
                    parts.append(torch.einsum("mhd,hbd->mhb", dqe, kbar[r]) * scal)
                res.append(torch.cat(parts, 0))
            out[L] = torch.stack(res, 0)
        return out

    def _store_unit_block(self, b, kind, l, Cg, mass, chk):
        """Cg [R, Hx, G, E]: the unit-group contributions of one writer layer for sample b (and its pair later)."""
        a = self.cfgs[b]["anat"]
        o = self.out[b]
        o.setdefault("_C", {})[(kind, l)] = Cg                                              # kept for the pair projection of the partner
        o.setdefault("_mass", {})[(kind, l)] = mass
        o.setdefault("_chk", {})[(kind, l)] = chk

    def _finish_layer(self, kind, l):
        """project every anatomy sample's (and its pair's) contributions of this writer layer."""
        for b in self._samples():
            o = self.out[b]; a = self.cfgs[b]["anat"]
            if (kind, l) not in o.get("_C", {}):
                continue
            refs = [("", b)] + [(pre_ + "_", ix) for pre_, ix in self._pairs(a)]
            for who, src in refs:
                if src is None:
                    continue
                Cg = self.out[src]["_C"][(kind, l)]                                          # [R, Hx, G, E]
                R, Hx, G, E = Cg.shape
                flat = Cg.reshape(R, Hx * G, E)
                dz = self._project(b, flat)                                                  # L -> [R, Hx*G, nh, nb]
                for L, z in dz.items():
                    z = z.view(R, Hx, G, z.shape[2], z.shape[3])
                    hm = z[:, :, :, a["late_mean"][L]].mean(3)                                # [R, Hx, G, nb]
                    if L in a["gz_late"]:
                        o.setdefault(who + "_gz", {})[(kind, l, L)] = hm.permute(1, 2, 0, 3).cpu()   # [Hx, G, R, nb]
                    o.setdefault(who + "_uz", {})[(kind, l, L)] = hm.sum(2).permute(1, 0, 2).cpu()  # [Hx, R, nb]
                    if L in a["perhead"]:
                        o.setdefault(who + "_uhz", {})[(kind, l, L)] = z.sum(2).permute(1, 2, 0, 3).cpu()   # [Hx, nh, R, nb]
        if kind in ("fa", "dn"):                                                             # END - pair: gate part and read part
            for b in self._samples():
                o = self.out[b]; a = self.cfgs[b]["anat"]
                if (kind, l) not in o.get("_gr", {}):
                    continue
                for pre_, src in self._pairs(a):
                  tag_ = "" if pre_ == "pair" else pre_ + "_"
                  Ab, gb, W = o["_gr"][(kind, l)]; As, gs_, _ = self.out[src]["_gr"][(kind, l)]
                  Cgate = torch.einsum("rhgd,ehd->rhge", (gb - gs_)[:, :, None] * Ab, W)
                  Cread = torch.einsum("rhgd,ehd->rhge", gs_[:, :, None] * (Ab - As), W)
                  for nm_, Cx in ((tag_ + "gate", Cgate), (tag_ + "read", Cread)):
                    R, Hx, G, E = Cx.shape
                    dz = self._project(b, Cx.reshape(R, Hx * G, E))
                    for L, z in dz.items():
                        z = z.view(R, Hx, G, z.shape[2], z.shape[3])
                        hm = z[:, :, :, a["late_mean"][L]].mean(3)
                        o.setdefault("_" + nm_ + "uz", {})[(kind, l, L)] = hm.sum(2).permute(1, 0, 2).cpu()
                        if kind == "fa" and L in a["gz_late"]:
                            o.setdefault("_" + nm_ + "gz", {})[(kind, l, L)] = hm.permute(1, 2, 0, 3).cpu()
                  del Cgate, Cread
            for b in self._samples():
                self.out[b].get("_gr", {}).pop((kind, l), None)
        for b in self._samples():                                                            # the contributions themselves are not kept
            o = self.out[b]
            if (kind, l) not in o.get("_C", {}):
                continue
            Cg = o["_C"].pop((kind, l))
            o.setdefault("_tot", {})[(kind, l)] = Cg.sum((1, 2))                               # [R, E] (layer total, for the checks)
            o.setdefault("_unorm", {})[(kind, l)] = Cg.sum(2).norm(dim=-1).cpu()               # [R, Hx]

    # ---------------------------------------------------------------- full-attention writer layers
    def _fa_post(self, l):
        def h(mod, args, kwargs, out):
            if self.mode != "anat":
                return None
            sm = self._samples()
            if not sm:
                return None
            t_out = out[0] if isinstance(out, tuple) else out
            Wo = mod.o_proj.weight.float().view(-1, mod.config.num_attention_heads, mod.head_dim)   # [E, nh, hd], once per call
            for b in sm:
                c = self.cfgs[b]; a = c["anat"]; rows = c["rows"]; arows = a["rows"]
                ai = [rows.index(t) for t in arows]                                           # anatomy rows must be among the sample's rows
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
                for e_ in ([] if ed is None else ed if isinstance(ed, list) else [ed]):          # the pins exactly as QPathB applies them
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
                mask = torch.arange(S, device=hs.device)[None, :] > torch.tensor(arows, device=hs.device)[:, None]
                att = torch.softmax(lg.masked_fill(mask[:, None, :], float("-inf")), -1)          # [Ra, nh, S]
                grp = torch.tensor(a["group"], device=hs.device); G = a["ng"]
                sg = torch.sigmoid(gt)                                                             # [Ra, nh, hd]
                Cg = torch.zeros(len(arows), nh, G, Wo.shape[0], device=hs.device)
                mass = torch.zeros(len(arows), nh, G, device=hs.device)
                Ag = torch.zeros(len(arows), nh, G, hd, device=hs.device)
                for g in range(G):
                    sel = (grp == g).nonzero().flatten()
                    if len(sel) == 0:
                        continue
                    Ag[:, :, g] = torch.einsum("rhs,shd->rhd", att[:, :, sel], V[sel])
                    Cg[:, :, g] = torch.einsum("rhd,ehd->rhe", Ag[:, :, g] * sg, Wo)
                    mass[:, :, g] = att[:, :, sel].sum(-1)
                self.out[b].setdefault("_gr", {})[("fa", l)] = (Ag, sg, Wo)
                tot = Cg.sum((1, 2))
                if mod.o_proj.bias is not None:
                    tot = tot + mod.o_proj.bias.float()
                ref = t_out[b, arows].float()
                chk = ((tot - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-12)).cpu()
                self._store_unit_block(b, "fa", l, Cg, mass.cpu(), chk)
            self._finish_layer("fa", l)
            return None
        return h

    # ---------------------------------------------------------------- Gated DeltaNet writer layers
    def _dn_pre(self, l):
        def h(mod, args, kwargs):
            self.cur_dn = l
            if self.mode == "anat" and l in self.dn and self._samples():
                hs = args[0] if args else kwargs["hidden_states"]
                self.dnin[l] = hs
            return None
        return h

    def _chunk(self, query, key, value, g, beta, **kw):
        if self.mode == "anat" and self.cur_dn in self.dn and self._samples():
            self.dncap[self.cur_dn] = (query.detach(), key.detach(), value.detach(), g.detach(), beta.detach())
        return self._prev_chunk(query, key, value, g=g, beta=beta, **kw)

    def _dn_post(self, l):
        def h(mod, args, kwargs, out):
            if self.mode != "anat" or l not in self.dncap:
                return None
            sm = self._samples()
            query, key, value, gg, bb = self.dncap.pop(l)
            hs_in = self.dnin.pop(l)
            t_out = out[0] if isinstance(out, tuple) else out
            a0 = self.cfgs[sm[0]]["anat"]; arows = a0["rows"]
            assert all(self.cfgs[b]["anat"]["rows"] == arows and self.cfgs[b]["anat"]["group"] == a0["group"] for b in sm), "one layout per pass"
            bi = torch.tensor(sm, device=query.device)
            q = query[bi].float(); k = key[bi].float(); v = value[bi].float(); g_ = gg[bi].float(); be = bb[bi].float()   # [B', S, H, d]
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / (q.shape[-1] ** .5)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            Bp, S, H, dk = q.shape; dv = v.shape[-1]; R = len(arows); G = a0["ng"]
            grp = a0["group"]
            idx = {t: i for i, t in enumerate(arows)}
            r = torch.zeros(Bp, R, H, dk, device=q.device)
            Og = torch.zeros(Bp, R, H, G, dv, device=q.device)
            for s in range(max(arows), -1, -1):
                if s in idx:
                    r[:, idx[s]] = q[:, s]
                kr = (r * k[:, s][:, None]).sum(-1)                                          # [B', R, H]
                gs = grp[s]
                if gs >= 0:
                    Og[:, :, :, gs] += (be[:, s][:, None] * kr)[..., None] * v[:, s][:, None]
                r = torch.exp(g_[:, s])[:, None, :, None] * (r - (be[:, s][:, None, :, None] * kr[..., None]) * k[:, s][:, None])
            core = Og.sum(3)                                                                  # [B', R, H, dv]
            rms = torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + mod.norm.variance_epsilon)  # [B', R, H, 1]
            z = mod.in_proj_z(hs_in[bi][:, arows]).float().view(Bp, R, H, dv)
            gz = torch.nn.functional.silu(z) * mod.norm.weight.float()                        # [B', R, H, dv]
            Wout = mod.out_proj.weight.float().view(-1, H, dv)                                  # [E, H, dv]
            for j, b in enumerate(sm):
                mg = Og[j] * rms[j][:, :, None]                                               # [R, H, G, dv] normalised group cores
                self.out[b].setdefault("_gr", {})[("dn", l)] = (mg, gz[j], Wout)
                ng = mg * gz[j][:, :, None]                                                   # [R, H, G, dv]
                Cg = torch.einsum("rhgd,ehd->rhge", ng, Wout)
                mass = ng.norm(dim=-1)                                                        # [R, H, G]
                tot = Cg.sum((1, 2))
                if mod.out_proj.bias is not None:
                    tot = tot + mod.out_proj.bias.float()
                ref = t_out[b, arows].float()
                chk = ((tot - ref).abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-12)).cpu()
                self._store_unit_block(b, "dn", l, Cg, mass.cpu(), chk)
            self._finish_layer("dn", l)
            return None
        return h

    # ---------------------------------------------------------------- MLP writer layers
    def _mlp_post(self, l):
        def h(mod, args, kwargs, out):
            if self.mode != "anat":
                return None
            sm = self._samples()
            if not sm:
                return None
            t = out[0] if isinstance(out, tuple) else out
            for b in sm:
                arows = self.cfgs[b]["anat"]["rows"]
                Cg = t[b, arows].float()[:, None, None, :]                                     # [R, 1, 1, E]
                self._store_unit_block(b, "mlp", l, Cg, torch.ones(len(arows), 1, 1), torch.zeros(len(arows)))
            self._finish_layer("mlp", l)
            return None
        return h

    # ---------------------------------------------------------------- checks and packing (after pass 2)
    def finish(self, b):
        """pack sample b's anatomy into numpy arrays (see the module docstring); also runs the Jacobian checks on the band's total."""
        import numpy as np
        o = self.out[b]; a = self.cfgs[b]["anat"]
        d0 = self.stat[(b, self.late[0], "x")].device
        tot = sum(t.to(d0) for t in o["_tot"].values())                                     # [R, E] the band's total at the rows
        chk_jac, chk_lin = [], []
        for L in self.late:
            x = self.stat[(b, L, "x")]; tj = tot.to(x.device)
            rj, rl = [], []
            for r in range(x.shape[0]):
                jc = self._jvp(L, x[r], tj[r][None])[0]
                eps = 1e-3 * float(x[r].norm() / tj[r].norm().clamp_min(1e-12))
                fd = (self._qfun(L, (x[r] + eps * tj[r])[None]) - self._qfun(L, (x[r] - eps * tj[r])[None]))[0] / (2 * eps)
                rm = (self._qfun(L, x[r][None]) - self._qfun(L, (x[r] - tj[r])[None]))[0]
                rj.append(float((jc - fd).norm() / jc.norm().clamp_min(1e-12)))
                rl.append(float((jc - rm).norm() / rm.norm().clamp_min(1e-12)))
            chk_jac.append(rj); chk_lin.append(rl)
        res = dict(chk_jac=np.array(chk_jac, np.float32), chk_lin=np.array(chk_lin, np.float32))
        kinds = [("fa", self.fa), ("dn", self.dn)]
        for who in [""] + [pre_ + "_" for pre_, _ in self._pairs(a)]:
            if (who + "_uz") not in o:
                continue
            for kind, Ls in kinds:
                if Ls and a["gz_late"]:
                    res[who + kind + "_gz"] = np.stack([np.stack([o[who + "_gz"][(kind, l, L)].numpy() for L in a["gz_late"]], 2) for l in Ls]).astype(np.float16)
            uz = []; uhz = []
            for kind, l, hh in self.units:
                uz.append(np.stack([o[who + "_uz"][(kind, l, L)][hh].numpy() for L in self.late]))           # [nL, R, nb]
                if a["perhead"]:
                    uhz.append(np.stack([o[who + "_uhz"][(kind, l, L)][hh].numpy() for L in a["perhead"]]))   # [nP, nh, R, nb]
            res[who + "u_z"] = np.stack(uz).astype(np.float16)
            if uhz:
                res[who + "u_hz"] = np.stack(uhz).astype(np.float16)
        hu = [(kind, l, hh) for kind, l, hh in self.units if kind in ("fa", "dn")]
        for pre_, _ in self._pairs(a):
            tag_ = "" if pre_ == "pair" else pre_ + "_"
            if "_" + tag_ + "gateuz" not in o:
                continue
            for nm_ in (tag_ + "gate", tag_ + "read"):
                res[nm_ + "_uz"] = np.stack([np.stack([o["_" + nm_ + "uz"][(kind, l, L)][hh].numpy() for L in self.late]) for kind, l, hh in hu]).astype(np.float16)
                if self.fa and a["gz_late"]:
                    res[nm_ + "_fa_gz"] = np.stack([np.stack([o["_" + nm_ + "gz"][("fa", l, L)].numpy() for L in a["gz_late"]], 2) for l in self.fa]).astype(np.float16)
        if self.fa:
            res["fa_mass"] = np.stack([o["_mass"][("fa", l)].permute(1, 2, 0).numpy() for l in self.fa]).astype(np.float16)   # [nFA, H, G, R]
            res["chk_fa"] = np.stack([o["_chk"][("fa", l)].numpy() for l in self.fa]).astype(np.float32)
        if self.dn:
            res["dn_mass"] = np.stack([o["_mass"][("dn", l)].permute(1, 2, 0).numpy() for l in self.dn]).astype(np.float16)
            res["chk_dn"] = np.stack([o["_chk"][("dn", l)].numpy() for l in self.dn]).astype(np.float32)
        res["u_norm"] = np.stack([o["_unorm"][(kind, l)][:, hh].numpy() for kind, l, hh in self.units]).astype(np.float32)   # [nU, R]
        return res

    def clear(self):
        self.out = {}; self.stat = {}; self.dncap = {}; self.dnin = {}
