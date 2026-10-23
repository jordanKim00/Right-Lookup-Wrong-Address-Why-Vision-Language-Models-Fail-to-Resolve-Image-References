"""DeltaNet read-row value edit (10-22, for S888): the outputs of chosen Gated DeltaNet layers at the given rows are recomputed with the post-conv
values of some positions replaced by other positions' (token by token), every other position keeping the model's own output.  The layer's
q, k, v, g, beta and z are recomputed from its input exactly as Qwen3_5GatedDeltaNet.forward does (no cache, no padding); the core recurrence
runs twice through the module's own chunk_gated_delta_rule (the fla kernel when fla was importable at load time) - with the original and with
the edited values - and the rows get t + out_proj(norm(o_edit, z)) - out_proj(norm(o_orig, z)), so an identity map adds exactly zero.
cfg per sample: prow=[rows], vmap={layer: [(dst positions, src positions), ...]} (or None)."""
from __future__ import annotations

import threading
import torch
import torch.nn.functional as F


class DNValueEdit:
    cfgs = property(lambda self: getattr(self._tl, "cfgs", None), lambda self, x: setattr(self._tl, "cfgs", x))

    def __init__(self, v, layers):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        self.M = M; self._tl = threading.local(); self.hooks = []; self.calls = 0
        for l in layers:
            la = v.layers[l].linear_attn
            self.hooks.append(la.register_forward_hook(self._post(l), with_kwargs=True))

    def set(self, cfgs):
        self.cfgs = cfgs

    def remove(self):
        for h in self.hooks:
            h.remove()

    def _qkv(self, mod, x):
        M = self.M; B, S, _ = x.shape
        mixed = mod.in_proj_qkv(x).transpose(1, 2)
        mixed = M.causal_conv1d_fn(mixed, mod.conv1d.weight.squeeze(1), mod.conv1d.bias, activation=mod.activation).transpose(1, 2)
        q, k, vv = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
        q = q.reshape(B, S, -1, mod.head_k_dim); k = k.reshape(B, S, -1, mod.head_k_dim); vv = vv.reshape(B, S, -1, mod.head_v_dim)
        beta = mod.in_proj_b(x).sigmoid()
        g = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(x).float() + mod.dt_bias)
        if mod.num_v_heads // mod.num_k_heads > 1:
            q = q.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2); k = k.repeat_interleave(mod.num_v_heads // mod.num_k_heads, dim=2)
        z = mod.in_proj_z(x).reshape(B, S, -1, mod.head_v_dim)
        return q, k, vv, g, beta, z

    def _post(self, l):
        def h(mod, args, kwargs, out):
            cfgs = self.cfgs
            if cfgs is None:
                return None
            todo = [(b, c) for b, c in enumerate(cfgs) if c is not None and l in (c.get("vmap") or {})]
            if not todo:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            t = out[0] if isinstance(out, tuple) else out
            t2 = t.clone()
            for b, c in todo:
                x = hs[b:b + 1]
                q, k, vv, g, beta, z = self._qkv(mod, x)
                v2 = vv.clone()
                for dst, src in c["vmap"][l]:
                    v2[:, torch.tensor(dst, device=x.device)] = vv[:, torch.tensor(src, device=x.device)]
                o1, _ = self.M.torch_chunk_gated_delta_rule(q, k, vv, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
                o2, _ = self.M.torch_chunk_gated_delta_rule(q, k, v2, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
                rows = torch.tensor(c["prow"], device=x.device); hd = mod.head_v_dim
                def fin(o):
                    oo = o[0, rows].reshape(-1, hd); zz = z[0, rows].reshape(-1, hd)
                    return mod.out_proj(mod.norm(oo, zz).reshape(len(c["prow"]), -1))
                corr = fin(o2).float() - fin(o1).float()
                t2[b, rows] = (t[b, rows].float() + corr).to(t2.dtype)
            self.calls += 1
            return (t2,) + tuple(out[1:]) if isinstance(out, tuple) else t2
        return h
