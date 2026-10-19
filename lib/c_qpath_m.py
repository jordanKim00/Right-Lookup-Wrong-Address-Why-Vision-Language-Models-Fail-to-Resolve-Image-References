"""Reading-mode query edits for S839 (gap 1), added to QPath._qsub at run time by patch() - exp/c_qpath.py itself is not changed.
New qsub parts (post-RoPE query, per head; U = span of the centred block-mean image keys, as in c_qpath):
  'menu'         direction m = mean key over image tokens - mean key over the menu-word tokens (spec['menu_pos']), made orthogonal to U; the
                 query's component along m is set to the donor's (picture vs menu-word preference; the block address is untouched)
  'textall'      the same with every non-image token before the row's position (spec['text_end'] = last such position + 1) as the text set
  'norm'         every head's query is rescaled to the donor's norm, direction kept
  'perp_nomenu'  the non-address part (complement of U) is taken from the donor except along m, where the recipient's component is kept
  'wsub' / 'tsub' / 'rsub'  (S842) the donor-minus-recipient difference restricted to W (within-picture token pattern), T (text keys) or the
                 non-address rest outside W and T is added (rank spec['rank'], default 16; text before spec['text_end'])
Donors are given as in c_qpath ([(w, q after q_norm, before RoPE)])."""
from __future__ import annotations

import torch
from c_qpath import QPath

_orig = QPath._qsub


def _basis(ke, own, nb, h):
    Kb = torch.stack([ke[own == b].mean(0) for b in range(nb)]); C = Kb - Kb.mean(0, keepdim=True)
    U, s_, _ = torch.linalg.svd(C[:, h].T, full_matrices=False)
    return U[:, s_ > 1e-6 * s_.max()]


def _dir(ke, own, txt, h, U):
    m = ke[own >= 0][:, h].mean(0) - ke[txt][:, h].mean(0)
    m = m - U @ (U.T @ m)
    return m / (m.norm() + 1e-12)


def _sub_bases(ke, own, nb, h, r, text_end):
    """S842: per head, U (address), W (within-picture token pattern: top-r directions of image-token keys minus their block mean, made
    orthogonal to U) and T (text keys: top-r directions of the centred keys of non-image tokens before text_end, orthogonal to U and W)."""
    U = _basis(ke, own, nb, h)
    img = own >= 0
    Kb = torch.stack([ke[own == b][:, h].mean(0) for b in range(nb)])
    dev_ = ke[img][:, h] - Kb[own[img]]
    dev_ = dev_ - (dev_ @ U) @ U.T
    W = torch.linalg.svd(dev_.T, full_matrices=False)[0][:, :r]
    txt = (own < 0) & (torch.arange(len(own), device=ke.device) < text_end)
    tk = ke[txt][:, h]; tk = tk - tk.mean(0, keepdim=True)
    tk = tk - (tk @ U) @ U.T; tk = tk - (tk @ W) @ W.T
    T = torch.linalg.svd(tk.T, full_matrices=False)[0][:, :r]
    return U, W, T


def _qsub_m(self, qe, ke, spec, cr, sr, rd):
    part = spec.get("part")
    if part in ("wsub", "tsub", "rsub"):
        own = torch.tensor(spec["owner"], device=qe.device); nb = spec["nb"]; r = spec.get("rank", 16)
        don = sum(w * QPath._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["donors"])
        out = qe.clone()
        for h in range(qe.shape[1]):
            U, W, T = _sub_bases(ke, own, nb, h, r, spec["text_end"])
            d_ = don[:, h] - qe[:, h]
            if part == "wsub":
                out[:, h] = qe[:, h] + (d_ @ W) @ W.T
            elif part == "tsub":
                out[:, h] = qe[:, h] + (d_ @ T) @ T.T
            else:                                                               # the non-address rest outside W and T
                dr = d_ - (d_ @ U) @ U.T - (d_ @ W) @ W.T - (d_ @ T) @ T.T
                out[:, h] = qe[:, h] + dr
        return out
    if part not in ("menu", "textall", "norm", "perp_nomenu"):
        return _orig(self, qe, ke, spec, cr, sr, rd)
    own = torch.tensor(spec["owner"], device=qe.device); nb = spec["nb"]
    don = sum(w * QPath._rope(q.to(qe.device).float(), cr, sr, rd) for w, q in spec["donors"])
    out = qe.clone()
    if part == "norm":
        return qe * (don.norm(dim=-1, keepdim=True) / (qe.norm(dim=-1, keepdim=True) + 1e-12))
    if part in ("menu", "perp_nomenu"):
        txt = torch.zeros(len(own), dtype=torch.bool, device=qe.device); txt[torch.tensor(spec["menu_pos"], device=qe.device)] = True
    else:
        txt = (own < 0) & (torch.arange(len(own), device=qe.device) < spec["text_end"])
    for h in range(qe.shape[1]):
        U = _basis(ke, own, nb, h); mh = _dir(ke, own, txt, h, U)
        if part in ("menu", "textall"):
            out[:, h] = qe[:, h] + ((don[:, h] - qe[:, h]) @ mh)[:, None] * mh[None, :]
        else:                                                                   # perp_nomenu
            a_rec = (qe[:, h] @ U) @ U.T
            dperp = don[:, h] - (don[:, h] @ U) @ U.T
            dperp = dperp - (dperp @ mh)[:, None] * mh[None, :]
            out[:, h] = a_rec + dperp + (qe[:, h] @ mh)[:, None] * mh[None, :]
    return out


def patch():
    """Route the new parts through QPath._qsub (QPathB calls QPath._qsub(QPath, ...)); other parts keep the original code."""
    QPath._qsub = _qsub_m
