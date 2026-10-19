"""Micro-batch pipelining for a model spread over several cards with a balanced device map (10-16; efficiency only).  A batch is cut into
`parts` consecutive micro-batches run in threads: while one micro-batch is on card 1 the next is on card 0, so the cards work at once
instead of in turn.  Each sample's computation is unchanged (samples never interact); only the batch composition of a kernel call differs, as
with any change of batch size.  run1(inps, qs) must keep its per-call state thread-local (QPathB does since 10-16) and return
(logits [n, V], store {sample index: ...}).  Out-of-memory in any part raises in the caller (who may split and retry).
CHECK 10-16 21:2x (exp/pipe_check.py on 27B, logs/w27b_pipecheck2.log): 4 parts bit-identical to the sequential batch (1.30x), but 8 and 12 parts
change logits (up to .69) and block scores (up to 5.4) - a thread race somewhere in the model's kernels.  NOT USED for experiments."""
from __future__ import annotations

import threading
import torch


def pipelined(run1, inps, qs, parts=4):
    n = len(inps)
    if parts <= 1 or n < 2:
        return run1(inps, qs)
    parts = min(parts, n)
    bd = [round(i * n / parts) for i in range(parts + 1)]
    res, err = [None] * parts, [None] * parts

    def work(i):
        try:
            res[i] = run1(inps[bd[i]:bd[i + 1]], qs[bd[i]:bd[i + 1]])
        except BaseException as e:                                          # re-raised in the caller
            err[i] = e
    th = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(parts)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for e in err:
        if e is not None:
            raise e
    lg = torch.cat([r[0].to(res[0][0].device) for r in res], 0)
    st = {}
    for i, r in enumerate(res):
        for b, x in r[1].items():
            st[b + bd[i]] = x
    return lg, st
