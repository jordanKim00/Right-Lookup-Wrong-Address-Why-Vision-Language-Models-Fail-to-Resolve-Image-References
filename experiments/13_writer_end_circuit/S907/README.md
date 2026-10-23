# S907

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S907 (registered in knowledge/DECISIONS.md before its data) - can the frozen circuit-structured models predict every reference at an unseen N? (S870 asked it for N 5 and 7; S899 / S902 / S905 confirmed the route structure; N10 is outside every fit.) One scene = ten pictures with all ten names as the menu (layout N 10, stated = shown). Phase A only: natural END(k) for every k = 1..10 and START(q) for every q = 1..10 (no pins), the L47 matcher-head block scores of the ten read rows (F_early input), late block scores, menu logits and answers. Seed 90700, 48 scenes. Usage: python s907_grid10.py MO

Code: `code/s907_analyze.py`, `code/s907_grid10.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: none of its own (see below)
