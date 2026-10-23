# S860

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S860 (head-level causal test after S859; registered in knowledge/DECISIONS.md before its data) - which components of the writer suppress the early address (un-anchoring), which favour the predecessor (selection), what sets the gain, and does the computation transfer across anchors? Eight real pictures, stated count 8, the early lookup pinned (address part U of the ten read rows at the early look layers 35-47 <- START(p)), as S844-S859. One forward pass per (k, pins): k 5 with pins (3, 5) and (4, 7), k 3 with pins (3, 5) and (7); every pass holds, for each pin p, E END(k) pinned at p R START(p)

Code: `code/s860_analyze.py`, `code/s860_heads.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s860_analysis_s860_27b.json`, `data/s860_s860_27b_heads.json`, `data/s862_frozen_hashes.json`
