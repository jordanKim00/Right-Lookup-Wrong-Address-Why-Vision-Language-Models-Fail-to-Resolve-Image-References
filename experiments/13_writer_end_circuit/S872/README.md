# S872

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S872 (registered in knowledge/DECISIONS.md before its data; advisor review 4, 10-21: an address Jacobian lens / writer-effect lens, after the Jacobian lens of Gurnee et al. 2026) - does the model's own sensitivity, applied to each scene's pre-writer END - START state difference, compute that scene's writer effect? One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown). END(5) (E) and START(p) (R) pinned at p in {3, 4, 5} (U part at 35-47 <- the natural START(p)); p 4 = identity (target 4), p 5 = one step. Objectives (16): z_b = the L5

Code: `code/s872_analyze.py`, `code/s872_jlens.py`, `code/s872_jlens_v1.py`, `code/s872b_analyze.py`, `code/s872b_fallback.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s872_analysis_s872v4_27b.json`, `data/s872_s872b_27b_jlens_partial.json`, `data/s872_s872v4_27b_jlens.json`, `data/s872_s872v4_27b_jlens_partial.json`, `data/s872_s872v4pare_27b_jlens.json`, `data/s872_s872v4seq_27b_jlens.json`
