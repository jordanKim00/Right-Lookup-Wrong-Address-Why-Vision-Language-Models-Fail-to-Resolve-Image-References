# S865

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S865 (registered in knowledge/DECISIONS.md before its data; S862 post hoc: the band's END part and the pinned one-back are N-invariant (D4) while natural copy-anchor behaviour moves with the needed distance (A4) - does the N-dependent part of the answer live in the early address?) - cross-layout early-address swap in natural runs. One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (N 7-9, stated = shown). Image keys of a block depend only on the tokens before it, so block b is the same picture with the same keys in every layout. Per layout N_r: natural

Code: `code/s865_analyze.py`, `code/s865_xlayout.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s865_analysis_s865_27b.json`, `data/s865_s865_27b_xlayout.json`
