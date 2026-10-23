# S910

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S910 (registered in knowledge/DECISIONS.md before its data; adversarial review 10-22, item D1) - pinned vs natural at the same anchor. S905: pinned at the last picture, END(2)'s band shifts the late address (W_b 1.21) but the answer stays (.90); S902: at the natural last-picture anchor END(2)'s band moves the answer (one step .58, W 2.46). Does pinning change what the band writes (the write depends on the form of the early address), or does it only deepen the writer-free margin? Natural runs at N 8 on new scenes; phase A: natural END(k) for k in {2, 3, 5, 6} and START(q) for every q (their L35

Code: `code/s910_analyze.py`, `code/s910_pinnat.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s910_analysis_s910_27b.json`, `data/s910_s910_27b_pinnat.json`
