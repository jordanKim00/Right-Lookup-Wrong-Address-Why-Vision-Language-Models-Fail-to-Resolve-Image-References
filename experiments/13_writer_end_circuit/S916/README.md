# S916

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S916 collector (registered in knowledge/DECISIONS.md before the data) - where does END(2)'s end-side answer come from when the read-row vector and the early address are both removed (S911 R3: .73 stays at the end)? Natural runs at one N (default 8), k in --ks (1, 2, 3, 5): per scene and k one batch of END(k), START(k) (the source) and six residual row swaps at decoder-layer inputs (QPathK rpatch, the source's running residual of the same pass): RS35 / RS47 / RS51 / RS55 - the ten read rows ('?', the turn end, the assistant prefix) <- START(k)'s at layer 35 / 47 / 51 / 55; TS35 - the other ques

Code: `code/s916_analyze.py`, `code/s916_bypass.py`

Shared modules: `lib/c_qpath_k.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s916_analysis_s916_27b.json`, `data/s916_s916_27b_bypass.json`
