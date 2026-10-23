# S917

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S917 collector (registered in knowledge/DECISIONS.md before the data) - is the writer's effect on the answer transmitted through the late full-attention reads? (S914 post hoc: the L51 block profile's link to the answer differs with and without the writer.) Natural runs at N8 (fresh scenes), k in --ks (2-6), at END(k)'s natural L47 anchor a. Per scene and k one batch of 12: E (END(k)), R (START(a): the band source), B (E_BANDR - the band 46-49 read-row outputs <- R's: the writer-free run); late read swaps with the head-patch tool (every head's gated output at the read rows of the late full-atte

Code: `code/s917_analyze.py`, `code/s917_lateroute.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s917_analysis_s917_27b.json`
