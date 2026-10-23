# S918

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S918 collector (registered in knowledge/DECISIONS.md before the data; advisor review 2, 10-22: address vs reading strength) - at a fixed early address, does how the address is read (the picture-vs-text attention mass and the output gate at the read rows of the early full-attention layers 35, 39, 43, 47) carry the END-conditioned departure? Natural runs at N8 (fresh scenes), k in --ks (2-6), at END(k)'s natural L47 anchor a_k. Phase A: natural END(k) and START(q) - each a donor of its read-row query (address), picture mass and output gate (per layer, row, head). Phase B per k, every run pinned 

Code: `code/s918_analyze.py`, `code/s918_reading.py`

Shared modules: `lib/c_qpath_rm.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s918_analysis_s918_27b.json`
