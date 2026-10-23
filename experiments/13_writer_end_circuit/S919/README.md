# S919

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S919 collector (registered in knowledge/DECISIONS.md before the data) - S918 R3: with END's early query and reading strength given, the START state still departs .26 less than END; is that remaining END state carried by the band's MLP / DeltaNet outputs (layers 46-49)? Natural runs at N8 (fresh scenes), k in --ks (2-6), at END(k)'s natural L47 anchor a_k. Phase A: natural END(k) and START(q) (donors of query, picture mass, output gate). Phase B per k one batch of 12: E (natural END(k): the band source, batch index 0), S (natural START(a_k)), EEE (END(k) with its own query at 35-47), SEE (START

Code: `code/s919_analyze.py`, `code/s919_bandstate.py`

Shared modules: `lib/c_qpath_rm.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s919_analysis_s919_27b.json`
