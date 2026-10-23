# S886

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S886 (registered in knowledge/DECISIONS.md before its data; after S881 / p21 (the writer's late query change carries predecessor content (+) and anchor content (-)), S883 (the anchor heads' END part is what they read; attention changes do not set the per-scene size) and S885): which band component writes which content? Capture, at the read rows B1 / B4 / answer, the outputs of every band component - the token mixer (DN46, FA47, DN48, DN49) and the MLP of layers 46-49 - in E (END(5)) and R (START(p)) pinned at p in {3, 4, 5}, the residual entering layer 51 in E and E_BANDR (the late query map's

Code: `code/s886_analyze.py`, `code/s886_compmap.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s886_analysis_s886_27b.json`, `data/s886_s886_27b_compmap.json`
