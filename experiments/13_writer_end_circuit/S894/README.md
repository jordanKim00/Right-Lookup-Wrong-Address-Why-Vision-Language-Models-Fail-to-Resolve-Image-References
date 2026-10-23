# S894

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S894 (registered in knowledge/DECISIONS.md before its data; predictions frozen and hashed before the run; after the post hoc finding that END pulls FA47's self-address heads one block end-ward - anchor -> successor - although the address part of the query is pinned to START's, and S891: the composition map predicts FA47 attention interventions) - is the end-ward pull of the self-address heads part of the writer's step? FA47 read-row attention edits (exp/s891 FAAttnEdit plus a partial move): R_SA2S40 / R_SA2S100 (START: 40% / all of the self-address heads' anchor mass moved onto the successor b

Code: `code/s894_analyze.py`, `code/s894_endpull.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s894_analysis_s894_27b.json`, `data/s894_s894_27b_endpull.json`
