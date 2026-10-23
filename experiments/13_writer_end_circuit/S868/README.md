# S868

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S868 (registered in knowledge/DECISIONS.md before its data; user 10-21 'the complete algorithmic circuit of END'; the post hoc reading of S860's k3 end-pin cells, §972 supplement) - near-end END answers come from the early end route, and the writer's k-switched step breaks an exact target. One scene = nine pictures with all nine names as the menu; layouts show its first N pictures (N 8 and 9, stated = shown). Natural END(k), k in {1, 2, 3} (targets N, N - 1, N - 2), and per k: PIN_NM1 the read rows' early queries (U part at 35-47) <- the natural START(N - 1) (a sharp N - 1 anchor, no end-route

Code: `code/s868_analyze.py`, `code/s868_nearend.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s868_analysis_s868_27b.json`, `data/s868_s868_27b_nearend.json`
