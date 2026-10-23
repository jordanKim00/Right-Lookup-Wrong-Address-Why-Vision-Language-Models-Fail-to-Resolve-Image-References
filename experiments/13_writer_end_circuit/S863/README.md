# S863

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S863 (registered in knowledge/DECISIONS.md before its data; the advisor's open gap 10-20 'where does k control the strength of leaving and predecessor selection') - the k-locus decomposition of the writer's step in pinned runs (design revised after the co-analysis and mechanist critiques, 10-20 23:4x, before any data). One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown), as S861 / S862. The step is off at k = 2 and on from k = 3 (S859, pinned). Recipients E(k, p) = END(k) with the early lookup pinned at p in {4, 5} (U part at 35-47 <- ST

Code: `code/s863_analyze.py`, `code/s863_klocus.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s863_analysis_s863_27b.json`
