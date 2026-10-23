# S866

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S866 (registered in knowledge/DECISIONS.md before its data; user 10-21 'the complete algorithmic circuit of END is the top priority') - sufficiency of the writer's END output: does the band's END computation alone make a pointer (START) run step back from its anchor? Removal (S860-S862) showed necessity and exchange showed anchor-keying; this transplants END's parts into START at the same pinned anchor. One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown), as S861-S865. Pinned references at p in {3, 4, 5} (U part at 35-47 <- the nat

Code: `code/s866_analyze.py`, `code/s866_suff.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s866_analysis_s866_27b.json`
