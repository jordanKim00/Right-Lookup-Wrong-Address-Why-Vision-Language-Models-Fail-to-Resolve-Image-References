# S873

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S873 (registered in knowledge/DECISIONS.md before its data; causal tests of S872's Jacobian lens - advisor review 4 'lens readouts must match causal patching' and review 3 'same address, different writer'; the swap test of Gurnee et al. 2026) - does each scene's writer effect follow the pre-writer END - START state difference or the scene it acts in, do the lens's address directions carry the writer's input, and are the lens's address coordinates causally used? Scenes = S872's (seed 87200, 48 scenes: nine pictures, all nine names as the menu, the first eight shown); END(5) (E) and START(p) (R)

Code: `code/s873_analyze.py`, `code/s873_transplant.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s873_analysis_s873_27b.json`, `data/s873_s873_27b_transplant.json`
