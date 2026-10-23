# S871

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S871 (registered in knowledge/DECISIONS.md before its data; advisor review 3, 10-21: 'can the pre-writer state predict each run's writer effect?') - State-Conditioned Writer Prediction, observation run. One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown). Pinned references at p in {3, 4, 5} (U part at 35-47 <- the natural START(p)): E(k) = END(k) for k in {3, 4, 5, 6} (the writer's step on), R = START(p) at its own address. Arm per E: E_BANDR (the band's eight component outputs at the ten read rows <- R; the writer's END part remov

Code: `code/s871_analyze.py`, `code/s871_writerstate.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s871_analysis_s871_27b.json`
