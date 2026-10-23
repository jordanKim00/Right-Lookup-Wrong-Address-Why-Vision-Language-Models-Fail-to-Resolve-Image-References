# S869

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S869 (registered in knowledge/DECISIONS.md before its data; advisor review 10-21 'compute why the same writer intervention flips some runs and not others') - the writer's flip condition per run: the answer moves from the anchor p to p - 1 when the writer's dose-scaled contribution exceeds the base margin. One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown). Pinned references at p in {3, 4, 5} (U part at 35-47 <- the natural START(p)): E = END(5), R = START(p) at its own address. Dose arms (exp/c_qpath_i 'cpatch_lin' on the band's e

Code: `code/s869_analyze.py`, `code/s869_dose.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s869_analysis_s869_27b.json`, `data/s869_s869_27b_dose.json`
