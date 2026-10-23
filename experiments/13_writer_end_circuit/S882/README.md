# S882

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S882 (registered in knowledge/DECISIONS.md before its data; post hoc synthesis of S872 arms and S878: the writer = h16's scene-constant front-ward shift + the anchor-checking heads (self-address, anti-self) and MLP46 whose END part varies across scenes) - is that core circuit sufficient for the writer's per-scene address effect? Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R) pinned at p in {3, 4, 5}. Arms (the band's read-row outputs; everything <- R except the kept parts, which keep E's own outputs): E_BANDR all band outputs <- R (the writer removed; reference for W); KEEP_CORE ML

Code: `code/s882_analyze.py`, `code/s882_core.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s882_analysis_s882_27b.json`, `data/s882_s882_27b_core.json`
