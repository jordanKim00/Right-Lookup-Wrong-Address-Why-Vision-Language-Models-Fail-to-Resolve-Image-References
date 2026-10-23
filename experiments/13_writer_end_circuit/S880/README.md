# S880

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S880 (registered in knowledge/DECISIONS.md before its data; follows S873 A1 'the writer magnitude follows the context scene') - which pictures of the scene set the writer's magnitude? Hybrid scenes at the input: scene c with one picture slot b replaced by a photo of c's unshown ninth shape (the name pool has nine shapes and every scene shows eight of them; the slot's old shape becomes the unshown one, the menu of nine names is unchanged), everything else c's. Scenes = S872's (seed 87200, 48; nine pictures, first eight shown, N 8); END(5) (E) and START(p) (R) pinned at p in {3, 4, 5} with each 

Code: `code/s880_analyze.py`, `code/s880_slotswap.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s880_analysis_s880_27b.json`, `data/s880_s880_27b_slotswap.json`
