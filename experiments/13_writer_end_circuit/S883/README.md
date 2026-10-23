# S883

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S883 (registered in knowledge/DECISIONS.md before its data; after S882: the anchor-checking FA47 heads carry much of the per-scene writer variation but the band acts jointly) - where in those heads is their END part: in what they read (attention pattern and values) or in their output gate? Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R) pinned at p in {3, 4, 5}. Per (scene, pin) one pass of 12 runs: E, R, E_BANDR and, for the head groups ANCH = self-address + anti-self heads (1, 4, 12, 17, 20, 0, 2, 5, 13, 15) and H16 (16), at FA47's read rows (exp/c_headpatch.py FAHeadPatchPinned, 

Code: `code/s883_analyze.py`, `code/s883_readgate.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s883_analysis_s883_27b.json`
