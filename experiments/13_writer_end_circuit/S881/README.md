# S881

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S881 (registered in knowledge/DECISIONS.md before its data; S878 post hoc p20: the writer's late query update = a shared front-ward shift + a component along the anchor picture's own key that sets the per-scene magnitude, r .72-.84) - causal test: remove only that component. Scenes = S872's (seed 87200, 48); END(5) (E), START(p) (R), E_BANDR (the band's read-row outputs <- R) pinned at p in {3, 4, 5}. At every late full-attention layer (51, 55, 59, 63) a forward hook recomputes the read rows' attention (rows qm, B0-B4, A4; float32, q_norm / k_norm, partial RoPE, GQA, output gate, o_proj - as e

Code: `code/s881_analyze.py`, `code/s881_anchorrm.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s881_analysis_s881_27b.json`
