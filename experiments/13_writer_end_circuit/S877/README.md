# S877

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S877 (registered in knowledge/DECISIONS.md before its data; follows S873 A1 'the writer magnitude follows the context scene' and the post hoc p19 'the context dependence is already present at L51') - which part of the context sets the per-scene writer magnitude: the band's processing of the scene's pictures (layers 46-49 read the image positions through FA47 and the DeltaNet memory) or the late lookup's key geometry (L51 on)? Scenes = S872's (seed 87200, 48); END(5) (E) / START(p) (R) pinned at p in {3, 4, 5} as S872 / S873; donors d = c + 1, c + 17 (mod 48). The prefix = every position before

Code: `code/s877_analyze.py`, `code/s877_context.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s877_analysis_s877_27b.json`, `data/s877_s877_27b_context.json`
