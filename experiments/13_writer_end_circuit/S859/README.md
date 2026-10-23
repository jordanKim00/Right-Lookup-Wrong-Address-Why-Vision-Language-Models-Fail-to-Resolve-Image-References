# S859

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S859 (descriptive anatomy; registered in knowledge/DECISIONS.md before its data) - which heads of the writer band (27B: DeltaNet 46 / 48 / 49, full attention 47, the four MLPs) write the end-relative late update, from which source tokens, and what each head's write does to the late lookup's block scores (FA 51 / 55 / 59 / 63) through the direct path. Passive: the exp/c_anat.py hooks only read (two passes per batch, logits of the two passes compared bit for bit). Scenes: eight real pictures and a menu of nine names; the same scene is also shown with its first six pictures only (N_a 6, same menu

Code: `code/s859_analyze.py`, `code/s859_anat.py`

Shared modules: `lib/c_anat.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s859_analysis_s859_27b.json`, `data/s859_s859_27b_anat.json`
