# S861

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S861 (registered in knowledge/DECISIONS.md before its data; user review 10-20 - three parts, judged separately) - does the S860 circuit act in natural runs, is h16's predecessor signal the one language 'before' uses, and where does k act on the units? One scene = nine pictures with all nine names as the menu; layouts show its first 7, 8 or 9 pictures (stated count = shown count). A (natural, no pins; patches at the ten read rows; donor = the same scene's START of the same ordinal, same layout): N7 END(4) and N9 END(5) (identity), N8 END(5) (target 4 = k - 1: the step helps), N8 END(3) (target 

Code: `code/s861_analyze.py`, `code/s861_verify.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_i.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s861_analysis_s861_27b.json`, `data/s861_s861_27b_verify.json`
