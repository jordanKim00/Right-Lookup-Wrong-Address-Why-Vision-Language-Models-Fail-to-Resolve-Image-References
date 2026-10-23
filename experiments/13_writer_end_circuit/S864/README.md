# S864

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S864 (registered in knowledge/DECISIONS.md before its data; the advisor's 'core generalisation' 10-20, S861-B redone with pins - S861-B was invalid because the natural early anchors of BEF5 and END5 matched in only .375 of scenes; design revised after the four agents' critiques 10-21 00:2x, before any data) - operator identity of the writer's predecessor step. One scene = nine pictures with all nine names as the menu; layout = its first eight pictures (N 8, stated = shown), as S861-S863. Forms (ten question tokens each, rows align): END5 'What is in the fifth image from the last?' (target 4), 

Code: `code/s864_analyze.py`, `code/s864_opident.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s864_analysis_s864_27b.json`
