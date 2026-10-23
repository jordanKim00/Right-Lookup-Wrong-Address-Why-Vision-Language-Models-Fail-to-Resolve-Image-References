# S879

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S879 (registered in knowledge/DECISIONS.md before its data; advisor review 5 'boundary-route reinforcement': why N 5 END(3) is answered .88 while N 7 END(4) breaks) - in the middle identity cells (target j* = N + 1 - k = k) the copy route (ordinal k) and the end route (N - (N - k)) point at the same picture; does the end route's support protect the middle identity from the writer's END step, and is that protection specific to small N? One scene = nine pictures with all nine names as the menu; layouts show the first N pictures (N 5, 7, 9; stated = shown); cells END(3) at N 5, END(4) at N 7, END

Code: `code/s879_analyze.py`, `code/s879_reinforce.py`

Shared modules: `lib/c_qpath_k.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s879_analysis_s879_27b.json`, `data/s879_s879_27b_reinforce.json`
