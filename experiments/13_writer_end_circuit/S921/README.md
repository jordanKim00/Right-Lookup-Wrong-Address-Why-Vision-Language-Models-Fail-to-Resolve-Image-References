# S921

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S921 collector (registered in knowledge/DECISIONS.md before the data; advisor review 3: S917 left the late menu / picture division open because removing the menu reads disturbed the writer-free run) - which late read route carries the writer's effect on the answer? Position-restricted swaps of the late full-attention layers' read-row outputs (exp/c_routeswap.py: every head's gated output part over the picture tokens, the menu-name tokens or the remaining positions <- the other run's; the three parts add up to S917's full swap), which leave the answer machinery intact. Natural runs at N8 (fresh

Code: `code/s921_analyze.py`, `code/s921_routesplit.py`

Shared modules: `lib/c_qpath_k.py`, `lib/c_routeswap.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s921_analysis_s921_27b.json`, `data/s921_s921_27b_routesplit.json`
