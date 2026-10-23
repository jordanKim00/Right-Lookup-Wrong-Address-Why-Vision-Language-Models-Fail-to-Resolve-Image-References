# S915

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S915 collector (registered in knowledge/DECISIONS.md before the data; the advisor's sufficiency demand, 10-22) - is the early-address profile a sufficient state for the later computation? Hold the early address fixed and swap the remaining hidden state. Natural runs at one N: END(k) for every k ('from the last'; with --forms last,end also 'from the end'), START(q) for every q; every natural run's L35-47 read-row queries are kept as donors. Pinned runs replace every head's read-row query at the pinned layers by the donor's (S829 qsub part='all'; the picture keys are the same in every prompt of 

Code: `code/s915_analyze.py`, `code/s915_cross.py`

Shared modules: `lib/c_qpath_k.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s915_analysis_s915A_27b.json`, `data/s915_s915smA2_27b_cross.json`, `data/s915_s915smA3_27b_cross.json`, `data/s915_s915smB3_27b_cross.json`
