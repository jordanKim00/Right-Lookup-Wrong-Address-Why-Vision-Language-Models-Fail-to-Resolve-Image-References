# S896

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S896 (registered in knowledge/DECISIONS.md before its data; after the decomposition T1(E) = T1(R) + [T1(E_BANDR) - T1(R)] + W ~ -2.99 + 1.70 + 2.62 (S880 base): 39% of the END-START shift of the late address arises outside the writer band) - where is the non-band part: in the read rows' pre-band state or at the question / menu text positions that the late layers read? Residual row patches (exp/c_qpath_* 'rpatch': at the input of a layer, the sample's residual at the given positions <- START's) - read rows (rowsC) or text TXT (every position from the first token that differs between the prompts

Code: `code/s896_analyze.py`, `code/s896_nonband.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s896_analysis_s896_27b.json`, `data/s896_s896_27b_nonband.json`
