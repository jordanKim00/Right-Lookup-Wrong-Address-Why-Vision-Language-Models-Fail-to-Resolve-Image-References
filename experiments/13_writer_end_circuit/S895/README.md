# S895

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S895 (registered in knowledge/DECISIONS.md before its data; after S887 (layer 46's outputs restore only a third of the self-address heads' anchor attention) and the post hoc finding that END pulls the self-address heads one block end-ward (anchor -> successor) despite the pinned address) - which upstream write makes FA47's self-address heads move end-ward? Direct path into FA47's read-row queries and gates only (as S887; residual stream unchanged): the residual entering layer 47 with a group of upstream outputs replaced by START's - PRE35 (the residual entering layer 35: everything before the 

Code: `code/s895_analyze.py`, `code/s895_qpathsweep.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s895_analysis_s895_27b.json`, `data/s895_s895_27b_qpathsweep.json`
