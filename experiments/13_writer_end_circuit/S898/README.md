# S898

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S898 (registered in knowledge/DECISIONS.md before its data; after S895: information written into the read rows before layer 35 restores 70% of the self-address heads' anchor attention through FA47's queries) - where within layers 0-34 is it written? Same direct path into FA47's read-row queries and gates (residual stream unchanged): the residual entering layer 47 with a group of upstream read-row writes replaced by START's - PRE20 (the residual entering layer 20), L20_25 (outputs of layers 20-25), L26 (the ordinal layer), L27_30 and L31_34 (the frame-read band), PRE35 (S895's arm, replication)

Code: `code/s898_analyze.py`, `code/s898_qpathpre.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s898_analysis_s898_27b.json`, `data/s898_s898_27b_qpathpre.json`
