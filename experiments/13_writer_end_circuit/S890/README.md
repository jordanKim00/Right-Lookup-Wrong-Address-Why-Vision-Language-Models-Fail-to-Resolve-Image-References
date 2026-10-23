# S890

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S890 (registered in knowledge/DECISIONS.md before its data; after S887: MLP46's END write, read through FA47's read-row queries and gates, carries .69 of FA47's END part, but recovers only 16% of the self-address heads' anchor attention) - which part of that path does the work: the queries or the output gates, and in which FA47 heads? Same path patch as S887 (exp/s887_qpath46.py FA47QueryPath; residual entering layer 47 with the layer-46 MLP output <- START's), now restricted to the queries (Q) or the gates (G) and to head groups: self-address {1, 4, 12, 17, 20}, anti-self {0, 2, 5, 13, 15}, h

Code: `code/s890_analyze.py`, `code/s890_qpathsplit.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s890_analysis_s890_27b.json`, `data/s890_s890_27b_qpathsplit.json`
