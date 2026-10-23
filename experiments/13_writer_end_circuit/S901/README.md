# S901

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S901 (registered in knowledge/DECISIONS.md before its data) - how the predecessor's context enters the anchor's representation. S889: the anchor picture's FA47 values carry the context of the picture before it, and FA47 passes it to the late query (K1: transplanting the anchor values of a hybrid whose predecessor differs removes 17% of the step). Which token-mixing route writes that context into the anchor's image rows by layer 47: DeltaNet recurrence or full attention among image tokens, and at which depth? Per (scene, pin): the HP hybrid H (picture p - 1 <- a photo of the scene's unshown nin

Code: `code/s901_analyze.py`, `code/s901_ctxroute.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s901_analysis_s901_27b.json`, `data/s901_s901_27b_ctxroute.json`
