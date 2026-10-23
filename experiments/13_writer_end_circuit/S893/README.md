# S893

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S893 (registered in knowledge/DECISIONS.md before its data; after S888: redirecting the anchor's content at the read rows through FA and DeltaNet layers 24-63 moves START's late block scores toward f (L63 up to 78% of the gap) but never its answer; S875: the readout's menu-word path (.44) outweighs its image path (.28)) - does the menu-word route hold START's answer? Late readout edit at FA51 / 55 / 59 / 63 (read rows, float32 recompute, pins not involved, keys and values unchanged; exp/s891_attnpredict.FAAttnEdit): every head's attention mass on the nine menu-name tokens (NOMENU) or on all pi

Code: `code/s893_analyze.py`, `code/s893_readroute.py`

Shared modules: `lib/c_dnedit.py`, `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s893_analysis_s893_27b.json`, `data/s893_s893_27b_readroute.json`
