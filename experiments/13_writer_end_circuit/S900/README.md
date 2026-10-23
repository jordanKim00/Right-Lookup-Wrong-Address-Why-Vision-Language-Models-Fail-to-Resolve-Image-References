# S900

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S900 (registered in knowledge/DECISIONS.md before its data) - two dissections in one pass. (A) h3 (S897: MLP46's path into FA47 runs through h3 .22 and h14 .19; h3 is a copy-type head that reads the anchor instead of the text under END - why does that add to the step?): MLP46 -> h3 path through h3's queries only (QMq_h3) or gate only (QMg_h3), and a context transplant for h3 alone (h3 reads the anchor's FA47 values from the S889 hybrid whose predecessor differs; HPV_h3) against the same transplant for all heads (HPV_all, S889's HP). (B) Readout (S893: the late image and menu routes substitute 

Code: `code/s900_analyze.py`, `code/s900_h3readout.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s900_analysis_s900_27b.json`, `data/s900_s900_27b_h3readout.json`
