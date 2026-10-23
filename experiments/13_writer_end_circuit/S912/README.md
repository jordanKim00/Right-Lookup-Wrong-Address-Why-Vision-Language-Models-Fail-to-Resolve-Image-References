# S912

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S912 (registered in knowledge/DECISIONS.md before its data; adversarial review 10-22, item D3) - which writer components carry the identity collapse on fresh natural scenes, and why does N5 END(3) survive? S879: restoring START's band repairs the middle identity cells (.40 -> .96, .06 -> .94) while N5 END(3) keeps .875 with its anchor on the target; S899: N9 END(5) keeps the copy anchor on the target (.85) and is correct .07. The sub-mechanisms were measured only on S872's 48 reused pinned scenes. Natural runs (no pins) at one N per run with its middle identity cell k_id (N5 END(3), N7 END(4),

Code: `code/s912_analyze.py`, `code/s912_idrepair.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s912_analysis_s912n7_27b.json`, `data/s912_s912n5_27b_idrepair.json`, `data/s912_s912n7_27b_idrepair.json`, `data/s912_s912n9_27b_idrepair.json`
