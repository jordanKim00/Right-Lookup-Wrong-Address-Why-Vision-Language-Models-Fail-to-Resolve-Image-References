# S772

Category: `06_read_interchange`. Route-complete read interchanges: later tokens read one token as in a donor question, through full attention, the DeltaNet state and its convolution.

From the main script's docstring:

> S772: faithfulness of the END circuit on new scenes, both image counts and a second END phrasing. Pre-registration in knowledge/DECISIONS.md (2026-10-14 01:1x) before any S772 data; analysis exp/s772_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count; phrasing A "from the last" / "from the first", phrasing B "from the end" / "from the start" (one-token frame words, the ordinal four tokens before). Per scene and k = 2..6: END(k), START(k) and START(k') with k' = N + 1 - k (the image END(k) asks for; START(k') = START(k) when k' = k). Arms per (scene, k)

Re-measured with the clean readout in: S783b (files tagged with the re-run name are kept here).

Code: `code/s772_analyze.py`, `code/s772_faith.py`, `code/s772_run.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s772_analysis.json`, `data/s772_analysis_bare.json`, `data/s772_s772_27b7A_faith.json`, `data/s772_s772_27b7B_faith.json`, `data/s772_s772_27b8A_faith.json`, `data/s772_s772_27b8B_faith.json`, `data/s772_s783f_27b7A_faith.json`, `data/s772_s783f_27b7B_faith.json`, `data/s772_s783f_27b8A_faith.json`, `data/s772_s783f_27b8B_faith.json`, `data/s772_sm_s772_faith.json`, `data/s772_sm_s772b_faith.json`
